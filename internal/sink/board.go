package sink

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

// boardMCPClient posts to a board MCP server via Streamable HTTP JSON-RPC.
// The board (rendezvous) is stateful: initialize must be called first to
// obtain an Mcp-Session-Id, which is then sent on every subsequent call.
type boardMCPClient struct {
	baseURL   string
	token     string
	sessionID string
	mu        sync.Mutex
	hc        *http.Client
}

func newBoardMCPClient(baseURL, token string) *boardMCPClient {
	return &boardMCPClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		token:   token,
		hc:      &http.Client{Timeout: 15 * time.Second},
	}
}

func (c *boardMCPClient) sessionHeader() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sessionID
}

func (c *boardMCPClient) setSession(id string) {
	c.mu.Lock()
	c.sessionID = id
	c.mu.Unlock()
}

// initialize sends the MCP initialize handshake and stores the session ID.
func (c *boardMCPClient) initialize(ctx context.Context) error {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      0,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{},
			"clientInfo":      map[string]any{"name": "ai-sre-sink", "version": "1.0"},
		},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return fmt.Errorf("board MCP: initialize: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("board MCP: initialize: HTTP %d: %s", resp.StatusCode, raw)
	}
	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		c.setSession(sid)
		slog.Info("board sink: session initialized", "session", sid)
	}
	return nil
}

func (c *boardMCPClient) createThread(ctx context.Context, title, body string) (string, error) {
	text, err := c.call(ctx, "create_thread", map[string]any{
		"title": title,
		"body":  body,
	})
	if err != nil {
		return "", err
	}
	// content[0].text contains JSON: {"thread_id": "..."}
	var result struct {
		ThreadID string `json:"thread_id"`
	}
	if jsonErr := json.Unmarshal([]byte(text), &result); jsonErr == nil && result.ThreadID != "" {
		return result.ThreadID, nil
	}
	// Fallback: bare string returned directly.
	return strings.TrimSpace(text), nil
}

func (c *boardMCPClient) reply(ctx context.Context, threadID, body string) error {
	_, err := c.call(ctx, "reply", map[string]any{
		"thread_id": threadID,
		"body":      body,
	})
	return err
}

// call invokes a board tool, initializing the session on the first call or after
// a session error, then retrying once.
func (c *boardMCPClient) call(ctx context.Context, tool string, args map[string]any) (string, error) {
	if c.sessionHeader() == "" {
		if err := c.initialize(ctx); err != nil {
			return "", err
		}
	}
	out, err := c.callOnce(ctx, tool, args)
	if err != nil && isBoardSessionError(err) {
		slog.Info("board sink: session error, re-initializing", "err", err)
		if initErr := c.initialize(ctx); initErr != nil {
			return "", initErr
		}
		return c.callOnce(ctx, tool, args)
	}
	return out, err
}

func (c *boardMCPClient) callOnce(ctx context.Context, tool string, args map[string]any) (string, error) {
	reqBody, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params":  map[string]any{"name": tool, "arguments": args},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	if sid := c.sessionHeader(); sid != "" {
		req.Header.Set("Mcp-Session-Id", sid)
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return "", fmt.Errorf("board MCP: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("board MCP: HTTP %d: %s", resp.StatusCode, raw)
	}

	// Handle SSE responses.
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		raw = extractBoardSSEData(raw)
	}

	var r struct {
		Result *struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return string(raw), nil
	}
	if r.Error != nil {
		return "", fmt.Errorf("board MCP: %s", r.Error.Message)
	}
	if r.Result == nil {
		return string(raw), nil
	}
	if r.Result.IsError {
		var parts []string
		for _, c := range r.Result.Content {
			parts = append(parts, c.Text)
		}
		return "", fmt.Errorf("board MCP tool error: %s", strings.Join(parts, "; "))
	}
	for _, c := range r.Result.Content {
		if c.Type == "text" && c.Text != "" {
			return c.Text, nil
		}
	}
	return "", nil
}

func isBoardSessionError(err error) bool {
	s := err.Error()
	return strings.Contains(s, "HTTP 404") ||
		(strings.Contains(s, "HTTP 400") && strings.Contains(s, "session"))
}

func extractBoardSSEData(body []byte) []byte {
	var last []byte
	for _, line := range bytes.Split(body, []byte("\n")) {
		if bytes.HasPrefix(line, []byte("data: ")) {
			last = bytes.TrimPrefix(line, []byte("data: "))
		}
	}
	return last
}
