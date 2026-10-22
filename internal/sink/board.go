package sink

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// boardMCPClient posts to a board MCP server via JSON-RPC.
type boardMCPClient struct {
	baseURL string
	token   string
	hc      *http.Client
}

func newBoardMCPClient(baseURL, token string) *boardMCPClient {
	return &boardMCPClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		token:   token,
		hc:      &http.Client{Timeout: 15 * time.Second},
	}
}

func (c *boardMCPClient) createThread(ctx context.Context, title, body string) (string, error) {
	return c.callString(ctx, "create_thread", map[string]any{
		"title": title,
		"body":  body,
	})
}

func (c *boardMCPClient) reply(ctx context.Context, threadID, body string) error {
	_, err := c.callString(ctx, "reply", map[string]any{
		"thread_id": threadID,
		"body":      body,
	})
	return err
}

func (c *boardMCPClient) callString(ctx context.Context, method string, params map[string]any) (string, error) {
	reqBody, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params":  map[string]any{"name": method, "arguments": params},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
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
	var r struct {
		Result *struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
		Error *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return string(raw), nil
	}
	if r.Error != nil {
		return "", fmt.Errorf("board MCP: %s", r.Error.Message)
	}
	if r.Result != nil && len(r.Result.Content) > 0 {
		return r.Result.Content[0].Text, nil
	}
	return "", nil
}
