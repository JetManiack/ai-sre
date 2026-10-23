// Package tools provides ToolDispatcher implementations for the investigator.
//
// The main implementation routes calls through MCP (Model Context Protocol)
// servers over Streamable HTTP. This keeps the agent decoupled from specific
// backends: k8s, VictoriaMetrics, VictoriaLogs, etc. are all MCP servers.
package tools

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

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/investigate"
)

// mcpClient calls a single MCP server over Streamable HTTP.
// Stateful servers (e.g. mcp-victoriametrics) require an MCP session:
// initialize → Mcp-Session-Id header on every subsequent call.
type mcpClient struct {
	name      string
	baseURL   string
	token     string
	tools     map[string]bool // nil = unknown (before listTools)
	defs      []mcpToolDef    // full tool definitions from tools/list
	sessionID string          // empty = stateless
	mu        sync.Mutex      // protects sessionID
	hc        *http.Client
}

func newMCPClient(srv config.MCPServer) *mcpClient {
	return &mcpClient{
		name:    srv.Name,
		baseURL: strings.TrimRight(srv.URL, "/"),
		token:   srv.Token,
		hc:      &http.Client{Timeout: 60 * time.Second},
	}
}

func (c *mcpClient) handles(tool string) bool {
	if c.tools == nil {
		return false
	}
	return c.tools[tool]
}

// sessionHeader returns the Mcp-Session-Id header value, or "" if stateless.
func (c *mcpClient) sessionHeader() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sessionID
}

// setSession stores the session ID from a server response.
func (c *mcpClient) setSession(id string) {
	c.mu.Lock()
	c.sessionID = id
	c.mu.Unlock()
}

// addAuthHeaders adds Authorization and Mcp-Session-Id headers.
func (c *mcpClient) addAuthHeaders(req *http.Request) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	if sid := c.sessionHeader(); sid != "" {
		req.Header.Set("Mcp-Session-Id", sid)
	}
}

// initialize sends the MCP initialize handshake and stores the session ID.
// Must be called before tools/list and tools/call on stateful servers.
func (c *mcpClient) initialize(ctx context.Context) error {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      0,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{},
			"clientInfo":      map[string]any{"name": "ai-sre", "version": "1.0"},
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
		return fmt.Errorf("mcp %s: initialize: %w", c.name, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("mcp %s: initialize: HTTP %d: %s", c.name, resp.StatusCode, raw)
	}
	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		c.setSession(sid)
		slog.Info("mcp server: session initialized", "server", c.name, "session", sid)
		// Fire-and-forget notifications/initialized.
		go func() {
			nctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			c.sendNotification(nctx, "notifications/initialized")
		}()
	}
	return nil
}

// sendNotification sends a JSON-RPC notification (no response expected).
func (c *mcpClient) sendNotification(ctx context.Context, method string) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
		"params":  map[string]any{},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	c.addAuthHeaders(req)
	resp, err := c.hc.Do(req)
	if err == nil {
		resp.Body.Close()
	}
}

// call invokes tools/call on the MCP server and returns the text result.
// On a session error (404 / 400), it re-initializes and retries once.
func (c *mcpClient) call(ctx context.Context, tool string, args json.RawMessage) (string, error) {
	out, err := c.callOnce(ctx, tool, args)
	if err != nil && isSessionError(err) {
		slog.Info("mcp: session error, re-initializing", "server", c.name, "err", err)
		if initErr := c.initialize(ctx); initErr == nil {
			return c.callOnce(ctx, tool, args)
		}
	}
	return out, err
}

func (c *mcpClient) callOnce(ctx context.Context, tool string, args json.RawMessage) (string, error) {
	body, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      tool,
			"arguments": args,
		},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	c.addAuthHeaders(req)

	resp, err := c.hc.Do(req)
	if err != nil {
		return "", fmt.Errorf("mcp %s: %w", c.name, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("mcp %s: HTTP %d: %s", c.name, resp.StatusCode, raw)
	}

	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/event-stream") {
		raw = extractSSEData(raw)
	}
	return parseMCPResult(raw)
}

// isSessionError returns true if the error looks like a session-not-found response.
func isSessionError(err error) bool {
	s := err.Error()
	return strings.Contains(s, "HTTP 404") ||
		(strings.Contains(s, "HTTP 400") && strings.Contains(s, "session"))
}

// extractSSEData returns the JSON payload from the last non-empty "data: " line.
func extractSSEData(body []byte) []byte {
	var last []byte
	for _, line := range bytes.Split(body, []byte("\n")) {
		if bytes.HasPrefix(line, []byte("data: ")) {
			last = bytes.TrimPrefix(line, []byte("data: "))
		}
	}
	return last
}

// parseMCPResult extracts the text content from a JSON-RPC response.
func parseMCPResult(raw []byte) (string, error) {
	var r struct {
		Result *struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return string(raw), nil
	}
	if r.Error != nil {
		return "", fmt.Errorf("mcp error: %s", r.Error.Message)
	}
	if r.Result == nil {
		return string(raw), nil
	}
	if r.Result.IsError {
		var texts []string
		for _, c := range r.Result.Content {
			texts = append(texts, c.Text)
		}
		return "", fmt.Errorf("tool error: %s", strings.Join(texts, "; "))
	}
	var parts []string
	for _, c := range r.Result.Content {
		if c.Type == "text" && c.Text != "" {
			parts = append(parts, c.Text)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// mcpToolDef is a tool definition as returned by tools/list.
type mcpToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// listTools calls tools/list on the MCP server.
// For stateful servers (no tools returned without a session), it initializes
// a session first and retries.
func (c *mcpClient) listTools(ctx context.Context) error {
	if err := c.listToolsOnce(ctx); err != nil {
		return err
	}
	// If no tools returned, the server may be stateful — init session and retry.
	if len(c.tools) == 0 {
		slog.Info("mcp: no tools without session, trying initialize", "server", c.name)
		if err := c.initialize(ctx); err != nil {
			slog.Warn("mcp: initialize failed", "server", c.name, "err", err)
			return nil // accept stateless with no tools
		}
		return c.listToolsOnce(ctx)
	}
	return nil
}

func (c *mcpClient) listToolsOnce(ctx context.Context) error {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/list",
		"params":  map[string]any{},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	c.addAuthHeaders(req)

	resp, err := c.hc.Do(req)
	if err != nil {
		return fmt.Errorf("mcp %s: tools/list: %w", c.name, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 256*1024))

	// Handle SSE responses from some servers.
	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/event-stream") {
		raw = extractSSEData(raw)
	}

	var r struct {
		Result *struct {
			Tools []mcpToolDef `json:"tools"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || r.Result == nil {
		return nil // treat as empty, not an error
	}
	c.tools = make(map[string]bool, len(r.Result.Tools))
	c.defs = r.Result.Tools
	for _, t := range r.Result.Tools {
		c.tools[t.Name] = true
	}
	if len(c.tools) > 0 {
		slog.Info("mcp server: tools listed", "server", c.name, "count", len(c.tools))
	}
	return nil
}

// Dispatcher routes ToolCall to the appropriate MCP server.
// memory_recall is served locally from the MemoryStore.
// All other tools are served by MCP servers discovered via tools/list.
type Dispatcher struct {
	servers []*mcpClient
	memory  investigate.MemoryStore
}

// NewDispatcher creates a Dispatcher from config.
// It initializes MCP sessions and calls tools/list on each server.
func NewDispatcher(ctx context.Context, servers []config.MCPServer, memory investigate.MemoryStore) (*Dispatcher, error) {
	d := &Dispatcher{memory: memory}
	for _, srv := range servers {
		c := newMCPClient(srv)
		if err := c.listTools(ctx); err != nil {
			slog.Warn("mcp: tools/list failed, server skipped", "server", srv.Name, "err", err)
			continue
		}
		// Apply explicit tool filter if configured (empty = all tools).
		if len(srv.Tools) > 0 {
			filtered := make(map[string]bool, len(srv.Tools))
			var filteredDefs []mcpToolDef
			for _, t := range srv.Tools {
				if c.tools[t] {
					filtered[t] = true
				}
			}
			for _, def := range c.defs {
				if filtered[def.Name] {
					filteredDefs = append(filteredDefs, def)
				}
			}
			c.tools = filtered
			c.defs = filteredDefs
		}
		d.servers = append(d.servers, c)
	}
	return d, nil
}

// DiscoveredTools returns all tools discovered from MCP servers.
// The agent offers exactly these tools to the LLM — no hardcoded additions.
func (d *Dispatcher) DiscoveredTools() []investigate.Tool {
	var out []investigate.Tool
	for _, s := range d.servers {
		for _, def := range s.defs {
			schema := def.InputSchema
			if len(schema) == 0 {
				schema = []byte(`{"type":"object","properties":{}}`)
			}
			out = append(out, investigate.Tool{
				Name:        def.Name,
				Description: def.Description,
				InputSchema: schema,
			})
		}
	}
	return out
}

// Dispatch routes a ToolCall to the right MCP server, or handles memory_recall locally.
func (d *Dispatcher) Dispatch(ctx context.Context, call investigate.ToolCall) (string, error) {
	if call.Name == "memory_recall" {
		return d.dispatchMemoryRecall(call.Input)
	}
	for _, s := range d.servers {
		if s.handles(call.Name) {
			return s.call(ctx, call.Name, call.Input)
		}
	}
	return "", fmt.Errorf("tools dispatcher: no server handles tool %q", call.Name)
}

func (d *Dispatcher) dispatchMemoryRecall(input json.RawMessage) (string, error) {
	var p struct {
		Workload  string `json:"workload"`
		Namespace string `json:"namespace"`
		Limit     int    `json:"limit"`
	}
	if err := json.Unmarshal(input, &p); err != nil {
		return "", fmt.Errorf("memory_recall: parse input: %w", err)
	}
	if p.Limit == 0 {
		p.Limit = 5
	}
	invs, err := d.memory.RecallInvestigations(p.Workload, p.Namespace, p.Limit)
	if err != nil {
		return "[]", nil // non-fatal
	}
	b, _ := json.Marshal(invs)
	return string(b), nil
}
