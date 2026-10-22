// Package tools provides ToolDispatcher implementations for the investigator.
//
// The main implementation routes calls through MCP (Model Context Protocol)
// servers over Streamable HTTP. This keeps the agent decoupled from specific
// backends: k8s, Loki, Thanos, incident.io, etc. are all MCP servers.
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
	"time"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/investigate"
)

// mcpClient calls a single MCP server over Streamable HTTP.
type mcpClient struct {
	name    string
	baseURL string
	token   string
	tools   map[string]bool // nil = all allowed
	hc      *http.Client
}

func newMCPClient(srv config.MCPServer) *mcpClient {
	c := &mcpClient{
		name:    srv.Name,
		baseURL: strings.TrimRight(srv.URL, "/"),
		token:   srv.Token,
		hc:      &http.Client{Timeout: 60 * time.Second},
	}
	if len(srv.Tools) > 0 {
		c.tools = make(map[string]bool, len(srv.Tools))
		for _, t := range srv.Tools {
			c.tools[t] = true
		}
	}
	return c
}

func (c *mcpClient) handles(tool string) bool {
	if c.tools == nil {
		return false // unknown until we list; handled in Dispatcher
	}
	return c.tools[tool]
}

// call invokes tools/call on the MCP server and returns the text result.
func (c *mcpClient) call(ctx context.Context, tool string, args json.RawMessage) (string, error) {
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
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

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

	// Handle SSE (text/event-stream) response: extract last "data: {...}" line.
	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/event-stream") {
		raw = extractSSEData(raw)
	}

	return parseMCPResult(raw)
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
		// Not JSON-RPC — return raw string (some servers return plain text).
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

// listTools calls tools/list on the MCP server and populates c.tools.
func (c *mcpClient) listTools(ctx context.Context) error {
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
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return fmt.Errorf("mcp %s listTools: %w", c.name, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))

	var r struct {
		Result *struct {
			Tools []struct{ Name string } `json:"tools"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || r.Result == nil {
		return nil // treat as ok, just can't route to this server
	}
	c.tools = make(map[string]bool, len(r.Result.Tools))
	for _, t := range r.Result.Tools {
		c.tools[t.Name] = true
	}
	slog.Info("mcp server: tools listed", "server", c.name, "count", len(c.tools))
	return nil
}

// Dispatcher routes ToolCall to the appropriate MCP server.
// memory_recall is served locally from the MemoryStore.
type Dispatcher struct {
	servers []*mcpClient
	memory  investigate.MemoryStore
}

// NewDispatcher creates a Dispatcher from config.
// It calls tools/list on each server that has no explicit tool list.
func NewDispatcher(ctx context.Context, servers []config.MCPServer, memory investigate.MemoryStore) (*Dispatcher, error) {
	d := &Dispatcher{memory: memory}
	for _, srv := range servers {
		c := newMCPClient(srv)
		if c.tools == nil {
			// Discover tools from the server.
			if err := c.listTools(ctx); err != nil {
				slog.Warn("mcp: tools/list failed, server skipped", "server", srv.Name, "err", err)
				continue
			}
		}
		d.servers = append(d.servers, c)
	}
	return d, nil
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

// filterSources returns sources matching any of the given kinds.
func filterSources(sources []config.Source, kinds ...string) []config.Source {
	set := make(map[string]bool, len(kinds))
	for _, k := range kinds {
		set[k] = true
	}
	var out []config.Source
	for _, s := range sources {
		if set[s.Kind] {
			out = append(out, s)
		}
	}
	return out
}
