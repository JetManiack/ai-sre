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
	"regexp"
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
	// Streamable HTTP requires both types; some servers (kubernetes-mcp-server)
	// return 400 when only application/json is present.
	req.Header.Set("Accept", "application/json, text/event-stream")
	c.addAuthHeaders(req)

	resp, err := c.hc.Do(req)
	if err != nil {
		return fmt.Errorf("mcp %s: tools/list: %w", c.name, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 256*1024))

	if resp.StatusCode != http.StatusOK {
		slog.Warn("mcp: tools/list non-200", "server", c.name, "status", resp.StatusCode, "body", string(raw[:min(len(raw), 200)]))
		return nil // non-fatal: server stays excluded from routing
	}

	// Handle SSE responses.
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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
// Tool names are prefixed with the server name to avoid collisions when multiple
// servers expose identically-named tools (e.g. both victoriametrics and
// victorialogs export "query"). The prefix format is "<server>__<tool>".
// The agent offers exactly these tools to the LLM — no hardcoded additions.
func (d *Dispatcher) DiscoveredTools() []investigate.Tool {
	var out []investigate.Tool
	// Track bare names to detect collisions.
	seen := map[string]int{}
	for _, s := range d.servers {
		for _, def := range s.defs {
			seen[def.Name]++
		}
	}
	for _, s := range d.servers {
		for _, def := range s.defs {
			schema := def.InputSchema
			if len(schema) == 0 {
				schema = []byte(`{"type":"object","properties":{}}`)
			}
			name := def.Name
			// Prefix only when the bare name collides across servers.
			if seen[def.Name] > 1 {
				name = s.name + "__" + def.Name
			}
			out = append(out, investigate.Tool{
				Name:        name,
				Description: def.Description,
				InputSchema: schema,
			})
		}
	}
	return out
}

// Dispatch routes a ToolCall to the right MCP server, or handles memory_recall locally.
// If the tool name contains a server prefix (<server>__<tool>), the prefix is stripped
// and the call is routed to that specific server.
func (d *Dispatcher) Dispatch(ctx context.Context, call investigate.ToolCall) (string, error) {
	if call.Name == "memory_recall" {
		return d.dispatchMemoryRecall(call.Input)
	}
	// Handle prefixed names: "metrics__query" → server "metrics", tool "query".
	serverName, toolName := splitToolName(call.Name)
	if serverName != "" {
		for _, s := range d.servers {
			if s.name == serverName && s.handles(toolName) {
				if toolName == "list_documents" {
					return d.dispatchListDocuments(ctx, s, toolName, call.WorkloadHint, call.AlertHint, call.Input)
				}
				return s.call(ctx, toolName, call.Input)
			}
		}
		return "", fmt.Errorf("tools dispatcher: server %q does not handle tool %q", serverName, toolName)
	}
	for _, s := range d.servers {
		if s.handles(call.Name) {
			if call.Name == "list_documents" {
				return d.dispatchListDocuments(ctx, s, call.Name, call.WorkloadHint, call.AlertHint, call.Input)
			}
			return s.call(ctx, call.Name, call.Input)
		}
	}
	return "", fmt.Errorf("tools dispatcher: no server handles tool %q", call.Name)
}

// splitToolName splits "server__tool" into ("server", "tool").
// Returns ("", name) if there is no prefix.
func splitToolName(name string) (server, tool string) {
	if idx := strings.Index(name, "__"); idx > 0 {
		return name[:idx], name[idx+2:]
	}
	return "", name
}

// reIdentifier matches tokens that look like code identifiers: dotted paths,
// CamelCase, snake_case, or all-caps abbreviations. These confuse Outline's
// AND-based FTS and should be stripped on retry.
var reIdentifier = regexp.MustCompile(`\b(?:[A-Z][a-z]+(?:[A-Z][a-z]*)+|[a-zA-Z]+[._][a-zA-Z][a-zA-Z0-9._]*|[A-Z]{3,})\b`)

// reDanglingDot strips residual ".word" fragments left after identifier removal
// (e.g. "RealmEntity.components" → " .components" after CamelCase strip).
var reDanglingDot = regexp.MustCompile(`\s*\.[a-zA-Z][a-zA-Z0-9_]*`)

// listDocumentsIsEmpty reports whether an Outline MCP list_documents response
// contains no document results. The Outline MCP server returns a bare JSON
// array of {document:{id:…},context:…} objects, so an empty result is simply
// a JSON value with no "id" substring. We check for the absence of "id" rather
// than for a specific wrapper key to stay robust against Outline API changes.
func listDocumentsIsEmpty(out string) bool {
	trimmed := strings.TrimSpace(out)
	if trimmed == "" || trimmed == "null" || trimmed == "[]" || trimmed == "{}" {
		return true
	}
	return !strings.Contains(out, `"id"`)
}

// reBoilerplate matches common boilerplate phrases that appear in ai-sre
// alert summaries but carry no KB search signal.
var reBoilerplate = regexp.MustCompile(`(?i)\b(what is the root cause|investigate|check|please|alert|workload|namespace|cluster|evidence|summary)\b`)

// reSentenceSep splits alert summary text into fragments at punctuation
// boundaries that typically delimit semantic clauses.
var reSentenceSep = regexp.MustCompile(`[;:?.!\n]+`)

// alertHintFragments splits the alert summary into fragments, strips
// identifiers, drops boilerplate and very short fragments, deduplicates
// against existing candidates, and returns the survivors sorted
// longest-first. This gives a targeted sub-sentence (e.g.
// "Cannot lazily initialize collection of role") before a whole sentence
// that would AND every word against Outline's FTS and always return empty.
func alertHintFragments(hint, originalQuery, strippedQuery string) []string {
	if hint == "" {
		return nil
	}
	parts := reSentenceSep.Split(hint, -1)
	seen := map[string]bool{originalQuery: true, strippedQuery: true, "": true}

	type frag struct {
		text  string
		words int
	}
	var frags []frag
	for _, p := range parts {
		s := reIdentifier.ReplaceAllString(p, " ")
		s = reDanglingDot.ReplaceAllString(s, " ")
		s = reBoilerplate.ReplaceAllString(s, " ")
		s = strings.Join(strings.Fields(s), " ")
		if len(strings.Fields(s)) < 3 || seen[s] {
			continue
		}
		seen[s] = true
		frags = append(frags, frag{s, len(strings.Fields(s))})
	}
	// Sort longest-first so the most specific fragment is tried first.
	for i := 0; i < len(frags); i++ {
		for j := i + 1; j < len(frags); j++ {
			if frags[j].words > frags[i].words {
				frags[i], frags[j] = frags[j], frags[i]
			}
		}
	}
	out := make([]string, 0, len(frags))
	for _, f := range frags {
		out = append(out, f.text)
	}
	return out
}

// kbQueryFallbacks generates progressively broader query strings:
//  0. Alert summary with identifiers stripped (if non-empty and different from query).
//  1. Original query with identifier tokens removed.
//  2. First half of remaining words.
//  3. workload + first word (if workload is known and different from step 2).
//  4. First word only.
func kbQueryFallbacks(query, workload, alertHint string) []string {
	noIdents := reIdentifier.ReplaceAllString(query, " ")
	noIdents = reDanglingDot.ReplaceAllString(noIdents, " ")
	noIdents = strings.TrimSpace(noIdents)
	// Collapse multiple spaces.
	noIdents = strings.Join(strings.Fields(noIdents), " ")

	words := strings.Fields(noIdents)
	var out []string

	// Step 0: alert summary split into fragments at punctuation (:;.?\n),
	// stripped of identifiers, sorted longest-first, tried in order.
	// This gives "Cannot lazily initialize collection of role" before the
	// long whole-sentence that ANDs every word and always returns empty.
	for _, frag := range alertHintFragments(alertHint, query, noIdents) {
		out = append(out, frag)
	}

	if noIdents != query && noIdents != "" {
		out = append(out, noIdents)
	}
	if len(words) > 2 {
		half := strings.Join(words[:len(words)/2+1], " ")
		if half != noIdents {
			out = append(out, half)
		}
	}
	// Prefer "workload firstWord" over bare "firstWord" when the workload name
	// is known and different — avoids overly generic single-word queries.
	if len(words) > 0 {
		firstWord := words[0]
		if workload != "" && workload != firstWord {
			candidate := workload + " " + firstWord
			if (len(out) == 0 || out[len(out)-1] != candidate) && candidate != query {
				out = append(out, candidate)
			}
		}
		// Only add the bare first word if it's not already the original query
		// (retrying with the same string is pointless).
		if firstWord != query && (len(out) == 0 || out[len(out)-1] != firstWord) {
			out = append(out, firstWord)
		}
	}
	return out
}

// outlineHit is the shape of one entry in an Outline MCP list_documents result.
// The MCP server returns a bare JSON array of these.
type outlineHit struct {
	Document struct {
		ID    string `json:"id"`
		Title string `json:"title"`
	} `json:"document"`
	Context string `json:"context"`
}

// kbTopID returns the ID of the first document in an Outline MCP
// list_documents result, or "" if the result is empty or unparseable.
func kbTopID(out string) string {
	var hits []outlineHit
	if err := json.Unmarshal([]byte(out), &hits); err != nil || len(hits) == 0 {
		return ""
	}
	return hits[0].Document.ID
}

// kbHitIsRelevant reports whether the top hit's context snippet contains at
// least 2 words from effectiveQuery that are not the workload/component name.
// Requiring 2 words prevents false positives from a single generic term.
func kbHitIsRelevant(out, effectiveQuery, workload string) bool {
	var hits []outlineHit
	if err := json.Unmarshal([]byte(out), &hits); err != nil || len(hits) == 0 {
		return false
	}
	ctx := strings.ToLower(hits[0].Context)
	workloadLower := strings.ToLower(workload)
	matches := 0
	for _, word := range strings.Fields(strings.ToLower(effectiveQuery)) {
		if len(word) < 4 {
			continue // skip short stop-words
		}
		if word == workloadLower {
			continue
		}
		if strings.Contains(ctx, word) {
			matches++
			if matches >= 2 {
				return true
			}
		}
	}
	return false
}

// dispatchListDocuments calls list_documents and, on an empty result,
// automatically retries with progressively broader queries:
//   0. Alert summary with identifiers stripped.
//   1. Model query with identifiers stripped.
//   2. Word-halved.
//   3. workload + firstWord.
//   4. firstWord only.
//
// When a non-empty, relevant result is found, the top document is auto-fetched.
func (d *Dispatcher) dispatchListDocuments(ctx context.Context, srv *mcpClient, toolName, workload, alertHint string, input json.RawMessage) (string, error) {
	var p struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(input, &p); err != nil || p.Query == "" {
		return srv.call(ctx, toolName, input)
	}

	effectiveQuery := p.Query
	out, err := srv.call(ctx, toolName, input)
	if err == nil && listDocumentsIsEmpty(out) {
		for _, q := range kbQueryFallbacks(p.Query, workload, alertHint) {
			slog.Info("mcp: list_documents empty, retrying", "original", p.Query, "retry", q)
			retryInput, jerr := json.Marshal(map[string]string{"query": q})
			if jerr != nil {
				break
			}
			out, err = srv.call(ctx, toolName, json.RawMessage(retryInput))
			if err != nil || !listDocumentsIsEmpty(out) {
				effectiveQuery = q
				break
			}
		}
	}
	if err != nil || listDocumentsIsEmpty(out) {
		return out, err
	}

	// Auto-fetch the top document only when the hit's context snippet contains
	// at least one non-trivial, non-workload word from the effective query.
	// This avoids attaching unrelated articles that only matched on the
	// component name.
	if id := kbTopID(out); id != "" && srv.handles("fetch") && kbHitIsRelevant(out, effectiveQuery, workload) {
		fetchInput, jerr := json.Marshal(map[string]string{"id": id})
		if jerr == nil {
			fetched, ferr := srv.call(ctx, "fetch", json.RawMessage(fetchInput))
			if ferr == nil && fetched != "" {
				out = out + "\n\n--- full text of top result ---\n" + fetched
				slog.Info("mcp: list_documents auto-fetched top document", "id", id)
			}
		}
	}
	return out, err
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
