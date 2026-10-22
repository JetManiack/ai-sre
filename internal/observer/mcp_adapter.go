package observer

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
)

// MCPAdapter polls an MCP server for anomalies by calling a "check_anomalies"
// tool or equivalent. Each MCP server in the observer section can define its
// own detection logic (e.g. a Loki server that runs metric-over-log queries).
//
// The tool must return a JSON array of candidates:
//
//	[{"kind":"error_rate_spike","severity":"high","workload":{"name":"gateway","namespace":"ai"},...}]
//
// This lets site-specific detection rules live in the MCP server, not in the
// agent binary — the binary just polls and deserialises.
type MCPAdapter struct {
	name    string
	baseURL string
	token   string
	toolName string
	source  string
	hc      *http.Client

	cfg config.ObserverConfig
}

// NewMCPAdapter creates an adapter that polls an MCP server for candidates.
// toolName is the MCP tool to call (e.g. "check_anomalies"). source is the
// Workload.Source to stamp on returned candidates.
func NewMCPAdapter(name, baseURL, token, toolName, source string, cfg config.ObserverConfig) *MCPAdapter {
	return &MCPAdapter{
		name:     name,
		baseURL:  strings.TrimRight(baseURL, "/"),
		token:    token,
		toolName: toolName,
		source:   source,
		cfg:      cfg,
		hc:       &http.Client{Timeout: 30 * time.Second},
	}
}

func (a *MCPAdapter) Name() string { return a.name }

func (a *MCPAdapter) Poll(ctx context.Context) ([]Candidate, error) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      a.toolName,
			"arguments": map[string]any{},
		},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if a.token != "" {
		req.Header.Set("Authorization", "Bearer "+a.token)
	}
	resp, err := a.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mcp adapter %s: %w", a.name, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("mcp adapter %s: HTTP %d", a.name, resp.StatusCode)
	}

	var r struct {
		Result *struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || r.Result == nil {
		return nil, fmt.Errorf("mcp adapter %s: bad response: %s", a.name, raw)
	}
	var text string
	for _, c := range r.Result.Content {
		if c.Type == "text" {
			text = c.Text
			break
		}
	}
	if text == "" {
		return nil, nil
	}

	var candidates []Candidate
	if err := json.Unmarshal([]byte(text), &candidates); err != nil {
		slog.Warn("mcp adapter: could not parse candidates", "adapter", a.name, "err", err, "text", text)
		return nil, nil
	}
	// Stamp source and fingerprints if missing.
	for i := range candidates {
		if candidates[i].Workload.Source == "" {
			candidates[i].Workload.Source = a.source
		}
		if candidates[i].ID == "" {
			candidates[i].ID = Fingerprint(candidates[i].Workload, candidates[i].Kind, candidates[i].Summary)
		}
		if candidates[i].ObservedAt.IsZero() {
			candidates[i].ObservedAt = time.Now()
		}
	}
	return candidates, nil
}

// LogPatternAdapter is a simple observer adapter that scans log lines from an
// MCP Loki/VictoriaLogs server and emits a KindLogPattern candidate on a match.
// It calls "logs_query" on the MCP server every poll cycle.
type LogPatternAdapter struct {
	name        string
	mcpBaseURL  string
	token       string
	source      string
	namespace   string
	patterns    []string
	hc          *http.Client
}

func NewLogPatternAdapter(name, mcpBaseURL, token, source, namespace string, patterns []string) *LogPatternAdapter {
	return &LogPatternAdapter{
		name:       name,
		mcpBaseURL: strings.TrimRight(mcpBaseURL, "/"),
		token:      token,
		source:     source,
		namespace:  namespace,
		patterns:   patterns,
		hc:         &http.Client{Timeout: 30 * time.Second},
	}
}

func (a *LogPatternAdapter) Name() string { return "log-pattern:" + a.name }

func (a *LogPatternAdapter) Poll(ctx context.Context) ([]Candidate, error) {
	args := map[string]any{
		"source":    a.source,
		"namespace": a.namespace,
		"since":     "5m",
	}
	text, err := a.mcpCall(ctx, "logs_query", args)
	if err != nil {
		return nil, err
	}
	var out []Candidate
	for _, line := range strings.Split(text, "\n") {
		if pattern, matched := matchesAny(line, a.patterns); matched {
			wl := Workload{Kind: "unknown", Name: a.name, Namespace: a.namespace, Source: a.source}
			c := Candidate{
				ID:          Fingerprint(wl, KindLogPattern, pattern),
				ObservedAt:  time.Now(),
				Kind:        KindLogPattern,
				Severity:    SeverityHigh,
				Workload:    wl,
				Summary:     fmt.Sprintf("log pattern matched: %q", pattern),
				Evidence:    []string{line},
			}
			out = append(out, c)
			break // one candidate per poll per adapter
		}
	}
	return out, nil
}

func (a *LogPatternAdapter) mcpCall(ctx context.Context, tool string, args map[string]any) (string, error) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params":  map[string]any{"name": tool, "arguments": args},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.mcpBaseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if a.token != "" {
		req.Header.Set("Authorization", "Bearer "+a.token)
	}
	resp, err := a.hc.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("mcp %s: HTTP %d", tool, resp.StatusCode)
	}
	var r struct {
		Result *struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || r.Result == nil {
		return string(raw), nil
	}
	for _, c := range r.Result.Content {
		if c.Type == "text" {
			return c.Text, nil
		}
	}
	return "", nil
}

func matchesAny(text string, patterns []string) (string, bool) {
	lower := strings.ToLower(text)
	for _, p := range patterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return p, true
		}
	}
	return "", false
}
