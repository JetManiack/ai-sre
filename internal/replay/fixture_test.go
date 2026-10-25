package replay

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/JetManiack/ai-sre/internal/investigate"
)

func TestLoad_MemoryLeak(t *testing.T) {
	f, err := Load("../../testdata/fixtures/memory-leak")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if f.Candidate.ID != "syn-memleak-001" {
		t.Errorf("candidate ID: got %q", f.Candidate.ID)
	}
	if f.Candidate.Kind != "memory_growth" {
		t.Errorf("candidate kind: got %q", f.Candidate.Kind)
	}
	if len(f.Responses) == 0 {
		t.Error("no tool responses loaded")
	}
	if f.Expected.Action != "escalate" {
		t.Errorf("expected action escalate, got %q", f.Expected.Action)
	}
}

func TestLoad_GatewayAll500(t *testing.T) {
	f, err := Load("../../testdata/fixtures/gateway-all-500")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if f.Candidate.Kind != "error_rate_spike" {
		t.Errorf("kind: got %q", f.Candidate.Kind)
	}
	if f.Candidate.RawMetrics["success_ratio_1h"] != 0.0 {
		t.Errorf("success_ratio_1h should be 0")
	}
}

func TestLoad_ModelAliasRemoved(t *testing.T) {
	f, err := Load("../../testdata/fixtures/model-alias-removed")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(f.Expected.RootCauseContainsAll) == 0 {
		t.Error("expected root_cause_contains_all to be non-empty")
	}
}

func TestDispatcher_Match(t *testing.T) {
	responses := []ToolResponse{
		{
			Tool:       "metrics_query",
			InputMatch: json.RawMessage(`{"query":"up"}`),
			Output:     "1",
		},
		{
			Tool:       "k8s_top",
			InputMatch: json.RawMessage(`{}`),
			Output:     "pod cpu memory",
		},
	}
	d := NewDispatcher(responses)

	out, err := d.Dispatch(context.Background(), investigate.ToolCall{
		Name:  "metrics_query",
		Input: json.RawMessage(`{"query":"up","source":"prom"}`),
	})
	if err != nil {
		t.Fatalf("Dispatch metrics_query: %v", err)
	}
	if out != "1" {
		t.Errorf("output: got %q", out)
	}

	out, err = d.Dispatch(context.Background(), investigate.ToolCall{
		Name:  "k8s_top",
		Input: json.RawMessage(`{"namespace":"default"}`),
	})
	if err != nil {
		t.Fatalf("Dispatch k8s_top: %v", err)
	}
	if out != "pod cpu memory" {
		t.Errorf("output: got %q", out)
	}
}

func TestDispatcher_NoMatch(t *testing.T) {
	d := NewDispatcher([]ToolResponse{
		{Tool: "metrics_query", InputMatch: json.RawMessage(`{"query":"up"}`), Output: "1"},
	})
	_, err := d.Dispatch(context.Background(), investigate.ToolCall{
		Name:  "k8s_top",
		Input: json.RawMessage(`{"namespace":"x"}`),
	})
	if err == nil {
		t.Error("expected error for unmatched tool")
	}
}

func TestDispatcher_UsedOnce(t *testing.T) {
	d := NewDispatcher([]ToolResponse{
		{Tool: "logs_query", InputMatch: json.RawMessage(`{}`), Output: "line1"},
		{Tool: "logs_query", InputMatch: json.RawMessage(`{}`), Output: "line2"},
	})

	out1, err := d.Dispatch(context.Background(), investigate.ToolCall{Name: "logs_query", Input: json.RawMessage(`{}`)})
	if err != nil || out1 != "line1" {
		t.Fatalf("first call: err=%v out=%q", err, out1)
	}
	out2, err := d.Dispatch(context.Background(), investigate.ToolCall{Name: "logs_query", Input: json.RawMessage(`{}`)})
	if err != nil || out2 != "line2" {
		t.Fatalf("second call: err=%v out=%q", err, out2)
	}
}

func TestInputMatches(t *testing.T) {
	tests := []struct {
		pattern string
		actual  string
		want    bool
	}{
		{`{}`, `{"a":"b"}`, true},
		{`{"a":"b"}`, `{"a":"b","c":"d"}`, true},
		{`{"a":"b"}`, `{"a":"x"}`, false},
		{`{"a":"b"}`, `{"c":"d"}`, false},
		{`null`, `{"a":"b"}`, true},
		// decoded comparison: 50 == 50.0
		{`{"limit":50}`, `{"limit":50}`, true},
	}
	for _, tt := range tests {
		got := inputMatches(json.RawMessage(tt.pattern), json.RawMessage(tt.actual))
		if got != tt.want {
			t.Errorf("inputMatches(%s, %s) = %v, want %v", tt.pattern, tt.actual, got, tt.want)
		}
	}
}

func TestInputContainsMatches(t *testing.T) {
	tests := []struct {
		contains map[string]string
		actual   string
		want     bool
	}{
		{nil, `{"query":"up"}`, true},
		{map[string]string{"query": "up"}, `{"query":"sum(up)"}`, true},
		{map[string]string{"query": "down"}, `{"query":"sum(up)"}`, false},
		{map[string]string{"source": "vlogs"}, `{"source":"vlogs","query":"x"}`, true},
		{map[string]string{"missing": "x"}, `{"query":"x"}`, false},
	}
	for _, tt := range tests {
		got := inputContainsMatches(tt.contains, json.RawMessage(tt.actual))
		if got != tt.want {
			t.Errorf("inputContainsMatches(%v, %s) = %v, want %v", tt.contains, tt.actual, got, tt.want)
		}
	}
}

func TestCheckExpected(t *testing.T) {
	exp := ExpectedOutcome{
		Action:               "escalate",
		ConfidenceMin:        0.6,
		RootCauseContains:    []string{"alias", "model"},
		RootCauseContainsAll: []string{"ModelNotMapped", "litellm"},
	}

	dec := func(action investigate.ActionKind, confidence float64, rootCause string) investigate.InvestigationResult {
		return investigate.InvestigationResult{RecommendedAction: action, Confidence: confidence, RootCauseRU: rootCause}
	}
	// happy path
	if err := CheckExpected(exp, dec("escalate", 0.9, "ModelNotMapped error in litellm alias config")); err != nil {
		t.Errorf("expected nil, got: %v", err)
	}
	// wrong action
	if err := CheckExpected(exp, dec("pod_restart", 0.9, "ModelNotMapped litellm")); err == nil {
		t.Error("expected error for wrong action")
	}
	// low confidence
	if err := CheckExpected(exp, dec("escalate", 0.5, "ModelNotMapped litellm")); err == nil {
		t.Error("expected error for low confidence")
	}
	// contains_all fails (missing "litellm")
	if err := CheckExpected(exp, dec("escalate", 0.9, "ModelNotMapped issue")); err == nil {
		t.Error("expected error for missing contains_all token")
	}
	// contains passes with any match
	expAny := ExpectedOutcome{Action: "escalate", ConfidenceMin: 0.0, RootCauseContains: []string{"alias", "xyz"}}
	if err := CheckExpected(expAny, dec("escalate", 0.1, "alias removed")); err != nil {
		t.Errorf("expected nil (any match), got: %v", err)
	}
}
