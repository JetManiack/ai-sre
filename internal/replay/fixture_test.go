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
	if len(f.Expected.RootCauseContains) == 0 {
		t.Error("expected root_cause_contains to be non-empty")
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
	}
	for _, tt := range tests {
		got := inputMatches(json.RawMessage(tt.pattern), json.RawMessage(tt.actual))
		if got != tt.want {
			t.Errorf("inputMatches(%s, %s) = %v, want %v", tt.pattern, tt.actual, got, tt.want)
		}
	}
}
