// Package replay loads test fixtures for the --replay flag and CI testing.
//
// A fixture directory contains:
//
//	candidate.json       – the observer.Candidate that triggered investigation
//	tool_responses.jsonl – one JSON object per line, each a ToolResponse
//	expected.json        – the ExpectedOutcome the test asserts
//
// The ToolDispatcher built by [NewDispatcher] replays tool responses in order,
// keyed by (tool_name, input). An unmatched call returns an error.
package replay

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/JetManiack/ai-sre/internal/investigate"
	"github.com/JetManiack/ai-sre/internal/observer"
)

// ToolResponse is one line in tool_responses.jsonl.
// It maps a (name, input) pair to a canned output.
type ToolResponse struct {
	// Tool is the tool name (e.g. "metrics_query").
	Tool string `json:"tool"`
	// InputMatch is a JSON object whose keys must all appear with semantically equal
	// values in the actual tool input (decoded comparison, not raw bytes). Extra keys
	// in the actual input are ignored. An empty object {} matches any input.
	InputMatch json.RawMessage `json:"input_match"`
	// InputContains is an optional map of key → substring. For each pair, the actual
	// input must have that key, and its string value must contain the substring.
	// Useful for matching free-form query strings from a live model.
	InputContains map[string]string `json:"input_contains,omitempty"`
	// Output is the string the dispatcher returns.
	Output string `json:"output"`
	// Error, if non-empty, causes the dispatcher to return this as an error.
	Error string `json:"error,omitempty"`
}

// ExpectedOutcome is what the test asserts about the resulting Decision.
type ExpectedOutcome struct {
	// Action is the required action kind.
	Action investigate.ActionKind `json:"action"`
	// ConfidenceMin is the minimum required confidence (0–1).
	ConfidenceMin float64 `json:"confidence_min"`
	// RootCauseContains lists substrings where AT LEAST ONE must appear in
	// RootCauseRU (case-insensitive). Use for broad "is the right topic" checks.
	RootCauseContains []string `json:"root_cause_contains,omitempty"`
	// RootCauseContainsAll lists substrings where ALL must appear in RootCauseRU
	// (case-insensitive). Use for precise "must mention X and Y" assertions.
	RootCauseContainsAll []string `json:"root_cause_contains_all,omitempty"`
}

// Fixture is a fully loaded test case.
type Fixture struct {
	Dir       string
	Candidate observer.Candidate
	Responses []ToolResponse
	Expected  ExpectedOutcome
}

// Load reads a fixture directory and returns a Fixture.
func Load(dir string) (*Fixture, error) {
	f := &Fixture{Dir: dir}

	candData, err := os.ReadFile(filepath.Join(dir, "candidate.json"))
	if err != nil {
		return nil, fmt.Errorf("replay: read candidate.json: %w", err)
	}
	if err := json.Unmarshal(candData, &f.Candidate); err != nil {
		return nil, fmt.Errorf("replay: parse candidate.json: %w", err)
	}

	respFile, err := os.Open(filepath.Join(dir, "tool_responses.jsonl"))
	if err != nil {
		return nil, fmt.Errorf("replay: open tool_responses.jsonl: %w", err)
	}
	defer respFile.Close()

	sc := bufio.NewScanner(respFile)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		var tr ToolResponse
		if err := json.Unmarshal([]byte(line), &tr); err != nil {
			return nil, fmt.Errorf("replay: parse tool_responses.jsonl: %w", err)
		}
		f.Responses = append(f.Responses, tr)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("replay: scan tool_responses.jsonl: %w", err)
	}

	expData, err := os.ReadFile(filepath.Join(dir, "expected.json"))
	if err != nil {
		return nil, fmt.Errorf("replay: read expected.json: %w", err)
	}
	if err := json.Unmarshal(expData, &f.Expected); err != nil {
		return nil, fmt.Errorf("replay: parse expected.json: %w", err)
	}

	return f, nil
}

// inputMatches returns true if all keys in pattern appear with semantically equal
// values in actual (decoded via interface{}, not raw-byte comparison).
// Both must be JSON objects. An empty or null pattern matches everything.
func inputMatches(pattern, actual json.RawMessage) bool {
	if len(pattern) == 0 || string(pattern) == "{}" || string(pattern) == "null" {
		return true
	}
	var p, a map[string]any
	if err := json.Unmarshal(pattern, &p); err != nil {
		return false
	}
	if err := json.Unmarshal(actual, &a); err != nil {
		return false
	}
	for k, pv := range p {
		av, ok := a[k]
		if !ok {
			return false
		}
		if !reflect.DeepEqual(pv, av) {
			return false
		}
	}
	return true
}

// inputContainsMatches returns true if for every (key, substr) in contains,
// the actual input has that key and its string value contains substr (case-sensitive).
func inputContainsMatches(contains map[string]string, actual json.RawMessage) bool {
	if len(contains) == 0 {
		return true
	}
	var a map[string]any
	if err := json.Unmarshal(actual, &a); err != nil {
		return false
	}
	for k, substr := range contains {
		v, ok := a[k]
		if !ok {
			return false
		}
		s, ok := v.(string)
		if !ok {
			return false
		}
		if !strings.Contains(s, substr) {
			return false
		}
	}
	return true
}

// Dispatcher implements investigate.ToolDispatcher using pre-loaded ToolResponses.
// Each response is consumed once (in-order across responses with the same tool).
type Dispatcher struct {
	responses []ToolResponse
	used      []bool
}

// NewDispatcher creates a Dispatcher from a slice of responses.
func NewDispatcher(responses []ToolResponse) *Dispatcher {
	return &Dispatcher{responses: responses, used: make([]bool, len(responses))}
}

// Dispatch finds the first unused response that matches the tool name and input.
func (d *Dispatcher) Dispatch(_ context.Context, call investigate.ToolCall) (string, error) {
	for i, r := range d.responses {
		if d.used[i] {
			continue
		}
		if r.Tool != call.Name {
			continue
		}
		if !inputMatches(r.InputMatch, call.Input) {
			continue
		}
		if !inputContainsMatches(r.InputContains, call.Input) {
			continue
		}
		d.used[i] = true
		if r.Error != "" {
			return "", fmt.Errorf("%s", r.Error)
		}
		return r.Output, nil
	}
	return "", fmt.Errorf("replay: no response for tool=%q input=%s", call.Name, call.Input)
}

// CheckExpected validates an InvestigationResult against an ExpectedOutcome.
// Returns a non-nil error describing the first failed assertion.
// For root_cause_* checks the search corpus is root_cause_ru + escalate_message
// so that models which embed the diagnosis in the escalation text still pass.
func CheckExpected(exp ExpectedOutcome, r investigate.InvestigationResult) error {
	if r.RecommendedAction != exp.Action {
		return fmt.Errorf("action: got %q, want %q", r.RecommendedAction, exp.Action)
	}
	if r.Confidence < exp.ConfidenceMin {
		return fmt.Errorf("confidence: got %.2f, want >= %.2f", r.Confidence, exp.ConfidenceMin)
	}
	// Search both root_cause_ru and escalate_message for diagnostic substrings.
	corpus := strings.ToLower(r.RootCauseRU + " " + r.EscalateMessage)
	if len(exp.RootCauseContains) > 0 {
		matched := false
		for _, substr := range exp.RootCauseContains {
			if strings.Contains(corpus, strings.ToLower(substr)) {
				matched = true
				break
			}
		}
		if !matched {
			return fmt.Errorf("root_cause_ru+escalate_message: none of %v matched in %q", exp.RootCauseContains, corpus)
		}
	}
	for _, substr := range exp.RootCauseContainsAll {
		if !strings.Contains(corpus, strings.ToLower(substr)) {
			return fmt.Errorf("root_cause_ru+escalate_message: required substring %q not found", substr)
		}
	}
	return nil
}
