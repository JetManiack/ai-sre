package investigate

import (
	"encoding/json"
	"testing"
)

func TestOpenAIToolWireFormat(t *testing.T) {
	tools := openAITools(toolSchemas[:2]) // just k8s_describe + k8s_logs

	b, err := json.Marshal(tools)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var wire []map[string]json.RawMessage
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(wire) != 2 {
		t.Fatalf("got %d tools, want 2", len(wire))
	}

	first := wire[0]
	// Must have "type": "function"
	var typ string
	if err := json.Unmarshal(first["type"], &typ); err != nil || typ != "function" {
		t.Errorf("type: got %s, want \"function\"", first["type"])
	}
	// Must have "function" key
	if first["function"] == nil {
		t.Error("missing 'function' key")
	}
	// Must NOT have top-level "input_schema" (Anthropic format)
	if first["input_schema"] != nil {
		t.Error("found 'input_schema' at top level — wrong format (Anthropic, not OpenAI)")
	}

	var fn map[string]json.RawMessage
	if err := json.Unmarshal(first["function"], &fn); err != nil {
		t.Fatalf("unmarshal function: %v", err)
	}
	if fn["parameters"] == nil {
		t.Error("function.parameters is missing")
	}
	if fn["input_schema"] != nil {
		t.Error("function.input_schema present — should be function.parameters")
	}
}

func TestRawToolCallArgumentsIsString(t *testing.T) {
	rc := RawToolCall{
		ID:   "tc1",
		Type: "function",
	}
	rc.Function.Name = "k8s_top"
	rc.Function.Arguments = `{"namespace":"default"}`

	b, err := json.Marshal(rc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var wire map[string]json.RawMessage
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var fn map[string]json.RawMessage
	if err := json.Unmarshal(wire["function"], &fn); err != nil {
		t.Fatalf("unmarshal function: %v", err)
	}

	// arguments must be a JSON string, not an object
	var argStr string
	if err := json.Unmarshal(fn["arguments"], &argStr); err != nil {
		t.Fatalf("arguments must be a string, got %s: %v", fn["arguments"], err)
	}
	// and that string must contain valid JSON
	var inner map[string]any
	if err := json.Unmarshal([]byte(argStr), &inner); err != nil {
		t.Errorf("arguments string is not valid JSON: %v", err)
	}
	if inner["namespace"] != "default" {
		t.Errorf("namespace: got %v", inner["namespace"])
	}
}

func TestResponseFormatOnlyOnFinalStep(t *testing.T) {
	// Verify that a non-final ChatRequest has no response_format set.
	// This is enforced in agent.go's loop; here we just document the expectation
	// by checking that the zero value marshals without it.
	req := ChatRequest{
		Model:    "test",
		Messages: []Message{{Role: "user", Content: "hello"}},
		Tools:    openAITools(toolSchemas[:1]),
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatal(err)
	}
	if wire["response_format"] != nil && string(wire["response_format"]) != "null" {
		t.Errorf("non-final request must not have response_format, got %s", wire["response_format"])
	}
}
