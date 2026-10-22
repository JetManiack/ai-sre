package investigate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ChatRequest is an OpenAI-compatible chat completions request.
// Tools are serialised in the OpenAI {type:function,function:{…}} wire format.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []OpenAITool    `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
}

// Message is a single conversation turn.
type Message struct {
	Role       string       `json:"role"`
	Content    string       `json:"content,omitempty"`
	ToolCalls  []RawToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
}

// RawToolCall is the wire format for a tool call inside an assistant message.
// In OpenAI format, Function.Arguments is a JSON-encoded string, not an object.
type RawToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // "function"
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // JSON string, e.g. "{\"workload\":\"gruh\"}"
	} `json:"function"`
}

// ResponseFormat requests structured JSON output.
type ResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// ChatResponse is the relevant subset of an OpenAI chat completions response.
type ChatResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
}

// OpenAIClient is an OpenAI-compatible HTTP client.
type OpenAIClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// NewOpenAIClient creates a client. baseURL should end without a trailing slash.
func NewOpenAIClient(baseURL, apiKey string) *OpenAIClient {
	return &OpenAIClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 5 * time.Minute},
	}
}

// ChatCompletionWithTools sends a chat completion request and returns the response.
func (c *OpenAIClient) ChatCompletionWithTools(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return ChatResponse{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatResponse{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return ChatResponse{}, fmt.Errorf("llm: HTTP %d: %s", resp.StatusCode, raw)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content   string        `json:"content"`
				ToolCalls []RawToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return ChatResponse{}, fmt.Errorf("llm: parse response: %w", err)
	}
	if len(out.Choices) == 0 {
		return ChatResponse{}, fmt.Errorf("llm: empty choices")
	}
	ch := out.Choices[0]

	var toolCalls []ToolCall
	for _, tc := range ch.Message.ToolCalls {
		// OpenAI returns arguments as a JSON string; decode it into raw JSON.
		input := json.RawMessage(tc.Function.Arguments)
		toolCalls = append(toolCalls, ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	return ChatResponse{
		Content:      ch.Message.Content,
		ToolCalls:    toolCalls,
		FinishReason: ch.FinishReason,
	}, nil
}
