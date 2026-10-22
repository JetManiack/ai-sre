// Package investigate runs the LLM investigator+decision agent.
//
// The agent receives a Candidate from the observer and runs a bounded tool loop:
//  1. Calls tools read-only (k8s, metrics, logs, host, memory_recall)
//  2. After ≤MaxSteps or when ready, produces a Decision (json_schema)
//
// The LLM decides both the diagnosis AND the action (escalate, restart, etc.).
// The executor (decision package) validates the action against ActionsAllowlist
// before execution — the LLM never executes anything directly.
package investigate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"text/template"
	"time"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/observer"
)

// ToolDispatcher executes a tool call and returns the (possibly trimmed) output.
// Implementations live in internal/tools/.
type ToolDispatcher interface {
	Dispatch(ctx context.Context, call ToolCall) (string, error)
}

// MemoryStore provides long-term recall and storage for investigations.
type MemoryStore interface {
	RecallInvestigations(workload, namespace string, limit int) ([]PastInvestigation, error)
	SaveInvestigation(inv PastInvestigation) error
}

// PastInvestigation is a persisted investigation record.
type PastInvestigation struct {
	ID          string
	WorkloadName string
	Namespace   string
	CandidateKind string
	RootCauseRU string
	Action      string
	Confidence  float64
	OccurredAt  time.Time
}

// Agent runs the investigator+decision loop for a single candidate.
type Agent struct {
	cfg        config.InvestigateConfig
	dispatcher ToolDispatcher
	memory     MemoryStore
	llm        llmClient
}

// llmClient is the minimal interface to the OpenAI-compatible endpoint.
type llmClient interface {
	ChatCompletionWithTools(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

// New creates an Agent.
func New(cfg config.InvestigateConfig, dispatcher ToolDispatcher, memory MemoryStore, llm llmClient) *Agent {
	return &Agent{cfg: cfg, dispatcher: dispatcher, memory: memory, llm: llm}
}

// Investigate runs the full investigator+decision loop for a candidate.
// It returns a Decision even on partial failure (confidence = 0, action = escalate).
func (a *Agent) Investigate(ctx context.Context, c observer.Candidate) Decision {
	ctx, cancel := context.WithTimeout(ctx, a.cfg.TotalTimeout)
	defer cancel()

	msgs := []Message{
		{Role: "system", Content: a.systemPrompt(c)},
		{Role: "user", Content: a.userPrompt(c)},
	}

	// Track tool call signatures for dup-guard.
	seen := map[string]int{}
	var lastErr error

	for step := 0; step < a.cfg.MaxSteps; step++ {
		// Mandatory final-answer step at MaxSteps-1.
		forceFinal := step == a.cfg.MaxSteps-1

		req := ChatRequest{
			Model:    a.cfg.Model,
			Messages: msgs,
			Tools:    toolSchemas,
			ResponseFormat: &ResponseFormat{
				Type:       "json_schema",
				JSONSchema: decisionSchema(),
			},
		}
		if forceFinal {
			req.Tools = nil // no more tool calls: produce the final answer now
		}
		req.MaxTokens = a.cfg.MaxTokensStep
		if forceFinal {
			req.MaxTokens = a.cfg.MaxTokensFinal
		}

		stepCtx, stepCancel := context.WithTimeout(ctx, a.cfg.StepTimeout)
		resp, err := a.llm.ChatCompletionWithTools(stepCtx, req)
		stepCancel()
		if err != nil {
			lastErr = err
			slog.Warn("investigate: llm step error", "step", step, "err", err)
			break
		}

		// Final answer: parse Decision from content.
		if resp.FinishReason == "stop" || resp.FinishReason == "end_turn" || forceFinal {
			d, err := parseDecision(resp.Content)
			if err != nil {
				slog.Warn("investigate: parse decision failed", "err", err, "content", resp.Content)
				return failedDecision(c, fmt.Sprintf("parse failed: %v", err))
			}
			_ = a.memory.SaveInvestigation(PastInvestigation{
				ID:            c.ID + "-" + fmt.Sprint(time.Now().UnixNano()),
				WorkloadName:  c.Workload.Name,
				Namespace:     c.Workload.Namespace,
				CandidateKind: string(c.Kind),
				RootCauseRU:   d.RootCauseRU,
				Action:        string(d.Action),
				Confidence:    d.Confidence,
				OccurredAt:    time.Now(),
			})
			return d
		}

		// Tool call step.
		if len(resp.ToolCalls) == 0 {
			// Model produced content but didn't finish and didn't call a tool — unusual.
			msgs = append(msgs, Message{Role: "assistant", Content: resp.Content})
			continue
		}

		// Append assistant turn with tool calls (convert to wire format).
		var rawCalls []RawToolCall
		for _, tc := range resp.ToolCalls {
			var rc RawToolCall
			rc.ID = tc.ID
			rc.Type = "function"
			rc.Function.Name = tc.Name
			rc.Function.Arguments = tc.Input
			rawCalls = append(rawCalls, rc)
		}
		msgs = append(msgs, Message{Role: "assistant", ToolCalls: rawCalls})

		// Execute each tool call and append results.
		for _, tc := range resp.ToolCalls {
			sig := tc.Name + "|" + string(tc.Input)
			seen[sig]++
			if seen[sig] > 1 {
				// Dup-guard: same tool+args twice → force final answer next step.
				slog.Info("investigate: dup tool call, forcing final", "tool", tc.Name)
				msgs = append(msgs, Message{
					Role:       "tool",
					ToolCallID: tc.ID,
					Content:    "[duplicate call — please produce your final answer now]",
				})
				continue
			}

			out, err := a.dispatcher.Dispatch(ctx, ToolCall{Name: tc.Name, Input: tc.Input})
			if err != nil {
				slog.Warn("investigate: tool dispatch error", "tool", tc.Name, "err", err)
				out = fmt.Sprintf("[error: %v]", err)
			}
			// Trim to budget.
			out = trimOutput(out, a.cfg.MaxToolOutputTokens)
			msgs = append(msgs, Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    out,
			})
		}
	}

	reason := "max steps reached"
	if lastErr != nil {
		reason = lastErr.Error()
	}
	return failedDecision(c, reason)
}

var systemPromptTmpl = template.Must(template.New("sys").Parse(`You are an autonomous SRE agent. Your job is to investigate an anomaly detected in a service, determine the root cause, and decide what to do.

Rules:
- Use tools to collect evidence before concluding.
- You may call memory_recall first to check past investigations of this workload.
- Never read secrets. Never write or modify anything — that is handled by the executor.
- Think step by step; call one or two tools per step.
- When you are confident, or at step 7, produce your final JSON answer.
- Your action field determines what happens: "escalate" means post to the board; other actions are executed only if they are in the operator's allowlist.
- EscalateMessage should be a complete board post in Russian, including evidence and suggested fix, ready to send.
- EscalateTo should be the relevant person: use "on-call-dev" for code bugs, "on-call-ops" for operator decisions.

{{if .SystemPrompt}}
Site context:
{{.SystemPrompt}}
{{end}}`))

func (a *Agent) systemPrompt(c observer.Candidate) string {
	var buf bytes.Buffer
	_ = systemPromptTmpl.Execute(&buf, a.cfg)
	return buf.String()
}

var userPromptTmpl = template.Must(template.New("user").Parse(`Anomaly detected:
- Workload: {{.Workload.Kind}}/{{.Workload.Name}} (namespace: {{.Workload.Namespace}}, source: {{.Workload.Source}})
- Kind: {{.Kind}}
- Severity: {{.Severity}}
- Summary: {{.Summary}}
- Evidence:
{{range .Evidence}}  - {{.}}
{{end}}
Investigate and produce a Decision.`))

func (a *Agent) userPrompt(c observer.Candidate) string {
	var buf bytes.Buffer
	_ = userPromptTmpl.Execute(&buf, c)
	return buf.String()
}

func parseDecision(content string) (Decision, error) {
	content = strings.TrimSpace(content)
	// Strip markdown fences if the model wrapped the JSON.
	if strings.HasPrefix(content, "```") {
		lines := strings.Split(content, "\n")
		var inner []string
		for _, l := range lines {
			if strings.HasPrefix(l, "```") {
				continue
			}
			inner = append(inner, l)
		}
		content = strings.Join(inner, "\n")
	}
	var d Decision
	if err := json.Unmarshal([]byte(content), &d); err != nil {
		return Decision{}, err
	}
	if d.Action == "" {
		d.Action = ActionEscalate
	}
	return d, nil
}

func failedDecision(c observer.Candidate, reason string) Decision {
	return Decision{
		RootCauseRU:     fmt.Sprintf("Расследование не завершено: %s", reason),
		Evidence:        c.Evidence,
		Confidence:      0,
		SuggestedFixRU:  "Проверьте вручную.",
		Action:          ActionEscalate,
		EscalateTo:      "on-call-ops",
		EscalateMessage: fmt.Sprintf("⚠️ Агент не смог завершить расследование для %s/%s (%s): %s\n\nПроверьте вручную.", c.Workload.Namespace, c.Workload.Name, c.Kind, reason),
	}
}

// trimOutput truncates output to approximately maxTokens tokens (≈4 chars/token).
func trimOutput(s string, maxTokens int) string {
	maxBytes := maxTokens * 4
	if len(s) <= maxBytes {
		return s
	}
	// Keep the tail — recent lines are more informative.
	truncated := s[len(s)-maxBytes:]
	// Align to line boundary.
	if idx := strings.Index(truncated, "\n"); idx >= 0 {
		truncated = truncated[idx+1:]
	}
	return "[...trimmed...]\n" + truncated
}

// decisionSchema returns the JSON schema for the final Decision output.
func decisionSchema() json.RawMessage {
	return mustJSON(map[string]any{
		"name":   "Decision",
		"strict": true,
		"schema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"root_cause_ru":    map[string]any{"type": "string"},
				"evidence":         map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				"confidence":       map[string]any{"type": "number"},
				"suggested_fix_ru": map[string]any{"type": "string"},
				"next_checks":      map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				"action":           map[string]any{"type": "string", "enum": []string{"escalate", "pod_restart", "rollout_restart", "scale", "systemd_restart"}},
				"action_target":    map[string]any{"type": "string"},
				"action_namespace": map[string]any{"type": "string"},
				"escalate_to":      map[string]any{"type": "string"},
				"escalate_message": map[string]any{"type": "string"},
			},
			"required": []string{"root_cause_ru", "evidence", "confidence", "suggested_fix_ru", "action"},
		},
	})
}
