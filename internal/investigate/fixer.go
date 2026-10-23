package investigate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"text/template"
	"time"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/observer"
)

// Fixer runs Phase 2: a short verification tool loop after an action is executed.
// It checks whether the action resolved the issue and returns a FixResult.
type Fixer struct {
	cfg        config.InvestigateConfig
	tools      []Tool
	dispatcher ToolDispatcher
	llm        llmClient
}

// NewFixer creates a Fixer that shares the same tools and dispatcher as the Agent.
func NewFixer(cfg config.InvestigateConfig, tools []Tool, dispatcher ToolDispatcher, llm llmClient) *Fixer {
	return &Fixer{cfg: cfg, tools: tools, dispatcher: dispatcher, llm: llm}
}

// Verify runs up to 3 tool steps to confirm actionTaken resolved the issue.
// Returns a FixResult even on partial failure.
func (f *Fixer) Verify(ctx context.Context, c observer.Candidate, inv InvestigationResult, actionTaken ActionKind) FixResult {
	timeout := f.cfg.TotalTimeout / 3
	if timeout < 30*time.Second {
		timeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	msgs := []Message{
		{Role: "system", Content: f.systemPrompt(c, inv, actionTaken)},
		{Role: "user", Content: "Verify that the action resolved the issue. Use 1–2 tool calls then produce your FixResult."},
	}

	const maxSteps = 3
	seen := map[string]int{}
	var lastErr error

	for step := 0; step < maxSteps; step++ {
		forceFinal := step == maxSteps-1

		req := ChatRequest{Model: f.cfg.Model, Messages: msgs}
		if forceFinal {
			req.ResponseFormat = &ResponseFormat{
				Type:       "json_schema",
				JSONSchema: fixResultSchema(actionTaken),
			}
			req.MaxTokens = f.cfg.MaxTokensFinal
		} else {
			req.Tools = openAITools(f.activeTools())
			req.MaxTokens = f.cfg.MaxTokensStep
		}

		stepCtx, stepCancel := context.WithTimeout(ctx, f.cfg.StepTimeout)
		resp, err := f.llm.ChatCompletionWithTools(stepCtx, req)
		stepCancel()
		if err != nil {
			lastErr = err
			slog.Warn("fixer: llm step error", "step", step, "err", err)
			break
		}

		if forceFinal || resp.FinishReason == "stop" || resp.FinishReason == "end_turn" {
			if resp.Content != "" {
				r, err := parseFixResult(resp.Content)
				if err != nil {
					slog.Warn("fixer: parse result failed", "err", err)
					return failedFix(actionTaken, fmt.Sprintf("parse failed: %v", err))
				}
				return r
			}
			// No content on final step — fall through to next iteration to force final
			continue
		}

		if len(resp.ToolCalls) == 0 {
			msgs = append(msgs, Message{Role: "assistant", Content: resp.Content})
			continue
		}

		var rawCalls []RawToolCall
		for _, tc := range resp.ToolCalls {
			var rc RawToolCall
			rc.ID = tc.ID
			rc.Type = "function"
			rc.Function.Name = tc.Name
			rc.Function.Arguments = string(tc.Input)
			rawCalls = append(rawCalls, rc)
		}
		msgs = append(msgs, Message{Role: "assistant", ToolCalls: rawCalls})

		for _, tc := range resp.ToolCalls {
			sig := tc.Name + "|" + string(tc.Input)
			seen[sig]++
			if seen[sig] > 1 {
				msgs = append(msgs, Message{Role: "tool", ToolCallID: tc.ID, Content: "[duplicate — skipped]"})
				continue
			}
			slog.Info("fixer: tool call", "step", step, "tool", tc.Name)
			out, err := f.dispatcher.Dispatch(ctx, ToolCall{Name: tc.Name, Input: tc.Input})
			if err != nil {
				out = fmt.Sprintf("[error: %v]", err)
			}
			out = trimOutput(out, f.cfg.MaxToolOutputTokens)
			msgs = append(msgs, Message{Role: "tool", ToolCallID: tc.ID, Content: out})
		}
	}

	reason := "max steps reached"
	if lastErr != nil {
		reason = lastErr.Error()
	}
	return failedFix(actionTaken, reason)
}

var fixerSystemTmpl = template.Must(template.New("fixer").Parse(
	`You are an SRE verification agent running Phase 2: post-action check.

Current time (UTC): {{.NowUTC}}

Action executed: {{.Action}} on {{.Kind}}/{{.Name}} (namespace: {{.Namespace}})

Phase 1 diagnosis:
- Root cause: {{.RootCauseRU}}
- Suggested fix: {{.SuggestedFixRU}}

Check whether the action resolved the issue.
Use 1–2 tool calls (logs, metrics, or pod state). Then output FixResult JSON.
If still broken, set success=false and write escalate_message in Russian (3–5 sentences).`))

func (f *Fixer) systemPrompt(c observer.Candidate, inv InvestigationResult, action ActionKind) string {
	var buf bytes.Buffer
	_ = fixerSystemTmpl.Execute(&buf, struct {
		NowUTC         string
		Action         ActionKind
		Kind, Name, Namespace string
		RootCauseRU    string
		SuggestedFixRU string
	}{
		NowUTC:         time.Now().UTC().Format(time.RFC3339),
		Action:         action,
		Kind:           c.Workload.Kind,
		Name:           c.Workload.Name,
		Namespace:      c.Workload.Namespace,
		RootCauseRU:    inv.RootCauseRU,
		SuggestedFixRU: inv.SuggestedFixRU,
	})
	return buf.String()
}

func (f *Fixer) activeTools() []Tool {
	if len(f.tools) > 0 {
		return f.tools
	}
	return toolSchemas
}

func parseFixResult(content string) (FixResult, error) {
	var r FixResult
	if err := json.Unmarshal([]byte(content), &r); err != nil {
		return FixResult{}, err
	}
	return r, nil
}

func failedFix(action ActionKind, reason string) FixResult {
	return FixResult{
		ActionTaken:     action,
		Success:         false,
		VerificationRU:  fmt.Sprintf("Проверка не завершена: %s", reason),
		EscalateMessage: fmt.Sprintf("⚠️ Действие %s было выполнено, но проверка результата не завершилась: %s\n\nПроверьте вручную.", action, reason),
	}
}

func fixResultSchema(actionTaken ActionKind) json.RawMessage {
	return mustJSON(map[string]any{
		"name":   "FixResult",
		"strict": true,
		"schema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action_taken":     map[string]any{"type": "string"},
				"success":          map[string]any{"type": "boolean"},
				"verification_ru":  map[string]any{"type": "string", "minLength": 10, "maxLength": 500},
				"escalate_message": map[string]any{"type": "string", "maxLength": 600},
			},
			"required": []string{"action_taken", "success", "verification_ru"},
		},
	})
}
