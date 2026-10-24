// Package sink sends incident reports to configured output channels.
package sink

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/investigate"
	"github.com/JetManiack/ai-sre/internal/observer"
)

// IncidentReport is the full output of one investigation+fix cycle.
type IncidentReport struct {
	Candidate   observer.Candidate
	Investigation investigate.InvestigationResult
	// Fix is nil when the recommended action was escalate (no execution).
	Fix *investigate.FixResult
}

// Sink sends an IncidentReport to an output channel.
type Sink interface {
	Send(ctx context.Context, r IncidentReport) error
}

// Multi fans out to multiple sinks. Errors are logged but don't abort others.
type Multi struct {
	sinks []Sink
}

func NewMulti(sinks ...Sink) *Multi { return &Multi{sinks: sinks} }

func (m *Multi) Send(ctx context.Context, r IncidentReport) error {
	var errs []string
	for _, s := range m.sinks {
		if err := s.Send(ctx, r); err != nil {
			slog.Warn("sink: send error", "err", err)
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("sink errors: %s", strings.Join(errs, "; "))
	}
	return nil
}

// StdoutJSON writes the report as a JSON line to stdout.
type StdoutJSON struct{}

func (s StdoutJSON) Send(_ context.Context, r IncidentReport) error {
	inv := r.Investigation
	out := map[string]any{
		"workload":   r.Candidate.Workload.Name,
		"namespace":  r.Candidate.Workload.Namespace,
		"kind":       r.Candidate.Kind,
		"severity":   r.Candidate.Severity,
		"action":     inv.RecommendedAction,
		"confidence": inv.Confidence,
		"root_cause": inv.RootCauseRU,
		"suggested":  inv.SuggestedFixRU,
	}
	if r.Fix != nil {
		out["fix_success"] = r.Fix.Success
		out["fix_verification"] = r.Fix.VerificationRU
	}
	b, _ := json.Marshal(out)
	_, err := fmt.Fprintf(os.Stdout, "%s\n", b)
	return err
}

// BoardSink posts to the coordination board via MCP HTTP.
// Posts when: recommended_action=escalate OR fix failed.
type BoardSink struct {
	client *boardMCPClient
	cfg    config.BoardSinkConfig
}

func NewBoardSink(cfg config.BoardSinkConfig) *BoardSink {
	return &BoardSink{
		cfg:    cfg,
		client: newBoardMCPClient(cfg.URL, cfg.Token),
	}
}

func (s *BoardSink) Send(ctx context.Context, r IncidentReport) error {
	inv := r.Investigation
	shouldPost := inv.RecommendedAction == investigate.ActionEscalate ||
		(r.Fix != nil && !r.Fix.Success)
	if !shouldPost {
		return nil
	}
	if s.cfg.URL == "" {
		slog.Warn("sink: board URL not configured, skipping escalation")
		return nil
	}

	sev := string(r.Candidate.Severity)
	if minSev := s.cfg.MinSeverity; minSev != "" && severityBelow(sev, minSev) {
		return nil
	}

	body := inv.EscalateMessage
	if r.Fix != nil && !r.Fix.Success {
		body = r.Fix.EscalateMessage
		if body == "" {
			body = fmt.Sprintf("⚠️ %s/%s: действие %s не решило проблему.\n\nПричина: %s\n\nПроверка: %s",
				r.Candidate.Workload.Namespace, r.Candidate.Workload.Name,
				r.Fix.ActionTaken, inv.RootCauseRU, r.Fix.VerificationRU)
		}
	}
	if body == "" {
		// Detect a failed/empty investigation (confidence=0, empty fields).
		if inv.Confidence == 0 && inv.RootCauseRU == "" {
			body = fmt.Sprintf("⚠️ [ai-sre] расследование не завершено для %s/%s\n\nАлерт: %s\n\nПроверьте вручную — агент не смог сформировать ответ (пустое поле root_cause). Возможные причины: контекст превысил max_tokens, или модель вернула только reasoning_content. Смотрите логи пода ai-sre.",
				r.Candidate.Workload.Namespace, r.Candidate.Workload.Name, r.Candidate.Summary)
		} else {
			body = fmt.Sprintf("🚨 %s/%s: %s\n\nПричина: %s\n\nРекомендация: %s",
				r.Candidate.Workload.Namespace, r.Candidate.Workload.Name,
				r.Candidate.Summary, inv.RootCauseRU, inv.SuggestedFixRU)
		}
	}
	// Add @mention only when a human is actually needed.
	// Fail-safe: always page on failed fix or failed investigation (confidence==0)
	// regardless of NoProblemFound — those are unambiguous failure modes that
	// must not be suppressible by LLM output.
	// Suppress the mention only when the model explicitly concluded no problem
	// was found (NoProblemFound=true) AND the investigation succeeded
	// (confidence>0, no failed fix).
	alwaysPage := (r.Fix != nil && !r.Fix.Success) || inv.Confidence == 0
	needsMention := inv.EscalateTo != "" && (alwaysPage || !inv.NoProblemFound)
	if needsMention {
		body = fmt.Sprintf("@%s %s", inv.EscalateTo, body)
	}

	threadID := s.cfg.ThreadID
	if threadID == "" {
		title := fmt.Sprintf("[ai-sre] %s/%s %s", r.Candidate.Workload.Namespace, r.Candidate.Workload.Name, r.Candidate.Kind)
		tid, err := s.client.createThread(ctx, title, body)
		if err != nil {
			return fmt.Errorf("board sink: create thread: %w", err)
		}
		slog.Info("sink: board thread created", "thread_id", tid, "workload", r.Candidate.Workload.Name)
		return nil
	}
	return s.client.reply(ctx, threadID, body)
}

// WebhookSink POST-s a JSON payload to a generic HTTP endpoint.
// Always posts the full report.
type WebhookSink struct {
	client *webhookClient
}

func NewWebhookSink(cfg config.WebhookConfig) *WebhookSink {
	return &WebhookSink{client: newWebhookClient(cfg.URL, cfg.Token)}
}

func (s *WebhookSink) Send(ctx context.Context, r IncidentReport) error {
	inv := r.Investigation
	payload := map[string]any{
		"workload":         r.Candidate.Workload.Name,
		"namespace":        r.Candidate.Workload.Namespace,
		"kind":             string(r.Candidate.Kind),
		"severity":         string(r.Candidate.Severity),
		"action":           string(inv.RecommendedAction),
		"confidence":       inv.Confidence,
		"root_cause_ru":    inv.RootCauseRU,
		"suggested_fix_ru": inv.SuggestedFixRU,
		"evidence":         inv.Evidence,
		"escalate_to":      inv.EscalateTo,
		"escalate_message": inv.EscalateMessage,
	}
	if r.Fix != nil {
		payload["fix_action"] = string(r.Fix.ActionTaken)
		payload["fix_success"] = r.Fix.Success
		payload["fix_verification_ru"] = r.Fix.VerificationRU
	}
	return s.client.post(ctx, payload)
}

// severityBelow returns true if a < b in Low<Medium<High<Critical order.
func severityBelow(a, b string) bool {
	order := map[string]int{"low": 0, "medium": 1, "high": 2, "critical": 3}
	return order[strings.ToLower(a)] < order[strings.ToLower(b)]
}

// Build constructs the sink stack from config.
func Build(cfg config.SinksConfig) Sink {
	var sinks []Sink
	if cfg.StdoutJSON {
		sinks = append(sinks, StdoutJSON{})
	}
	if cfg.Board.Enabled && cfg.Board.URL != "" {
		sinks = append(sinks, NewBoardSink(cfg.Board))
	}
	if cfg.Webhook.Enabled && cfg.Webhook.URL != "" {
		sinks = append(sinks, NewWebhookSink(cfg.Webhook))
	}
	if len(sinks) == 0 {
		sinks = append(sinks, StdoutJSON{})
	}
	return NewMulti(sinks...)
}
