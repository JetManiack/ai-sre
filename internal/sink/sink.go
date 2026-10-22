// Package sink sends investigation decisions to configured output channels.
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

// Sink sends a Decision to an output channel.
type Sink interface {
	Send(ctx context.Context, c observer.Candidate, d investigate.Decision) error
}

// Multi fans out to multiple sinks. Errors are logged but don't abort others.
type Multi struct {
	sinks []Sink
}

func NewMulti(sinks ...Sink) *Multi { return &Multi{sinks: sinks} }

func (m *Multi) Send(ctx context.Context, c observer.Candidate, d investigate.Decision) error {
	var errs []string
	for _, s := range m.sinks {
		if err := s.Send(ctx, c, d); err != nil {
			slog.Warn("sink: send error", "err", err)
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("sink errors: %s", strings.Join(errs, "; "))
	}
	return nil
}

// StdoutJSON writes the decision as a JSON line to stdout.
type StdoutJSON struct{}

func (s StdoutJSON) Send(_ context.Context, c observer.Candidate, d investigate.Decision) error {
	out := map[string]any{
		"workload":   c.Workload.Name,
		"namespace":  c.Workload.Namespace,
		"kind":       c.Kind,
		"severity":   c.Severity,
		"action":     d.Action,
		"confidence": d.Confidence,
		"root_cause": d.RootCauseRU,
		"suggested":  d.SuggestedFixRU,
	}
	b, _ := json.Marshal(out)
	_, err := fmt.Fprintf(os.Stdout, "%s\n", b)
	return err
}

// BoardSink posts escalations to the coordination board via MCP HTTP.
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

func (s *BoardSink) Send(ctx context.Context, c observer.Candidate, d investigate.Decision) error {
	if d.Action != investigate.ActionEscalate {
		return nil // only escalations go to board
	}
	if s.cfg.URL == "" {
		slog.Warn("sink: board URL not configured, skipping escalation")
		return nil
	}

	sev := string(c.Severity)
	if minSev := s.cfg.MinSeverity; minSev != "" && severityBelow(sev, minSev) {
		return nil
	}

	body := d.EscalateMessage
	if body == "" {
		body = fmt.Sprintf("🚨 %s/%s: %s\n\nPричина: %s\n\nРекомендация: %s",
			c.Workload.Namespace, c.Workload.Name,
			c.Summary, d.RootCauseRU, d.SuggestedFixRU)
	}
	// Add @mention if configured.
	if d.EscalateTo != "" {
		body = fmt.Sprintf("@%s %s", d.EscalateTo, body)
	}

	threadID := s.cfg.ThreadID
	if threadID == "" {
		// Create a new thread per workload.
		title := fmt.Sprintf("[ai-sre] %s/%s %s", c.Workload.Namespace, c.Workload.Name, c.Kind)
		tid, err := s.client.createThread(ctx, title, body)
		if err != nil {
			return fmt.Errorf("board sink: create thread: %w", err)
		}
		slog.Info("sink: board thread created", "thread_id", tid, "workload", c.Workload.Name)
		return nil
	}
	return s.client.reply(ctx, threadID, body)
}

// WebhookSink POST-s a JSON payload to a generic HTTP endpoint.
type WebhookSink struct {
	client *webhookClient
}

func NewWebhookSink(cfg config.WebhookConfig) *WebhookSink {
	return &WebhookSink{client: newWebhookClient(cfg.URL, cfg.Token)}
}

func (s *WebhookSink) Send(ctx context.Context, c observer.Candidate, d investigate.Decision) error {
	payload := map[string]any{
		"workload":         c.Workload.Name,
		"namespace":        c.Workload.Namespace,
		"kind":             string(c.Kind),
		"severity":         string(c.Severity),
		"action":           string(d.Action),
		"confidence":       d.Confidence,
		"root_cause_ru":    d.RootCauseRU,
		"suggested_fix_ru": d.SuggestedFixRU,
		"evidence":         d.Evidence,
		"escalate_to":      d.EscalateTo,
		"escalate_message": d.EscalateMessage,
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
	if cfg.Board.URL != "" {
		sinks = append(sinks, NewBoardSink(cfg.Board))
	}
	if cfg.Webhook.Enabled && cfg.Webhook.URL != "" {
		sinks = append(sinks, NewWebhookSink(cfg.Webhook))
	}
	if len(sinks) == 0 {
		// Always log to stdout by default.
		sinks = append(sinks, StdoutJSON{})
	}
	return NewMulti(sinks...)
}
