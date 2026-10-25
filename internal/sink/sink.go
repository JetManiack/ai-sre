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
	Candidate     observer.Candidate
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
	}
	if r.Fix != nil {
		payload["fix_action"] = string(r.Fix.ActionTaken)
		payload["fix_success"] = r.Fix.Success
		payload["fix_verification_ru"] = r.Fix.VerificationRU
	}
	return s.client.post(ctx, payload)
}

// Build constructs the sink stack from config.
func Build(cfg config.SinksConfig) Sink {
	var sinks []Sink
	if cfg.StdoutJSON {
		sinks = append(sinks, StdoutJSON{})
	}
	if cfg.Webhook.Enabled && cfg.Webhook.URL != "" {
		sinks = append(sinks, NewWebhookSink(cfg.Webhook))
	}
	if len(sinks) == 0 {
		sinks = append(sinks, StdoutJSON{})
	}
	return NewMulti(sinks...)
}
