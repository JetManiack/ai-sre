package observer

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// WebhookReceiver is an HTTP server that accepts Grafana/Alertmanager alert
// webhooks and pushes them as Candidates onto the observer channel.
// Start it with Serve(); it runs until ctx is cancelled.
type WebhookReceiver struct {
	addr      string
	token     string
	candidates chan<- Candidate
}

// NewWebhookReceiver creates a receiver on addr (e.g. ":8081").
// If token is non-empty, incoming requests must have Authorization: Bearer <token>.
func NewWebhookReceiver(addr, token string, candidates chan<- Candidate) *WebhookReceiver {
	return &WebhookReceiver{addr: addr, token: token, candidates: candidates}
}

// Serve starts the HTTP server. Blocks until ctx is cancelled.
func (r *WebhookReceiver) Serve(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/webhook", r.handleWebhook)
	srv := &http.Server{Addr: r.addr, Handler: mux}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	slog.Info("observer: webhook receiver starting", "addr", r.addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

// grafanaPayload is the subset of Grafana unified alerting webhook we care about.
// https://grafana.com/docs/grafana/latest/alerting/manage-notifications/webhook-notifier/
type grafanaPayload struct {
	Status string `json:"status"` // "firing" | "resolved"
	Alerts []struct {
		Status      string            `json:"status"`
		Labels      map[string]string `json:"labels"`
		Annotations map[string]string `json:"annotations"`
		StartsAt    time.Time         `json:"startsAt"`
		GeneratorURL string           `json:"generatorURL"`
	} `json:"alerts"`
}

func (r *WebhookReceiver) handleWebhook(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if r.token != "" {
		auth := req.Header.Get("Authorization")
		if auth != "Bearer "+r.token {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
	}
	body, err := io.ReadAll(io.LimitReader(req.Body, 256*1024))
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}

	var payload grafanaPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		slog.Warn("webhook: parse error", "err", err)
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}

	for _, alert := range payload.Alerts {
		if alert.Status != "firing" {
			continue
		}
		c := alertToCandidate(alert.Labels, alert.Annotations, alert.StartsAt)
		select {
		case r.candidates <- c:
			slog.Info("webhook: candidate enqueued", "id", c.ID, "workload", c.Workload.Name)
		default:
			slog.Warn("webhook: candidate channel full, dropping", "id", c.ID)
		}
	}
	w.WriteHeader(http.StatusOK)
}

func alertToCandidate(labels, annotations map[string]string, startsAt time.Time) Candidate {
	wl := Workload{
		Kind:      coalesce(labels["workload_kind"], labels["job"], "Deployment"),
		Name:      coalesce(labels["deployment"], labels["statefulset"], labels["pod"], labels["workload"], labels["alertname"]),
		Namespace: coalesce(labels["namespace"], labels["exported_namespace"], "default"),
		Source:    "grafana-alert",
	}
	summary := coalesce(annotations["summary"], annotations["message"], labels["alertname"])
	sev := grafanaSeverity(labels["severity"])
	id := Fingerprint(wl, KindAlertWebhook, labels["alertname"])
	return Candidate{
		ID:         id,
		ObservedAt: startsAt,
		Kind:       KindAlertWebhook,
		Severity:   sev,
		Workload:   wl,
		Summary:    summary,
		Evidence:   []string{summary},
	}
}

func grafanaSeverity(s string) Severity {
	switch s {
	case "critical":
		return SeverityCritical
	case "warning":
		return SeverityMedium
	case "info":
		return SeverityLow
	default:
		return SeverityHigh
	}
}

func coalesce(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return "unknown"
}
