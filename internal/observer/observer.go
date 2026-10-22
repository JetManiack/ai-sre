package observer

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/JetManiack/ai-sre/internal/config"
)

// Observer watches k8s state, metrics, and logs for anomalies.
// It emits Candidates on the output channel; dedup is done via the QuietStore.
type Observer struct {
	cfg        config.ObserverConfig
	quiet      QuietStore
	candidates chan<- Candidate
	adapters   []Adapter
}

// QuietStore checks and sets quiet windows for fingerprints.
type QuietStore interface {
	IsQuiet(fingerprint string) (bool, error)
	SetQuiet(fingerprint string, until time.Time) error
}

// Adapter is a pluggable anomaly source (k8s, metrics, logs, host).
type Adapter interface {
	// Name returns a short identifier for logging.
	Name() string
	// Poll runs one detection cycle and returns any new candidates.
	// Implementations must return quickly — heavy I/O belongs here, not in a goroutine.
	Poll(ctx context.Context) ([]Candidate, error)
}

// New creates an Observer. candidates is written to when an anomaly passes dedup.
func New(cfg config.ObserverConfig, quiet QuietStore, candidates chan<- Candidate) *Observer {
	return &Observer{cfg: cfg, quiet: quiet, candidates: candidates}
}

// Register adds an adapter to the observer.
func (o *Observer) Register(a Adapter) {
	o.adapters = append(o.adapters, a)
}

// Run starts the observer loop. Blocks until ctx is cancelled.
func (o *Observer) Run(ctx context.Context) {
	ticker := time.NewTicker(o.cfg.Interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			o.pollAll(ctx)
		}
	}
}

func (o *Observer) pollAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, a := range o.adapters {
		a := a
		wg.Add(1)
		go func() {
			defer wg.Done()
			candidates, err := a.Poll(ctx)
			if err != nil {
				slog.Warn("observer: adapter poll error", "adapter", a.Name(), "err", err)
				return
			}
			for _, c := range candidates {
				o.emit(c)
			}
		}()
	}
	wg.Wait()
}

func (o *Observer) emit(c Candidate) {
	quiet, err := o.quiet.IsQuiet(c.ID)
	if err != nil {
		slog.Warn("observer: quiet check error", "fingerprint", c.ID, "err", err)
	}
	if quiet {
		return
	}
	// Set quiet window immediately to prevent duplicate investigations.
	_ = o.quiet.SetQuiet(c.ID, time.Now().Add(o.cfg.QuietWindowDefault))
	select {
	case o.candidates <- c:
	default:
		slog.Warn("observer: candidate channel full, dropping", "id", c.ID, "workload", c.Workload.Name)
	}
}

// Fingerprint builds a stable dedup ID for a candidate.
func Fingerprint(workload Workload, kind Kind, detail string) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%s|%s|%s", workload.Source, workload.Namespace, workload.Name, kind, detail)
	return fmt.Sprintf("%x", h.Sum(nil))[:16]
}

// MatchesFailOpen returns true if any of the fail-open patterns appear in text.
func MatchesFailOpen(text string, patterns []string) (string, bool) {
	lower := strings.ToLower(text)
	for _, p := range patterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return p, true
		}
	}
	return "", false
}
