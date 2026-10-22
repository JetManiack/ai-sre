// Package observer detects anomalies across k8s state, metrics, and logs,
// emitting Candidates for the investigator when something looks wrong.
package observer

import "time"

// Kind classifies what triggered the candidate.
type Kind string

const (
	KindOOMKill        Kind = "oom_kill"
	KindCrashLoop      Kind = "crash_loop"
	KindMemoryGrowth   Kind = "memory_growth"
	KindErrorRateSpike Kind = "error_rate_spike"
	KindLogPattern     Kind = "log_pattern"
	KindPodPending     Kind = "pod_pending"
	KindReadinessDrop  Kind = "readiness_drop"
	KindAlertWebhook   Kind = "alert_webhook"
)

// Severity mirrors the sentinel's model.
type Severity string

const (
	SeverityLow      Severity = "low"
	SeverityMedium   Severity = "medium"
	SeverityHigh     Severity = "high"
	SeverityCritical Severity = "critical"
)

// Workload identifies the affected service (platform-agnostic).
type Workload struct {
	// Kind is "Deployment", "StatefulSet", "DaemonSet", "Pod", "systemd-unit", etc.
	Kind      string `json:"kind"`
	Name      string `json:"name"`
	Namespace string `json:"namespace"` // empty on bare Linux
	// Source is the configured log/metrics source name (matches config LogSource.Name).
	Source string `json:"source"`
}

// Candidate is an anomaly detected by the Observer that warrants investigation.
type Candidate struct {
	ID         string    `json:"id"`
	ObservedAt time.Time `json:"observed_at"`
	Kind       Kind      `json:"kind"`
	Severity   Severity  `json:"severity"`
	Workload   Workload  `json:"workload"`
	Summary    string    `json:"summary"`  // one-line human-readable description
	Evidence   []string  `json:"evidence"` // masked log lines, metric values, k8s event snippets
	// RawMetrics holds numeric values for the detector (e.g. growth_rate_mb_per_hour).
	RawMetrics map[string]float64 `json:"raw_metrics,omitempty"`
}
