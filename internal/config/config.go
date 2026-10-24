// Package config loads and validates the ai-sre configuration.
package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration structure.
type Config struct {
	// State is the SQLite database path for fingerprints, baselines and quiet windows.
	State string `yaml:"state"`

	// MCPServers lists MCP servers whose tools the agent may call.
	MCPServers []MCPServer `yaml:"mcp_servers"`

	// Sources lists log/metrics backends (used by observer adapters for polling).
	Sources []Source `yaml:"sources"`

	// Kubernetes configures the k8s adapter. Omit on bare Linux.
	Kubernetes KubernetesConfig `yaml:"kubernetes"`

	// Observer configures anomaly detection.
	Observer ObserverConfig `yaml:"observer"`

	// Investigate configures the LLM investigator+decision agent.
	Investigate InvestigateConfig `yaml:"investigate"`

	// Actions lists what the agent is allowed to execute autonomously.
	// Empty by default (escalate-only mode).
	Actions ActionsConfig `yaml:"actions"`

	// Sinks configure where findings and decisions are sent.
	Sinks SinksConfig `yaml:"sinks"`
}

// MCPServer is an MCP (Model Context Protocol) server that exposes tools.
// The agent calls its tools via Streamable HTTP (POST /mcp, SSE or JSON response).
type MCPServer struct {
	// Name is a short identifier (e.g. "incident-io", "loki", "thanos").
	Name string `yaml:"name"`
	// URL is the base URL of the MCP server (e.g. http://mcp-loki:8080).
	URL string `yaml:"url"`
	// Token is an optional Bearer token for Authorization header.
	Token string `yaml:"token"`
	// Tools lists which tool names from this server are available to the agent.
	// Empty means all tools the server advertises.
	Tools []string `yaml:"tools"`
}

// Source is a log or metrics backend (used by observer adapters, not the agent tools).
type Source struct {
	Name   string `yaml:"name"`
	Kind   string `yaml:"kind"` // "victorialogs", "loki", "prometheus", "victoriametrics"
	URL    string `yaml:"url"`
	Token  string `yaml:"token"`
	CAFile string `yaml:"ca_file"`
}

// KubernetesConfig controls the k8s informer adapter.
type KubernetesConfig struct {
	// Kubeconfig path; empty = in-cluster.
	Kubeconfig string `yaml:"kubeconfig"`
	// Namespaces to watch; empty = all.
	Namespaces []string `yaml:"namespaces"`
}

// ObserverConfig controls anomaly detection thresholds.
type ObserverConfig struct {
	// Interval is how often metrics and logs are polled (default 1m).
	Interval time.Duration `yaml:"interval"`
	// QuietWindowDefault is the dedup window after a candidate fires (default 1h).
	QuietWindowDefault time.Duration `yaml:"quiet_window_default"`
	// MemoryGrowthThresholdMBPerHour triggers a candidate when exceeded (default 50).
	MemoryGrowthThresholdMBPerHour float64 `yaml:"memory_growth_threshold_mb_per_hour"`
	// ErrorRateThresholdPct triggers on HTTP 4xx/5xx rate above this % (default 5).
	ErrorRateThresholdPct float64 `yaml:"error_rate_threshold_pct"`
	// FailOpenPatterns are substrings in logs that always trigger a candidate.
	FailOpenPatterns []string `yaml:"fail_open_patterns"`
	// AlertWebhook configures an HTTP receiver for Grafana/Alertmanager webhooks.
	AlertWebhook AlertWebhookConfig `yaml:"alert_webhook"`
	// Workloads is a per-workload configuration overlay (probes, known-benign, etc.).
	Workloads []WorkloadConfig `yaml:"workloads"`
}

// AlertWebhookConfig configures the Grafana/Alertmanager webhook receiver.
type AlertWebhookConfig struct {
	// Enabled starts the HTTP receiver when true.
	Enabled bool `yaml:"enabled"`
	// Addr is the listen address (default ":8081").
	Addr string `yaml:"addr"`
	// Token is an optional bearer token for incoming requests.
	Token string `yaml:"token"`
}

// WorkloadConfig is a per-service observer overlay.
type WorkloadConfig struct {
	// Name matches Deployment/StatefulSet/systemd-unit name.
	Name      string `yaml:"name"`
	Namespace string `yaml:"namespace"`
	// Probes are synthetic HTTP checks for functional health (not just liveness).
	Probes []ProbeConfig `yaml:"probes"`
	// KnownBenign patterns suppress candidates for this workload only.
	KnownBenign []string `yaml:"known_benign"`
}

// ProbeConfig is a synthetic HTTP probe.
type ProbeConfig struct {
	// URL to GET; may use template variables: {{.Name}}, {{.Namespace}}.
	URL            string        `yaml:"url"`
	ExpectStatus   int           `yaml:"expect_status"`   // default 200
	ExpectBodyContains string    `yaml:"expect_body_contains"`
	Timeout        time.Duration `yaml:"timeout"`         // default 10s
}

// InvestigateConfig controls the LLM investigator+decision agent.
type InvestigateConfig struct {
	Enabled        bool   `yaml:"enabled"`
	BaseURL        string `yaml:"base_url"`
	APIKey         string `yaml:"api_key"`
	Model          string `yaml:"model"`
	// SystemPrompt is prepended to every investigation (site-specific context).
	SystemPrompt string `yaml:"system_prompt"`

	// EscalateToCodeBugs is the board actor to escalate to for application/code bugs.
	// Defaults to "on-call-dev" if empty.
	EscalateToCodeBugs string `yaml:"escalate_to_code_bugs"`
	// EscalateToInfra is the board actor to escalate to for operator/infra decisions.
	// Defaults to "on-call-ops" if empty.
	EscalateToInfra string `yaml:"escalate_to_infra"`

	MaxSteps       int           `yaml:"max_steps"`        // default 8
	MaxContextTokens int         `yaml:"max_context_tokens"` // default 16000
	MaxToolOutputTokens int      `yaml:"max_tool_output_tokens"` // default 1500
	MaxTokensStep  int           `yaml:"max_tokens_step"`   // default 256
	MaxTokensFinal int           `yaml:"max_tokens_final"`  // default 1200
	StepTimeout    time.Duration `yaml:"step_timeout"`      // default 60s
	TotalTimeout   time.Duration `yaml:"total_timeout"`     // default 240s
	// Concurrency is max simultaneous investigations (default 1).
	Concurrency int `yaml:"concurrency"`
	// DebugRequestDump enables logging of message content previews (first 200
	// chars per message) in logFinalRequestDump. Disabled by default: tool
	// outputs and prompt content can contain sensitive infrastructure data.
	DebugRequestDump bool `yaml:"debug_request_dump"`
}

// ActionsConfig lists autonomous actions the agent may execute.
// All fields default to false (escalate-only by default).
type ActionsConfig struct {
	AllowPodRestart     bool `yaml:"allow_pod_restart"`
	AllowRolloutRestart bool `yaml:"allow_rollout_restart"`
	// AllowScale permits scaling a deployment up or to its configured baseline.
	// The agent may NOT scale below baseline.
	AllowScale bool `yaml:"allow_scale"`
	// AllowSystemdRestart permits `systemctl restart <unit>` on bare Linux.
	AllowSystemdRestart bool `yaml:"allow_systemd_restart"`

	// K8sWriteURL is the base URL of a k8s write MCP server.
	// Required for pod_restart, rollout_restart, scale.
	// If empty, those actions fail gracefully and the decision is escalated.
	K8sWriteURL   string `yaml:"k8s_write_url"`
	K8sWriteToken string `yaml:"k8s_write_token"`
}

// SinksConfig lists output channels.
type SinksConfig struct {
	StdoutJSON bool            `yaml:"stdout_json"`
	Board      BoardSinkConfig `yaml:"board"`
	Webhook    WebhookConfig   `yaml:"webhook"`
}

// BoardSinkConfig configures the coordination board MCP client.
type BoardSinkConfig struct {
	Enabled     bool   `yaml:"enabled"`
	URL         string `yaml:"url"`
	Token       string `yaml:"token"`
	MinSeverity string `yaml:"min_severity"`
	// ThreadID pins all escalations to a specific board thread.
	// If empty, a new thread is created per workload.
	ThreadID string `yaml:"thread_id"`
	// MentionActors maps severity levels to board actor IDs to @-mention.
	// e.g. {"critical": "operator-actor-id"}
	MentionActors map[string]string `yaml:"mention_actors"`
}

// WebhookConfig is a generic HTTP POST sink.
type WebhookConfig struct {
	Enabled bool   `yaml:"enabled"`
	URL     string `yaml:"url"`
	Token   string `yaml:"token"`
}

// Load reads and parses the YAML config at path.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config: read %s: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("config: parse %s: %w", path, err)
	}
	setDefaults(&cfg)
	return &cfg, nil
}

func setDefaults(cfg *Config) {
	if cfg.State == "" {
		cfg.State = "data/aisre.db"
	}
	o := &cfg.Observer
	if o.Interval == 0 {
		o.Interval = time.Minute
	}
	if o.QuietWindowDefault == 0 {
		o.QuietWindowDefault = time.Hour
	}
	if o.MemoryGrowthThresholdMBPerHour == 0 {
		o.MemoryGrowthThresholdMBPerHour = 50
	}
	if o.ErrorRateThresholdPct == 0 {
		o.ErrorRateThresholdPct = 5
	}
	if len(o.FailOpenPatterns) == 0 {
		o.FailOpenPatterns = []string{
			"OOMKilled", "automigrate", "connection refused",
			"LocalServerException", "x509", "no such host",
			"permission denied", "No api key passed in",
			"key not allowed to access model", "EngineDeadError",
		}
	}
	inv := &cfg.Investigate
	if inv.MaxSteps == 0 {
		inv.MaxSteps = 8
	}
	if inv.MaxContextTokens == 0 {
		inv.MaxContextTokens = 16000
	}
	if inv.MaxToolOutputTokens == 0 {
		inv.MaxToolOutputTokens = 1500
	}
	if inv.MaxTokensStep == 0 {
		inv.MaxTokensStep = 256
	}
	if inv.MaxTokensFinal == 0 {
		inv.MaxTokensFinal = 1200
	}
	if inv.StepTimeout == 0 {
		inv.StepTimeout = 60 * time.Second
	}
	if inv.TotalTimeout == 0 {
		inv.TotalTimeout = 240 * time.Second
	}
	if inv.Concurrency == 0 {
		inv.Concurrency = 1
	}
}
