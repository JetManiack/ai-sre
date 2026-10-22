package investigate

import "encoding/json"

// Tool is a read-only diagnostic tool the investigator can call.
// All tools are defined at startup; the LLM picks from this fixed list.
// No bash, no writes, no secret reads.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// toolSchemas is the canonical tool list exposed to the LLM.
// The executor validates every call against this list before dispatch.
var toolSchemas = []Tool{
	{
		Name:        "k8s_describe",
		Description: "Describe a k8s object (pod, deployment, statefulset, node, etc.). Returns kubectl-describe-style output.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"kind":      map[string]any{"type": "string", "description": "e.g. Pod, Deployment, StatefulSet"},
				"name":      map[string]any{"type": "string"},
				"namespace": map[string]any{"type": "string"},
			},
			"required": []string{"kind", "name", "namespace"},
		}),
	},
	{
		Name:        "k8s_logs",
		Description: "Fetch recent logs from a pod/container. Returns at most max_lines lines.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pod":       map[string]any{"type": "string"},
				"namespace": map[string]any{"type": "string"},
				"container": map[string]any{"type": "string", "description": "optional; omit for single-container pods"},
				"lines":     map[string]any{"type": "integer", "description": "default 100, max 500"},
				"since":     map[string]any{"type": "string", "description": "Go duration, e.g. 1h, 30m"},
			},
			"required": []string{"pod", "namespace"},
		}),
	},
	{
		Name:        "k8s_events",
		Description: "List recent k8s events in a namespace, optionally filtered by involved object.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"namespace":       map[string]any{"type": "string"},
				"involved_object": map[string]any{"type": "string", "description": "optional name filter"},
			},
			"required": []string{"namespace"},
		}),
	},
	{
		Name:        "k8s_top",
		Description: "Current CPU and memory usage for pods in a namespace.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"namespace": map[string]any{"type": "string"},
			},
			"required": []string{"namespace"},
		}),
	},
	{
		Name:        "metrics_query",
		Description: "Run a PromQL/MetricsQL instant or range query against Prometheus/VictoriaMetrics. Returns aggregated results.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":  map[string]any{"type": "string", "description": "PromQL expression"},
				"source": map[string]any{"type": "string", "description": "source name from config; omit for default"},
				"range":  map[string]any{"type": "string", "description": "Go duration for range query, e.g. 2h; omit for instant"},
				"step":   map[string]any{"type": "string", "description": "step for range query, e.g. 5m; omit for instant"},
			},
			"required": []string{"query"},
		}),
	},
	{
		Name:        "logs_query",
		Description: "Query VictoriaLogs or Loki for log lines matching a LogsQL/LogQL expression.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":  map[string]any{"type": "string", "description": "LogsQL or LogQL expression"},
				"source": map[string]any{"type": "string", "description": "source name from config; omit for default"},
				"since":  map[string]any{"type": "string", "description": "Go duration, e.g. 2h"},
				"limit":  map[string]any{"type": "integer", "description": "max lines, default 50"},
			},
			"required": []string{"query"},
		}),
	},
	{
		Name:        "host_cmd",
		Description: "Run a read-only host command. Only a fixed allowlist is accepted; only the unit/service name is taken from this call.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"cmd":  map[string]any{"type": "string", "enum": []string{"systemctl_status", "journalctl", "ss_lnt", "df_h"}},
				"unit": map[string]any{"type": "string", "description": "systemd unit name (for systemctl_status and journalctl only)"},
			},
			"required": []string{"cmd"},
		}),
	},
	{
		Name:        "memory_recall",
		Description: "Recall past investigations for a workload from the agent's long-term memory (SQLite). Returns previous root causes and actions taken.",
		InputSchema: mustJSON(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"workload":  map[string]any{"type": "string"},
				"namespace": map[string]any{"type": "string"},
				"limit":     map[string]any{"type": "integer", "description": "default 5"},
			},
			"required": []string{"workload"},
		}),
	},
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// ToolCall is a single LLM→executor request.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}
