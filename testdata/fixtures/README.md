# ai-sre fixture format

Fixtures drive the `--replay <dir>` mode and CI tests. Each fixture is a directory with three files.

## File: `candidate.json`

A single `observer.Candidate` JSON object — the anomaly the observer detected.

```json
{
  "id":          "string — stable dedup fingerprint",
  "observed_at": "RFC3339 timestamp",
  "kind":        "memory_growth | error_rate_spike | oom_kill | crash_loop | log_pattern | pod_pending | readiness_drop | alert_webhook",
  "severity":    "low | medium | high | critical",
  "workload": {
    "kind":      "Deployment | StatefulSet | DaemonSet | Pod | systemd-unit",
    "name":      "string",
    "namespace": "string (empty on bare Linux)",
    "source":    "string — log/metrics source name from config"
  },
  "summary":  "string — one-line human description",
  "evidence": ["string", "..."],
  "raw_metrics": {"key": 0.0}
}
```

## File: `tool_responses.jsonl`

One JSON object per line. Each line is a canned tool response. The dispatcher
matches calls in order, consuming each response once. `input_match` is a partial
match — only the listed keys must equal the actual call; extra keys are ignored.
An empty `{}` matches any input.

```jsonl
{"tool": "memory_recall",   "input_match": {"workload": "myapp"}, "output": "[]"}
{"tool": "metrics_query",   "input_match": {"query": "up"},       "output": "1"}
{"tool": "k8s_top",         "input_match": {},                    "output": "pod cpu memory"}
{"tool": "logs_query",      "input_match": {},                    "output": "line1\nline2"}
{"tool": "k8s_describe",    "input_match": {"kind":"Pod"},        "output": "Name: mypod ..."}
{"tool": "k8s_events",      "input_match": {},                    "output": "Normal ..."}
{"tool": "host_cmd",        "input_match": {"cmd":"df_h"},        "output": "Filesystem ..."}
{"tool": "memory_recall",   "input_match": {},                    "error": "db unavailable"}
{"tool": "logs_query",      "input_match": {"source":"vlogs"},    "input_contains": {"query": "500"}, "output": "..."}
```

**`input_match`** — partial decoded comparison (not raw bytes). All listed keys must
equal the actual call's values. `50` and `50.0` are equal. Extra keys are ignored.

**`input_contains`** (optional) — for each `key: substr`, the actual input must have
that key with a string value containing the substring. Use this when the full query
string differs between a mock and a live model (e.g. match on `"source":"vlogs"` in
`input_match`, and `"query":"500"` in `input_contains`).

Available tools: `memory_recall`, `metrics_query`, `logs_query`, `k8s_describe`,
`k8s_logs`, `k8s_events`, `k8s_top`, `host_cmd`.

## File: `expected.json`

Assertions the test makes on the resulting `Decision`.

```json
{
  "action":                  "escalate | pod_restart | rollout_restart | scale | systemd_restart",
  "confidence_min":          0.6,
  "root_cause_contains":     ["substring1", "substring2"],
  "root_cause_contains_all": ["must-have-1", "must-have-2"]
}
```

**`root_cause_contains`** — at least ONE substring must appear (case-insensitive) in
`root_cause_ru`. Use for broad topic checks, e.g. `["ModelNotMapped", "litellm", "500"]`.

**`root_cause_contains_all`** — ALL substrings must appear. Use for precise checks where
the diagnosis must name specific signals, e.g. `["ModelNotMapped", "litellm"]`. Prefer
this over a long `root_cause_contains` list to avoid false positives.

## Synthetic fixture conventions

- **No real names.** Use generic names: `myapp`, `myapp-worker`, `llm-gateway`,
  `inference-client`. No cluster hostnames, namespace names, IPs, or SA names
  from any real deployment.
- **One scenario per directory.** Name the directory after the failure class:
  `memory-leak`, `gateway-all-500`, `model-alias-removed`, `oom-restart`, etc.
- **Tool responses cover the golden path.** The mock LLM in tests calls
  `memory_recall` first, then produces a final answer. Add responses for any tool
  the golden path calls; unmatched calls return an error (which the agent handles
  as a tool error and continues).

## Bringing real fixtures (site repos)

Real recordings live in your site repo under `infra/ai-sre/fixtures/`. Use the
same three-file layout. Run them with:

```
aisre investigate --replay infra/ai-sre/fixtures/case6-client/ <workload>
```

Real data never goes into this public repo.
