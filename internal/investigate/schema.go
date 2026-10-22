package investigate

// Decision is the structured output of the investigator+decision agent.
// The LLM fills this after its tool loop; the executor acts on it.
type Decision struct {
	// RootCauseRU is a concise root-cause statement in Russian.
	RootCauseRU string `json:"root_cause_ru"`
	// Evidence is a list of masked log lines / metric values that support the diagnosis.
	Evidence []string `json:"evidence"`
	// Confidence is 0.0–1.0.
	Confidence float64 `json:"confidence"`
	// SuggestedFixRU describes what should be done, in Russian.
	SuggestedFixRU string `json:"suggested_fix_ru"`
	// NextChecks are further checks recommended if action is not taken.
	NextChecks []string `json:"next_checks,omitempty"`

	// Action is what the agent decides to do.
	Action ActionKind `json:"action"`
	// ActionTarget identifies the resource to act on (pod name, deployment, unit…).
	ActionTarget string `json:"action_target,omitempty"`
	// ActionNamespace is the k8s namespace of the target (empty on bare Linux).
	ActionNamespace string `json:"action_namespace,omitempty"`

	// EscalateTo is a board actor ID or nickname to @-mention on escalation.
	// The agent fills this based on the kind of fix needed (code bug → developer,
	// infra issue → operator).
	EscalateTo string `json:"escalate_to,omitempty"`
	// EscalateMessage is the board post body to send. Pre-written by the LLM.
	EscalateMessage string `json:"escalate_message,omitempty"`
}

// ActionKind is what the agent decides to do after diagnosis.
type ActionKind string

const (
	// ActionEscalate posts a diagnosis to the board / Telegram and stops.
	ActionEscalate ActionKind = "escalate"
	// ActionPodRestart restarts the affected pod.
	ActionPodRestart ActionKind = "pod_restart"
	// ActionRolloutRestart does a rollout restart of the Deployment/StatefulSet.
	ActionRolloutRestart ActionKind = "rollout_restart"
	// ActionScale scales the Deployment to ActionTarget replicas (never below baseline).
	ActionScale ActionKind = "scale"
	// ActionSystemdRestart restarts a systemd unit on bare Linux.
	ActionSystemdRestart ActionKind = "systemd_restart"
)
