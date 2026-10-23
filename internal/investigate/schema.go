package investigate

// InvestigationResult is the structured output of Phase 1 (read-only ReAct loop).
// The LLM fills this after its tool loop; the executor and fixer act on it.
type InvestigationResult struct {
	// RootCauseRU is a concise root-cause statement in Russian.
	RootCauseRU string `json:"root_cause_ru"`
	// Evidence is a list of log lines / metric values that support the diagnosis.
	Evidence []string `json:"evidence"`
	// Confidence is 0.0–1.0.
	Confidence float64 `json:"confidence"`
	// SuggestedFixRU describes what should be done, in Russian.
	SuggestedFixRU string `json:"suggested_fix_ru"`
	// NextChecks are further checks recommended if action is not taken.
	NextChecks []string `json:"next_checks,omitempty"`

	// RecommendedAction is what the agent recommends.
	// The executor validates it against the allowlist before acting.
	RecommendedAction ActionKind `json:"recommended_action"`
	// ActionTarget identifies the resource (pod name, deployment, unit…).
	// Always clamped to the originating Candidate workload — never LLM-supplied.
	ActionTarget string `json:"action_target,omitempty"`
	// ActionNamespace is the k8s namespace. Always clamped to Candidate.Workload.Namespace.
	ActionNamespace string `json:"action_namespace,omitempty"`

	// EscalateTo is a board actor ID / nickname to @-mention.
	EscalateTo string `json:"escalate_to,omitempty"`
	// EscalateMessage is the board post body. Pre-written by the LLM.
	EscalateMessage string `json:"escalate_message,omitempty"`
}

// FixResult is the structured output of Phase 2 (verify after action).
type FixResult struct {
	// ActionTaken is the action that was actually executed.
	ActionTaken ActionKind `json:"action_taken"`
	// Success is true when verification confirms the issue is resolved.
	Success bool `json:"success"`
	// VerificationRU describes what was checked and what was observed, in Russian.
	VerificationRU string `json:"verification_ru"`
	// EscalateMessage is filled when Success=false, describing what still needs attention.
	EscalateMessage string `json:"escalate_message,omitempty"`
}

// ActionKind is what the agent recommends after diagnosis.
type ActionKind string

const (
	// ActionEscalate posts a diagnosis to the board / Telegram and stops.
	ActionEscalate ActionKind = "escalate"
	// ActionPodRestart restarts the affected pod.
	ActionPodRestart ActionKind = "pod_restart"
	// ActionRolloutRestart does a rollout restart of the Deployment/StatefulSet.
	ActionRolloutRestart ActionKind = "rollout_restart"
	// ActionScale scales the Deployment (never below baseline).
	ActionScale ActionKind = "scale"
	// ActionSystemdRestart restarts a systemd unit on bare Linux.
	ActionSystemdRestart ActionKind = "systemd_restart"
)

// Decision is kept as a type alias for backward compatibility with replay tests.
// New code should use InvestigationResult directly.
type Decision = InvestigationResult
