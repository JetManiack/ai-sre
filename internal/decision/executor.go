// Package decision executes autonomous actions decided by the LLM agent.
// Every action is validated against the operator allowlist before execution.
// The k8s write operations go through a dedicated MCP server (if configured),
// or directly through the k8s REST API for in-cluster deployments.
package decision

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/investigate"
)

// Executor validates a Decision against the allowlist and executes the action.
type Executor struct {
	cfg    config.ActionsConfig
	k8sMCP *mcpWriteClient // nil = no write MCP configured
}

// mcpWriteClient is a thin MCP client for k8s write operations.
type mcpWriteClient struct {
	baseURL string
	token   string
	hc      *http.Client
}

// NewExecutor creates an Executor.
// k8sWriteMCP is optional; if empty no autonomous k8s actions are possible.
func NewExecutor(cfg config.ActionsConfig, k8sWriteURL, k8sWriteToken string) *Executor {
	e := &Executor{cfg: cfg}
	if k8sWriteURL != "" {
		e.k8sMCP = &mcpWriteClient{
			baseURL: strings.TrimRight(k8sWriteURL, "/"),
			token:   k8sWriteToken,
			hc:      &http.Client{Timeout: 60 * time.Second},
		}
	}
	return e
}

// Execute validates and runs the action in inv. Returns nil if the action was
// executed (or is escalate-only). Returns an error if the action is not in the
// allowlist or execution failed.
func (e *Executor) Execute(ctx context.Context, inv investigate.InvestigationResult) error {
	switch inv.RecommendedAction {
	case investigate.ActionEscalate:
		// Escalate is always permitted — handled by the sink, not here.
		return nil
	case investigate.ActionPodRestart:
		if !e.cfg.AllowPodRestart {
			return fmt.Errorf("pod_restart not in allowlist")
		}
		return e.podRestart(ctx, inv.ActionNamespace, inv.ActionTarget)
	case investigate.ActionRolloutRestart:
		if !e.cfg.AllowRolloutRestart {
			return fmt.Errorf("rollout_restart not in allowlist")
		}
		return e.rolloutRestart(ctx, inv.ActionNamespace, inv.ActionTarget)
	case investigate.ActionScale:
		if !e.cfg.AllowScale {
			return fmt.Errorf("scale not in allowlist")
		}
		// ActionTarget encodes "name:replicas" to keep InvestigationResult schema simple.
		return e.scale(ctx, inv.ActionNamespace, inv.ActionTarget)
	case investigate.ActionSystemdRestart:
		if !e.cfg.AllowSystemdRestart {
			return fmt.Errorf("systemd_restart not in allowlist")
		}
		return e.systemdRestart(ctx, inv.ActionTarget)
	default:
		return fmt.Errorf("unknown action kind: %q", inv.RecommendedAction)
	}
}

func (e *Executor) podRestart(ctx context.Context, namespace, podName string) error {
	if e.k8sMCP == nil {
		return fmt.Errorf("pod_restart: no k8s write MCP configured")
	}
	slog.Info("executor: pod_restart", "namespace", namespace, "pod", podName)
	args := map[string]any{"namespace": namespace, "name": podName}
	return e.k8sMCP.call(ctx, "k8s_delete_pod", args)
}

func (e *Executor) rolloutRestart(ctx context.Context, namespace, name string) error {
	if e.k8sMCP == nil {
		return fmt.Errorf("rollout_restart: no k8s write MCP configured")
	}
	slog.Info("executor: rollout_restart", "namespace", namespace, "deployment", name)
	args := map[string]any{"namespace": namespace, "name": name, "kind": "Deployment"}
	return e.k8sMCP.call(ctx, "k8s_rollout_restart", args)
}

func (e *Executor) scale(ctx context.Context, namespace, target string) error {
	if e.k8sMCP == nil {
		return fmt.Errorf("scale: no k8s write MCP configured")
	}
	// target format: "name:replicas"
	parts := strings.SplitN(target, ":", 2)
	if len(parts) != 2 {
		return fmt.Errorf("scale: invalid target %q, want name:replicas", target)
	}
	slog.Info("executor: scale", "namespace", namespace, "deployment", parts[0], "replicas", parts[1])
	args := map[string]any{"namespace": namespace, "name": parts[0], "replicas": parts[1]}
	return e.k8sMCP.call(ctx, "k8s_scale", args)
}

func (e *Executor) systemdRestart(ctx context.Context, unit string) error {
	if err := investigate.ValidateUnitName(unit); err != nil {
		return fmt.Errorf("systemd_restart: %w", err)
	}
	slog.Info("executor: systemd_restart", "unit", unit)
	// On bare Linux we exec systemctl directly.
	// This path only runs when AllowSystemdRestart is true and the agent
	// is deployed as a systemd service itself (not in a container).
	return runSystemctl(ctx, "restart", unit)
}

// call calls a tool on the write MCP server.
func (c *mcpWriteClient) call(ctx context.Context, tool string, args map[string]any) error {
	body, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params":  map[string]any{"name": tool, "arguments": args},
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/mcp", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return fmt.Errorf("k8s write MCP: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("k8s write MCP: HTTP %d: %s", resp.StatusCode, raw)
	}
	var r struct {
		Error *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(raw, &r); err == nil && r.Error != nil {
		return fmt.Errorf("k8s write MCP: %s", r.Error.Message)
	}
	return nil
}
