package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/urfave/cli/v3"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/investigate"
	"github.com/JetManiack/ai-sre/internal/observer"
	"github.com/JetManiack/ai-sre/internal/replay"
	"github.com/JetManiack/ai-sre/internal/state"
	"github.com/JetManiack/ai-sre/internal/tools"
)

func investigateCmd() *cli.Command {
	return &cli.Command{
		Name:      "investigate",
		Usage:     "one-shot investigation of a workload (for replays, CI, manual use)",
		ArgsUsage: "<workload-name>",
		Flags: []cli.Flag{
			&cli.StringFlag{Name: "config", Aliases: []string{"c"}, Value: "/etc/ai-sre/config.yaml"},
			&cli.StringFlag{Name: "namespace", Aliases: []string{"n"}, Value: "default"},
			&cli.StringFlag{Name: "kind", Value: "Deployment", Usage: "workload kind"},
			&cli.StringFlag{Name: "source", Value: "", Usage: "log/metrics source name from config"},
			&cli.StringFlag{Name: "replay", Value: "", Usage: "run from a fixture directory instead of live data"},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			replayDir := cmd.String("replay")
			if replayDir != "" {
				return runReplay(ctx, cmd, replayDir)
			}

			name := cmd.Args().First()
			if name == "" {
				return cli.Exit("workload name required", 1)
			}
			return runLive(ctx, cmd, name)
		},
	}
}

// runLive performs a one-shot investigation against live MCP servers.
// It builds a synthetic Candidate from the CLI flags and runs the full agent loop.
func runLive(ctx context.Context, cmd *cli.Command, name string) error {
	cfgPath := cmd.String("config")
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	if !cfg.Investigate.Enabled {
		return fmt.Errorf("investigate.enabled is false in config")
	}

	db, err := state.Open(cfg.State)
	if err != nil {
		return fmt.Errorf("state: %w", err)
	}
	defer db.Close()

	mem := &stateMemory{db: db}
	dispatcher, err := tools.NewDispatcher(ctx, cfg.MCPServers, mem)
	if err != nil {
		return fmt.Errorf("dispatcher: %w", err)
	}
	discoveredTools := dispatcher.DiscoveredTools()
	if len(discoveredTools) == 0 {
		return fmt.Errorf("no tools discovered from MCP servers; check mcp_servers config")
	}

	llmClient := investigate.NewOpenAIClient(cfg.Investigate.BaseURL, cfg.Investigate.APIKey)
	agent := investigate.NewWithTools(cfg.Investigate, cfg.Sources, discoveredTools, dispatcher, mem, llmClient)

	c := observer.Candidate{
		ID:         fmt.Sprintf("live-%d", time.Now().UnixNano()),
		ObservedAt: time.Now(),
		Kind:       observer.KindAlertWebhook,
		Severity:   observer.SeverityHigh,
		Workload: observer.Workload{
			Kind:      cmd.String("kind"),
			Name:      name,
			Namespace: cmd.String("namespace"),
			Source:    cmd.String("source"),
		},
		Summary:  fmt.Sprintf("manual investigation: %s/%s", cmd.String("namespace"), name),
		Evidence: []string{},
	}
	slog.Info("aisre: live investigate", "workload", name, "namespace", c.Workload.Namespace,
		"tools", len(discoveredTools))

	result := agent.Investigate(ctx, c)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

func runReplay(ctx context.Context, cmd *cli.Command, dir string) error {
	f, err := replay.Load(dir)
	if err != nil {
		return fmt.Errorf("replay: %w", err)
	}
	slog.Info("aisre: replay", "dir", dir, "candidate", f.Candidate.ID,
		"workload", f.Candidate.Workload.Name, "kind", f.Candidate.Kind)

	cfgPath := cmd.String("config")
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	if !cfg.Investigate.Enabled {
		return fmt.Errorf("investigate.enabled is false in config; set it to run replay")
	}

	db, err := state.Open(cfg.State)
	if err != nil {
		return fmt.Errorf("state: %w", err)
	}
	defer db.Close()

	mem := &stateMemory{db: db}
	dispatcher := replay.NewDispatcher(f.Responses)
	llmClient := investigate.NewOpenAIClient(cfg.Investigate.BaseURL, cfg.Investigate.APIKey)
	agent := investigate.New(cfg.Investigate, dispatcher, mem, llmClient)

	result := agent.Investigate(ctx, f.Candidate)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		return fmt.Errorf("encode result: %w", err)
	}

	// Validate against expected outcome if present.
	if err := replay.CheckExpected(f.Expected, result); err != nil {
		slog.Warn("aisre: replay assertion failed", "err", err)
		fmt.Fprintf(os.Stderr, "ASSERTION FAILED: %v\n", err)
		os.Exit(2)
	}
	slog.Info("aisre: replay OK", "action", result.RecommendedAction, "confidence", result.Confidence)
	return nil
}

// stateMemory adapts state.DB to investigate.MemoryStore.
type stateMemory struct{ db *state.DB }

func (m *stateMemory) RecallInvestigations(workload, namespace string, limit int) ([]investigate.PastInvestigation, error) {
	rows, err := m.db.RecallInvestigations(workload, namespace, limit)
	if err != nil {
		return nil, err
	}
	out := make([]investigate.PastInvestigation, len(rows))
	for i, r := range rows {
		out[i] = investigate.PastInvestigation{
			ID:            r.ID,
			WorkloadName:  r.Workload,
			Namespace:     r.Namespace,
			CandidateKind: r.CandidateKind,
			RootCauseRU:   r.RootCauseRU,
			Action:        r.Action,
			Confidence:    r.Confidence,
			OccurredAt:    r.OccurredAt,
		}
	}
	return out, nil
}

func (m *stateMemory) SaveInvestigation(inv investigate.PastInvestigation) error {
	return m.db.SaveInvestigation(inv.ID, inv.WorkloadName, inv.Namespace,
		inv.CandidateKind, inv.RootCauseRU, inv.Action, inv.Confidence, inv.OccurredAt)
}
