package main

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/urfave/cli/v3"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/decision"
	"github.com/JetManiack/ai-sre/internal/investigate"
	"github.com/JetManiack/ai-sre/internal/observer"
	"github.com/JetManiack/ai-sre/internal/sink"
	"github.com/JetManiack/ai-sre/internal/state"
	"github.com/JetManiack/ai-sre/internal/tools"
)

func serveCmd() *cli.Command {
	return &cli.Command{
		Name:  "serve",
		Usage: "run the Grafana webhook receiver + investigator daemon",
		Flags: []cli.Flag{
			&cli.StringFlag{Name: "config", Aliases: []string{"c"}, Value: "/etc/ai-sre/config.yaml", Usage: "config file path"},
		},
		Action: runServe,
	}
}

func runServe(ctx context.Context, cmd *cli.Command) error {
	cfg, err := config.Load(cmd.String("config"))
	if err != nil {
		return err
	}

	db, err := state.Open(cfg.State)
	if err != nil {
		return err
	}
	defer db.Close()

	mem := &stateMemory{db: db}

	// Build the tool dispatcher (MCP servers + local memory_recall).
	dispatcher, err := tools.NewDispatcher(ctx, cfg.MCPServers, mem)
	if err != nil {
		return err
	}

	// In serve mode we require at least one MCP server to advertise tools.
	discoveredTools := dispatcher.DiscoveredTools()
	if len(discoveredTools) == 0 {
		return fmt.Errorf("serve: no tools discovered from MCP servers; check mcp_servers config and server connectivity")
	}

	llmClient := investigate.NewOpenAIClient(cfg.Investigate.BaseURL, cfg.Investigate.APIKey)

	newAgent := func() *investigate.Agent {
		return investigate.NewWithTools(cfg.Investigate, cfg.Sources, discoveredTools, dispatcher, mem, llmClient)
	}
	newFixer := func() *investigate.Fixer {
		return investigate.NewFixer(cfg.Investigate, discoveredTools, dispatcher, llmClient)
	}

	executor := decision.NewExecutor(cfg.Actions, cfg.Actions.K8sWriteURL, cfg.Actions.K8sWriteToken)
	sinkStack := sink.Build(cfg.Sinks)

	candidateCh := make(chan observer.Candidate, 32)

	var wg sync.WaitGroup

	// Grafana/Alertmanager webhook receiver is the sole input source.
	// Requires alert_webhook.enabled: true and a non-empty token in config.
	wh := cfg.Observer.AlertWebhook
	if !wh.Enabled {
		return fmt.Errorf("serve: observer.alert_webhook.enabled must be true — it is the only input source")
	}
	if wh.Token == "" {
		return fmt.Errorf("serve: observer.alert_webhook.token must be set — unauthenticated webhook is not allowed")
	}
	addr := wh.Addr
	if addr == "" {
		addr = ":8081"
	}
	recv := observer.NewWebhookReceiver(addr, wh.Token, candidateCh)
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := recv.Serve(ctx); err != nil {
			slog.Warn("observer: webhook receiver error", "err", err)
		}
	}()

	// Investigator worker pool.
	concurrency := cfg.Investigate.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	slog.Info("aisre: serve starting",
		"addr", addr,
		"concurrency", concurrency,
		"model", cfg.Investigate.Model,
		"tools", len(discoveredTools),
	)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			processWorker(ctx, candidateCh, newAgent, newFixer, executor, sinkStack)
		}()
	}

	<-ctx.Done()
	slog.Info("aisre: shutting down")
	wg.Wait()
	return nil
}

// processWorker receives candidates and runs both investigation phases.
func processWorker(
	ctx context.Context,
	ch <-chan observer.Candidate,
	newAgent func() *investigate.Agent,
	newFixer func() *investigate.Fixer,
	exec *decision.Executor,
	sinkStack sink.Sink,
) {
	for {
		select {
		case <-ctx.Done():
			return
		case c, ok := <-ch:
			if !ok {
				return
			}
			report := processCandidate(ctx, c, newAgent(), newFixer(), exec)
			if err := sinkStack.Send(ctx, report); err != nil {
				slog.Warn("sink: send error", "err", err)
			}
		}
	}
}

// processCandidate runs Phase 1 (investigate) then Phase 2 (execute + verify).
func processCandidate(
	ctx context.Context,
	c observer.Candidate,
	agent *investigate.Agent,
	fixer *investigate.Fixer,
	exec *decision.Executor,
) sink.IncidentReport {
	slog.Info("investigate: received candidate",
		"id", c.ID,
		"workload", c.Workload.Name,
		"namespace", c.Workload.Namespace,
		"kind", c.Kind,
		"severity", c.Severity,
	)

	// Phase 1: investigate.
	inv := agent.Investigate(ctx, c)
	slog.Info("investigate: phase 1 complete",
		"workload", c.Workload.Name,
		"action", inv.RecommendedAction,
		"confidence", inv.Confidence,
	)

	// Escalate or low confidence → no execution, just report.
	if inv.RecommendedAction == investigate.ActionEscalate || inv.Confidence < 0.7 {
		return sink.IncidentReport{Candidate: c, Investigation: inv}
	}

	// Phase 2: execute action.
	if err := exec.Execute(ctx, inv); err != nil {
		slog.Warn("investigate: action failed", "action", inv.RecommendedAction, "err", err)
		inv.RecommendedAction = investigate.ActionEscalate
		inv.EscalateMessage += "\n\n⚠️ Автономное действие не выполнено: " + err.Error()
		return sink.IncidentReport{Candidate: c, Investigation: inv}
	}

	// Phase 2: verify the fix.
	fix := fixer.Verify(ctx, c, inv, inv.RecommendedAction)
	slog.Info("investigate: phase 2 complete",
		"workload", c.Workload.Name,
		"action", fix.ActionTaken,
		"success", fix.Success,
	)

	return sink.IncidentReport{Candidate: c, Investigation: inv, Fix: &fix}
}
