package main

import (
	"context"
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
		Usage: "run the observer+investigator daemon",
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

	// Build the LLM client.
	llmClient := investigate.NewOpenAIClient(cfg.Investigate.BaseURL, cfg.Investigate.APIKey)

	// Build the investigator agent factory.
	newAgent := func() *investigate.Agent {
		return investigate.NewWithSources(cfg.Investigate, cfg.Sources, dispatcher, mem, llmClient)
	}

	// Build the action executor.
	executor := decision.NewExecutor(cfg.Actions, cfg.Actions.K8sWriteURL, cfg.Actions.K8sWriteToken)

	// Build the output sink stack.
	sinkStack := sink.Build(cfg.Sinks)

	// Build the observer.
	candidateCh := make(chan observer.Candidate, 32)
	obs := observer.New(cfg.Observer, db, candidateCh)
	registerAdapters(obs, cfg)

	// Start the observer loop.
	go func() {
		slog.Info("aisre: observer starting", "interval", cfg.Observer.Interval)
		obs.Run(ctx)
	}()

	// Start the investigator worker pool.
	concurrency := cfg.Investigate.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	slog.Info("aisre: investigator starting", "concurrency", concurrency, "model", cfg.Investigate.Model)

	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			investigateWorker(ctx, candidateCh, newAgent, executor, sinkStack)
		}()
	}

	<-ctx.Done()
	slog.Info("aisre: shutting down")
	wg.Wait()
	return nil
}

// investigateWorker receives candidates, investigates each one, executes the
// decided action, and sends the result to sinks.
func investigateWorker(
	ctx context.Context,
	ch <-chan observer.Candidate,
	newAgent func() *investigate.Agent,
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
			slog.Info("investigate: received candidate",
				"id", c.ID,
				"workload", c.Workload.Name,
				"namespace", c.Workload.Namespace,
				"kind", c.Kind,
				"severity", c.Severity,
			)
			agent := newAgent()
			d := agent.Investigate(ctx, c)
			slog.Info("investigate: decision",
				"workload", c.Workload.Name,
				"action", d.Action,
				"confidence", d.Confidence,
			)

			// Execute autonomous action (no-op for escalate).
			if err := exec.Execute(ctx, d); err != nil {
				slog.Warn("investigate: action failed", "action", d.Action, "err", err)
				// Downgrade to escalate so we don't silently drop the finding.
				d.Action = investigate.ActionEscalate
				d.EscalateMessage += "\n\n⚠️ Автономное действие не выполнено: " + err.Error()
			}

			// Send to sinks (board, webhook, stdout).
			if err := sinkStack.Send(ctx, c, d); err != nil {
				slog.Warn("investigate: sink error", "err", err)
			}
		}
	}
}

// registerAdapters wires observer adapters from config.
func registerAdapters(obs *observer.Observer, cfg *config.Config) {
	for _, srv := range cfg.MCPServers {
		// A server with a "check_anomalies" tool acts as an observer adapter.
		hasCheckAnomalies := false
		for _, t := range srv.Tools {
			if t == "check_anomalies" {
				hasCheckAnomalies = true
				break
			}
		}
		if hasCheckAnomalies {
			a := observer.NewMCPAdapter(srv.Name, srv.URL, srv.Token, "check_anomalies", srv.Name, cfg.Observer)
			obs.Register(a)
			slog.Info("observer: registered MCP adapter", "server", srv.Name)
		}
	}
	// Log-pattern adapters from fail_open_patterns across all sources.
	for _, src := range cfg.Sources {
		if len(cfg.Observer.FailOpenPatterns) > 0 {
			a := observer.NewLogPatternAdapter(
				src.Name, src.URL, src.Token, src.Name, "",
				cfg.Observer.FailOpenPatterns,
			)
			obs.Register(a)
			slog.Info("observer: registered log-pattern adapter", "source", src.Name)
		}
	}
}
