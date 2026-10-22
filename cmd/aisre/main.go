// ai-sre — autonomous SRE agent.
//
// Observes k8s state, metrics, and logs for anomalies; investigates them with a
// bounded LLM tool loop; and either acts autonomously (within the configured
// allowlist) or escalates with a complete diagnosis.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/urfave/cli/v3"
)

func main() {
	app := &cli.Command{
		Name:  "aisre",
		Usage: "autonomous SRE agent — observe, investigate, act or escalate",
		Commands: []*cli.Command{
			serveCmd(),
			investigateCmd(),
		},
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := app.Run(ctx, os.Args); err != nil {
		slog.Error("aisre", "err", err)
		os.Exit(1)
	}
}
