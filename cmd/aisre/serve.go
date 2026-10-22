package main

import (
	"context"
	"log/slog"

	"github.com/urfave/cli/v3"
)

func serveCmd() *cli.Command {
	return &cli.Command{
		Name:  "serve",
		Usage: "run the observer+investigator daemon",
		Flags: []cli.Flag{
			&cli.StringFlag{Name: "config", Aliases: []string{"c"}, Value: "/etc/ai-sre/config.yaml", Usage: "config file path"},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			slog.Info("aisre: serve starting", "config", cmd.String("config"))
			// TODO: load config, init state, start observer, start investigator queue
			<-ctx.Done()
			slog.Info("aisre: serve stopping")
			return nil
		},
	}
}
