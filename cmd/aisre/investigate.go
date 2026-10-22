package main

import (
	"context"
	"log/slog"

	"github.com/urfave/cli/v3"
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
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			name := cmd.Args().First()
			if name == "" {
				return cli.Exit("workload name required", 1)
			}
			slog.Info("aisre: investigate", "workload", name, "namespace", cmd.String("namespace"))
			// TODO: load config, build investigator, run one-shot investigation, print JSON result
			return nil
		},
	}
}
