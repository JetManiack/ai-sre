package decision

import (
	"context"
	"fmt"
	"os/exec"
)

// runSystemctl runs "systemctl <verb> <unit>" with a context.
func runSystemctl(ctx context.Context, verb, unit string) error {
	cmd := exec.CommandContext(ctx, "systemctl", verb, unit)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("systemctl %s %s: %w: %s", verb, unit, err, out)
	}
	return nil
}
