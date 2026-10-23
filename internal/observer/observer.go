package observer

import (
	"crypto/sha256"
	"fmt"
	"strings"
)

// Fingerprint builds a stable dedup ID for a candidate.
func Fingerprint(workload Workload, kind Kind, detail string) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%s|%s|%s", workload.Source, workload.Namespace, workload.Name, kind, detail)
	return fmt.Sprintf("%x", h.Sum(nil))[:16]
}

// MatchesFailOpen returns true if any of the fail-open patterns appear in text.
func MatchesFailOpen(text string, patterns []string) (string, bool) {
	lower := strings.ToLower(text)
	for _, p := range patterns {
		if strings.Contains(lower, strings.ToLower(p)) {
			return p, true
		}
	}
	return "", false
}
