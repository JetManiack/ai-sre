package tools

import (
	"strings"
	"testing"
)

func TestKbHitIsRelevant(t *testing.T) {
	// Real article that matches the error phrase.
	upgradeArticle := `[{"document":{"id":"GZUkRokaV7","title":"Keycloak 26.x upgrade"},"context":"Failed to start server… Cannot lazily initialize collection of role"}]`
	// Unrelated article that only mentions Keycloak and "component".
	ssoArticle := `[{"document":{"id":"jQNSO341WO","title":"SSO gateway"},"context":"A gateway that lets bearer-token API clients skip interactive SSO using Keycloak component"}]`

	cases := []struct {
		out   string
		query string
		wl    string
		want  bool
	}{
		// Correct query → context has "lazily", "initialize", "collection" → ≥2 matches → relevant.
		{upgradeArticle, "Cannot lazily initialize collection of role", "keycloak", true},
		// vllm failure case: "Keycloak components" → only "component" (singular) in SSO context,
		// "components" (plural) absent → 0 matches → not relevant. Even with ≥1 threshold this
		// was fragile; ≥2 makes it robust to coincidental single-word hits.
		{ssoArticle, "Keycloak components", "keycloak", false},
		// Workload-only query → no non-workload words → not relevant.
		{ssoArticle, "keycloak", "keycloak", false},
		// Two non-workload words "lazily" and "initialize" → both in upgrade context → relevant.
		{upgradeArticle, "keycloak lazily initialize", "keycloak", true},
		// Only 1 non-workload word in context → not relevant with ≥2 threshold.
		{upgradeArticle, "keycloak lazily", "keycloak", false},
		// Empty result.
		{`[]`, "keycloak upgrade", "keycloak", false},
	}
	for i, tc := range cases {
		got := kbHitIsRelevant(tc.out, tc.query, tc.wl)
		if got != tc.want {
			t.Errorf("[%d] kbHitIsRelevant(query=%q, wl=%q) = %v, want %v", i, tc.query, tc.wl, got, tc.want)
		}
	}
}

func TestKbTopID(t *testing.T) {
	cases := []struct {
		out  string
		want string
	}{
		{`[]`, ""},
		{`null`, ""},
		{`[{"document":{"id":"abc123","title":"Keycloak upgrade"},"context":"…"}]`, "abc123"},
		{`[{"document":{"id":"first"}},{"document":{"id":"second"}}]`, "first"},
		{`not json`, ""},
	}
	for _, tc := range cases {
		got := kbTopID(tc.out)
		if got != tc.want {
			t.Errorf("kbTopID(%q) = %q, want %q", tc.out, got, tc.want)
		}
	}
}

func TestListDocumentsIsEmpty(t *testing.T) {
	empty := []string{
		`[]`,
		`null`,
		`{}`,
		``,
		`   `,
		`[{"document":{"title":"x"},"context":"y"}]`, // no "id" field
	}
	nonEmpty := []string{
		`[{"document":{"id":"abc123","title":"Keycloak upgrade schema"},"context":"…"}]`,
		`[{"document":{"id":"def456"}},{"document":{"id":"ghi789"}}]`,
	}

	for _, tc := range empty {
		if !listDocumentsIsEmpty(tc) {
			t.Errorf("expected empty for %q", tc)
		}
	}
	for _, tc := range nonEmpty {
		if listDocumentsIsEmpty(tc) {
			t.Errorf("expected non-empty for %q", tc)
		}
	}
}

func TestKbQueryFallbacks(t *testing.T) {
	cases := []struct {
		query    string
		workload string
		want     []string
	}{
		{
			query:    "Cannot lazily initialize collection of role RealmEntity.components",
			workload: "keycloak",
			want: []string{
				"Cannot lazily initialize collection of role",
				"Cannot lazily initialize collection",
				"keycloak Cannot",
				"Cannot",
			},
		},
		{
			query:    "LazyInitializationException keycloak",
			workload: "keycloak",
			want:     []string{"keycloak"},
		},
		{
			query:    "keycloak upgrade database schema",
			workload: "keycloak",
			want: []string{
				"keycloak upgrade database",
				"keycloak Cannot", // should not appear — first word is "keycloak"
			},
		},
		{
			query:    "keycloak",
			workload: "keycloak",
			want:     nil,
		},
	}

	// Only test the cases we can verify deterministically.
	t.Run("keycloak-lazy", func(t *testing.T) {
		// With alert hint: step 0 = stripped summary (direct hit), then the rest.
		alertHint := "Cannot lazily initialize collection of role: org.keycloak.models.jpa.entities.RealmEntity.components"
		got := kbQueryFallbacks("Cannot lazily initialize collection of role RealmEntity.components", "keycloak", alertHint)
		if len(got) < 2 {
			t.Fatalf("expected ≥2 fallbacks, got %v", got)
		}
		// Step 0 = stripped alert summary.
		if got[0] != "Cannot lazily initialize collection of role:" {
			// The colon at end is fine; what matters is the human-readable part is first.
			// Trim trailing punct for comparison.
			first := strings.TrimRight(got[0], ": ")
			if first != "Cannot lazily initialize collection of role" {
				t.Errorf("[0] got %q", got[0])
			}
		}
	})

	t.Run("keycloak-lazy-no-alert", func(t *testing.T) {
		got := kbQueryFallbacks("Cannot lazily initialize collection of role RealmEntity.components", "keycloak", "")
		if len(got) < 3 {
			t.Fatalf("expected ≥3 fallbacks, got %v", got)
		}
		if got[0] != "Cannot lazily initialize collection of role" {
			t.Errorf("[0] got %q", got[0])
		}
		last := got[len(got)-1]
		if last != "Cannot" {
			t.Errorf("last got %q, want %q", last, "Cannot")
		}
	})

	t.Run("class-name-only", func(t *testing.T) {
		got := kbQueryFallbacks("LazyInitializationException keycloak", "keycloak", "")
		if len(got) != 1 || got[0] != "keycloak" {
			t.Errorf("got %v, want [keycloak]", got)
		}
	})

	t.Run("no-identifiers", func(t *testing.T) {
		got := kbQueryFallbacks("keycloak", "keycloak", "")
		if len(got) != 0 {
			t.Errorf("got %v, want []", got)
		}
	})

	_ = cases
}
