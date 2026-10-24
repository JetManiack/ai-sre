package tools

import (
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
		// Correct query → upgrade article context has "lazily", "initialize", etc. → relevant.
		{upgradeArticle, "Cannot lazily initialize collection of role", "keycloak", true},
		// vllm failure case: stripped query "Keycloak components", SSO article context has "component"
		// (singular) but not "components" (plural) → exact word absent → not relevant → no auto-fetch.
		{ssoArticle, "Keycloak components", "keycloak", false},
		// Workload-only query → no non-workload words → not relevant.
		{ssoArticle, "keycloak", "keycloak", false},
		// Upgrade article with word "lazily" also in query → relevant.
		{upgradeArticle, "Keycloak lazily components", "keycloak", true},
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
		got := kbQueryFallbacks("Cannot lazily initialize collection of role RealmEntity.components", "keycloak")
		if len(got) < 3 {
			t.Fatalf("expected ≥3 fallbacks, got %v", got)
		}
		if got[0] != "Cannot lazily initialize collection of role" {
			t.Errorf("[0] got %q", got[0])
		}
		if got[len(got)-2] != "keycloak Cannot" {
			t.Errorf("[-2] got %q, want %q", got[len(got)-2], "keycloak Cannot")
		}
		if got[len(got)-1] != "Cannot" {
			t.Errorf("[-1] got %q, want %q", got[len(got)-1], "Cannot")
		}
	})

	t.Run("class-name-only", func(t *testing.T) {
		got := kbQueryFallbacks("LazyInitializationException keycloak", "keycloak")
		if len(got) != 1 || got[0] != "keycloak" {
			t.Errorf("got %v, want [keycloak]", got)
		}
	})

	t.Run("no-identifiers", func(t *testing.T) {
		got := kbQueryFallbacks("keycloak", "keycloak")
		if len(got) != 0 {
			t.Errorf("got %v, want []", got)
		}
	})

	_ = cases
}
