package replay_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/JetManiack/ai-sre/internal/config"
	"github.com/JetManiack/ai-sre/internal/investigate"
	"github.com/JetManiack/ai-sre/internal/replay"
)

// noopMemory satisfies investigate.MemoryStore.
type noopMemory struct{}

func (noopMemory) RecallInvestigations(_, _ string, _ int) ([]investigate.PastInvestigation, error) {
	return nil, nil
}
func (noopMemory) SaveInvestigation(_ investigate.PastInvestigation) error { return nil }

// mockLLM drives the agent toward a specific Decision without a real LLM.
type mockLLM struct {
	toolSequence  []func(msgs []investigate.Message) investigate.ChatResponse
	finalDecision investigate.Decision
	step          int
}

func (m *mockLLM) ChatCompletionWithTools(_ context.Context, req investigate.ChatRequest) (investigate.ChatResponse, error) {
	if m.step < len(m.toolSequence) {
		resp := m.toolSequence[m.step](req.Messages)
		m.step++
		return resp, nil
	}
	b, _ := json.Marshal(m.finalDecision)
	return investigate.ChatResponse{Content: string(b), FinishReason: "stop"}, nil
}

func toolCallResp(name string, args map[string]string) investigate.ChatResponse {
	argsB, _ := json.Marshal(args)
	return investigate.ChatResponse{
		ToolCalls: []investigate.ToolCall{{ID: "tc1", Name: name, Input: argsB}},
	}
}

func testConfig() config.InvestigateConfig {
	return config.InvestigateConfig{
		Model:               "test-model",
		MaxSteps:            8,
		MaxToolOutputTokens: 2000,
		MaxTokensStep:       256,
		MaxTokensFinal:      1200,
		StepTimeout:         5 * time.Second,
		TotalTimeout:        30 * time.Second,
	}
}

func runFixtureTest(t *testing.T, fixtureDir string) {
	t.Helper()
	f, err := replay.Load(fixtureDir)
	if err != nil {
		t.Fatalf("load fixture %s: %v", fixtureDir, err)
	}

	d := replay.NewDispatcher(f.Responses)
	decision := investigate.Decision{
		RootCauseRU:     buildSyntheticRootCause(f),
		Evidence:        f.Candidate.Evidence,
		Confidence:      0.85,
		SuggestedFixRU:  "Проверьте конфигурацию и откатите изменения.",
		Action:          f.Expected.Action,
		EscalateTo:      "on-call-ops",
		EscalateMessage: "Синтетическое сообщение эскалации.",
	}

	llm := &mockLLM{
		toolSequence: []func([]investigate.Message) investigate.ChatResponse{
			func(_ []investigate.Message) investigate.ChatResponse {
				return toolCallResp("memory_recall", map[string]string{
					"workload":  f.Candidate.Workload.Name,
					"namespace": f.Candidate.Workload.Namespace,
				})
			},
		},
		finalDecision: decision,
	}

	agent := investigate.New(testConfig(), d, noopMemory{}, llm)
	result := agent.Investigate(context.Background(), f.Candidate)

	if result.Action != f.Expected.Action {
		t.Errorf("action: got %q, want %q", result.Action, f.Expected.Action)
	}
	if result.Confidence < f.Expected.ConfidenceMin {
		t.Errorf("confidence: got %.2f, want >= %.2f", result.Confidence, f.Expected.ConfidenceMin)
	}
	if len(f.Expected.RootCauseContains) > 0 {
		lower := strings.ToLower(result.RootCauseRU)
		matched := false
		for _, substr := range f.Expected.RootCauseContains {
			if strings.Contains(lower, strings.ToLower(substr)) {
				matched = true
				break
			}
		}
		if !matched {
			t.Errorf("root_cause_ru %q does not contain any of %v", result.RootCauseRU, f.Expected.RootCauseContains)
		}
	}
}

func buildSyntheticRootCause(f *replay.Fixture) string {
	if len(f.Expected.RootCauseContains) > 0 {
		return "Обнаружена проблема: " + f.Expected.RootCauseContains[0]
	}
	return "Аномалия обнаружена, требует расследования."
}

func TestAgentReplay_MemoryLeak(t *testing.T) {
	runFixtureTest(t, "../../testdata/fixtures/memory-leak")
}

func TestAgentReplay_GatewayAll500(t *testing.T) {
	runFixtureTest(t, "../../testdata/fixtures/gateway-all-500")
}

func TestAgentReplay_ModelAliasRemoved(t *testing.T) {
	runFixtureTest(t, "../../testdata/fixtures/model-alias-removed")
}
