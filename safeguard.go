package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/tinfoilsh/tinfoil-go"
	"gopkg.in/yaml.v3"
)

//go:embed policies.yaml
var policiesYAML []byte

const (
	safeguardModel       = "gpt-oss-safeguard-120b"
	safeguardTemperature = 0.0
)

// policyStage tags where a policy applies: on the user input or on the model
// output.
type policyStage string

const (
	stageInput  policyStage = "input"
	stageOutput policyStage = "output"
)

// loadedPolicy is a single policy parsed from policies.yaml.
type loadedPolicy struct {
	name  string
	stage policyStage
	text  string
}

// policiesFile mirrors the on-disk shape of policies.yaml.
type policiesFile struct {
	Version  int `yaml:"version"`
	Policies map[string]struct {
		Stage string `yaml:"stage"`
		Text  string `yaml:"text"`
	} `yaml:"policies"`
}

// checkResult is the structured output enforced by the safeguard model.
type checkResult struct {
	Violation bool   `json:"violation"`
	Rationale string `json:"rationale"`
}

// violation is a positive safeguard hit from a specific policy.
type violation struct {
	policy    string
	rationale string
}

// checkResultSchema is the JSON schema for structured output enforcement.
var checkResultSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"violation": map[string]any{
			"type":        "boolean",
			"description": "Whether the content violates the policy",
		},
		"rationale": map[string]any{
			"type":        "string",
			"description": "Brief explanation of the classification decision",
		},
	},
	"required":             []string{"violation", "rationale"},
	"additionalProperties": false,
}

// safeguardClient wraps a Tinfoil client for safety classification calls.
type safeguardClient struct {
	client   *tinfoil.Client
	policies []loadedPolicy
}

// newSafeguardClient creates a safeguard client and loads policies from the
// embedded policies.yaml. Panics if the policies are invalid — a bad policy
// bundle is a build-time / deployment-time bug, not a runtime condition.
func newSafeguardClient(client *tinfoil.Client) *safeguardClient {
	sg := &safeguardClient{client: client}
	if err := sg.loadPolicies(); err != nil {
		panic(fmt.Sprintf("safeguard: %v", err))
	}
	return sg
}

func (s *safeguardClient) loadPolicies() error {
	var pf policiesFile
	if err := yaml.Unmarshal(policiesYAML, &pf); err != nil {
		return fmt.Errorf("failed to parse policies.yaml: %w", err)
	}
	for name, p := range pf.Policies {
		if p.Text == "" {
			return fmt.Errorf("policy %q has empty text", name)
		}
		stage := policyStage(p.Stage)
		if stage != stageInput && stage != stageOutput {
			return fmt.Errorf("policy %q has invalid stage %q (want input or output)", name, p.Stage)
		}
		s.policies = append(s.policies, loadedPolicy{name: name, stage: stage, text: p.Text})
	}
	if len(s.policies) == 0 {
		return fmt.Errorf("no policies loaded")
	}
	// Deterministic order so "first violation wins" is stable.
	slices.SortFunc(s.policies, func(a, b loadedPolicy) int {
		return cmpString(a.name, b.name)
	})
	return nil
}

func cmpString(a, b string) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

// checkStage runs every policy matching stage against content, in order.
// Returns the first violation, or nil if all pass.
func (s *safeguardClient) checkStage(ctx context.Context, stage policyStage, content string) (*violation, error) {
	for _, p := range s.policies {
		if p.stage != stage {
			continue
		}
		res, err := s.check(ctx, p.text, content)
		if err != nil {
			return nil, fmt.Errorf("policy %q: %w", p.name, err)
		}
		if res.Violation {
			return &violation{policy: p.name, rationale: res.Rationale}, nil
		}
	}
	return nil, nil
}

// check runs the safeguard model against content using a policy.
func (s *safeguardClient) check(ctx context.Context, policy, content string) (*checkResult, error) {
	resp, err := s.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: shared.ChatModel(safeguardModel),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(policy),
			openai.UserMessage(content),
		},
		Temperature: openai.Float(safeguardTemperature),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "check_result",
					Schema: checkResultSchema,
					Strict: openai.Bool(true),
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("safeguard call failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("safeguard returned no choices")
	}

	var result checkResult
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &result); err != nil {
		return nil, fmt.Errorf("failed to parse safeguard response: %w", err)
	}
	return &result, nil
}
