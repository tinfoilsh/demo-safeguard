package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/tinfoilsh/tinfoil-go"
)

//go:embed policy.txt
var safeguardPolicy string

const (
	safeguardModelDefault = "gpt-oss-safeguard-120b"
	safeguardTemperature  = 0.0
)

// checkResult is the structured output enforced by the safeguard model.
type checkResult struct {
	Violation bool   `json:"violation"`
	Rationale string `json:"rationale"`
}

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
	client *tinfoil.Client
	model  string
}

func newSafeguardClient(client *tinfoil.Client, model string) *safeguardClient {
	if model == "" {
		model = safeguardModelDefault
	}
	return &safeguardClient{client: client, model: model}
}

// check runs the safeguard model against content using the embedded policy.
// Returns the parsed result, or an error if the call or parsing fails.
func (s *safeguardClient) check(ctx context.Context, content string) (*checkResult, error) {
	resp, err := s.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: shared.ChatModel(s.model),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(safeguardPolicy),
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
