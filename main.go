package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/tinfoilsh/tinfoil-go"
)

const (
	proxyModel        = "gpt-oss-120b"
	listenAddrDefault = ":8080"
)

type config struct {
	tinfoilAPIKey   string
	listenAddr      string
	userCacheSecret string
}

func loadConfig() config {
	return config{
		tinfoilAPIKey:   os.Getenv("TINFOIL_API_KEY"),
		listenAddr:      envOr("LISTEN_ADDR", listenAddrDefault),
		userCacheSecret: os.Getenv("TINFOIL_USER_CACHE_SECRET"),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	cfg := loadConfig()

	flag.StringVar(&cfg.listenAddr, "listen", cfg.listenAddr, "listen address")
	flag.Parse()

	if cfg.tinfoilAPIKey == "" {
		log.Fatal("TINFOIL_API_KEY is required")
	}

	opts := []option.RequestOption{option.WithAPIKey(cfg.tinfoilAPIKey)}
	if cfg.userCacheSecret != "" {
		opts = append(opts, option.WithJSONSet("user_cache_secret", cfg.userCacheSecret))
	}
	client, err := tinfoil.NewClient(opts...)
	if err != nil {
		log.Fatalf("failed to create tinfoil client: %v", err)
	}

	sg := newSafeguardClient(client)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "ok")
	})
	mux.HandleFunc("/v1/chat/completions", chatHandler(client, sg))

	log.Printf("demo-safeguard listening on %s (proxy=%s safeguard=%s)",
		cfg.listenAddr, proxyModel, safeguardModel)
	srv := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

// chatHandler returns a handler that proxies chat completions to gpt-oss with
// safeguard checks on both input and output.
func chatHandler(client *tinfoil.Client, sg *safeguardClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := readBody(w, r)
		if err != nil {
			writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("could not read request body: %v", err), "")
			return
		}

		// Parse streaming intent and policy group from the raw JSON. The
		// openai-go params type does not surface `stream` as a field (it is
		// set out-of-band via WithJSONSet), and `policy_group` is a custom
		// extension field, so both are decoded separately.
		meta := parseRequestMeta(body)

		var params openai.ChatCompletionNewParams
		if err := json.Unmarshal(body, &params); err != nil {
			writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("invalid request body: %v", err), "")
			return
		}

		// Resolve which policy group to apply for this request.
		group, ok := sg.resolveGroup(meta.PolicyGroup)
		if !ok {
			writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("unknown policy_group %q", meta.PolicyGroup), "")
			return
		}

		// 1. Safeguard the input: concatenate every message's text content.
		inputText := extractMessagesText(params.Messages)
		if inputText != "" {
			v, err := sg.checkStage(r.Context(), stageInput, group, inputText)
			if err != nil {
				writeOpenAIError(w, http.StatusBadGateway, "server_error",
					"input safeguard check failed: "+err.Error(), "")
				return
			}
			if v != nil {
				writeOpenAIError(w, http.StatusForbidden, "safeguard_violation",
					fmt.Sprintf("input blocked by safeguard policy %q: %s", v.policy, v.rationale), "input")
				return
			}
		}

		// 2. Proxy to gpt-oss, always buffered (stream off) so the output can
		// be safeguarded before anything reaches the caller.
		params.Model = openai.ChatModel(proxyModel)
		completion, err := client.Chat.Completions.New(r.Context(), params)
		if err != nil {
			status, errType, msg := upstreamErrorInfo(err)
			writeOpenAIError(w, status, errType, msg, "")
			return
		}
		if len(completion.Choices) == 0 {
			writeOpenAIError(w, http.StatusBadGateway, "server_error",
				"upstream model returned no choices", "")
			return
		}

		// 3. Safeguard the output.
		outputText := extractCompletionText(completion)
		if outputText != "" {
			v, err := sg.checkStage(r.Context(), stageOutput, group, outputText)
			if err != nil {
				writeOpenAIError(w, http.StatusBadGateway, "server_error",
					"output safeguard check failed: "+err.Error(), "")
				return
			}
			if v != nil {
				writeOpenAIError(w, http.StatusForbidden, "safeguard_violation",
					fmt.Sprintf("output blocked by safeguard policy %q: %s", v.policy, v.rationale), "output")
				return
			}
		}

		// 4. Safeguard the turn: check all turn-stage policies in the group
		// against the combined input + output. This lets policies reason about
		// the full conversation (e.g. did the model help commit a crime).
		if inputText != "" || outputText != "" {
			turnText := formatTurnText(inputText, outputText)
			v, err := sg.checkStage(r.Context(), stageTurn, group, turnText)
			if err != nil {
				writeOpenAIError(w, http.StatusBadGateway, "server_error",
					"turn safeguard check failed: "+err.Error(), "")
				return
			}
			if v != nil {
				writeOpenAIError(w, http.StatusForbidden, "safeguard_violation",
					fmt.Sprintf("turn blocked by safeguard policy %q: %s", v.policy, v.rationale), "turn")
				return
			}
		}

		// 5. Respond. Buffer the full completion, then stream it back to the
		// caller if they asked for streaming.
		if meta.Stream {
			writeStream(w, completion)
		} else {
			writeJSON(w, http.StatusOK, completion)
		}
	}
}

// readBody reads and returns the full request body, capped at maxBody.
func readBody(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	const maxBody = 32 << 20 // 32 MiB
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	return io.ReadAll(r.Body)
}

// requestMeta holds fields parsed from the raw request body that the
// openai-go params type does not surface.
type requestMeta struct {
	Stream      bool   `json:"stream"`
	PolicyGroup string `json:"policy_group"`
}

// parseRequestMeta extracts stream and policy_group from the raw JSON body.
func parseRequestMeta(body []byte) requestMeta {
	var m requestMeta
	_ = json.Unmarshal(body, &m)
	return m
}

// extractMessagesText concatenates the text content of every message, prefixed
// with its role, so the safeguard sees the full conversational context.
func extractMessagesText(msgs []openai.ChatCompletionMessageParamUnion) string {
	var b strings.Builder
	for _, m := range msgs {
		role := "unknown"
		if r := m.GetRole(); r != nil {
			role = *r
		}
		text := messageContentText(m.GetContent().AsAny())
		if text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%s: %s", role, text)
	}
	return b.String()
}

// messageContentText extracts plain text from a message content union, which
// may be a plain string, a slice of text content parts, or a slice of mixed
// content parts.
func messageContentText(content any) string {
	switch v := content.(type) {
	case *string:
		if v != nil {
			return *v
		}
	case *[]openai.ChatCompletionContentPartUnionParam:
		var b strings.Builder
		for _, part := range *v {
			if t := part.GetText(); t != nil {
				b.WriteString(*t)
			}
		}
		return b.String()
	case *[]openai.ChatCompletionContentPartTextParam:
		var b strings.Builder
		for _, part := range *v {
			b.WriteString(part.Text)
		}
		return b.String()
	}
	return ""
}

// formatTurnText combines input and output into a single text for turn-stage
// policies, which reason about the full conversation.
func formatTurnText(input, output string) string {
	if input == "" {
		return "Model output:\n" + output
	}
	if output == "" {
		return "User input:\n" + input
	}
	return fmt.Sprintf("User input:\n%s\n\nModel output:\n%s", input, output)
}

// extractCompletionText pulls the assistant text out of a buffered completion.
func extractCompletionText(c *openai.ChatCompletion) string {
	var b strings.Builder
	for _, choice := range c.Choices {
		b.WriteString(choice.Message.Content)
	}
	return b.String()
}

// --- response writers ---

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeOpenAIError writes an OpenAI-shaped error body.
func writeOpenAIError(w http.ResponseWriter, status int, errType, message, param string) {
	type errDetail struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param,omitempty"`
		Code    string `json:"code,omitempty"`
	}
	body := struct {
		Error errDetail `json:"error"`
	}{
		Error: errDetail{
			Message: message,
			Type:    errType,
			Param:   param,
			Code:    errType,
		},
	}
	writeJSON(w, status, body)
}

// streamChunk is a minimal chat.completion.chunk for the buffered→streamed path.
type streamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []streamDelta `json:"choices"`
	Usage   any           `json:"usage,omitempty"`
}

type streamDelta struct {
	Index        int64          `json:"index"`
	Delta        map[string]any `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

// writeStream re-emits a buffered completion as an SSE stream of chunks.
func writeStream(w http.ResponseWriter, c *openai.ChatCompletion) {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	id := c.ID
	created := c.Created
	model := c.Model

	// Opening chunk: announce the assistant role.
	writeSSE(w, streamChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []streamDelta{{Index: 0, Delta: map[string]any{"role": "assistant", "content": ""}}},
	})
	flush()

	// Content chunks, split on word boundaries to simulate streaming.
	content := ""
	if len(c.Choices) > 0 {
		content = c.Choices[0].Message.Content
	}
	finishReason := "stop"
	if len(c.Choices) > 0 && c.Choices[0].FinishReason != "" {
		finishReason = c.Choices[0].FinishReason
	}
	for _, piece := range splitContent(content) {
		writeSSE(w, streamChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []streamDelta{{Index: 0, Delta: map[string]any{"content": piece}}},
		})
		flush()
	}

	// Terminal chunk with finish_reason.
	writeSSE(w, streamChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []streamDelta{{Index: 0, Delta: map[string]any{}, FinishReason: &finishReason}},
	})
	// Usage chunk, when present.
	if c.Usage.TotalTokens > 0 {
		writeSSE(w, streamChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []streamDelta{}, Usage: c.Usage,
		})
	}
	flush()

	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flush()
}

// writeSSE marshals a chunk and writes it as an SSE data: line.
func writeSSE(w http.ResponseWriter, chunk streamChunk) {
	data, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
}

// splitContent breaks content into word-sized pieces for a streaming feel.
func splitContent(content string) []string {
	if content == "" {
		return nil
	}
	words := strings.Fields(content)
	pieces := make([]string, 0, len(words))
	for i, w := range words {
		if i == 0 {
			pieces = append(pieces, w)
		} else {
			pieces = append(pieces, " "+w)
		}
	}
	return pieces
}

// upstreamErrorInfo maps an error from the upstream model call to an HTTP
// status, OpenAI error type, and message. Tinfoil/openai-go errors carry the
// upstream status code and body, which we pass through.
func upstreamErrorInfo(err error) (int, string, string) {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		status := apiErr.StatusCode
		if status < 400 || status >= 600 {
			status = http.StatusBadGateway
		}
		errType := apiErr.Type
		if errType == "" {
			errType = "server_error"
		}
		msg := apiErr.Message
		if msg == "" {
			msg = err.Error()
		}
		return status, errType, msg
	}
	return http.StatusBadGateway, "server_error", "upstream model call failed: " + err.Error()
}
