package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
)

func TestExtractMessagesText(t *testing.T) {
	// Build params from JSON the way the real handler does (UnmarshalJSON
	// populates the Role field; the SystemMessage/UserMessage constructors
	// leave Role as a zero value with a marshal-time default).
	body := `{"messages":[` +
		`{"role":"system","content":"You are helpful."},` +
		`{"role":"user","content":"Hello there"},` +
		`{"role":"user","content":[{"type":"text","text":"part one"},{"type":"text","text":"part two"}]}` +
		`]}`
	var params openai.ChatCompletionNewParams
	if err := json.Unmarshal([]byte(body), &params); err != nil {
		t.Fatal(err)
	}
	got := extractMessagesText(params.Messages)
	want := "system: You are helpful.\nuser: Hello there\nuser: part onepart two"
	if got != want {
		t.Errorf("extractMessagesText:\n got=%q\nwant=%q", got, want)
	}
}

func TestStreamRequested(t *testing.T) {
	cases := []struct {
		body string
		want bool
	}{
		{`{"stream":true,"model":"x"}`, true},
		{`{"stream":false,"model":"x"}`, false},
		{`{"model":"x"}`, false},
		{`not json`, false},
	}
	for _, c := range cases {
		if got := streamRequested([]byte(c.body)); got != c.want {
			t.Errorf("streamRequested(%s)=%v want %v", c.body, got, c.want)
		}
	}
}

func TestSplitContent(t *testing.T) {
	pieces := splitContent("Hello world from gpt-oss")
	want := []string{"Hello", " world", " from", " gpt-oss"}
	if len(pieces) != len(want) {
		t.Fatalf("len=%d want %d: %v", len(pieces), len(want), pieces)
	}
	for i := range want {
		if pieces[i] != want[i] {
			t.Errorf("pieces[%d]=%q want %q", i, pieces[i], want[i])
		}
	}
}

func TestWriteStreamEmitsSSE(t *testing.T) {
	rec := httptest.NewRecorder()
	completion := &openai.ChatCompletion{
		ID:      "chatcmpl-test",
		Model:   "gpt-oss-120b",
		Created: 1234,
		Choices: []openai.ChatCompletionChoice{
			{Index: 0, FinishReason: "stop", Message: openai.ChatCompletionMessage{Content: "Hi there"}},
		},
	}
	writeStream(rec, completion)

	resp := rec.Result()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type=%s", ct)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Errorf("missing [DONE]:\n%s", body)
	}
	// Should have a role chunk, content chunks, and a stop chunk.
	if !strings.Contains(body, `"role":"assistant"`) {
		t.Errorf("missing role chunk:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing stop chunk:\n%s", body)
	}
	// Parse each data line as JSON.
	lines := strings.Split(body, "\n")
	var chunks int
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if !strings.HasPrefix(l, "data: ") || l == "data: [DONE]" {
			continue
		}
		var c streamChunk
		if err := json.Unmarshal([]byte(strings.TrimPrefix(l, "data: ")), &c); err != nil {
			t.Fatalf("bad chunk JSON %q: %v", l, err)
		}
		if c.Object != "chat.completion.chunk" {
			t.Errorf("object=%s", c.Object)
		}
		chunks++
	}
	if chunks < 3 {
		t.Errorf("expected at least 3 chunks, got %d:\n%s", chunks, body)
	}
}

func TestWriteOpenAIError(t *testing.T) {
	rec := httptest.NewRecorder()
	writeOpenAIError(rec, http.StatusForbidden, "safeguard_violation",
		"input blocked by safeguard: prompt injection detected", "input")

	resp := rec.Result()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Param   string `json:"param"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Type != "safeguard_violation" {
		t.Errorf("type=%s", body.Error.Type)
	}
	if body.Error.Param != "input" {
		t.Errorf("param=%s", body.Error.Param)
	}
	if !strings.Contains(body.Error.Message, "prompt injection") {
		t.Errorf("message=%s", body.Error.Message)
	}
}
