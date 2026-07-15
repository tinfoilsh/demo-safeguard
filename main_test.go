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

func TestExtractCompletionText(t *testing.T) {
	// Plain content, no tool calls.
	c := &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{Content: "hello world"}},
		},
	}
	if got := extractCompletionText(c); got != "hello world" {
		t.Errorf("plain content: got %q want %q", got, "hello world")
	}

	// Content alongside a function tool call.
	c = &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{
				Content: "let me check",
				ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
					{Type: "function", Function: openai.ChatCompletionMessageFunctionToolCallFunction{
						Name: "get_weather", Arguments: `{"location":"Paris"}`,
					}},
				},
			}},
		},
	}
	got := extractCompletionText(c)
	want := "let me check\ntool_call(function): name=get_weather arguments={\"location\":\"Paris\"}"
	if got != want {
		t.Errorf("content+tool:\n got=%q\nwant=%q", got, want)
	}

	// Tool-call only with empty content — must not be skipped.
	c = &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{
				ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
					{Type: "function", Function: openai.ChatCompletionMessageFunctionToolCallFunction{
						Name: "delete_file", Arguments: `{"path":"/etc/passwd"}`,
					}},
				},
			}},
		},
	}
	got = extractCompletionText(c)
	want = `tool_call(function): name=delete_file arguments={"path":"/etc/passwd"}`
	if got != want {
		t.Errorf("tool-only:\n got=%q\nwant=%q", got, want)
	}

	// Custom tool call variant.
	c = &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{
				ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
					{Type: "custom", Custom: openai.ChatCompletionMessageCustomToolCallCustom{
						Name: "lookup", Input: `{"q":"bread"}`,
					}},
				},
			}},
		},
	}
	got = extractCompletionText(c)
	want = `tool_call(custom): name=lookup input={"q":"bread"}`
	if got != want {
		t.Errorf("custom tool:\n got=%q\nwant=%q", got, want)
	}
}

func TestLoadPolicies(t *testing.T) {
	sg := &safeguardClient{groups: make(map[string][]string)}
	if err := sg.loadPolicies(); err != nil {
		t.Fatalf("loadPolicies: %v", err)
	}
	if len(sg.policies) < 3 {
		t.Fatalf("expected at least 3 policies, got %d", len(sg.policies))
	}

	bySurface := map[policySurface][]string{}
	for _, p := range sg.policies {
		bySurface[p.surface] = append(bySurface[p.surface], p.name)
	}
	if len(bySurface[surfaceUserMessage]) == 0 {
		t.Error("no user_message-surface policies")
	}
	if len(bySurface[surfaceModelMessage]) == 0 {
		t.Error("no model_message-surface policies")
	}
	if len(bySurface[surfaceTurn]) == 0 {
		t.Error("no turn-surface policies")
	}

	found := false
	for _, p := range sg.policies {
		if p.name == "trained_dolphins" && p.surface == surfaceTurn {
			found = true
		}
	}
	if !found {
		t.Error("trained_dolphins policy not found on turn surface")
	}

	if len(sg.groups) < 2 {
		t.Fatalf("expected at least 2 groups, got %d", len(sg.groups))
	}
	if sg.defaultGroup == "" {
		t.Error("no default group set")
	}
	if _, ok := sg.groups[sg.defaultGroup]; !ok {
		t.Errorf("default group %q not in groups", sg.defaultGroup)
	}
}

func TestResolveGroup(t *testing.T) {
	sg := &safeguardClient{groups: make(map[string][]string)}
	if err := sg.loadPolicies(); err != nil {
		t.Fatalf("loadPolicies: %v", err)
	}

	g, ok := sg.resolveGroup("")
	if !ok {
		t.Fatal(`resolveGroup("") should succeed`)
	}
	if g != sg.defaultGroup {
		t.Errorf(`resolveGroup("")=%q want %q`, g, sg.defaultGroup)
	}

	for name := range sg.groups {
		g, ok := sg.resolveGroup(name)
		if !ok || g != name {
			t.Errorf("resolveGroup(%q)=%q,%v want %q,true", name, g, ok, name)
		}
	}

	_, ok = sg.resolveGroup("nonexistent")
	if ok {
		t.Error(`resolveGroup("nonexistent") should fail`)
	}
}

func TestParseRequestMeta(t *testing.T) {
	cases := []struct {
		body   string
		stream bool
		group  string
	}{
		{`{"stream":true,"policy_group":"group2"}`, true, "group2"},
		{`{"stream":false,"model":"x"}`, false, ""},
		{`{"model":"x"}`, false, ""},
		{`{"policy_group":"group1"}`, false, "group1"},
		{`not json`, false, ""},
	}
	for _, c := range cases {
		m := parseRequestMeta([]byte(c.body))
		if m.Stream != c.stream {
			t.Errorf("parseRequestMeta(%s).Stream=%v want %v", c.body, m.Stream, c.stream)
		}
		if m.PolicyGroup != c.group {
			t.Errorf("parseRequestMeta(%s).PolicyGroup=%q want %q", c.body, m.PolicyGroup, c.group)
		}
	}
}

func TestFormatTurnText(t *testing.T) {
	got := formatTurnText("hello", "world")
	want := "User input:\nhello\n\nModel output:\nworld"
	if got != want {
		t.Errorf("formatTurnText both:\n got=%q\nwant=%q", got, want)
	}
	if got := formatTurnText("", "world"); got != "Model output:\nworld" {
		t.Errorf("formatTurnText empty input: %q", got)
	}
	if got := formatTurnText("hello", ""); got != "User input:\nhello" {
		t.Errorf("formatTurnText empty output: %q", got)
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
	if !strings.Contains(body, `"role":"assistant"`) {
		t.Errorf("missing role chunk:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing stop chunk:\n%s", body)
	}
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
