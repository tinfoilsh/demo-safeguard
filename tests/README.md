# tests

Smoke tests for the demo-safeguard proxy. Exercises every policy group with a
clean request (should pass) and requests designed to trip each policy (should
be blocked).

## Run

Start the proxy from the repo root:

```bash
TINFOIL_API_KEY=... go run .
```

Then:

```bash
pip install -r tests/requirements.txt
python tests/smoke.py                  # all groups
GROUP=group1 python tests/smoke.py     # one group
HOST=http://localhost:8080 python tests/smoke.py
```

## Cases

| group  | case             | expect  | policy           | note            |
| ------ | ---------------- | ------- | ---------------- | --------------- |
| group1 | clean            | pass    |                  |                 |
| group1 | prompt injection | blocked | prompt_injection | deterministic   |
| group1 | baked bread      | blocked | baked_bread      | model-dependent |
| group1 | tool-call bread  | blocked | baked_bread      | model-dependent |
| group2 | clean            | pass    |                  |                 |
| group2 | prompt injection | blocked | prompt_injection | deterministic   |
| group2 | trained dolphins | blocked | trained_dolphins | model-dependent |

Model-dependent cases (baked bread, trained dolphins, tool-call bread) rely on
the upstream model producing content that trips the policy. They may not always
fire — the suite does not fail on those. Only the deterministic cases (prompt
injection, clean) cause a non-zero exit.

The tool-call bread case defines a `bake` tool and asks the model to use it.
The safeguard inspects the tool call's arguments (not just prose) under the
`model_message` surface — so even if the model produces no text content, a
tool call with bread-y args is caught.
