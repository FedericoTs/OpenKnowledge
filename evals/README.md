# Evaluation sets

Two sets, deliberately in separate directories because they have different
schemas and different runners. Mixing them in one folder means a loader either
picks up a file it cannot parse, or silently skips it - and a silently skipped
eval is one that stops testing without telling you.

| | | |
|---|---|---|
| `golden/` | question → expected answer and citations | `openknowledge eval` |
| `conflicts/` | document pairs → contradiction or not | `openknowledge eval-conflicts` |

`golden/` needs a model configured to mean anything. `conflicts/` does not - it
is deterministic and runs in CI.

Add files freely to either directory; both runners load every `*.yaml` they
find.
