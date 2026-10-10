# CaptionForge Lightweight Nodes — prototype reliability pass

This feature branch adds two standalone ComfyUI nodes without changing the
CaptionForge A–D pipeline. Both pass the original `IMAGE` tensor through.

## Targeted VLM Query

Ask 1–10 separate image-grounded questions through an existing Ollama VLM,
with stable `answer_1` … `answer_10` pins. Slots beyond `slot_count` and
blank questions return empty strings. Each active question is an independent
Ollama `/api/chat` request; no chat history is shared.

The node reuses `config/captionforge_ollama_models.json`, including the
`custom` model-tag escape hatch.

### Seed and recovery settings

| Widget | Default | Behavior |
|---|---|---|
| `seed` | `-1` | Omit Ollama's seed option; a nonnegative value sends `options.seed`. Frontends that support it can provide after-generate seed controls. |
| `retry_count` | `1` | Extra requests **only** when Ollama responds without final `message.content`; 0–3 retries. |
| `retry_seed_mode` | `increment` | On empty-response retry, increase an explicit seed, wrapping at 2,147,483,647; `same` reuses it. `-1` remains unseeded. |
| `error_policy` | `skip_failed` | After failed attempts, return an empty answer. Alternatives are `insert_unknown` and `stop_on_error`. |
| `thinking_mode` | `model_default` | Omit the top-level Ollama `think` flag, or send explicit true/false. Support varies by model/server version. |
| `diagnostics` | `errors` | Prints empty-response metadata: `done_reason`, generated token counts, thinking **length** (not thinking text). `verbose` also logs success metadata. |
| `cache_successful_answers` | `true` | Reuse successful answers to unchanged individual questions within this ComfyUI server session, up to 128 entries total. Only successful strings are cached. |
| `force_regenerate` | `false` | When true, opt out of ComfyUI execution caching and bypass internal answer cache each queue run. |

Retries after an empty final answer automatically increase `num_predict`
from the configured `max_new_tokens` to `2x`, `3x`, etc.,
capped at 8192. The retry is for the failed question only.
Transport, HTTP and configuration errors follow `error_policy` without
blind retries. A completed question stays cached when a later question fails,
if answer caching is enabled.

Changing the seed changes the cache key. If generation is nearly greedy
(e.g. temperature 0), a new seed may still produce an identical answer.

Failures never appear as exception strings in captions. `skip_failed`
returns an empty string, allowing downstream Caption Assemble to substitute
a manual fallback or skip the section entirely.

### Why the new widgets are at the bottom

Existing saved ComfyUI workflows serialize widget values in order. New
settings are appended **after** the existing ten question widgets to avoid
shifting old saved values. Existing answer pin types/names/positions remain
unchanged.

## Caption Assemble

One section per numbered slot. With `enabled_N=true`, use connected
`text_N` if nonblank; otherwise use `manual_N`. Disabled or empty
sections are skipped. `slot_count` controls section inclusion; join modes
are space, newline, double newline, comma-space, or custom delimiter.

Connect upstream answers to the optional `text_N` pins, not to `manual_N`,
so the manual widgets remain available for alternate text.

## Test and release checklist

From the CaptionForge repository:

```powershell
python -m unittest discover -s tests -p "test_lightweight_nodes.py" -v
```

Then restart the ComfyUI **server**, run several questions against
`gemma4:26b`, and test: fixed/unseeded seeds; explicit thinking modes;
empty-answer recovery; `skip_failed`; `force_regenerate`; and the
caption assembly order. The tests mock Ollama; real Ollama behavior and
ComfyUI frontend widget handling require separate local validation.

No automatic model download or release-version change is performed.
