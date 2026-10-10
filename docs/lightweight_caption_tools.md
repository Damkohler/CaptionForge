# CaptionForge Lightweight Nodes — prototype reliability pass

This feature branch adds two standalone ComfyUI nodes without changing the
CaptionForge A–D pipeline. Both pass the original `IMAGE` tensor through.

## Targeted VLM Query

Ask 1–10 separate image-grounded questions through an existing Ollama VLM,
with stable `answer_1` … `answer_10` pins. Each active question is an
independent Ollama `/api/chat` request; no chat history is shared.
The two modes for selecting slots are explained below.

The node reuses `config/captionforge_ollama_models.json`, including the
`custom` model-tag escape hatch.

### Slot selection (both lightweight nodes)

`slot_mode` offers two explicit options:

- `first_n` (**default; existing behavior**): only consider slots numbered
  1 through `slot_count`. Whitespace-only questions or sections are skipped,
  but content above the slot-count limit is ignored.
- `all_nonempty`: check all ten numbered slots regardless of `slot_count`.
  Targeted VLM Query asks only the nonblank questions. Caption Assemble
  includes only enabled, nonblank resolved sections. The `slot_count`
  widget is ignored in this mode.

**Numbers and wiring never change.** Question 5 always produces `answer_5`,
even if questions 1–4 are empty; Caption Assemble `text_5` corresponds to
section 5. Blank answer pins still return `""`, not shifted answers.
Sections remain ordered by slot number, not by connection order.

For example, to build a fixed character description followed by
image-specific pose/outfit/background details:

1. Put fixed character paragraphs in Caption Assemble sections 1–3.
2. Put VLM questions in Targeted VLM Query slots 5, 6, and 7, leaving 1–4 blank.
3. Connect `answer_5` → `text_5`, `answer_6` → `text_6`, and
   `answer_7` → `text_7`; optionally add headings as other manual sections.
4. Select `all_nonempty` on **both** nodes. Neither slot count needs adjusting.

The new `slot_mode` widgets were **appended at the bottom of each existing
widget sequence** to avoid shifting serialized fields in saved workflows.
They default to `first_n` so existing workflows retain their semantics.

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
settings, including `slot_mode`, are appended **after** the existing
ten question widgets to avoid shifting old saved values. Existing answer
pin types/names/positions remain unchanged.

## Caption Assemble — dual outputs and deterministic composition

Caption Assemble combines each enabled section's **manual** and **incoming**
text instead of treating the incoming STRING as an override. The original
`image` (unchanged) and `caption` output positions remain 0 and 1.
The new output at position 2 is `prose_caption`.

- `caption`: structured/outline caption, retaining manual section headings.
- `prose_caption`: text in the same section order, with optional omission of
  heading-only manual text. It is **deterministic**, not LLM-edited prose.
  Sentence grammar, repetitions, and semantics are not rewritten.

Connect Ollama `answer_N` to **`text_N`** in Caption Assemble, and type
headings, transitions, fixed character descriptions, or qualifiers into the
corresponding **`manual_N`** widgets. Neither overwrites the other.
Leaving either blank is valid. Sections with neither input nor manual text
are skipped, and disabled sections are omitted from both outputs.

The new global controls are appended after the existing `slot_mode` widget.
There are no new per-section controls.

| Control | Default | Meaning |
|---|---|---|
| `composition_mode` | `manual_then_input` | Arrange manual text before incoming text, or choose `input_then_manual` for qualifying phrases after it |
| `section_join_mode` | `newline` | Space, newline, or double newline **between** manual and incoming text when both are present; applies to both outputs |
| `join_mode` | `space` | Existing separator **between completed structured sections**: space, newline, double newline, comma-space or custom |
| `custom_delimiter` | one space | Used by existing structured `join_mode=custom` |
| `prose_join_mode` | `space` | Independent space, newline, double newline or comma-space **between prose sections** |
| `omit_headings_in_prose` | `true` | A single trimmed line of manual text ending in `:` is retained in `caption` but omitted from `prose_caption` |

### Colon heading convention

Only a **standalone, one-line manual field ending in a colon** is treated
as an outline heading, e.g. `The Pose:`. A manual paragraph such as
`Her outfit includes: ...` or a multi-line block ending in `:` is
preserved in both outputs. Text coming from the external `text_N` input
is never interpreted as a heading.

Example sections (both `slot_mode=all_nonempty` and
`composition_mode=manual_then_input`):

| Slot | Manual | External text |
|---|---|---|
| 1 | `A portrait of a guitarist.` | *(empty)* |
| 5 | `The Pose:` | `She holds both hands behind her head.` |
| 6 | `Outfit & Accessories:` | `She wears a black crop top.` |
| 7 | `The Setting:` | `A graffiti-covered urban alley.` |

With `section_join_mode=newline`, `join_mode=double_newline`,
and `prose_join_mode=space`:

`caption`:

```text
A portrait of a guitarist.

The Pose:
She holds both hands behind her head.

Outfit & Accessories:
She wears a black crop top.

The Setting:
A graffiti-covered urban alley.
```

`prose_caption`:

```text
A portrait of a guitarist. She holds both hands behind her head. She wears a black crop top. A graffiti-covered urban alley.
```

The prose output is intentionally mechanical. Use complete sentences or
fragments that naturally connect; the assembler does not infer linking words.

If the VLM returns an empty answer while `manual_N` contains only a
colon-terminated heading, that heading remains visible in the **structured**
output but is omitted from **prose**. This follows the existing choice to
preserve manual text independently of VLM success.

### Compatibility

The `IMAGE` and original structured `caption` pins keep their original
identities and positions; `prose_caption` is appended. The existing
widget and socket order is unchanged. The new global widgets appear at the
bottom of the node. **Intentional behavior change:** older workflows that
relied on a nonempty `text_N` replacing `manual_N` will now combine both
pieces. To recover an input-only section, simply clear its `manual_N`.

`first_n` and `all_nonempty`, the per-section enable/disable controls,
and the image passthrough work as before. No Ollama request or other model
is called from Caption Assemble.

## Test and release checklist

From the CaptionForge repository:

```powershell
python -m unittest discover -s tests -p "test_lightweight_nodes.py" -v
```

Then restart the ComfyUI **server**, run several questions against
`gemma4:26b`, and test: fixed/unseeded seeds; explicit thinking modes;
empty-answer recovery; `skip_failed`; `force_regenerate`; both slot
modes with sparse/blank questions; and both composition orders, heading omission, independent section/prose line
breaks, and caption assembly ordering. The tests mock Ollama; real Ollama
behavior and ComfyUI frontend widget handling require separate local validation.

No automatic model download or release-version change is performed.
