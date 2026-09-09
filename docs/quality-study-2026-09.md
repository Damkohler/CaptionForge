# Fixed-corpus caption quality study

Date: 2026-09-09

Baseline: `5568f74dbb103b99bd6603481a247aee6033f753`

## Question

Evaluate the current `_long.txt`, `_short.txt`, and `_taggy.txt` outputs before
the 1.0 release. In particular, determine whether short captions retain useful
LoRA-training detail and whether taggy captions are compact, faithful, and
readable enough to keep without redesign.

## Corpus and method

The fixed corpus is the 19-image local `Release_1.0.1_BigTest01` set. Despite
the historical folder name, this is pre-1.0 release-candidate evidence. It
contains real photographs, digital illustrations, anime-style art, glossy 3D
characters, a ball-jointed doll, portraits, full-body images, stage/runway
scenes, indoor and outdoor settings, simple and detailed backgrounds, varied
poses, and varied clothing/material descriptions.

The stored run used the production model path:

- Pass A: one Joy and one Qwen caption per image
- Pass B: `mistral-small:24b`
- Pass C: `gemma4:26b`
- Pass D: `mistral-small:24b`

The study combined mechanical measurements with a manual image/caption review.
The proposed refinement was then tested directly against all 19 validated long
captions with `mistral-small:24b`, seed 3, temperature 0.12, top-p 0.88, and
top-k 50.

## Existing output measurements

| Metric | Result |
| --- | ---: |
| Images with complete long/short/taggy triplets | 19 |
| Mean long length | 127.3 words |
| Mean short length | 81.6 words |
| Mean short/long ratio | 66.6% |
| Shorts that are exact prefixes of long | 19/19 |
| Shorts that drop content | 16/19 |
| Shorts that drop at least 20 words | 14/19 |
| Shorts identical to long | 3/19 |
| Mean taggy length | 67.8 words / 28.5 items |
| Taggy range | 18–36 items |
| Largest taggy output | 648 characters |
| Exact duplicate taggy items | 0 |
| Taggy outputs reaching a compaction limit | 0 |

## Findings

### Long

`_long` is appropriately treated as authoritative. It consistently contains
the richest set of subject, appearance, clothing, pose, setting, lighting,
framing, and style details. This study did not identify a downstream-formatting
reason to change Pass C.

Manual image review did find isolated Pass-C wording errors (for example, a
doll support stand described as a microphone stand). Those are validator
accuracy issues, not short/taggy divergence, and are outside this formatting
study.

### Short: material problem found

The deterministic implementation is sentence-prefix extraction, not semantic
compression. Every stored short is the beginning of its long caption. Because
the validator commonly places setting, lighting, framing, style, accessories,
and some pose/body details near the end, those categories are systematically
more likely to disappear.

Representative losses include:

- the entire outfit, exposed midriff/thighs, pose, and shiny fabric detail from
  the pink bikini/sarong illustration;
- mirror-selfie action, phone/hand placement, background, lighting, and
  anime/semi-realistic rendering details;
- body shape, side-view pose, runway, crowd, sign text, reflective floor,
  full-body framing, and doll-like style;
- sailor hat, high heels, salute, relaxed arm, full-body framing, and studio
  lighting;
- castle, lanterns, trees, moon, stars, river, lighting, and 3D style from the
  fairy image.

There is also a concrete limit defect: when a validated long caption is a
single sentence longer than 90 words, the current helper keeps the entire
sentence. Two corpus shorts therefore contain 113 and 119 words despite the
advertised 90-word maximum.

Conclusion: the existing `_short` is materially worse than `_long` for
training whenever useful categories occur late in the validated paragraph. It
is not reliable as the intended LoRA-length semantic summary.

### Taggy: acceptable with minor cleanup

`_taggy` is compact enough for the studied corpus and remains semantically
close to `_long`. The deterministic compactor did not truncate any result, so
it removed no useful content through item or character limits. Exact duplicate
items were absent.

Minor defects remain: occasional dangling fragments (`sides`, `left`,
`visible`), terminal sentence punctuation, and near-duplicates such as
`glossy skin`/`smooth skin` or `bokeh lights`/`bokeh effect`. These do not
justify a new taggy architecture. Terminal punctuation can be safely removed
during deterministic cleanup; semantic near-deduplication should not be made
aggressive because similar phrases can carry distinct material or appearance
information.

## Refinement experiment

The existing Pass-D formatter was asked to return two labeled derivatives in
the same call. The short output was constrained to three concise sentences
covering, when present:

1. subject/appearance and major clothing/material;
2. pose/body/accessories or unusual visible details;
3. setting/lighting/framing and medium/style.

Results:

| Metric | Result |
| --- | ---: |
| Dual responses parsed | 19/19 |
| Shorts within 90 words | 19/19 |
| Short word range | 47–77 |
| Mean short length | 60.9 words |
| Added model calls | 0 |
| Added image-validation calls | 0 |

Manual review found that the refined shorts consistently selected information
from across the whole validated caption rather than stopping after its opening.
They are meaningfully shorter while retaining a balanced training identity.
As expected for compression, not every minor detail survives; the full long and
taggy variants remain available when maximum recall is preferred.

## Decision

Adopt the dual-output Pass-D prompt as the smallest defensible fix:

- `_long` remains the untouched Pass-C authority.
- One existing Pass-D text-only call derives both `_short` and `_taggy`.
- The formatter may only compress validated content; it may not add, infer, or
  correct visual claims.
- Deterministic cleanup enforces the short limit and normalizes taggy output.
- Legacy/custom taggy-only formatter responses remain supported, using the
  corrected deterministic short helper as a fallback.
- Do not add another VLM pass or another LLM call.

This resolves the demonstrated short-caption failure without redesigning the
successful semantic pipeline or the acceptable taggy path.
