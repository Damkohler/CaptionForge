# Dataset export prototype

This opt-in export mode extends the existing Pipeline Planner and Orchestrator.

## Try it

1. Reload your user-managed ComfyUI instance at port **8189** to load the Python changes.
2. Load `assets/workflows/CaptionForge_FullWorkflow_Rel_v1.0.2.json`, or add fresh
   Planner and Orchestrator nodes to an existing workflow.
3. Set the Planner input path and output folder. The prototype follows a
   **1536-pixel** caption/Validator maximum, with no enlargement.
4. Enable **Dataset - export image and caption** on the Planner. The supplied
   canonical workflow exposes this control; ordinary node defaults leave it off.
5. Choose the divisor, image format, and training caption. Queue a small dataset.

When connected, the Planner owns **every** Dataset setting, including disabled,
zero, and blank values. Orchestrator controls apply only in standalone mode.
Older plans with no Dataset settings keep export disabled.

| Control on both nodes | Default | Behavior |
| --- | --- | --- |
| Dataset - export image and caption | Off | Enables paired image and plain TXT export |
| Dataset - output folder | Blank | Parent folder; blank uses the main Output folder |
| Dataset - max image size | 0 | Follows configured Validator size; positive values override export size only |
| Dataset - dimension divisor | 16 | Integer; rounds both edges down. 1 disables alignment |
| Dataset - image format | PNG | PNG or JPEG, RGB |
| Dataset - JPEG quality | 95 | Applies only to JPEG |
| Dataset - caption | short | short, long, or taggy for the matching plain TXT |

In planned runs, the effective Validator maximum comes from the Planner's
**Caption - max image size**. Standalone runs use **Validator - max image size**.
If both the export override and effective Validator limit are zero, there is no
long-edge cap; divisor alignment still applies. Neither mode enlarges images.

## Output and protection

For an input `portraits/photo.jpg`, PNG export creates:

```text
<selected output parent>/training_dataset/
  .captionforge-dataset.json
  files/portraits/
    photo.jpg.png
    photo.jpg.txt
    photo.jpg.captionforge.json
    photo.jpg_long.txt
    photo.jpg_short.txt
    photo.jpg_taggy.txt
```

The last three files follow **Final - write TXT sidecars**. The selected plain
training TXT is always written when Dataset export is enabled. With export on,
caption variants are written beside the exported image rather than into the
source archive. With export off, existing v1.0.1 sidecar behavior is preserved.

Original extensions remain in the export stem, so `photo.jpg` and `photo.png`
produce distinct image/caption pairs. Relative folders are retained. Optional
IMAGE inputs use a separate `optional/` namespace. Sources outside the declared
input root use an `external/` namespace derived from their parent directory.

The dataset folder must be empty or already owned by CaptionForge. Untracked
files are not overwritten even when overwrite is enabled. The input path may
not be inside the selected export dataset. Resolved destination paths must stay
within the dataset and cannot refer to the source image, including hard links.
All three witness scanners exclude marked datasets on subsequent runs, even
when export has since been disabled. Do not remove the ownership marker.

## Resize and resume

Both output dimensions are rounded down after proportional size calculation.
This introduces a small aspect-ratio adjustment without cropping. If either
edge would become zero, export reports an error and keeps the captions in the
final record; reduce the divisor to handle such images.

Validator pixels are reused when their dimensions exactly match the requested
export dimensions. Otherwise, export resizes directly from the original to
avoid repeated resampling. Validator retry sizes do not change export size.

With overwrite disabled, completed caption records can receive a new dataset
export without repeating B/C/D model calls. Existing pairs resume only when
source metadata, settings, caption, and output hashes match their receipt.
Changed or incomplete pairs require overwrite to regenerate.
When changing PNG/JPEG format for an existing dataset, choose another output
parent; the prototype refuses to leave duplicate training images sharing one caption.
Export failures
retain caption text and are reported in the final record. Each file is published
by replacement from a temporary file; a receipt written last records completion.
An interrupted pair is not treated as complete.

## Scope

This prototype does not encode training latents or caption embeddings. It does
not change release version numbers, production captioning defaults, or public
node outputs. Export paths and dimensions are included in `final_records` under
`dataset_export`. The existing JSONL audit pipeline remains available.

CPU checks cover resize limits, divisor errors, pairing, collisions, recursive
exclusion, Planner ownership, and caption-preserving resume. Live canvas and
real-model validation remain pending; port 8189 was not responding during development.
