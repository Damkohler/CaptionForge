# Repository structure

CaptionForge keeps ComfyUI-facing nodes separate from reusable backend code.
Only the explicit production packages and package data listed in
`pyproject.toml` are included in a distribution.

```text
CaptionForge/
├─ __init__.py                         ComfyUI registration entry point
├─ captionforge_version.py             authoritative release version
├─ README.md
├─ pyproject.toml
├─ config/
│  └─ captionforge_ollama_models.json  user-editable Ollama model tags
├─ nodes/
│  ├─ jlc_captionforge_pipeline_planner_node.py
│  ├─ jlc_captionforge_node.py         production B/C/D capstone
│  ├─ jlc_captionforge_template_options.py
│  ├─ captionforge_ollama_model_dropdowns.py
│  └─ caption_nodes/
│     ├─ jlc_captionforge_joy_caption_node.py
│     ├─ jlc_captionforge_qwen_caption_node.py
│     └─ jlc_captionforge_ollama_caption_node.py
├─ engines/
│  ├─ captionforge_pipeline_planner_engine.py
│  ├─ captionforge_prompt_defaults.py
│  ├─ captionforge_caption_prompt_kit.py
│  ├─ captionforge_joy_space_prompt_kit.py
│  ├─ captionforge_model_cache.py
│  ├─ jlc_joy_caption_engine.py
│  ├─ jlc_qwen_caption_engine.py
│  ├─ captionforge_distiller_engine.py  standalone reference/CLI engine
│  └─ captionforge_vlm_validator_engine.py  standalone reference/CLI engine
├─ assets/
│  ├─ icons/
│  └─ workflows/                       canonical JSON/API JSON/PNG workflows
├─ docs/                               release-facing studies
├─ tests/                              CPU contract tests
└─ web/                                ComfyUI frontend/icon support
```

Local `.backups/`, `.tests/`, `.vscode/`, `.bak/`, `.deprecated/`, and
experimental trees are archival or development-only. They are neither
registered by CaptionForge nor included in the production package.
