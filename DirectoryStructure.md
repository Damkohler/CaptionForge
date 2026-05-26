Refactor Plan:
Phase 1: add nodes/ and engines/run_plan/
Phase 2: move cache to engines/cache/
Phase 3: move Qwen/Joy engines to engines/captioning/
Phase 4: move Qwen/Joy node wrappers to nodes/
Phase 5: move claim/semantic engine to engines/semantic/


Proposed Folder/File Structure:

CaptionForge/
├─ __init__.py
├─ README.md
├─ .gitignore
├─ pyproject.toml                 # later, if useful
│
├─ nodes/
│  ├─ __init__.py
│  ├─ qwen_caption_node.py
│  ├─ joy_caption_node.py
│  ├─ qwen_caption_lite_node.py
│  ├─ joy_caption_lite_node.py
│  └─ captionforge_run_plan_node.py
│
├─ engines/
│  ├─ __init__.py
│  ├─ captioning/
│  │  ├─ __init__.py
│  │  ├─ qwen_caption_engine.py
│  │  └─ joy_caption_engine.py
│  │
│  ├─ semantic/
│  │  ├─ __init__.py
│  │  └─ claim_engine.py
│  │
│  ├─ cache/
│  │  ├─ __init__.py
│  │  └─ model_cache.py
│  │
│  └─ run_plan/
│     ├─ __init__.py
│     └─ captionforge_run_plan.py
│
├─ semantic_profiles/
│  ├─ general_v1.semantic_profile.json
│  ├─ female_character_v1.json
│  └─ experimental/
│
└─ web/
   ├─ captionforge_branding.js
   └─ ...