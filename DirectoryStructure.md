Project Folder/File Structure:

CaptionForge/
├─ __init__.py
├─ README.md
├─ pyproject.toml
│
├─ nodes/
│  ├─ __init__.py
│  ├─ captionforge_extra_options_CUI_node.py
│  ├─ jlc_captionforge_node.py
│  ├─ jlc_captionforge_pipeline_planner_node.py
|  ├─ caption_nodes
│  │  ├─ jlc_joy_caption_node.py
│  │  ├─ jlc_qwen_caption_node.py
│  │  ├─ jlc_ollama_caption_node.py
│
├─ engines/
│  ├─ __init__.py
│  ├─ captionforge_claim_engine.py
│  ├─ captionforge_model_cache.py
│  ├─ captionforge_pipeline_planner_engine.py
│  ├─ jlc_joy_caption_engine.py
│  ├─ jlc_qwen_caption_engine.py
│
├─ config/
│  ├─  captionforge_ollama_models.json
│
└─ web/
   ├─ jlc_captionforge_icons.js
   └─ ...