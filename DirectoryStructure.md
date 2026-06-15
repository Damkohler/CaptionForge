Project Folder/File Structure:

CaptionForge/
├─ __init__.py
├─ README.md
├─ .gitignore
├─ pyproject.toml
│
├─ nodes/
│  ├─ __init__.py
│  ├─ captionforge_extra_options_CUI_node.py
│  ├─ jlc_captionforge_node.py
│  ├─ jlc_captionforge_pipeline_planner_node.py
│  ├─ jlc_joy_caption_CUI_node.py
│  ├─ jlc_joy_caption_lite_CUI_node.py
│  ├─ jlc_qwen_caption_CUI_node.py
│  ├─ jlc_qwen_caption_lite_CUI_node.py
│  ├─ jlc_smolvlm_caption_CUI_node.py
│  ├─ jlc_smolvlm_caption_lite_CUI_node.py
│
├─ engines/
│  ├─ __init__.py
│  ├─ captionforge_claim_engine.py
│  ├─ captionforge_model_cache.py
│  ├─ captionforge_pipeline_planner_engine.py
│  ├─ CLI_Settings.txt
│  ├─ jlc_joy_caption_engine.py
│  ├─ jlc_qwen_caption_engine.py
│
├─ semantic_profiles/
│  ├─ female_character_v1.semantic_profile.json
│  ├─ general_v1.semantic_profile.json
│  ├─ image_v1_minimum.semantic_profile.json
│  ├─ experimental/
│  │  ├─ female_character_conservative_v1.semantic_profile.json
│  │  ├─ female_character_rich_overbroad_v1.semantic_profile.json
│
└─ web/
   ├─ jlc_captionforge_icons.js
   └─ ...