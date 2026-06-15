
from __future__ import annotations
MANIFEST={"name":"JLC BLIP Caption","version":(0,1,0),"author":"J. L. Córdova"}
import json, time
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from PIL import Image
import folder_paths
from ..engines.jlc_blip_caption_engine import BatchCaptionConfig, CaptionRecord, CleanupConfig, BlipCaptionConfig, BlipCaptionEngine, GenerationConfig, MEMORY_MODES, MODEL_REGISTRY, PROMPT_PRESETS, append_jsonl_records, load_existing_jsonl_images, probe_registry_model_download, record_to_json, resolve_prompt, timestamp, write_run_config_json, write_text_sidecar
from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs
JLC_BLIP_MODEL_ROOT=Path(folder_paths.models_dir)/"LLM"/"JLC_BLIPCaption"
def _tensor_to_pil(t):
    if t is None: return []
    if isinstance(t,torch.Tensor): t=t.detach().cpu()
    if t.ndim==3: t=t.unsqueeze(0)
    out=[]
    for img in t:
        arr=np.clip(img.numpy()*255.0,0,255).astype(np.uint8); out.append(Image.fromarray(arr).convert("RGB"))
    return out
def _lines(v): return [x.strip() for x in (v or "").splitlines() if x.strip()]
def _pairs(v):
    out=[]
    for line in (v or "").splitlines():
        line=line.strip()
        if line and not line.startswith("#") and "=>" in line:
            a,b=line.split("=>",1); a=a.strip(); b=b.strip()
            if a: out.append((a,b))
    return out
def _norm_plan(c):
    if c is None: return {}
    if isinstance(c,dict): return dict(c)
    if isinstance(c,str):
        try: o=json.loads(c.strip()); return o if isinstance(o,dict) else {}
        except Exception: return {}
    return {}
def _planned_jsonl(c):
    paths=_norm_plan(c).get("paths") if isinstance(_norm_plan(c).get("paths"),dict) else {}
    for k in ("caption_jsonl","pass_a_jsonl"):
        v=str(paths.get(k) or "").strip()
        if v: return Path(v)
    return None
def _jsonl(records): return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in records)
class JLC_BLIPCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "model":(list(MODEL_REGISTRY.keys()),{"default":"BLIP-2 FLAN-T5 XL"}),
            "memory_mode":(list(MEMORY_MODES.keys()),{"default":"Default"}),
            "input_path":("STRING",{"default":"","multiline":False}), "recursive":("BOOLEAN",{"default":True}), "filename_glob":("STRING",{"default":"*","multiline":False}),
            "prompt_preset":(list(PROMPT_PRESETS.keys()),{"default":"detailed"}), "custom_prompt":("STRING",{"default":"","multiline":True}),
            "max_new_tokens":("INT",{"default":256,"min":16,"max":2048,"step":8}), "num_beams":("INT",{"default":3,"min":1,"max":16,"step":1}),
            "temperature":("FLOAT",{"default":0.0,"min":0.0,"max":2.0,"step":0.01}), "top_p":("FLOAT",{"default":0.90,"min":0.0,"max":1.0,"step":0.01}), "top_k":("INT",{"default":50,"min":0,"max":500,"step":1}),
            "repetition_penalty":("FLOAT",{"default":1.0,"min":1.0,"max":2.0,"step":0.01}), "captions_per_image":("INT",{"default":1,"min":1,"max":100,"step":1}), "seed":("INT",{"default":-1,"min":-1,"max":0xFFFFFFFF,"step":1}),
            "max_size":("INT",{"default":768,"min":0,"max":4096,"step":64}), "output_dir":("STRING",{"default":"","multiline":False}),
            "write_txt":("BOOLEAN",{"default":True}), "write_jsonl":("BOOLEAN",{"default":False}), "also_jsonl":("BOOLEAN",{"default":False}), "write_run_config":("BOOLEAN",{"default":True}), "jsonl_filename":("STRING",{"default":"captions.jsonl","multiline":False}),
            "overwrite":("BOOLEAN",{"default":False}), "backup_existing":("BOOLEAN",{"default":True}), "dry_run":("BOOLEAN",{"default":False}), "limit":("INT",{"default":0,"min":0,"max":100000,"step":1}),
            "skip_existing_txt":("BOOLEAN",{"default":True}), "skip_existing_jsonl_images":("BOOLEAN",{"default":False}), "prefix":("STRING",{"default":"","multiline":False}), "suffix":("STRING",{"default":"","multiline":False}),
            "forbidden_phrases":("STRING",{"default":"","multiline":True}), "replace_pairs":("STRING",{"default":"","multiline":True}), "keep_loaded":("BOOLEAN",{"default":True}), "download_probe_only":("BOOLEAN",{"default":False}),
            },"optional":{"image":("IMAGE",{}),"captionforge_run_config":("CAPTIONFORGE_PIPELINE_PLAN",{})}}
    RETURN_TYPES=("CAPTIONFORGE_PIPELINE_PLAN","STRING","STRING","STRING"); RETURN_NAMES=("captionforge_run_config_out","caption","jsonl_records","resolved_prompt"); FUNCTION="caption"; CATEGORY="JLC/Captioning"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    def caption(self, model, memory_mode, input_path, recursive, filename_glob, prompt_preset, custom_prompt, max_new_tokens, num_beams, temperature, top_p, top_k, repetition_penalty, captions_per_image, seed, max_size, output_dir, write_txt, write_jsonl, also_jsonl, write_run_config, jsonl_filename, overwrite, backup_existing, dry_run, limit, skip_existing_txt, skip_existing_jsonl_images, prefix, suffix, forbidden_phrases, replace_pairs, keep_loaded, download_probe_only, image=None, captionforge_run_config=None):
        if download_probe_only: return (captionforge_run_config, probe_registry_model_download(model,JLC_BLIP_MODEL_ROOT), "", "")
        prompt=resolve_prompt(prompt_preset, custom_prompt)
        runs=expand_captionforge_runs(captionforge_run_config, model_key="blip", widget_captions_per_image=int(captions_per_image), widget_seed=int(seed), widget_temperature=float(temperature), widget_top_p=float(top_p), widget_top_k=int(top_k), widget_max_new_tokens=int(max_new_tokens), widget_max_size=int(max_size), widget_trigger_word="", widget_output_dir=output_dir, widget_input_path=input_path, widget_recursive=bool(recursive), widget_filename_glob=filename_glob)
        planned=bool(captionforge_run_config)
        if planned and not runs:
            status="[CaptionForge] BLIP Caption disabled by Pipeline Planner."; print(status); return (captionforge_run_config,status,"",prompt)
        first=runs[0]
        if planned: write_jsonl=True; also_jsonl=False; skip_existing_txt=False; skip_existing_jsonl_images=False; overwrite=True
        outdir=(first.output_dir or output_dir or "").strip(); inpath=(first.input_path or input_path or "").strip(); jsonl_filename=(jsonl_filename or "captions.jsonl").strip() or "captions.jsonl"
        planned_jsonl=_planned_jsonl(captionforge_run_config) if planned else None
        if planned_jsonl: outdir=str(planned_jsonl.parent); jsonl_filename=planned_jsonl.name
        eff_prefix=f"{first.trigger_word}, {prefix}".strip(", ") if first.trigger_word else prefix
        engine=BlipCaptionEngine(BlipCaptionConfig(model, "", str(JLC_BLIP_MODEL_ROOT), memory_mode, "bf16", "auto", False, bool(keep_loaded), "", True, int(first.max_size), prompt, True, True), GenerationConfig(int(first.max_new_tokens), int(num_beams), float(first.temperature), float(first.top_p), int(first.top_k), float(repetition_penalty), first.seed), CleanupConfig("",eff_prefix,suffix,_lines(forbidden_phrases),_pairs(replace_pairs)))
        all_records=[]; pil_images=_tensor_to_pil(image); batch_result=None; use_jsonl=bool(write_jsonl or also_jsonl)
        if pil_images:
            engine.load(); image_out=Path(outdir) if outdir else Path(folder_paths.get_output_directory())/"jlc_blip_caption"; image_out.mkdir(parents=True,exist_ok=True); jsonl_path=image_out/jsonl_filename
            if write_run_config and not inpath: write_run_config_json(image_out/f"jlc_blip_caption_run_config_{timestamp()}.json", engine.build_run_config(), bool(dry_run))
            seen=load_existing_jsonl_images(jsonl_path) if use_jsonl and skip_existing_jsonl_images else set()
            for idx,pil in enumerate(pil_images):
                src=f"comfy_image_{idx:04d}"; txt=image_out/f"{src}.txt"
                if len(runs)==1 and skip_existing_txt and write_txt and txt.exists() and not overwrite: print(f"[JLC BLIP Caption] Skipping existing TXT: {txt}"); continue
                if use_jsonl and skip_existing_jsonl_images and src in seen: print(f"[JLC BLIP Caption] Skipping existing JSONL image: {src}"); continue
                for r in runs:
                    engine.generation=GenerationConfig(int(r.max_new_tokens),int(num_beams),float(r.temperature),float(r.top_p),int(r.top_k),float(repetition_penalty),r.seed); engine.config.max_size=int(r.max_size); engine.config.prompt=prompt
                    engine.cleanup=CleanupConfig("", f"{r.trigger_word}, {prefix}".strip(", ") if r.trigger_word else prefix, suffix, _lines(forbidden_phrases), _pairs(replace_pairs))
                    run_txt=txt if len(runs)<=1 else image_out/f"{src}__cf_run_{r.ensemble_run_index:02d}.txt"
                    if skip_existing_txt and write_txt and run_txt.exists() and not overwrite: print(f"[JLC BLIP Caption] Skipping existing TXT: {run_txt}"); continue
                    t0=time.perf_counter(); final,raw=engine.caption_pil(pil); dt=time.perf_counter()-t0; print(f"[JLC BLIP Caption] Generation time run {r.ensemble_run_index}: {dt:.2f}s")
                    rec=CaptionRecord(src,final,raw,model,str(engine.local_model_path or ""),prompt,r.seed,r.temperature,r.top_p,r.top_k,int(num_beams),r.max_new_tokens,int(r.max_size),datetime.now().isoformat(timespec="seconds"),captionforge_pass="A",model_family="blip",ensemble_run_index=r.ensemble_run_index,image_key=src)
                    all_records.append(rec)
                    if use_jsonl: append_jsonl_records(jsonl_path,[rec],bool(dry_run))
                    if write_txt: write_text_sidecar(run_txt,rec.caption,bool(overwrite),bool(backup_existing),bool(dry_run))
                    print(f"[JLC BLIP Caption] Captioned IMAGE {idx+1}/{len(pil_images)} run {r.ensemble_run_index+1}/{len(runs)}: {src}")
        if inpath:
            batch=BatchCaptionConfig(inpath,bool(first.recursive if planned else recursive),first.filename_glob if planned else ((filename_glob or "*").strip() or "*"),output_dir=outdir,write_txt=bool(write_txt),write_jsonl=bool(write_jsonl),jsonl_filename=jsonl_filename,write_run_config=bool(write_run_config),overwrite=bool(overwrite),backup_existing=bool(backup_existing),dry_run=bool(dry_run),limit=int(limit),skip_existing_txt=bool(skip_existing_txt),skip_existing_jsonl_images=bool(skip_existing_jsonl_images))
            batch_result=engine.caption_batch(batch); all_records.extend(batch_result.records)
        if not pil_images and batch_result is None: raise RuntimeError("No image input found. Connect an IMAGE input or provide input_path pointing to an image file or folder.")
        if not keep_loaded: engine.unload()
        return (captionforge_run_config,"\n\n".join(r.caption for r in all_records if r.status=="ok"),_jsonl(all_records),prompt)
NODE_CLASS_MAPPINGS={"JLC_BLIPCaption":JLC_BLIPCaption}
NODE_DISPLAY_NAME_MAPPINGS={"JLC_BLIPCaption":"\u2003JLC BLIP Caption"}
