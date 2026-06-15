
from __future__ import annotations
MANIFEST={"name":"JLC BLIP Caption (Lite)","version":(0,1,0),"author":"J. L. Córdova"}
import json,time
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from PIL import Image
import folder_paths
from ..engines.jlc_blip_caption_engine import CaptionRecord, CleanupConfig, BlipCaptionConfig, BlipCaptionEngine, GenerationConfig, MEMORY_MODES, MODEL_REGISTRY, PROMPT_PRESETS, append_jsonl_records, record_to_json, resolve_prompt, timestamp, write_run_config_json, write_text_sidecar
from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs
JLC_BLIP_MODEL_ROOT=Path(folder_paths.models_dir)/"LLM"/"JLC_BLIPCaption"; DEFAULT_JSONL_FILENAME="captions.jsonl"; _SUF={".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
def _tensor_to_pil(t):
    if t is None: return []
    if isinstance(t,torch.Tensor): t=t.detach().cpu()
    if t.ndim==3: t=t.unsqueeze(0)
    return [Image.fromarray(np.clip(img.numpy()*255.0,0,255).astype(np.uint8)).convert("RGB") for img in t]
def _safe(v):
    s=[]
    for ch in v.replace("\\","/"):
        if ch.isalnum() or ch in {"-","_","."}: s.append(ch)
        elif ch=="/": s.append("__")
        else: s.append("_")
    return "".join(s).strip("._") or "image"
def _iter(path, recursive, glob):
    root=Path(str(path or "").strip())
    if not root: return []
    if not root.exists(): raise RuntimeError(f"CaptionForge input_path does not exist: {root}")
    if root.is_file(): return [(_safe(root.stem),root)] if root.suffix.lower() in _SUF else []
    paths=sorted(p for p in (root.rglob(glob or "*") if recursive else root.glob(glob or "*")) if p.is_file() and p.suffix.lower() in _SUF)
    return [(_safe(str(p.relative_to(root).with_suffix(""))),p) for p in paths]
def _open(p):
    with Image.open(p) as im: return im.convert("RGB")
def _norm(c):
    if c is None: return {}
    if isinstance(c,dict): return dict(c)
    if isinstance(c,str):
        try: o=json.loads(c.strip()); return o if isinstance(o,dict) else {}
        except Exception: return {}
    return {}
def _planned(c):
    paths=_norm(c).get("paths") if isinstance(_norm(c).get("paths"),dict) else {}
    for k in ("caption_jsonl","pass_a_jsonl"):
        v=str(paths.get(k) or "").strip()
        if v: return Path(v)
    return None
def _jsonl(records): return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in records)
def _txt(out,src,n,i): return out/f"{src}.txt" if n<=1 else out/f"{src}__cf_run_{i:02d}.txt"
class JLC_BLIPCaptionLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":(list(MODEL_REGISTRY.keys()),{"default":"BLIP-2 FLAN-T5 XL"}),"memory_mode":(list(MEMORY_MODES.keys()),{"default":"Default"}),"keep_loaded":("BOOLEAN",{"default":True}),"prompt_preset":(list(PROMPT_PRESETS.keys()),{"default":"detailed"}),"custom_prompt":("STRING",{"default":"","multiline":True}),"max_new_tokens":("INT",{"default":256,"min":16,"max":2048,"step":8}),"num_beams":("INT",{"default":3,"min":1,"max":16,"step":1}),"temperature":("FLOAT",{"default":0.0,"min":0.0,"max":2.0,"step":0.01}),"top_p":("FLOAT",{"default":0.90,"min":0.0,"max":1.0,"step":0.01}),"top_k":("INT",{"default":50,"min":0,"max":500,"step":1})},"optional":{"image":("IMAGE",{}),"captionforge_run_config":("CAPTIONFORGE_PIPELINE_PLAN",{})}}
    RETURN_TYPES=("CAPTIONFORGE_PIPELINE_PLAN","STRING","STRING","STRING"); RETURN_NAMES=("captionforge_run_config_out","caption","jsonl_records","resolved_prompt"); FUNCTION="caption"; CATEGORY="JLC/Captioning"
    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    def caption(self, model, memory_mode, keep_loaded, prompt_preset, custom_prompt, max_new_tokens, num_beams, temperature, top_p, top_k, image=None, captionforge_run_config=None):
        planned=bool(captionforge_run_config); prompt=resolve_prompt(prompt_preset, custom_prompt)
        runs=expand_captionforge_runs(captionforge_run_config, model_key="blip", widget_captions_per_image=1, widget_seed=-1, widget_temperature=float(temperature), widget_top_p=float(top_p), widget_top_k=int(top_k), widget_max_new_tokens=int(max_new_tokens), widget_max_size=768, widget_trigger_word="", widget_output_dir="", widget_input_path="", widget_recursive=True, widget_filename_glob="*")
        if planned and not runs:
            status="[CaptionForge] BLIP Caption Lite disabled by Pipeline Planner."; print(status); return (captionforge_run_config,status,"",prompt)
        first=runs[0]
        engine=BlipCaptionEngine(BlipCaptionConfig(model,"",str(JLC_BLIP_MODEL_ROOT),memory_mode,"bf16","auto",False,bool(keep_loaded),"",True,int(first.max_size),prompt,True,True),GenerationConfig(int(first.max_new_tokens),int(num_beams),float(first.temperature),float(first.top_p),int(first.top_k),1.0,first.seed),CleanupConfig("",f"{first.trigger_word}," if first.trigger_word else ""))
        direct=[(f"comfy_image_{i:04d}",pil) for i,pil in enumerate(_tensor_to_pil(image))]; files=[]
        if first.input_path: files=_iter(first.input_path, first.recursive, first.filename_glob)
        if not direct and not files: raise RuntimeError("No image input found. Connect an IMAGE input or provide input_path in the CaptionForge Run Plan.")
        pj=_planned(captionforge_run_config) if planned else None; out=Path(first.output_dir) if first.output_dir else Path(folder_paths.get_output_directory())/"jlc_blip_caption_lite"
        if pj: out=pj.parent
        if planned: out.mkdir(parents=True,exist_ok=True)
        jsonl_path=pj or (out/DEFAULT_JSONL_FILENAME); all_records=[]; engine.load()
        if planned: write_run_config_json(out/f"jlc_blip_caption_lite_run_config_{timestamp()}.json", engine.build_run_config(), False)
        def process(src,pil):
            for r in runs:
                engine.generation=GenerationConfig(int(r.max_new_tokens),int(num_beams),float(r.temperature),float(r.top_p),int(r.top_k),1.0,r.seed); engine.cleanup=CleanupConfig("",f"{r.trigger_word}," if r.trigger_word else ""); engine.config.max_size=int(r.max_size); engine.config.prompt=prompt
                t0=time.perf_counter(); final,raw=engine.caption_pil(pil); dt=time.perf_counter()-t0; print(f"[JLC BLIP Caption Lite] Generation time run {r.ensemble_run_index}: {dt:.2f}s")
                rec=CaptionRecord(src,final,raw,model,str(engine.local_model_path or ""),prompt,r.seed,r.temperature,r.top_p,r.top_k,int(num_beams),r.max_new_tokens,int(r.max_size),datetime.now().isoformat(timespec="seconds"),captionforge_pass="A",model_family="blip",ensemble_run_index=r.ensemble_run_index,image_key=src)
                all_records.append(rec)
                if planned: append_jsonl_records(jsonl_path,[rec],False); write_text_sidecar(_txt(out,src,len(runs),r.ensemble_run_index),rec.caption,True,False,False)
                print(f"[JLC BLIP Caption Lite] Captioned {src} run {r.ensemble_run_index+1}/{len(runs)}")
        for src,pil in direct: process(src,pil)
        for src,path in files: process(src,_open(path))
        if not keep_loaded: engine.unload()
        return (captionforge_run_config,"\n\n".join(r.caption for r in all_records if r.status=="ok"),_jsonl(all_records),prompt)
NODE_CLASS_MAPPINGS={"JLC_BLIPCaptionLite":JLC_BLIPCaptionLite}
NODE_DISPLAY_NAME_MAPPINGS={"JLC_BLIPCaptionLite":"\u2003JLC BLIP Caption (Lite)"}
