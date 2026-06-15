
from __future__ import annotations

MANIFEST = {"name":"JLC BLIP Caption Engine","version":(0,1,0),"author":"J. L. Córdova"}

import fnmatch, json, random, re, shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from PIL import Image
import torch

from .captionforge_model_cache import make_cache_key, get_cached_model, register_model, prepare_for_model_load, unload_after_run

@dataclass(frozen=True)
class BlipModelInfo:
    repo_id: str
    local_folder: str
    model_type: str
    notes: str = ""

MODEL_REGISTRY: dict[str, BlipModelInfo] = {
    "BLIP-2 FLAN-T5 XL": BlipModelInfo("Salesforce/blip2-flan-t5-xl", "blip2-flan-t5-xl", "blip2", "Advanced BLIP-2 checkpoint with FLAN-T5 XL."),
    "BLIP-2 OPT 2.7B": BlipModelInfo("Salesforce/blip2-opt-2.7b", "blip2-opt-2.7b", "blip2", "BLIP-2 checkpoint with OPT-2.7B."),
    "BLIP Large": BlipModelInfo("Salesforce/blip-image-captioning-large", "blip-image-captioning-large", "blip", "Classic BLIP large captioning baseline."),
    "BLIP Base": BlipModelInfo("Salesforce/blip-image-captioning-base", "blip-image-captioning-base", "blip", "Classic BLIP base captioning baseline."),
}

SUPPORTED_EXTENSIONS={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
PROMPT_PRESETS={
    "caption":"",
    "detailed":"Describe this image in detail.",
    "lora_literal":"Describe this image as a concise visual caption for image dataset training. Mention only visible subject, pose, clothing, appearance, style, lighting, and background.",
    "qa_detail":"Question: Describe this image in detailed visual terms. Answer:",
    "qa_lora":"Question: What visible details should be included in a text-to-image training caption for this image? Answer:",
}
DEFAULT_PROMPT_PRESET="detailed"
MEMORY_MODES={"Default":{}, "Balanced (8-bit)":{"load_in_8bit":True}}

@dataclass
class GenerationConfig:
    max_new_tokens:int=256
    num_beams:int=3
    temperature:float=0.0
    top_p:float=0.90
    top_k:int=50
    repetition_penalty:float=1.0
    seed:Optional[int]=None

@dataclass
class CleanupConfig:
    trigger:str=""
    prefix:str=""
    suffix:str=""
    forbidden_phrases:list[str]=field(default_factory=list)
    replacement_rules:list[tuple[str,str]]=field(default_factory=list)
    replace_case_insensitive:bool=True
    replace_whole_words_only:bool=False
    strip_boilerplate_prefixes:bool=True
    strip_trailing_period:bool=True

@dataclass
class BlipCaptionConfig:
    model_name:str="BLIP-2 FLAN-T5 XL"
    model_path:str=""
    model_root:str="models/LLM/JLC_BLIPCaption"
    memory_mode:str="Default"
    dtype:str="bf16"
    device:str="auto"
    trust_remote_code:bool=False
    keep_loaded:bool=True
    cache_policy:str=""
    quiet_transformers_load:bool=True
    max_size:int=768
    prompt:str=PROMPT_PRESETS[DEFAULT_PROMPT_PRESET]
    allow_download:bool=True
    use_comfy_model_management:bool=True

@dataclass
class BatchCaptionConfig:
    input_path:str=""
    recursive:bool=True
    filename_glob:str="*"
    extensions:set[str]=field(default_factory=lambda:set(SUPPORTED_EXTENSIONS))
    output_dir:str=""
    write_txt:bool=True
    write_jsonl:bool=False
    jsonl_filename:str="captions.jsonl"
    also_jsonl_path:str=""
    write_run_config:bool=True
    run_config_filename:str=""
    overwrite:bool=False
    backup_existing:bool=True
    dry_run:bool=False
    limit:int=0
    skip_existing_txt:bool=True
    skip_existing_jsonl_images:bool=False

@dataclass
class CaptionRecord:
    image:str
    caption:str
    raw_caption:str
    model_name:str
    model_path:str
    prompt:str
    seed:Optional[int]
    temperature:float
    top_p:float
    top_k:int
    num_beams:int
    max_new_tokens:int
    max_size:int
    timestamp:str
    status:str="ok"
    error:str=""
    captionforge_pass:str="A"
    model_family:str="blip"
    ensemble_run_index:int=0
    image_key:str=""

@dataclass
class BatchCaptionResult:
    records:list[CaptionRecord]=field(default_factory=list)
    skipped:int=0
    failed:int=0
    @property
    def processed(self): return len([r for r in self.records if r.status=="ok"])
    @property
    def captions_text(self): return "\n\n".join(r.caption for r in self.records if r.status=="ok")
    @property
    def jsonl_text(self): return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in self.records)

def timestamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def iso_timestamp(): return datetime.now().isoformat(timespec="seconds")
def safe_mkdir(path:Path): path.mkdir(parents=True, exist_ok=True)
def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_prompt(prompt_preset:str=DEFAULT_PROMPT_PRESET, custom_prompt:str="") -> str:
    custom=(custom_prompt or "").strip()
    return custom if custom else PROMPT_PRESETS.get(prompt_preset, PROMPT_PRESETS[DEFAULT_PROMPT_PRESET])

UNWANTED_PREFIXES=["The image depicts ","The image shows ","This image depicts ","This image shows ","An image of ","A photo of ","A photograph of ","The photo shows ","This photo shows "]
def normalize_caption(caption:str, strip_boilerplate_prefixes=True, strip_trailing_period=True)->str:
    text=str(caption or "").strip()
    if strip_boilerplate_prefixes:
        for p in UNWANTED_PREFIXES:
            if text.lower().startswith(p.lower()): text=text[len(p):].strip(); break
    text=text.replace("\n"," "); text=re.sub(r"\s+"," ",text).strip(); text=re.sub(r"\s+,",",",text); text=re.sub(r",\s*,+",",",text); text=re.sub(r"\s+\.",".",text)
    if strip_trailing_period and text.endswith("."): text=text[:-1].strip()
    return text

def apply_replacements(caption, rules, case_insensitive=True, whole_words_only=False):
    result=caption; flags=re.IGNORECASE if case_insensitive else 0
    for old,new in rules or []:
        old=old.strip()
        if not old: continue
        pat=re.escape(old); pat=(r"\b"+pat+r"\b") if whole_words_only else pat
        result=re.sub(pat, new.strip(), result, flags=flags)
    return result

def remove_forbidden_phrases(caption, forbidden_phrases):
    result=caption
    for phrase in forbidden_phrases or []:
        phrase=phrase.strip()
        if phrase: result=re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)
    result=re.sub(r"\s+,",",",result); result=re.sub(r",\s*,+",",",result); result=re.sub(r"\s+"," ",result)
    return result.strip(" ,")

def add_trigger_prefix_suffix(caption, trigger="", prefix="", suffix=""):
    parts=[]
    for item in (trigger,prefix):
        item=(item or "").strip().strip(" ,")
        if item: parts.append(item)
    if caption: parts.append(caption.lstrip(" ,"))
    final=", ".join(parts); final=re.sub(r",\s*,+",",",final).strip(" ,")
    suffix=(suffix or "").strip()
    return (final+suffix).strip() if suffix else final.strip()

def cleanup_caption(caption, config:CleanupConfig):
    text=normalize_caption(caption, config.strip_boilerplate_prefixes, config.strip_trailing_period)
    text=apply_replacements(text, config.replacement_rules, config.replace_case_insensitive, config.replace_whole_words_only)
    text=remove_forbidden_phrases(text, config.forbidden_phrases)
    text=normalize_caption(text, config.strip_boilerplate_prefixes, config.strip_trailing_period)
    return add_trigger_prefix_suffix(text, config.trigger, config.prefix, config.suffix)

def iter_image_files(input_path, recursive=True, filename_glob="*", extensions=None):
    p=Path(input_path); exts=extensions or set(SUPPORTED_EXTENSIONS); filename_glob=(filename_glob or "*").strip() or "*"
    if p.is_file():
        if p.suffix.lower() in exts and fnmatch.fnmatch(p.name, filename_glob): yield p
        return
    if not p.exists(): raise FileNotFoundError(f"input_path does not exist: {p}")
    if not p.is_dir(): raise NotADirectoryError(f"input_path is not a file or folder: {p}")
    iterator=p.rglob("*") if recursive else p.glob("*")
    for child in sorted(iterator):
        if child.is_file() and child.suffix.lower() in exts and fnmatch.fnmatch(child.name, filename_glob): yield child

def load_image_file(path): return Image.open(path).convert("RGB")
def resize_for_model(image, max_size:int):
    if max_size<=0: return image
    w,h=image.size; longest=max(w,h)
    if longest<=max_size: return image
    scale=max_size/float(longest)
    return image.resize((max(1,round(w*scale)), max(1,round(h*scale))), Image.Resampling.LANCZOS)

def sidecar_txt_path(image_path:Path, output_dir=None):
    if output_dir:
        out=Path(output_dir); safe_mkdir(out); return out/f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")
def backup_existing_file(path:Path, dry_run=False):
    if not path.exists(): return None
    b=path.with_name(f"{path.name}.bak_{timestamp()}")
    if not dry_run: shutil.copy2(path,b)
    return b
def write_text_sidecar(path:Path, text:str, overwrite=False, backup_existing=True, dry_run=False):
    if path.exists() and not overwrite: return False
    if dry_run: return True
    safe_mkdir(path.parent)
    if path.exists() and overwrite and backup_existing: backup_existing_file(path, False)
    path.write_text(text.rstrip()+"\n", encoding="utf-8"); return True

def load_existing_jsonl_images(path:Path):
    seen=set()
    if not path.exists(): return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try: rec=json.loads(line)
            except json.JSONDecodeError: continue
            val=rec.get("image") or rec.get("image_path") or rec.get("source")
            if val:
                seen.add(str(val))
                try: seen.add(str(Path(val)))
                except Exception: pass
    return seen

def record_to_json(record): return asdict(record)
def append_jsonl_records(path:Path, records, dry_run=False):
    if dry_run or not records: return
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(record_to_json(r), ensure_ascii=False)+"\n")
def json_safe(v):
    if isinstance(v,set): return sorted(v)
    if isinstance(v,Path): return str(v)
    if isinstance(v,dict): return {k:json_safe(x) for k,x in v.items()}
    if isinstance(v,list): return [json_safe(x) for x in v]
    if isinstance(v,tuple): return [json_safe(x) for x in v]
    return v
def write_run_config_json(path:Path, config, dry_run=False):
    if dry_run: return
    safe_mkdir(path.parent); path.write_text(json.dumps(json_safe(config), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def model_folder_has_weights(local_path:Path):
    if not local_path.exists() or not local_path.is_dir(): return False
    pats=["*.safetensors","*.bin","*.pt","*.pth","*.gguf","*.ckpt"]
    return any(any(local_path.rglob(p)) for p in pats)
def get_model_info(model_name):
    if model_name not in MODEL_REGISTRY: raise KeyError(f"Unknown BLIP model_name: {model_name}. Known models: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]
def get_registry_model_path(model_name, model_root): return Path(model_root)/get_model_info(model_name).local_folder

def download_registry_model_if_needed(model_name, model_root, metadata_only=False, allow_download=True):
    local_path=get_registry_model_path(model_name, model_root)
    if not metadata_only and model_folder_has_weights(local_path): return local_path
    if not allow_download:
        if metadata_only: safe_mkdir(local_path); return local_path
        raise FileNotFoundError(f"Model folder does not contain weights and allow_download=False: {local_path}")
    try: from huggingface_hub import snapshot_download
    except Exception as exc: raise RuntimeError("Missing dependency: huggingface_hub. Install: pip install huggingface_hub") from exc
    info=get_model_info(model_name); safe_mkdir(Path(model_root)); safe_mkdir(local_path)
    if metadata_only:
        print(f"[JLC BLIP Engine] Probe download for {info.repo_id} -> {local_path}")
        snapshot_download(repo_id=info.repo_id, local_dir=str(local_path), ignore_patterns=["*.safetensors","*.bin","*.pt","*.pth","*.gguf","*.onnx","*.ckpt","*.h5","*.msgpack","*.tflite"])
        return local_path
    print(f"[JLC BLIP Engine] Downloading {info.repo_id} -> {local_path}")
    snapshot_download(repo_id=info.repo_id, local_dir=str(local_path)); return local_path

def probe_registry_model_download(model_name, model_root):
    local_path=download_registry_model_if_needed(model_name, model_root, metadata_only=True, allow_download=True)
    files=[]
    try:
        for p in sorted(local_path.rglob("*")):
            if p.is_file(): files.append(str(p.relative_to(local_path)))
    except Exception: files=[]
    preview="\n".join(files[:40])
    if len(files)>40: preview += f"\n... plus {len(files)-40} more files"
    return f"JLC BLIP Caption download probe completed.\n\nModel: {model_name}\nFolder: {local_path}\n\nLarge model weight files were intentionally skipped.\n\nFiles found:\n{preview if preview else '(no files listed)'}"

def _try_get_comfy_model_management():
    try:
        import comfy.model_management as model_management
        return model_management
    except Exception: return None

def _unload_blip_bundle(bundle):
    model=bundle.get("model") if isinstance(bundle,dict) else None
    if model is not None:
        try: model.to("cpu")
        except Exception: pass
        try: del model
        except Exception: pass
    if isinstance(bundle,dict): bundle.clear()

class BlipCaptionEngine:
    def __init__(self, config:BlipCaptionConfig, generation=None, cleanup=None):
        self.config=config; self.generation=generation or GenerationConfig(); self.cleanup=cleanup or CleanupConfig()
        self.processor=None; self.model=None; self.local_model_path=None; self.model_size_bytes=None
        self._comfy_mm=_try_get_comfy_model_management() if config.use_comfy_model_management else None
        self.inference_device=self._resolve_inference_device(config.device); self.offload_device=self._resolve_offload_device(config.device)
    def resolve_model_path(self):
        if self.config.model_path.strip(): return Path(self.config.model_path).expanduser()
        return download_registry_model_if_needed(self.config.model_name, self.config.model_root, False, self.config.allow_download)
    def _cache_policy(self):
        p=getattr(self.config,"cache_policy",""); p=p.strip() if isinstance(p,str) else ""
        return p or ("evict_other_caption_models" if getattr(self.config,"keep_loaded",True) else "unload_after_run")
    def _cache_key(self, local_path:Path):
        info=get_model_info(self.config.model_name)
        return make_cache_key(role="caption", family=f"blip-{info.model_type}", model_path=str(local_path.resolve()), device=str(self.inference_device), quantization=self.config.memory_mode, dtype=str(self.config.dtype))
    def _resolve_inference_device(self, device):
        if self._comfy_mm is not None:
            try:
                dev=self._comfy_mm.get_torch_device()
                if dev.type!="cuda": raise RuntimeError("CaptionForge BLIP requires CUDA for release builds. Silent CPU fallback is disabled.")
                return dev
            except RuntimeError: raise
            except Exception: pass
        if device=="auto":
            if not torch.cuda.is_available(): raise RuntimeError("CaptionForge BLIP requires CUDA for release builds. Silent CPU fallback is disabled.")
            return torch.device("cuda")
        dev=torch.device(device)
        if dev.type!="cuda": raise RuntimeError(f"CaptionForge BLIP release mode does not allow inference device={device!r}.")
        return dev
    def _resolve_offload_device(self, device):
        if self._comfy_mm is not None:
            try: return self._comfy_mm.unet_offload_device()
            except Exception: pass
        return torch.device("cpu")
    @staticmethod
    def _resolve_dtype(dtype):
        t=dtype.lower().strip()
        if t=="auto": return "auto"
        if t in {"bf16","bfloat16"}: return torch.bfloat16
        if t in {"fp16","float16"}: return torch.float16
        if t in {"fp32","float32"}: return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")
    def _module_size(self, module):
        if self._comfy_mm is not None:
            try: return int(self._comfy_mm.module_size(module))
            except Exception: pass
        total=0
        try:
            for p in module.parameters(): total += p.numel()*p.element_size()
            for b in module.buffers(): total += b.numel()*b.element_size()
        except Exception: total=0
        return total
    def _free_memory(self, bytes_needed, device):
        if bytes_needed is None: return
        if self._comfy_mm is not None:
            try: self._comfy_mm.free_memory(bytes_needed, device); return
            except Exception: pass
        if device.type=="cuda" and torch.cuda.is_available(): torch.cuda.empty_cache()
    def _soft_empty_cache(self):
        if self._comfy_mm is not None:
            try: self._comfy_mm.soft_empty_cache(); return
            except Exception: pass
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    def load(self):
        local_path=self.resolve_model_path()
        if not local_path.exists(): raise FileNotFoundError(f"Model path does not exist: {local_path}")
        self.local_model_path=local_path; key=self._cache_key(local_path); cached=get_cached_model(key)
        if cached is not None:
            self.processor=cached.get("processor"); self.model=cached.get("model"); self.model_size_bytes=cached.get("model_size_bytes")
            print(f"[JLC BLIP Engine] Reusing cached model bundle: {local_path}"); return
        prepare_for_model_load(key, policy=self._cache_policy(), role="caption")
        if self.generation.seed is not None: set_seed(self.generation.seed)
        info=get_model_info(self.config.model_name)
        try:
            if info.model_type=="blip2":
                from transformers import Blip2Processor as ProcessorClass, Blip2ForConditionalGeneration as ModelClass
            elif info.model_type=="blip":
                from transformers import BlipProcessor as ProcessorClass, BlipForConditionalGeneration as ModelClass
            else: raise RuntimeError(f"Unsupported BLIP model_type: {info.model_type}")
        except Exception as exc: raise RuntimeError("Could not import BLIP Transformers classes. Update transformers/accelerate/pillow/huggingface_hub.") from exc
        torch_dtype=self._resolve_dtype(self.config.dtype)
        print(f"[JLC BLIP Engine] Loading processor: {local_path}")
        self.processor=ProcessorClass.from_pretrained(str(local_path), trust_remote_code=self.config.trust_remote_code)
        print(f"[JLC BLIP Engine] Loading model: {local_path} dtype={self.config.dtype} mode={self.config.memory_mode}")
        kwargs={"torch_dtype":torch_dtype, "trust_remote_code":self.config.trust_remote_code}
        if self.config.memory_mode=="Balanced (8-bit)":
            try: from transformers import BitsAndBytesConfig
            except Exception as exc: raise RuntimeError("BLIP Balanced (8-bit) requires bitsandbytes. Use Default or install bitsandbytes.") from exc
            kwargs["quantization_config"]=BitsAndBytesConfig(load_in_8bit=True); kwargs["device_map"]={"": self.inference_device.index or 0}
        elif self.config.memory_mode!="Default": raise ValueError(f"Unsupported memory_mode: {self.config.memory_mode}")
        self.model=ModelClass.from_pretrained(str(local_path), **kwargs); self.model.eval(); self.model_size_bytes=self._module_size(self.model)
        if self.config.memory_mode=="Default": self._free_memory(self.model_size_bytes, self.inference_device); self.model.to(self.inference_device)
        register_model(key, {"processor":self.processor,"model":self.model,"model_size_bytes":self.model_size_bytes,"local_path":str(local_path)}, family="blip", model_path=str(local_path), device=str(self.inference_device), quantization=self.config.memory_mode, role="caption", unload_fn=_unload_blip_bundle, keep=self._cache_policy()=="keep_this_model")
        print(f"[JLC BLIP Engine] Loaded model: {self.config.model_name}")
    def prepare_for_inference(self):
        if self.processor is None or self.model is None: self.load()
        if self.model is None: raise RuntimeError("BLIP model failed to load.")
        if self.config.memory_mode=="Default": self._free_memory(self.model_size_bytes, self.inference_device); self.model.to(self.inference_device)
    def cleanup_after_inference(self):
        if self._cache_policy()!="unload_after_run": return
        if self.model is not None and self.config.memory_mode=="Default": self.model.to(self.offload_device); self._soft_empty_cache()
    def unload(self):
        if self.local_model_path is not None: unload_after_run(self._cache_key(self.local_model_path), enabled=True)
        self.processor=None; self.model=None; self.model_size_bytes=None; self._soft_empty_cache()
    @torch.inference_mode()
    def caption_pil(self, image:Image.Image):
        if self.generation.seed is not None: set_seed(self.generation.seed)
        self.prepare_for_inference()
        if self.model is None or self.processor is None: raise RuntimeError("Model is not loaded.")
        image_for_model=resize_for_model(image.convert("RGB"), self.config.max_size); prompt=(self.config.prompt or "").strip()
        inputs=self.processor(images=image_for_model, text=prompt, return_tensors="pt") if prompt else self.processor(images=image_for_model, return_tensors="pt")
        inputs={k:(v.to(self.inference_device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        model_dtype=getattr(self.model,"dtype",None)
        if isinstance(model_dtype, torch.dtype) and "pixel_values" in inputs and torch.is_floating_point(inputs["pixel_values"]): inputs["pixel_values"]=inputs["pixel_values"].to(dtype=model_dtype)
        gen={"max_new_tokens":int(self.generation.max_new_tokens), "num_beams":max(1,int(self.generation.num_beams)), "do_sample":bool(self.generation.temperature>0), "use_cache":True}
        if self.generation.temperature>0: gen.update({"temperature":float(self.generation.temperature),"top_p":float(self.generation.top_p),"top_k":int(self.generation.top_k)})
        if self.generation.repetition_penalty and self.generation.repetition_penalty!=1.0: gen["repetition_penalty"]=float(self.generation.repetition_penalty)
        try: ids=self.model.generate(**inputs, **gen)
        finally: self.cleanup_after_inference()
        raw=self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return cleanup_caption(raw,self.cleanup), raw
    def caption_path(self, image_path):
        p=Path(image_path)
        if not p.exists(): raise FileNotFoundError(f"Image does not exist: {p}")
        final,raw=self.caption_pil(load_image_file(p))
        return CaptionRecord(str(p), final, raw, self.config.model_name, str(self.local_model_path or self.config.model_path), self.config.prompt, self.generation.seed, self.generation.temperature, self.generation.top_p, self.generation.top_k, self.generation.num_beams, self.generation.max_new_tokens, self.config.max_size, iso_timestamp(), captionforge_pass="A", model_family="blip", ensemble_run_index=0, image_key=str(p.resolve()))
    def caption_batch(self, batch:BatchCaptionConfig):
        if not batch.input_path.strip(): raise ValueError("BatchCaptionConfig.input_path is required.")
        result=BatchCaptionResult(); images=list(iter_image_files(batch.input_path,batch.recursive,batch.filename_glob,batch.extensions))
        if batch.limit and batch.limit>0: images=images[:int(batch.limit)]
        if not images: print(f"[JLC BLIP Engine] No images found in: {batch.input_path}"); return result
        output_dir=Path(batch.output_dir) if batch.output_dir.strip() else None; jsonl_path=None
        if batch.write_jsonl: jsonl_path=(output_dir or images[0].parent)/(batch.jsonl_filename.strip() or "captions.jsonl")
        also_jsonl_path=Path(batch.also_jsonl_path) if batch.also_jsonl_path.strip() else None
        seen=set()
        if batch.skip_existing_jsonl_images:
            if jsonl_path: seen.update(load_existing_jsonl_images(jsonl_path))
            if also_jsonl_path: seen.update(load_existing_jsonl_images(also_jsonl_path))
        if batch.write_run_config:
            base=output_dir or images[0].parent; name=batch.run_config_filename.strip() or f"jlc_blip_caption_run_config_{timestamp()}.json"; write_run_config_json(base/name, self.build_run_config(batch), batch.dry_run)
        print(f"[JLC BLIP Engine] Found {len(images)} image(s).")
        try:
            for idx,path in enumerate(images,1):
                try:
                    txt=sidecar_txt_path(path, output_dir); src=str(path)
                    if batch.skip_existing_txt and batch.write_txt and txt.exists() and not batch.overwrite: print(f"[{idx}/{len(images)}] SKIP existing TXT: {txt}"); result.skipped+=1; continue
                    if batch.skip_existing_jsonl_images and (src in seen or path.name in seen): print(f"[{idx}/{len(images)}] SKIP existing JSONL image: {src}"); result.skipped+=1; continue
                    print(f"[{idx}/{len(images)}] Captioning: {path}"); rec=self.caption_path(path); result.records.append(rec)
                    if batch.write_txt and not write_text_sidecar(txt, rec.caption, batch.overwrite, batch.backup_existing, batch.dry_run): result.skipped+=1; print(f"[{idx}/{len(images)}] SKIP existing TXT after caption: {txt}")
                    if jsonl_path: append_jsonl_records(jsonl_path,[rec],batch.dry_run)
                    if also_jsonl_path: append_jsonl_records(also_jsonl_path,[rec],batch.dry_run)
                except KeyboardInterrupt: raise
                except Exception as exc:
                    result.failed+=1; print(f"[JLC BLIP Engine] ERROR on {path}: {exc}")
                    err=CaptionRecord(str(path),"","",self.config.model_name,str(self.local_model_path or self.config.model_path),self.config.prompt,self.generation.seed,self.generation.temperature,self.generation.top_p,self.generation.top_k,self.generation.num_beams,self.generation.max_new_tokens,self.config.max_size,iso_timestamp(),status="error",error=str(exc),captionforge_pass="A",model_family="blip",ensemble_run_index=0,image_key=str(path.resolve()))
                    result.records.append(err)
                    if jsonl_path: append_jsonl_records(jsonl_path,[err],batch.dry_run)
                    if also_jsonl_path: append_jsonl_records(also_jsonl_path,[err],batch.dry_run)
        finally:
            if self._cache_policy()=="unload_after_run": self.unload()
        return result
    def build_run_config(self,batch=None):
        return {"timestamp":iso_timestamp(),"engine":"JLC BLIP Caption Engine","blip_config":asdict(self.config),"generation":asdict(self.generation),"cleanup":{**asdict(self.cleanup),"replacement_rules":[list(r) for r in self.cleanup.replacement_rules]},"batch":json_safe(asdict(batch)) if batch is not None else None}

def caption_one_image(image_path, blip_config, generation=None, cleanup=None):
    engine=BlipCaptionEngine(blip_config,generation,cleanup); engine.load()
    try: return engine.caption_path(image_path)
    finally:
        p=getattr(blip_config,"cache_policy",""); p=p.strip() if isinstance(p,str) else ""
        if not p: p="evict_other_caption_models" if getattr(blip_config,"keep_loaded",True) else "unload_after_run"
        if p=="unload_after_run": engine.unload()
