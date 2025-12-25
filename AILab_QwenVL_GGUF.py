# ComfyUI-QwenVL (GGUF) - 支持多图输入和本地文件选择

import base64
import gc
import io
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"


def _load_prompt_config():
    preset_prompts = ["🖼️ Detailed Description"]
    system_prompts: dict[str, str] = {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        preset_prompts = data.get("_preset_prompts") or preset_prompts
        system_prompts = data.get("_system_prompts") or system_prompts
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")

    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            system_prompts = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            preset_prompts = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    return preset_prompts, system_prompts


PRESET_PROMPTS, SYSTEM_PROMPTS = _load_prompt_config()


@dataclass(frozen=True)
class GGUFVLResolved:
    display_name: str
    repo_id: str | None
    alt_repo_ids: list[str]
    author: str | None
    repo_dirname: str
    model_filename: str
    mmproj_filename: str | None
    context_length: int
    image_max_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _resolve_base_dir(base_dir_value: str) -> Path:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    return Path(folder_paths.models_dir) / base_dir


def _safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def _model_name_to_filename_candidates(model_name: str) -> set[str]:
    raw = (model_name or "").strip()
    if not raw:
        return set()
    candidates = {raw, f"{raw}.gguf"}
    if " / " in raw:
        tail = raw.split(" / ", 1)[1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    if "/" in raw:
        tail = raw.rsplit("/", 1)[-1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    return candidates


def _load_gguf_vl_catalog():
    """加载GGUF模型配置"""
    if not GGUF_CONFIG_PATH.exists():
        return {"base_dir": "LLM/GGUF", "models": {}}
    try:
        with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception as exc:
        print(f"[QwenVL] gguf_models.json load failed: {exc}")
        return {"base_dir": "LLM/GGUF", "models": {}}

    base_dir = data.get("base_dir") or "LLM/GGUF"

    flattened: dict[str, dict] = {}

    repos = data.get("qwenVL_model") or data.get("vl_repos") or data.get("repos") or {}
    seen_display_names: set[str] = set()
    for repo_key, repo in repos.items():
        if not isinstance(repo, dict):
            continue
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
        repo_id = repo.get("repo_id") or (f"{author}/{repo_name}" if author and repo_name else None)
        alt_repo_ids = repo.get("alt_repo_ids") or []

        defaults = repo.get("defaults") or {}
        mmproj_file = repo.get("mmproj_file")
        model_files = repo.get("model_files") or []

        for model_file in model_files:
            display = Path(model_file).name
            if display in seen_display_names:
                display = f"{display} ({repo_key})"
            seen_display_names.add(display)
            flattened[display] = {
                **defaults,
                "author": author,
                "repo_dirname": repo_name,
                "repo_id": repo_id,
                "alt_repo_ids": alt_repo_ids,
                "filename": model_file,
                "mmproj_filename": mmproj_file,
            }

    legacy_models = data.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            flattened[name] = entry

    return {"base_dir": base_dir, "models": flattened}


GGUF_VL_CATALOG = _load_gguf_vl_catalog()


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _tensor_to_base64_png(tensor) -> str | None:
    """将张量转换为base64 PNG图像"""
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    pil_img = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _sample_video_frames(video, frame_count: int):
    """采样视频帧"""
    if video is None:
        return []
    if video.ndim != 4:
        return [video]
    total = int(video.shape[0])
    frame_count = max(int(frame_count), 1)
    if total <= frame_count:
        return [video[i] for i in range(total)]
    idx = np.linspace(0, total - 1, frame_count, dtype=int)
    return [video[i] for i in idx]


def _pick_device(device_choice: str) -> str:
    """选择设备"""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_single_file(repo_ids: list[str], filename: str, target_path: Path):
    """下载单个文件"""
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for repo_id in repo_ids:
        print(f"[QwenVL] Downloading {filename} from {repo_id} -> {target_path}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(target_path.parent),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL] hf_hub_download failed from {repo_id}: {exc}")
    else:
        raise FileNotFoundError(f"[QwenVL] Download failed for {filename}: {last_exc}")

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL] File not found after download: {target_path}")


def _resolve_model_entry(model_name: str) -> GGUFVLResolved:
    """解析模型条目"""
    all_models = GGUF_VL_CATALOG.get("models") or {}
    entry = all_models.get(model_name) or {}
    if not entry:
        wanted = _model_name_to_filename_candidates(model_name)
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break

    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []

    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else model_name)

    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")

    if not model_filename:
        raise ValueError(f"[QwenVL] gguf_vl_models.json entry missing 'filename' for: {model_name}")

    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    return GGUFVLResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=_safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 8192),
        image_max_tokens=_int("image_max_tokens", 4096),
        n_batch=_int("n_batch", 512),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
    )


def _get_local_gguf_files():
    """获取本地GGUF文件列表"""
    base_dir = _resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "llm/GGUF")
    gguf_files = []
    
    if base_dir.exists():
        # 递归查找所有.gguf文件
        for file_path in base_dir.rglob("*.gguf"):
            # 计算相对路径，便于显示
            try:
                rel_path = file_path.relative_to(base_dir)
                display_name = f"本地: {rel_path}"
                gguf_files.append((str(file_path), display_name))
            except ValueError:
                gguf_files.append((str(file_path), f"本地: {file_path.name}"))
    
    # 按文件名排序
    gguf_files.sort(key=lambda x: x[1])
    return gguf_files


def _get_local_mmproj_files():
    """获取本地mmproj文件列表"""
    base_dir = _resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "llm/GGUF")
    mmproj_files = [("无", "无 mmproj 文件")]
    
    if base_dir.exists():
        # 查找所有常见的mmproj文件扩展名
        mmproj_extensions = ['.mmproj', '.gguf', '.bin', '.safetensors']
        
        for file_path in base_dir.rglob("*"):
            if file_path.suffix.lower() in mmproj_extensions:
                # 检查文件名是否包含mmproj相关关键词
                filename_lower = file_path.name.lower()
                if any(keyword in filename_lower for keyword in ['mmproj', 'vision', 'clip', 'visual']):
                    try:
                        rel_path = file_path.relative_to(base_dir)
                        display_name = f"本地: {rel_path}"
                        mmproj_files.append((str(file_path), display_name))
                    except ValueError:
                        mmproj_files.append((str(file_path), f"本地: {file_path.name}"))
    
    # 按文件名排序
    mmproj_files.sort(key=lambda x: x[1])
    return mmproj_files


class QwenVLGGUFBase:
    """QwenVL GGUF基础类 - 支持多图输入和本地文件选择"""
    
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None

    def clear(self):
        """清理模型资源"""
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_backend(self):
        """加载后端库"""
        try:
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp is not available. Install the GGUF vision dependency first. See docs/GGUF_MANUAL_INSTALL.md"
            ) from exc

    def _load_model(
        self,
        model_source: str,  # 改为模型来源：可以是配置名称或本地路径
        mmproj_source: str,  # mmproj文件来源
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        is_local_file: bool = False,  # 是否使用本地文件
    ):
        """加载模型 - 支持配置模型和本地文件"""
        self._load_backend()

        # 判断模型来源类型
        if is_local_file:
            # 使用本地文件
            model_path = Path(model_source)
            mmproj_path = Path(mmproj_source) if mmproj_source and mmproj_source != "无" else None
            
            if not model_path.exists():
                raise FileNotFoundError(f"[QwenVL] 本地模型文件不存在: {model_path}")
            
            if mmproj_path and not mmproj_path.exists():
                print(f"[QwenVL] 警告: mmproj文件不存在: {mmproj_path}，将不使用视觉功能")
                mmproj_path = None
                
            # 使用默认配置值
            resolved = GGUFVLResolved(
                display_name=model_path.name,
                repo_id=None,
                alt_repo_ids=[],
                author=None,
                repo_dirname=model_path.parent.name,
                model_filename=model_path.name,
                mmproj_filename=mmproj_path.name if mmproj_path else None,
                context_length=8192,
                image_max_tokens=4096,
                n_batch=512,
                gpu_layers=-1,
                top_k=0,
                pool_size=4194304,
            )
        else:
            # 使用配置中的模型
            resolved = _resolve_model_entry(model_source)
            base_dir = _resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "llm/GGUF")

            author_dir = _safe_dirname(resolved.author or "")
            repo_dir = _safe_dirname(resolved.repo_dirname)
            target_dir = base_dir / author_dir / repo_dir

            model_path = target_dir / Path(resolved.model_filename).name
            mmproj_path = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None

            repo_ids: list[str] = []
            if resolved.repo_id:
                repo_ids.append(resolved.repo_id)
            repo_ids.extend(resolved.alt_repo_ids)

            if not model_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] GGUF model not found locally and no repo_id provided: {model_path}")
                _download_single_file(repo_ids, resolved.model_filename, model_path)

            if mmproj_path is not None and not mmproj_path.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] mmproj not found locally and no repo_id provided: {mmproj_path}")
                _download_single_file(repo_ids, resolved.mmproj_filename, mmproj_path)

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens

        has_mmproj = mmproj_path is not None and mmproj_path.exists()

        signature = (
            str(model_path),
            str(mmproj_path) if has_mmproj else "",
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            top_k_val,
            pool_size_val,
        )
        if self.llm is not None and self.current_signature == signature:
            return

        self.clear()

        from llama_cpp import Llama

        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler

                handler_cls = Qwen3VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                    handler_cls = Qwen25VLChatHandler
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Missing Qwen VL chat handler in llama_cpp. Install the correct fork/wheel. See docs/GGUF_MANUAL_INSTALL.md"
                    )

            mmproj_kwargs = {
                "clip_model_path": str(mmproj_path),
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "verbose": False,
            }
            mmproj_kwargs = _filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = img_max

        print(f"[QwenVL] Loading GGUF: {model_path.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx})")
        llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")
        self.llm = Llama(**llm_kwargs_filtered)
        self.current_signature = signature

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> str:
        """调用模型生成"""
        if images_b64:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if not img:
                    continue
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        start = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
            stop=["<|im_end|>", "<|im_start|>"],
        )
        elapsed = max(time.perf_counter() - start, 1e-6)

        usage = result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            tok_s = completion_tokens / elapsed
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                print(
                    f"[QwenVL] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s"
                )
            else:
                print(f"[QwenVL] Tokens: completion={completion_tokens}, time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")

        content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        cleaned = clean_model_output(str(content or ""), OutputCleanConfig(mode="text"))
        return cleaned.strip()

    def run(
        self,
        model_source: str,  # 模型来源：配置名称或本地路径
        mmproj_source: str,  # mmproj文件来源
        use_local_files: bool,  # 是否使用本地文件
        preset_prompt: str,
        custom_prompt: str,
        images: list,  # 图像列表
        video,
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
    ):
        """运行模型生成"""
        torch.manual_seed(int(seed))

        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()

        images_b64: list[str] = []
        
        # 处理多个图像输入
        if images:
            for image_tensor in images:
                if image_tensor is not None:
                    img = _tensor_to_base64_png(image_tensor)
                    if img:
                        images_b64.append(img)
                        
        # 处理视频输入
        if video is not None:
            for frame in _sample_video_frames(video, int(frame_count)):
                img = _tensor_to_base64_png(frame)
                if img:
                    images_b64.append(img)

        try:
            self._load_model(
                model_source=model_source,
                mmproj_source=mmproj_source,
                device=device,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                top_k=top_k,
                pool_size=pool_size,
                is_local_file=use_local_files,
            )
            if images_b64 and self.chat_handler is None:
                print("[QwenVL] Warning: images provided but this model entry has no mmproj_file; images will be ignored")
            
            # 打印图像信息
            if images_b64:
                print(f"[QwenVL] 输入 {len(images_b64)} 张图像")
            
            text = self._invoke(
                system_prompt=(
                    "You are a helpful vision-language assistant. "
                    "Answer directly with the final answer only. No <think> and no reasoning."
                ),
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    """基础版GGUF节点 - 支持多图输入和本地文件选择"""
    
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(edit gguf_models.json)"]
        default_model = model_keys[0] if model_keys else ""

        # 获取本地文件
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        return {
            "required": {
                "使用本地文件": ("BOOLEAN", {"default": False, "tooltip": "启用后使用本地GGUF文件，否则使用配置中的模型"}),
                "模型选择方式": (["从配置选择", "本地文件"], {"default": "从配置选择", "tooltip": "选择模型加载方式"}),
                "model_name": (model_keys, {"default": default_model, "tooltip": "从配置中选择模型"}),
                "本地模型文件": (["无"] + [display for _, display in local_gguf_files], {"default": "无", "tooltip": "选择本地GGUF文件"}),
                "本地mmproj文件": (["无"] + [display for _, display in local_mmproj_files], {"default": "无", "tooltip": "选择本地mmproj文件（视觉模型需要）"}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                # 支持最多4张图像输入
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        使用本地文件,
        模型选择方式,
        model_name,
        本地模型文件,
        本地mmproj文件,
        preset_prompt,
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        video=None,
    ):
        # 收集所有图像输入
        images = [image_1, image_2, image_3, image_4]
        images = [img for img in images if img is not None]  # 过滤掉None值
        
        # 确定模型加载方式
        use_local = 使用本地文件 or (模型选择方式 == "本地文件")
        
        # 获取实际文件路径
        if use_local:
            model_source = "无"
            mmproj_source = "无"
            
            # 查找模型文件路径
            local_gguf_files = _get_local_gguf_files()
            for file_path, display_name in local_gguf_files:
                if display_name == 本地模型文件:
                    model_source = file_path
                    break
            
            # 查找mmproj文件路径
            if 本地mmproj文件 != "无":
                local_mmproj_files = _get_local_mmproj_files()
                for file_path, display_name in local_mmproj_files:
                    if display_name == 本地mmproj文件:
                        mmproj_source = file_path
                        break
            else:
                mmproj_source = "无"
                
            if model_source == "无":
                raise ValueError("未选择有效的本地模型文件")
        else:
            model_source = model_name
            mmproj_source = "无"  # 配置中的模型会自动处理mmproj
        
        return self.run(
            model_source=model_source,
            mmproj_source=mmproj_source,
            use_local_files=use_local,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            images=images,
            video=video,
            frame_count=16,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device="auto",
            ctx=None,
            n_batch=None,
            gpu_layers=None,
            image_max_tokens=None,
            top_k=None,
            pool_size=None,
        )


class AILab_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
    """高级版GGUF节点 - 支持多图输入和本地文件选择"""
    
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(edit gguf_models.json)"]
        default_model = model_keys[0] if model_keys else ""

        # 获取本地文件
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        prompts = PRESET_PROMPTS or ["🖼️ Detailed Description"]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "使用本地文件": ("BOOLEAN", {"default": False, "tooltip": "启用后使用本地GGUF文件，否则使用配置中的模型"}),
                "模型选择方式": (["从配置选择", "本地文件"], {"default": "从配置选择", "tooltip": "选择模型加载方式"}),
                "model_name": (model_keys, {"default": default_model, "tooltip": "从配置中选择模型"}),
                "本地模型文件": (["无"] + [display for _, display in local_gguf_files], {"default": "无", "tooltip": "选择本地GGUF文件"}),
                "本地mmproj文件": (["无"] + [display for _, display in local_mmproj_files], {"default": "无", "tooltip": "选择本地mmproj文件（视觉模型需要）"}),
                "device": (device_options, {"default": "auto"}),
                "preset_prompt": (prompts, {"default": default_prompt}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "ctx": ("INT", {"default": 8192, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 256, "max": 1024000, "step": 256}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                # 支持最多6张图像输入
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        使用本地文件,
        模型选择方式,
        model_name,
        本地模型文件,
        本地mmproj文件,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        top_k,
        pool_size,
        keep_model_loaded,
        seed,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        video=None,
    ):
        # 收集所有图像输入
        images = [image_1, image_2, image_3, image_4, image_5, image_6]
        images = [img for img in images if img is not None]  # 过滤掉None值
        
        # 确定模型加载方式
        use_local = 使用本地文件 or (模型选择方式 == "本地文件")
        
        # 获取实际文件路径
        if use_local:
            model_source = "无"
            mmproj_source = "无"
            
            # 查找模型文件路径
            local_gguf_files = _get_local_gguf_files()
            for file_path, display_name in local_gguf_files:
                if display_name == 本地模型文件:
                    model_source = file_path
                    break
            
            # 查找mmproj文件路径
            if 本地mmproj文件 != "无":
                local_mmproj_files = _get_local_mmproj_files()
                for file_path, display_name in local_mmproj_files:
                    if display_name == 本地mmproj文件:
                        mmproj_source = file_path
                        break
            else:
                mmproj_source = "无"
                
            if model_source == "无":
                raise ValueError("未选择有效的本地模型文件")
        else:
            model_source = model_name
            mmproj_source = "无"  # 配置中的模型会自动处理mmproj
        
        return self.run(
            model_source=model_source,
            mmproj_source=mmproj_source,
            use_local_files=use_local,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            images=images,
            video=video,
            frame_count=frame_count,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            top_k=top_k,
            pool_size=pool_size,
        )


class AILab_QwenVL_GGUF_MultiImageChat(QwenVLGGUFBase):
    """多图对话节点 - 专门用于多图像对话场景"""
    
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(edit gguf_models.json)"]
        default_model = model_keys[0] if model_keys else ""

        # 获取本地文件
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        # 多图对话专用提示词
        multi_image_prompts = [
            "比较这些图片的相似之处和差异",
            "描述每张图片的内容",
            "这些图片讲述了一个什么故事？",
            "分析这些图片的共同主题",
            "为这些图片创作一个连贯的描述",
            "从这些图片中提取关键信息",
            "这些图片反映了什么趋势或概念？",
            "为每张图片生成简短的标题",
        ]

        return {
            "required": {
                "使用本地文件": ("BOOLEAN", {"default": False, "tooltip": "启用后使用本地GGUF文件，否则使用配置中的模型"}),
                "模型选择方式": (["从配置选择", "本地文件"], {"default": "从配置选择", "tooltip": "选择模型加载方式"}),
                "model_name": (model_keys, {"default": default_model, "tooltip": "从配置中选择模型"}),
                "本地模型文件": (["无"] + [display for _, display in local_gguf_files], {"default": "无", "tooltip": "选择本地GGUF文件"}),
                "本地mmproj文件": (["无"] + [display for _, display in local_mmproj_files], {"default": "无", "tooltip": "选择本地mmproj文件（视觉模型需要）"}),
                "prompt_type": (["预设", "自定义"], {"default": "预设"}),
                "preset_prompt": (multi_image_prompts, {"default": multi_image_prompts[0]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "输入多图分析的提示词"}),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            },
            "optional": {
                # 支持最多8张图像输入
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("多图分析结果",)
    FUNCTION = "process_multi_image"
    CATEGORY = "🧪AILab/QwenVL"

    def process_multi_image(
        self,
        使用本地文件,
        模型选择方式,
        model_name,
        本地模型文件,
        本地mmproj文件,
        prompt_type,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        seed,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
    ):
        # 收集所有图像输入
        images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
        images = [img for img in images if img is not None]  # 过滤掉None值
        
        # 确定使用的提示词
        if prompt_type == "自定义" and custom_prompt.strip():
            prompt = custom_prompt.strip()
        else:
            prompt = preset_prompt
        
        # 确定模型加载方式
        use_local = 使用本地文件 or (模型选择方式 == "本地文件")
        
        # 获取实际文件路径
        if use_local:
            model_source = "无"
            mmproj_source = "无"
            
            # 查找模型文件路径
            local_gguf_files = _get_local_gguf_files()
            for file_path, display_name in local_gguf_files:
                if display_name == 本地模型文件:
                    model_source = file_path
                    break
            
            # 查找mmproj文件路径
            if 本地mmproj文件 != "无":
                local_mmproj_files = _get_local_mmproj_files()
                for file_path, display_name in local_mmproj_files:
                    if display_name == 本地mmproj文件:
                        mmproj_source = file_path
                        break
            else:
                mmproj_source = "无"
                
            if model_source == "无":
                raise ValueError("未选择有效的本地模型文件")
        else:
            model_source = model_name
            mmproj_source = "无"  # 配置中的模型会自动处理mmproj
        
        print(f"[QwenVL] 多图分析: 输入 {len(images)} 张图像")
        if images:
            print(f"[QwenVL] 图像尺寸: {images[0].shape}")
        
        try:
            self._load_model(
                model_source=model_source,
                mmproj_source=mmproj_source,
                device="auto",
                ctx=None,
                n_batch=None,
                gpu_layers=None,
                image_max_tokens=None,
                top_k=None,
                pool_size=None,
                is_local_file=use_local,
            )
            
            # 转换图像为base64
            images_b64: list[str] = []
            for img_tensor in images:
                if img_tensor is not None:
                    img_b64 = _tensor_to_base64_png(img_tensor)
                    if img_b64:
                        images_b64.append(img_b64)
            
            # 生成多图分析
            text = self._invoke(
                system_prompt=(
                    "You are a helpful vision-language assistant specialized in multi-image analysis. "
                    "You can compare, contrast, and synthesize information from multiple images. "
                    "Provide detailed and structured analysis."
                ),
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                seed=seed,
            )
            return (text,)
        finally:
            # 默认清理模型以节省内存
            self.clear()


class AILab_QwenVL_GGUF_FileSelector:
    """GGUF文件选择器节点 - 专门用于浏览和选择本地文件"""
    
    @classmethod
    def INPUT_TYPES(cls):
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        return {
            "required": {
                "刷新文件列表": ("BOOLEAN", {"default": False, "tooltip": "刷新本地文件列表"}),
            },
            "optional": {
                "自定义搜索路径": ("STRING", {"default": "", "multiline": False, "placeholder": "输入自定义搜索路径（可选）"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("模型文件", "mmproj文件", "模型路径", "mmproj路径")
    FUNCTION = "browse_files"
    CATEGORY = "🧪AILab/QwenVL"
    
    def browse_files(self, 刷新文件列表, 自定义搜索路径=None):
        """浏览本地文件"""
        # 如果提供了自定义路径，添加到搜索路径
        if 自定义搜索路径 and 自定义搜索路径.strip():
            custom_path = Path(自定义搜索路径.strip())
            if custom_path.exists():
                print(f"[QwenVL] 使用自定义搜索路径: {custom_path}")
        
        # 刷新文件列表
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        # 返回第一个文件作为示例
        model_file_display = "无"
        mmproj_file_display = "无"
        model_file_path = ""
        mmproj_file_path = ""
        
        if local_gguf_files:
            model_file_path, model_file_display = local_gguf_files[0]
            print(f"[QwenVL] 找到 {len(local_gguf_files)} 个GGUF文件")
        
        if len(local_mmproj_files) > 1:  # 跳过第一个"无"选项
            mmproj_file_path, mmproj_file_display = local_mmproj_files[1]
        
        return (model_file_display, mmproj_file_display, model_file_path, mmproj_file_path)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_GGUF": AILab_QwenVL_GGUF,
    "AILab_QwenVL_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
    "AILab_QwenVL_GGUF_MultiImageChat": AILab_QwenVL_GGUF_MultiImageChat,
    "AILab_QwenVL_GGUF_FileSelector": AILab_QwenVL_GGUF_FileSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_GGUF": "QwenVL (GGUF) - 多图输入+本地文件",
    "AILab_QwenVL_GGUF_Advanced": "QwenVL Advanced (GGUF) - 多图输入+本地文件",
    "AILab_QwenVL_GGUF_MultiImageChat": "QwenVL 多图对话 (GGUF) - 本地文件",
    "AILab_QwenVL_GGUF_FileSelector": "QwenVL 文件选择器 (GGUF)",
}
