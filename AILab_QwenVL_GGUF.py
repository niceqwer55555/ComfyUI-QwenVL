# ComfyUI-QwenVL (GGUF) - 优化版：支持Qwen2.5-VL和多图分析

import base64
import gc
import io
import inspect
import json
import os
import time
import random
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
    model_family: str = "unknown"


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


def _detect_model_family_from_filename(filename: str) -> str:
    """从文件名检测模型家族"""
    filename_lower = filename.lower()
    
    # 精确匹配
    if "qwen2.5" in filename_lower or "qwen2_5" in filename_lower:
        return "qwen2.5-vl"
    elif "qwen3" in filename_lower:
        return "qwen3-vl"
    elif "qwen2" in filename_lower:
        return "qwen2-vl"
    elif "qwen-vl" in filename_lower:
        return "qwen-vl"
    elif "llava-1.6" in filename_lower:
        return "llava-1.6"
    elif "llava-1.5" in filename_lower:
        return "llava-1.5"
    elif "llava" in filename_lower:
        return "llava"
    elif "bakllava" in filename_lower:
        return "bakllava"
    elif "minicpm" in filename_lower:
        return "minicpm"
    elif "cogvlm" in filename_lower:
        return "cogvlm"
    elif "yi-vl" in filename_lower:
        return "yi-vl"
    elif "deepseek-vl" in filename_lower:
        return "deepseek-vl"
    
    # 模糊匹配
    if "qwen" in filename_lower and "vl" in filename_lower:
        return "qwen-vl"
    elif "vision" in filename_lower or "visual" in filename_lower:
        return "generic-vl"
    
    return "unknown"


def _detect_model_family_from_path(model_path: Path) -> str:
    """从模型文件路径更精确地检测模型家族"""
    if not model_path.exists():
        return "unknown"
    
    filename = model_path.name.lower()
    family = _detect_model_family_from_filename(filename)
    
    if family == "unknown":
        # 尝试读取文件头信息
        try:
            with open(model_path, 'rb') as f:
                # 读取前512字节
                header = f.read(512)
                header_str = header.decode('ascii', errors='ignore')
                
                # 在文件头中搜索关键词
                if 'qwen' in header_str.lower():
                    if '2.5' in header_str:
                        return "qwen2.5-vl"
                    elif '3' in header_str:
                        return "qwen3-vl"
                    elif '2' in header_str:
                        return "qwen2-vl"
                    else:
                        return "qwen-vl"
                elif 'llava' in header_str.lower():
                    if '1.6' in header_str:
                        return "llava-1.6"
                    elif '1.5' in header_str:
                        return "llava-1.5"
                    else:
                        return "llava"
                elif 'yi' in header_str.lower() and 'vl' in header_str.lower():
                    return "yi-vl"
                elif 'deepseek' in header_str.lower() and 'vl' in header_str.lower():
                    return "deepseek-vl"
        except:
            pass
    
    return family


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
        model_family = repo.get("model_family") or "unknown"

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
                "model_family": model_family,
            }

    legacy_models = data.get("models") or {}
    for name, entry in legacy_models.items():
        if isinstance(entry, dict):
            if "model_family" not in entry:
                # 根据文件名猜测模型家族
                filename = entry.get("filename", "")
                entry["model_family"] = _detect_model_family_from_filename(filename)
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


def _tensor_to_base64_png(tensor, resize_to=(448, 448)) -> str | None:
    """将张量转换为base64 PNG图像，可调整大小"""
    if tensor is None:
        return None
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # 转换为numpy数组
    array = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    # 创建PIL图像
    pil_img = Image.fromarray(array, mode="RGB")
    
    # 调整大小（保持宽高比）
    if resize_to:
        original_width, original_height = pil_img.size
        target_width, target_height = resize_to
        
        # 计算缩放比例
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        if new_width != original_width or new_height != original_height:
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"[QwenVL] 图像已调整大小: {original_width}x{original_height} -> {new_width}x{new_height}")
    
    # 转换为base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
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
    model_family = entry.get("model_family") or _detect_model_family_from_filename(model_filename or "")

    if not model_filename:
        raise ValueError(f"[QwenVL] gguf_vl_models.json entry missing 'filename' for: {model_name}")

    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    # 根据模型家族设置默认值
    if model_family == "qwen2.5-vl":
        default_ctx = 8192
        default_img_max = 4096
        default_n_batch = 512
        default_gpu_layers = -1
    elif model_family == "qwen3-vl":
        default_ctx = 8192
        default_img_max = 4096
        default_n_batch = 512
        default_gpu_layers = -1
    elif "llava" in model_family:
        default_ctx = 4096
        default_img_max = 2048
        default_n_batch = 256
        default_gpu_layers = -1
    else:
        default_ctx = 8192
        default_img_max = 4096
        default_n_batch = 512
        default_gpu_layers = -1

    return GGUFVLResolved(
        display_name=model_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=_safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", default_ctx),
        image_max_tokens=_int("image_max_tokens", default_img_max),
        n_batch=_int("n_batch", default_n_batch),
        gpu_layers=_int("gpu_layers", default_gpu_layers),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
        model_family=model_family,
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
        self.model_family = "unknown"
        self.handler_name = ""

    def clear(self):
        """清理模型资源"""
        if self.llm is not None:
            try:
                # 尝试清理llama.cpp内部资源
                if hasattr(self.llm, '_ctx'):
                    self.llm._ctx = None
                if hasattr(self.llm, '_model'):
                    self.llm._model = None
            except:
                pass
        
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self.model_family = "unknown"
        self.handler_name = ""
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"[QwenVL] 模型资源已清理")

    def _load_backend(self):
        """加载后端库"""
        try:
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp is not available. Install the GGUF vision dependency first. See docs/GGUF_MANUAL_INSTALL.md"
            ) from exc

    def _create_chat_handler(self, handler_cls, mmproj_path, img_max, model_family):
        """根据处理器类创建相应的处理器实例"""
        self.handler_name = handler_cls.__name__
        
        # 基础参数
        base_params = {
            "clip_model_path": str(mmproj_path),
            "verbose": False,
        }
        
        # 根据处理器类型和模型家族添加特定参数
        if self.handler_name == "Qwen25VLChatHandler":
            # Qwen2.5-VL处理器支持以下参数
            possible_params = {
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "image_min_tokens": 1024,
            }
        elif self.handler_name == "Qwen3VLChatHandler":
            # Qwen3-VL处理器参数
            possible_params = {
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "image_min_tokens": 1024,
            }
        elif "Llava" in self.handler_name:
            # LLaVA处理器参数
            possible_params = {
                # LLaVA通常不支持image_max_tokens参数
            }
        else:
            possible_params = {}
        
        # 检查处理器实际支持的参数
        try:
            sig = inspect.signature(handler_cls.__init__)
            supported_params = set(sig.parameters.keys())
        except Exception:
            supported_params = set(base_params.keys())
        
        # 构建最终的参数
        final_params = {}
        
        # 添加基础参数
        for key, value in base_params.items():
            if key in supported_params:
                final_params[key] = value
            else:
                print(f"[QwenVL] 警告: {self.handler_name} 不支持基础参数 {key}")
        
        # 添加可能的高级参数
        for key, value in possible_params.items():
            if key in supported_params:
                final_params[key] = value
            else:
                print(f"[QwenVL] 跳过 {self.handler_name} 不支持参数: {key}")
        
        print(f"[QwenVL] 使用 {self.handler_name} 处理 {model_family} 模型")
        print(f"[QwenVL] 参数列表: {list(final_params.keys())}")
        
        try:
            return handler_cls(**final_params)
        except Exception as e:
            print(f"[QwenVL] 创建处理器失败: {e}")
            # 尝试最小参数集
            try:
                minimal_params = {"clip_model_path": str(mmproj_path)}
                if "verbose" in supported_params:
                    minimal_params["verbose"] = False
                print(f"[QwenVL] 尝试最小参数集: {list(minimal_params.keys())}")
                return handler_cls(**minimal_params)
            except Exception as e2:
                print(f"[QwenVL] 最小参数集也失败: {e2}")
                raise

    def _load_model(
        self,
        model_source: str,  # 模型来源：配置名称或本地路径
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
                
            # 更精确的模型家族检测
            self.model_family = _detect_model_family_from_path(model_path)
            
            # 检查模型文件大小
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"[QwenVL] 模型文件大小: {file_size_mb:.1f} MB")
            
            # 如果文件名包含特定信息但检测为unknown，尝试进一步检测
            if self.model_family == "unknown":
                filename_lower = model_path.name.lower()
                if "qwen2.5" in filename_lower or "qwen2_5" in filename_lower:
                    self.model_family = "qwen2.5-vl"
                    print(f"[QwenVL] 根据文件名推断模型家族为: qwen2.5-vl")
                elif "qwen3" in filename_lower:
                    self.model_family = "qwen3-vl"
                    print(f"[QwenVL] 根据文件名推断模型家族为: qwen3-vl")
                    
            # 根据模型家族设置默认参数
            if self.model_family == "qwen2.5-vl":
                default_ctx = 8192
                default_img_max = 4096
                default_n_batch = 512
                default_gpu_layers = -1
            elif self.model_family == "qwen3-vl":
                default_ctx = 8192
                default_img_max = 4096
                default_n_batch = 512
                default_gpu_layers = -1
            elif "llava" in self.model_family:
                default_ctx = 4096
                default_img_max = 2048
                default_n_batch = 256
                default_gpu_layers = -1
            else:
                default_ctx = 8192
                default_img_max = 4096
                default_n_batch = 512
                default_gpu_layers = -1
            
            resolved = GGUFVLResolved(
                display_name=model_path.name,
                repo_id=None,
                alt_repo_ids=[],
                author=None,
                repo_dirname=model_path.parent.name,
                model_filename=model_path.name,
                mmproj_filename=mmproj_path.name if mmproj_path else None,
                context_length=default_ctx,
                image_max_tokens=default_img_max,
                n_batch=default_n_batch,
                gpu_layers=default_gpu_layers,
                top_k=0,
                pool_size=4194304,
                model_family=self.model_family,
            )
        else:
            # 使用配置中的模型
            resolved = _resolve_model_entry(model_source)
            self.model_family = resolved.model_family
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
            self.model_family,
        )
        if self.llm is not None and self.current_signature == signature:
            print(f"[QwenVL] 模型已加载，跳过重复加载")
            return

        self.clear()

        from llama_cpp import Llama

        self.chat_handler = None
        if has_mmproj:
            # 根据模型家族选择处理器
            if self.model_family == "qwen2.5-vl":
                handler_priority = [
                    ("Qwen25VLChatHandler", "from llama_cpp.llama_chat_format import Qwen25VLChatHandler"),
                    ("Qwen3VLChatHandler", "from llama_cpp.llama_chat_format import Qwen3VLChatHandler"),
                    ("LlavaChatHandler", "from llama_cpp.llama_chat_format import LlavaChatHandler"),
                ]
            elif self.model_family == "qwen3-vl":
                handler_priority = [
                    ("Qwen3VLChatHandler", "from llama_cpp.llama_chat_format import Qwen3VLChatHandler"),
                    ("Qwen25VLChatHandler", "from llama_cpp.llama_chat_format import Qwen25VLChatHandler"),
                    ("LlavaChatHandler", "from llama_cpp.llama_chat_format import LlavaChatHandler"),
                ]
            elif "llava" in self.model_family:
                if "1.6" in self.model_family:
                    handler_priority = [
                        ("Llava16ChatHandler", "from llama_cpp.llama_chat_format import Llava16ChatHandler"),
                        ("Llava15ChatHandler", "from llama_cpp.llama_chat_format import Llava15ChatHandler"),
                        ("LlavaChatHandler", "from llama_cpp.llama_chat_format import LlavaChatHandler"),
                    ]
                else:
                    handler_priority = [
                        ("Llava15ChatHandler", "from llama_cpp.llama_chat_format import Llava15ChatHandler"),
                        ("Llava16ChatHandler", "from llama_cpp.llama_chat_format import Llava16ChatHandler"),
                        ("LlavaChatHandler", "from llama_cpp.llama_chat_format import LlavaChatHandler"),
                    ]
            else:
                # 通用尝试顺序
                handler_priority = [
                    ("Qwen25VLChatHandler", "from llama_cpp.llama_chat_format import Qwen25VLChatHandler"),
                    ("Qwen3VLChatHandler", "from llama_cpp.llama_chat_format import Qwen3VLChatHandler"),
                    ("Llava15ChatHandler", "from llama_cpp.llama_chat_format import Llava15ChatHandler"),
                    ("Llava16ChatHandler", "from llama_cpp.llama_chat_format import Llava16ChatHandler"),
                    ("LlavaChatHandler", "from llama_cpp.llama_chat_format import LlavaChatHandler"),
                ]
            
            handler_cls = None
            handler_name = ""
            
            for hname, import_stmt in handler_priority:
                try:
                    exec(import_stmt)
                    handler_cls = eval(hname)
                    handler_name = hname
                    print(f"[QwenVL] 为 {self.model_family} 找到处理器: {handler_name}")
                    break
                except ImportError:
                    continue
                except Exception as e:
                    print(f"[QwenVL] 导入 {hname} 失败: {e}")
                    continue
            
            if handler_cls is None:
                try:
                    from llama_cpp.llama_chat_format import LlavaChatHandler
                    handler_cls = LlavaChatHandler
                    handler_name = "LlavaChatHandler"
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Missing vision chat handler in llama_cpp. Install the correct fork/wheel. See docs/GGUF_MANUAL_INSTALL.md"
                    )
            
            try:
                self.chat_handler = self._create_chat_handler(handler_cls, mmproj_path, img_max, self.model_family)
            except Exception as e:
                print(f"[QwenVL] 创建 {handler_name} 处理器失败: {e}")
                try:
                    print(f"[QwenVL] 尝试使用最小参数集创建处理器")
                    self.chat_handler = handler_cls(clip_model_path=str(mmproj_path), verbose=False)
                except Exception as e2:
                    print(f"[QwenVL] 最小参数也失败: {e2}")
                    print(f"[QwenVL] 警告: 无法创建视觉处理器，图像功能将不可用")
                    self.chat_handler = None
                    has_mmproj = False

        llm_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        
        # Qwen系列模型需要特定参数
        if self.model_family in ["qwen2.5-vl", "qwen3-vl"]:
            llm_kwargs["swa_full"] = True  # Qwen模型需要这个参数
        
        # 尝试添加 chat_handler
        if has_mmproj and self.chat_handler is not None:
            try:
                llm_kwargs["chat_handler"] = self.chat_handler
                if self.model_family in ["qwen2.5-vl", "qwen3-vl"]:
                    llm_kwargs["image_min_tokens"] = 1024
                    llm_kwargs["image_max_tokens"] = img_max
                print(f"[QwenVL] 已添加 {self.handler_name} 作为 chat_handler")
            except Exception as e:
                print(f"[QwenVL] 警告: 添加 chat_handler 失败: {e}")
                print(f"[QwenVL] 图像功能可能受限")

        print(f"[QwenVL] Loading GGUF: {model_path.name} ({self.model_family})")
        print(f"[QwenVL] Device: {device_kind}, GPU layers: {n_gpu_layers}, Context: {n_ctx}")
        
        # 过滤掉 Llama 不支持的参数
        llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        
        # 检查 chat_handler 是否被接受
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] 警告: 当前 llama_cpp 版本不支持 chat_handler 参数。"
                "这可能是因为您使用的是旧版本或不支持多模态的构建。"
                "请更新到支持多模态的 llama-cpp-python 版本。"
            )
            # 移除 chat_handler 相关参数
            for key in ["chat_handler", "image_min_tokens", "image_max_tokens"]:
                llm_kwargs_filtered.pop(key, None)
            
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] 警告: device=cuda 但 n_gpu_layers=0，模型将在 CPU 上运行")
            
        try:
            self.llm = Llama(**llm_kwargs_filtered)
            self.current_signature = signature
            print(f"[QwenVL] {self.model_family} 模型加载成功")
        except Exception as e:
            print(f"[QwenVL] 模型加载失败: {e}")
            # 尝试去掉可能的额外参数
            minimal_kwargs = {
                "model_path": str(model_path),
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "verbose": False,
            }
            try:
                self.llm = Llama(**minimal_kwargs)
                self.current_signature = signature
                print(f"[QwenVL] 使用最小参数集加载模型成功")
            except Exception as e2:
                raise RuntimeError(f"[QwenVL] 模型加载失败，请检查模型文件: {e2}")

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],  # 所有图像，按输入顺序
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> str:
        """调用模型生成 - 支持按顺序处理图像"""
        
        # 构建消息，图像按输入顺序附加到用户消息
        messages = []
        
        # 添加系统消息（仅文本）
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户消息（包含用户文本和所有图像）
        user_content = []
        
        # 添加用户文本提示
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        
        # 按输入顺序添加所有图像
        for i, img in enumerate(images_b64):
            if img:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        print(f"[QwenVL] 总共输入 {len(images_b64)} 张图像，将按输入顺序处理")
        
        start = time.perf_counter()
        try:
            # Qwen系列模型的特殊停止词
            stop_tokens = ["<|im_end|>", "<|im_start|>"]
            if self.model_family == "qwen2.5-vl":
                stop_tokens.extend(["<|endoftext|>", "用户:"])
            elif self.model_family == "qwen3-vl":
                stop_tokens.extend(["<|endoftext|>", "用户:", "assistant:"])
            
            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                repeat_penalty=float(repetition_penalty),
                seed=int(seed),
                stop=stop_tokens,
            )
        except Exception as e:
            print(f"[QwenVL] 生成失败: {e}")
            # 尝试简化调用
            try:
                print(f"[QwenVL] 尝试简化生成调用")
                result = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
            except Exception as e2:
                print(f"[QwenVL] 简化调用也失败: {e2}")
                return f"[错误] 生成失败: {e2}"
                
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
        system_prompt: str,    # 系统角色定义提示词
        user_prompt: str,      # 用户输入提示词
        images: list,          # 所有图像列表，按输入顺序
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

        # 处理所有图像，按输入顺序
        images_b64: list[str] = []
        if images:
            for i, image_tensor in enumerate(images):
                if image_tensor is not None:
                    img = _tensor_to_base64_png(image_tensor, resize_to=(448, 448))
                    if img:
                        images_b64.append(img)
                        print(f"[QwenVL] 图像{i+1}: 已转换")
        
        # 处理视频输入（视频通常作为用户输入的一部分）
        if video is not None:
            for frame in _sample_video_frames(video, int(frame_count)):
                img = _tensor_to_base64_png(frame, resize_to=(448, 448))
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
            
            total_images = len(images_b64)
            if total_images > 0 and self.chat_handler is None:
                print("[QwenVL] 警告: 提供了图像但模型没有视觉处理器，图像将被忽略")
            
            # 打印图像信息
            if self.chat_handler is not None and total_images > 0:
                print(f"[QwenVL] 总共输入 {total_images} 张图像，将按输入顺序处理")
            
            text = self._invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
            return (text,)
        except Exception as e:
            print(f"[QwenVL] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return (f"[错误] {str(e)}",)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    """基础版GGUF节点 - 默认使用本地文件，支持多图分析"""
    
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(edit gguf_models.json)"]
        default_model = model_keys[0] if model_keys else ""

        # 获取本地文件
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        # 设置默认值
        default_model_file = "无"
        default_mmproj_file = "无"
        
        if local_gguf_files:
            default_model_file = local_gguf_files[0][1]  # 第一个本地文件
        
        if len(local_mmproj_files) > 1:
            default_mmproj_file = local_mmproj_files[1][1]  # 跳过第一个"无"选项
        
        # 多图分析专用提示词
        multi_image_prompts = [
            "详细描述这张图片",
            "分析图片的艺术风格",
            "描述图片中的人物和场景",
            "提取图片的关键信息",
            "为图片创作一个故事",
            "分析图片的色彩和构图",
            "描述图片中的物体和关系",
            "为图片生成详细的描述"
        ]

        return {
            "required": {
                # 默认启用本地文件
                "使用本地文件": ("BOOLEAN", {"default": True, "tooltip": "启用后使用本地GGUF文件，否则使用配置中的模型"}),
                "模型选择方式": (["从配置选择", "本地文件"], {"default": "从配置选择", "tooltip": "选择模型加载方式"}),
                "model_name": (model_keys, {"default": default_model, "tooltip": "从配置中选择模型"}),
                "本地模型文件": (["无"] + [display for _, display in local_gguf_files], {"default": "无", "tooltip": "选择本地GGUF文件"}),
                "本地mmproj文件": (["无"] + [display for _, display in local_mmproj_files], {"default": "无", "tooltip": "选择本地mmproj文件（视觉模型需要）"}),
                
                # 提示词配置
                "分析模式": (["单图描述", "多图对比", "多图分析"], {"default": "单图描述", "tooltip": "选择分析模式"}),
                "预设提示词": (multi_image_prompts, {"default": multi_image_prompts[0], "tooltip": "选择预设的多图分析提示词"}),
                "自定义提示词": ("STRING", {"default": "", "multiline": True, "placeholder": "输入自定义分析提示词（可选）"}),
                "系统角色定义": ("STRING", {"default": "你是一个专业的视觉分析助手，擅长理解和描述图像内容。", "multiline": True, "placeholder": "定义AI的系统角色"}),
                
                # 基本参数
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 4096, "tooltip": "最大生成令牌数"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.1, "tooltip": "温度参数，控制随机性"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "保持模型加载以加速后续推理"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "tooltip": "随机种子，-1为随机"}),
            },
            "optional": {
                # 图像输入（支持多图）
                "图像_1": ("IMAGE", {"tooltip": "图像输入 1"}),
                "图像_2": ("IMAGE", {"tooltip": "图像输入 2"}),
                "图像_3": ("IMAGE", {"tooltip": "图像输入 3"}),
                "图像_4": ("IMAGE", {"tooltip": "图像输入 4"}),
                "图像_5": ("IMAGE", {"tooltip": "图像输入 5"}),
                "图像_6": ("IMAGE", {"tooltip": "图像输入 6"}),
                
                "video": ("IMAGE", {"tooltip": "视频输入（可选）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("分析结果",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        使用本地文件=True,
        模型选择方式="本地文件",
        model_name="无",
        本地模型文件="无",
        本地mmproj文件="无",
        分析模式="单图描述",
        预设提示词="详细描述这张图片",
        自定义提示词="",
        系统角色定义="你是一个专业的视觉分析助手，擅长理解和描述图像内容。",
        max_tokens=1024,
        temperature=0.7,
        keep_model_loaded=True,
        seed=-1,
        图像_1=None,
        图像_2=None,
        图像_3=None,
        图像_4=None,
        图像_5=None,
        图像_6=None,
        video=None,
    ):
        # 收集所有图像，按输入顺序
        images = [图像_1, 图像_2, 图像_3, 图像_4, 图像_5, 图像_6]
        images = [img for img in images if img is not None]
        
        # 根据分析模式调整提示词
        if 分析模式 == "多图对比":
            if not 自定义提示词.strip():
                base_prompt = "请比较和分析这些图片的相似之处和差异："
            else:
                base_prompt = 自定义提示词.strip()
        elif 分析模式 == "多图分析":
            if not 自定义提示词.strip():
                base_prompt = "请综合分析这些图片，描述它们共同的主题和各自的特点："
            else:
                base_prompt = 自定义提示词.strip()
        else:  # 单图描述
            if not 自定义提示词.strip():
                base_prompt = 预设提示词
            else:
                base_prompt = 自定义提示词.strip()
        
        # 如果有多个图像，自动调整提示词
        if len(images) > 1 and 分析模式 == "单图描述":
            base_prompt = f"请按顺序描述这{len(images)}张图片：{base_prompt}"
        
        # 根据图像数量调整系统角色
        if len(images) > 1:
            if "多图" not in 系统角色定义:
                系统角色定义 = f"{系统角色定义}你特别擅长多图分析和对比。"
        
        # 使用本地文件（默认启用）
        use_local = 使用本地文件  # 默认就是True
        
        # 获取实际文件路径
        model_source = "无"
        mmproj_source = "无"
        
        if use_local:
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
                raise ValueError("请选择有效的本地模型文件")
        else:
            # 使用配置中的模型
            model_source = model_name
            mmproj_source = "无"  # 配置模型会自动查找mmproj
        
        print(f"[QwenVL] 多图分析模式: {分析模式}")
        print(f"[QwenVL] 输入 {len(images)} 张图像，将按输入顺序处理")
        print(f"[QwenVL] 使用本地模型: {Path(model_source).name}")
        
        # 如果种子为-1，使用随机种子
        effective_seed = seed if seed != -1 else random.randint(1, 2**32 - 1)
        
        return self.run(
            model_source=model_source,
            mmproj_source=mmproj_source,
            use_local_files=use_local,
            system_prompt=系统角色定义,
            user_prompt=base_prompt,
            images=images,
            video=video,
            frame_count=8,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            seed=effective_seed,
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
    """高级版GGUF节点 - 默认使用本地文件，支持高级多图分析"""
    
    @classmethod
    def INPUT_TYPES(cls):
        all_models = GGUF_VL_CATALOG.get("models") or {}
        model_keys = sorted([key for key, entry in all_models.items() if (entry or {}).get("mmproj_filename")]) or ["(edit gguf_models.json)"]
        default_model = model_keys[0] if model_keys else ""
        
        # 获取本地文件
        local_gguf_files = _get_local_gguf_files()
        local_mmproj_files = _get_local_mmproj_files()
        
        # 设置默认值
        default_model_file = "无"
        default_mmproj_file = "无"
        
        if local_gguf_files:
            default_model_file = local_gguf_files[0][1]
        
        if len(local_mmproj_files) > 1:
            default_mmproj_file = local_mmproj_files[1][1]
        
        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list
        
        # 高级分析模式
        advanced_modes = [
            "单图详细描述",
            "多图对比分析", 
            "多图故事创作",
            "多图主题提取",
            "艺术风格分析",
            "技术细节分析",
            "情感氛围分析",
            "创意灵感生成"
        ]

        return {
            "required": {
                # 默认启用本地文件
                "使用本地文件": ("BOOLEAN", {"default": True, "tooltip": "启用后使用本地GGUF文件，否则使用配置中的模型"}),
                "模型选择方式": (["从配置选择", "本地文件"], {"default": "从配置选择", "tooltip": "选择模型加载方式"}),
                "model_name": (model_keys, {"default": default_model, "tooltip": "从配置中选择模型"}),
                "本地模型文件": (["无"] + [display for _, display in local_gguf_files], {"default": "无", "tooltip": "选择本地GGUF文件"}),
                "本地mmproj文件": (["无"] + [display for _, display in local_mmproj_files], {"default": "无", "tooltip": "选择本地mmproj文件（视觉模型需要）"}),
                
                # 高级分析配置
                "分析模式": (advanced_modes, {"default": advanced_modes[0], "tooltip": "选择高级分析模式"}),
                "自定义提示词": ("STRING", {"default": "", "multiline": True, "placeholder": "输入自定义分析提示词（可选）"}),
                "系统角色定义": ("STRING", {"default": "你是一个专业的视觉智能助手，具有深厚的艺术和技术分析能力。", "multiline": True, "placeholder": "定义AI的系统角色"}),
                
                # 高级参数
                "device": (device_options, {"default": "auto", "tooltip": "选择计算设备"}),
                "max_tokens": ("INT", {"default": 2048, "min": 512, "max": 8192, "tooltip": "最大生成令牌数"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.5, "step": 0.1, "tooltip": "温度参数，控制随机性"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "核采样参数"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "重复惩罚参数"}),
                "ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768, "step": 1024, "tooltip": "上下文长度"}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": "GPU层数，-1为自动"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "保持模型加载以加速后续推理"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1, "tooltip": "随机种子，-1为随机"}),
            },
            "optional": {
                # 支持更多图像输入
                "图像_1": ("IMAGE", {"tooltip": "图像输入 1"}),
                "图像_2": ("IMAGE", {"tooltip": "图像输入 2"}),
                "图像_3": ("IMAGE", {"tooltip": "图像输入 3"}),
                "图像_4": ("IMAGE", {"tooltip": "图像输入 4"}),
                "图像_5": ("IMAGE", {"tooltip": "图像输入 5"}),
                "图像_6": ("IMAGE", {"tooltip": "图像输入 6"}),
                "图像_7": ("IMAGE", {"tooltip": "图像输入 7"}),
                "图像_8": ("IMAGE", {"tooltip": "图像输入 8"}),
                "图像_9": ("IMAGE", {"tooltip": "图像输入 9"}),
                "图像_10": ("IMAGE", {"tooltip": "图像输入 10"}),
                
                "video": ("IMAGE", {"tooltip": "视频输入（可选）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("高级分析结果",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        使用本地文件=True,
        模型选择方式="本地文件",
        model_name="无",
        本地模型文件="无",
        本地mmproj文件="无",
        分析模式="单图详细描述",
        自定义提示词="",
        系统角色定义="你是一个专业的视觉智能助手，具有深厚的艺术和技术分析能力。",
        device="auto",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        ctx=8192,
        gpu_layers=-1,
        keep_model_loaded=True,
        seed=-1,
        图像_1=None,
        图像_2=None,
        图像_3=None,
        图像_4=None,
        图像_5=None,
        图像_6=None,
        图像_7=None,
        图像_8=None,
        图像_9=None,
        图像_10=None,
        video=None,
    ):
        # 收集所有图像，按输入顺序
        images = [图像_1, 图像_2, 图像_3, 图像_4, 图像_5, 图像_6, 图像_7, 图像_8, 图像_9, 图像_10]
        images = [img for img in images if img is not None]
        
        # 根据分析模式生成提示词
        mode_prompts = {
            "单图详细描述": "请详细描述这张图片，包括场景、物体、人物、色彩、风格等所有视觉元素。",
            "多图对比分析": "请对比分析这些图片，指出它们的相似之处、差异、共同主题和各自特点。",
            "多图故事创作": "请根据这些图片创作一个连贯的故事或叙事。",
            "多图主题提取": "请从这些图片中提取共同的主题、概念和视觉元素。",
            "艺术风格分析": "请分析这些图片的艺术风格、绘画技巧、色彩运用和构图特点。",
            "技术细节分析": "请分析这些图片的技术细节，包括光线、角度、焦点、分辨率等。",
            "情感氛围分析": "请描述这些图片传达的情感氛围和情绪感受。",
            "创意灵感生成": "请基于这些图片生成创意灵感和设计思路。"
        }
        
        # 确定使用的提示词
        if 自定义提示词.strip():
            user_prompt = 自定义提示词.strip()
        else:
            user_prompt = mode_prompts.get(分析模式, "请分析这些图片。")
        
        # 根据图像数量调整提示词
        if len(images) > 1:
            user_prompt = f"共有{len(images)}张图片。{user_prompt}"
        
        # 根据分析模式调整系统角色
        role_specializations = {
            "艺术风格分析": "艺术评论家",
            "技术细节分析": "技术分析师", 
            "情感氛围分析": "情感分析师",
            "创意灵感生成": "创意顾问"
        }
        
        specialization = role_specializations.get(分析模式, "视觉分析专家")
        if specialization not in 系统角色定义:
            系统角色定义 = f"你是{specialization}，{系统角色定义}"
        
        # 使用本地文件（默认启用）
        use_local = 使用本地文件
        
        # 获取实际文件路径
        model_source = "无"
        mmproj_source = "无"
        
        if use_local:
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
                raise ValueError("请选择有效的本地模型文件")
        else:
            # 使用配置中的模型
            model_source = model_name
            mmproj_source = "无"
        
        print(f"[QwenVL] 高级分析模式: {分析模式}")
        print(f"[QwenVL] 输入 {len(images)} 张图像")
        print(f"[QwenVL] 使用设备: {device}")
        
        # 如果种子为-1，使用随机种子
        effective_seed = seed if seed != -1 else random.randint(1, 2**32 - 1)
        
        return self.run(
            model_source=model_source,
            mmproj_source=mmproj_source,
            use_local_files=use_local,
            system_prompt=系统角色定义,
            user_prompt=user_prompt,
            images=images,
            video=video,
            frame_count=12,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=effective_seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=512,
            gpu_layers=gpu_layers,
            image_max_tokens=4096,
            top_k=40,
            pool_size=4194304,
        )


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_GGUF": AILab_QwenVL_GGUF,
    "AILab_QwenVL_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_GGUF": "QwenVL 多图分析 (GGUF)",
    "AILab_QwenVL_GGUF_Advanced": "QwenVL 高级多图分析 (GGUF)",
}
