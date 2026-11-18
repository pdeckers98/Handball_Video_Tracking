"""Local vision-language models behind one small interface.

All models run on the local GPU (GTX 1660 Super, 6 GB, Turing without tensor cores). Measured on
this card: fp16 matmuls are ~5x *slower* than fp32 (0.56 vs 4.6 TFLOPS), while bf16 is as fast as
fp32 (cuBLAS runs it through fp32 compute) at half the memory. So models run in **bf16**, and
larger ones load in 4-bit (bitsandbytes NF4) with bf16 compute.

Interface (:class:`VLM`):
- ``yes_probability(images, question)``: P("yes") from the next-token logits, so we get a
  score instead of a hard answer (lets us measure ROC-AUC and pick thresholds).
- ``generate(images, prompt)``: free text.
- ``point(image, target)``: (x, y) in 0..1 image coordinates, or ``None``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

# name -> (repo, loader, quantize to 4 bit)
MODELS: dict[str, tuple[str, str, bool]] = {
    "smolvlm2-500m": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "hf", False),
    "smolvlm2-2.2b": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "hf", True),  # bf16 OOMs
    "lfm2.5-vl-1.6b": ("LiquidAI/LFM2.5-VL-1.6B", "hf", False),
    "moondream2": ("vikhyatk/moondream2", "moondream", False),
    "qwen3-vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", "hf", False),
    "qwen3-vl-4b": ("Qwen/Qwen3-VL-4B-Instruct", "hf", True),
    "internvl3.5-1b": ("OpenGVLab/InternVL3_5-1B-HF", "hf", False),
    "internvl3.5-2b": ("OpenGVLab/InternVL3_5-2B-HF", "hf", False),
    "internvl3.5-4b": ("OpenGVLab/InternVL3_5-4B-HF", "hf", True),
}

MAX_INPUT_TOKENS = 6000  # guard: beyond this the 6 GB card spills to system RAM and crawls
VRAM_FRACTION = 0.92  # make PyTorch raise OOM instead of the Windows driver paging to RAM


def fp32_conv3d(model, dtype) -> int:
    """Run Conv3d layers in fp32: cuDNN has no fast bf16 Conv3d kernel on Turing (Qwen3-VL's
    patch embedding took 78 s of an 83 s query in bf16). The layers are tiny, so fp32 costs
    almost no memory. Outputs are cast back to ``dtype``. Conv2d in bf16 is fine (SmolVLM
    2 s/query) and is left alone, because some vision towers infer their dtype from it."""
    import torch

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            module.float()
            original = module.forward

            def forward(x, _original=original):
                return _original(x.float()).to(dtype)

            module.forward = forward
            count += 1
    return count


def compute_dtype():
    """bf16: same memory as fp16 but ~5x faster on this GPU (see module docstring)."""
    import torch

    return torch.bfloat16


POINT_PROMPT = (
    "Where is the {target} in this image? Answer only with JSON "
    '{{"x": <0-1000>, "y": <0-1000>}} giving its centre in coordinates normalized to 0-1000.'
)


def _to_pil(image) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))


def parse_point(text: str) -> tuple[float, float] | None:
    """Extract an (x, y) point in 0..1 from a model answer.

    Accepts JSON ``{"x":..,"y":..}``, Qwen-style ``"point_2d": [x, y]``, a box
    ``[x0, y0, x1, y1]`` (its centre is used) or two bare numbers. Values are read as 0-1000
    normalized unless all are <= 1.
    """
    # Skip digits inside words (the "2" in Qwen's "point_2d").
    numbers = [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)]
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        if isinstance(data, dict) and "x" in data and "y" in data:
            numbers = [float(data["x"]), float(data["y"])]
    except (ValueError, TypeError):
        pass
    if len(numbers) == 4:
        numbers = [(numbers[0] + numbers[2]) / 2, (numbers[1] + numbers[3]) / 2]
    if len(numbers) < 2:
        return None
    x, y = numbers[:2]
    scale = 1.0 if max(x, y) <= 1.0 else 1000.0
    x, y = x / scale, y / scale
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return x, y


@dataclass
class LoadInfo:
    name: str
    repo: str
    quantized: bool
    load_seconds: float
    vram_gb: float


class VLM:
    info: LoadInfo

    def yes_probability(self, images, question: str) -> float:
        raise NotImplementedError

    def generate(self, images, prompt: str, max_new_tokens: int = 64) -> str:
        raise NotImplementedError

    last_answer: str | None = None
    """Raw text of the most recent :meth:`point` answer (kept for re-parsing / debugging)."""

    def point(self, image, target: str) -> tuple[float, float] | None:
        self.last_answer = self.generate([image], POINT_PROMPT.format(target=target), 40)
        return parse_point(self.last_answer)

    def close(self) -> None:
        import gc

        import torch

        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()


class HFVLM(VLM):
    """Any transformers image-text-to-text chat model (SmolVLM2, Qwen3-VL, InternVL, LFM2-VL)."""

    def __init__(self, name: str, repo: str, quantize: bool):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        kwargs = {"dtype": compute_dtype(), "device_map": "cuda"}
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype(),
            )
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs).eval()
        fp32_conv3d(self.model, compute_dtype())
        self.info = LoadInfo(
            name, repo, quantize, time.perf_counter() - started,
            torch.cuda.max_memory_allocated() / 1e9,
        )
        tokenizer = self.processor.tokenizer
        self._yes_ids = self._token_ids(tokenizer, ["Yes", "yes", " Yes", " yes", "YES"])
        self._no_ids = self._token_ids(tokenizer, ["No", "no", " No", " no", "NO"])

    @staticmethod
    def _token_ids(tokenizer, words: list[str]) -> list[int]:
        ids = {tokenizer.encode(w, add_special_tokens=False)[0] for w in words}
        return sorted(ids)

    def _configure(self, n_images: int) -> dict:
        """Tile single images for detail; never tile multi-frame input (token blow-up:
        SmolVLM turns 16 small frames into ~14k tokens with tiling, ~1k without).

        Returns extra call-time processor kwargs (some processors ignore attribute changes).
        """
        image_processor = self.processor.image_processor
        single = n_images == 1
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = single  # SmolVLM honours the attribute
        if hasattr(image_processor, "crop_to_patches"):
            # InternVL: 16 frames tile into 48 crops (12k tokens) unless disabled per call.
            return {"crop_to_patches": single}
        if hasattr(image_processor, "temporal_patch_size") and single:
            # Qwen-VL: a 1280x720 frame is 3520 patches with full attention in the vision
            # tower (OOM on 6 GB); cap the pixel budget at 1024x576 (2304 patches).
            return {"size": {"longest_edge": 1024 * 576, "shortest_edge": 256 * 256}}
        if hasattr(image_processor, "max_image_size") and hasattr(image_processor, "size"):
            # SmolVLM: tile size is max_image_size; the long edge is resized to ``size``.
            edge = 1536 if single else image_processor.max_image_size["longest_edge"]
            image_processor.size = {"longest_edge": edge}
        if hasattr(image_processor, "max_num_patches") and not single:
            # LFM2-VL pads every frame to max_num_patches (1024) and encodes the batch at once,
            # a 1 GB attention block for 16 frames. 448 patches of 16 px fit a 448x252 frame.
            return {"do_image_splitting": False, "max_num_patches": 448, "max_image_tokens": 112}
        return {}

    def _inputs(self, images, prompt: str):
        processor_kwargs = self._configure(len(images))
        content = [{"type": "image", "image": _to_pil(im)} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs=processor_kwargs,
        )
        self.last_input_tokens = int(inputs["input_ids"].shape[1])
        if self.last_input_tokens > MAX_INPUT_TOKENS:
            raise ValueError(f"{self.last_input_tokens} input tokens > {MAX_INPUT_TOKENS}")
        return inputs.to(self.model.device)

    def yes_probability(self, images, question: str) -> float:
        import torch

        prompt = f"{question}\nAnswer with only yes or no."
        with torch.inference_mode():
            # Only the last position is needed; full logits (tokens x vocab) can exceed 1 GB.
            output = self.model(**self._inputs(images, prompt), logits_to_keep=1)
            logits = output.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        yes = probs[self._yes_ids].sum().item()
        no = probs[self._no_ids].sum().item()
        return yes / (yes + no) if yes + no > 0 else 0.5

    def generate(self, images, prompt: str, max_new_tokens: int = 64) -> str:
        import torch

        inputs = self._inputs(images, prompt)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new, skip_special_tokens=True).strip()


def mosaic(images, cols: int = 4) -> Image.Image:
    """Tile frames into one image (row-major, in time order) for single-image models."""
    images = [_to_pil(im) for im in images]
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(images):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return sheet


MOONDREAM_REVISION = "2025-01-09"


def _import_snapshot_package(path, package: str):
    """Import a folder of modules with relative imports as a package (no transformers)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        package, str(path / "__init__.py"), submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    return module


class MoondreamVLM(VLM):
    """Moondream 2. Single image only: several frames are tiled into a mosaic.

    Its Hugging Face wrapper (``trust_remote_code``) is incompatible with transformers 5, so
    the plain PyTorch model is built from the snapshot's own modules and weight loader.
    """

    def __init__(self, name: str, repo: str):
        import importlib
        import json
        from pathlib import Path

        import torch
        from huggingface_hub import snapshot_download

        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        path = Path(snapshot_download(repo, revision=MOONDREAM_REVISION))
        package = "moondream2_snapshot"
        _import_snapshot_package(path, package)
        config_mod = importlib.import_module(f"{package}.config")
        model_mod = importlib.import_module(f"{package}.moondream")
        from safetensors.torch import load_file

        raw = json.loads((path / "config.json").read_text()).get("config", {})
        model = model_mod.MoondreamModel(config_mod.MoondreamConfig.from_dict(raw))
        # Saved through the HF wrapper, so every key is prefixed with "model.".
        state = {
            k.removeprefix("model."): v.to(torch.float16)
            for k, v in load_file(str(path / "model.safetensors")).items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"moondream weights: missing {missing[:5]}, unexpected "
                               f"{unexpected[:5]}")
        self.model = model.to("cuda").eval()
        self.info = LoadInfo(
            name, repo, False, time.perf_counter() - started,
            torch.cuda.max_memory_allocated() / 1e9,
        )

    def _image(self, images):
        return _to_pil(images[0]) if len(images) == 1 else mosaic(images)

    def yes_probability(self, images, question: str) -> float:
        prompt = f"{question} Answer with only yes or no."
        answer = self.generate(images, prompt).lower()
        if answer.startswith("yes"):
            return 1.0
        if answer.startswith("no"):
            return 0.0
        return 0.5

    @staticmethod
    def _fast():
        """Moondream hard-codes fp16 tensors, so keep fp16 weights but run matmuls in bf16."""
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16)

    def generate(self, images, prompt: str, max_new_tokens: int = 64) -> str:
        settings = {"max_tokens": max_new_tokens, "temperature": 0}
        with self._fast():
            result = self.model.query(self._image(images), prompt, settings=settings)
        return result["answer"].strip()

    def point(self, image, target: str) -> tuple[float, float] | None:
        with self._fast():
            points = self.model.point(_to_pil(image), target)["points"]
        self.last_answer = str(points)
        if not points:
            return None
        return float(points[0]["x"]), float(points[0]["y"])


def load(name: str) -> VLM:
    repo, loader, quantize = MODELS[name]
    if loader == "moondream":
        return MoondreamVLM(name, repo)
    return HFVLM(name, repo, quantize)
