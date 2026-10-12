"""Model / tokenizer / device helpers shared by train, infer, export and forgetting."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_dtype():
    dev = get_device()
    if dev == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if dev == "mps" else torch.float32


def train_dtype():
    """bf16 weights where supported; otherwise fp32 weights + fp16 autocast (T4/P100 have no bf16)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, dtype, load_4bit: bool = False):
    kwargs = {}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=infer_dtype(),
        )
        kwargs["device_map"] = {"": 0}
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except (TypeError, ValueError):  # older transformers: `torch_dtype`
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
