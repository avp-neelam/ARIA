"""Automatic device selection: CUDA > Apple Silicon (MPS) > CPU.

Every training/search script in this repo should call get_device() once and
thread the returned torch.device through, rather than hardcoding "cpu" the
way the exploratory scripts elsewhere in this project do (those were built
in a CPU-only sandbox; this repo is meant to run on whatever hardware is
actually available).
"""
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Returns the best available torch.device.

    prefer="auto" (default): CUDA if available, else Apple Silicon MPS if
    available, else CPU. Pass prefer="cpu"/"cuda"/"mps" to force a specific
    backend (raises RuntimeError if that backend isn't actually available on
    this machine, rather than silently falling back).
    """
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if prefer != "auto":
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device cuda but torch.cuda.is_available() is False.")
        if prefer == "mps" and not has_mps:
            raise RuntimeError("MPS requested via --device mps but torch.backends.mps.is_available() is False.")
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device.index or 0)})"
        except Exception:
            return "cuda"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
