"""
ATHENA Utilities
================
Shared utility functions for the ATHENA system.
"""

import logging
import random
import secrets
from typing import Optional, Any, Dict, List
from datetime import datetime, timezone
import hashlib
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for ATHENA system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

    logger = logging.getLogger("athena")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_device(preference: str = "auto") -> str:
    """
    Get the best available device for computation.

    Args:
        preference: Device preference ("auto", "cuda", "mps", "cpu")

    Returns:
        Device string for PyTorch
    """
    if preference != "auto":
        return preference

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique identifier.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique identifier string
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    random_part = secrets.token_hex(4)

    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    return f"{timestamp}_{random_part}"


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for cosine_similarity")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches.

    Args:
        items: List of items
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any], _depth: int = 0, max_depth: int = 20) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary
        max_depth: Maximum recursion depth (default 20)

    Returns:
        Merged dictionary
    """
    if _depth >= max_depth:
        return {**base, **override}

    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value, _depth=_depth + 1, max_depth=max_depth)
        else:
            result[key] = value

    return result


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime to string.

    Args:
        dt: Datetime object (defaults to now)
        format_str: Format string

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime(format_str)


def safe_json_dumps(obj: Any, default: Any = str) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: Object to serialize
        default: Default serializer for non-serializable objects

    Returns:
        JSON string
    """
    return json.dumps(obj, default=default, ensure_ascii=False)


def safe_json_loads(s: str, default: Any = None) -> Any:
    """
    Safely deserialize JSON string.

    Args:
        s: JSON string
        default: Default value if parsing fails

    Returns:
        Deserialized object or default
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


class Timer:
    """Simple timer context manager for measuring execution time."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        import time
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else __import__("time").perf_counter()
        return end - self.start_time

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.4f}s"
