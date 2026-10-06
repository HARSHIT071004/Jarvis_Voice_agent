"""Wake-word model bootstrap.

Downloads the sherpa-onnx open-vocabulary KWS model once and encodes the
configured wake phrase into a keywords file. Open-vocabulary KWS means we can
use an arbitrary phrase ("HEY JARVIS") with zero training.

Model: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (English, ~19 MB)
Source: https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("jarvis.wake.setup")

MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_NAME}.tar.bz2"
)


class WakeModelError(Exception):
    """Raised when the wake-word model cannot be prepared."""


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading wake-word model (one-time, ~19 MB)...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise WakeModelError(f"Model download failed: {exc}") from exc


def encode_keywords(
    model_dir: Path,
    wake_word: str,
    score: float,
    threshold: float,
    out_path: Path | None = None,
) -> Path:
    """Convert a plain-text wake phrase into sherpa-onnx keyword tokens.

    Uses sherpa_onnx.utils.text2token in-process; the packaged CLI executable
    crashes on some Windows setups (0xC0000005) when shelling out.

    Output line format: `▁HE Y ▁PR I Y AN K A :1.5 #0.25`
    """
    from sherpa_onnx.utils import text2token

    out_path = out_path or model_dir / "keywords.txt"
    phrase = wake_word.strip().upper()
    extras = f" :{score} #{threshold}"

    try:
        token_lists = text2token(
            [phrase],
            tokens=str(model_dir / "tokens.txt"),
            tokens_type="bpe",
            bpe_model=str(model_dir / "bpe.model"),
        )
    except Exception as exc:
        raise WakeModelError(f"Could not encode wake phrase: {exc}") from exc

    if not token_lists or not token_lists[0]:
        raise WakeModelError("Keyword encoding produced no tokens")

    encoded = " ".join(str(t) for t in token_lists[0]) + extras + "\n"
    out_path.write_text(encoded, encoding="utf-8")
    logger.info(
        "Wake phrase encoded: %s",
        encoded.strip().replace("\u2581", " ").replace("  ", " "),
    )
    return out_path


def ensure_model(
    base_dir: Path,
    wake_word: str,
    score: float = 1.5,
    threshold: float = 0.25,
) -> tuple[Path, Path]:
    """Return (model_dir, keywords_file), downloading/encoding as needed."""
    model_dir = base_dir / MODEL_NAME
    required = ["encoder-epoch-12-avg-2-chunk-16-left-64.onnx", "tokens.txt", "bpe.model"]

    if not all((model_dir / name).exists() for name in required):
        base_dir.mkdir(parents=True, exist_ok=True)
        archive = base_dir / f"{MODEL_NAME}.tar.bz2"
        if not archive.exists():
            _download(MODEL_URL, archive)
        logger.info("Extracting wake-word model...")
        try:
            with tarfile.open(archive, "r:bz2") as tar:
                tar.extractall(path=base_dir, filter="data")
        except Exception as exc:
            archive.unlink(missing_ok=True)
            raise WakeModelError(f"Model extraction failed: {exc}") from exc
        finally:
            archive.unlink(missing_ok=True)
        if not model_dir.exists():
            raise WakeModelError("Model archive did not contain the expected directory")

    keywords = encode_keywords(model_dir, wake_word, score, threshold)
    return model_dir, keywords
