"""
Step 2: Pixel Art Snapper アルゴリズム実装（高速版）
元実装: https://github.com/Hugo-Dz/spritefusion-pixel-snapper
- K-Means → PIL内蔵 quantize()（Cで実装、大幅高速化）
- resample ループ → numpy完全ベクトル化
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class PixelSnapConfig:
    k_colors: int = 16
    output_scale: int = 1
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8
    max_input_size: int = 1024  # 入力画像の最大辺(px)。超えたらリサイズ


def _quantize_pil(img_rgb: Image.Image, k: int) -> np.ndarray:
    """PIL内蔵のmedian-cut量子化（K-Meansより10〜50倍高速）。"""
    quantized = img_rgb.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=0)
    return np.array(quantized.convert("RGB"), dtype=np.uint8)


def compute_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col_profile = np.sum(np.abs(gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)), axis=0)
    row_profile = np.sum(np.abs(gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)), axis=1)
    return col_profile, row_profile


def estimate_step_size(profile: np.ndarray, cfg: PixelSnapConfig) -> float | None:
    threshold = profile.max() * cfg.peak_threshold_multiplier
    peaks = [i for i in range(1, len(profile) - 1)
             if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > threshold]
    filtered: list[int] = []
    for p in peaks:
        if not filtered or p - filtered[-1] >= cfg.peak_distance_filter:
            filtered.append(p)
    if len(filtered) < 2:
        return None
    return float(np.median(np.diff(filtered)))


def walk(profile: np.ndarray, step_size: float, cfg: PixelSnapConfig) -> list[int]:
    cuts = [0]
    pos = step_size
    while pos < len(profile):
        window = max(cfg.walker_min_search_window, step_size * cfg.walker_search_window_ratio)
        lo = max(0, int(pos - window))
        hi = min(len(profile) - 1, int(pos + window))
        seg = profile[lo:hi + 1]
        li = np.argmax(seg)
        if seg[li] > seg.mean() * cfg.walker_strength_threshold:
            cuts.append(lo + li)
        else:
            cuts.append(int(round(pos)))
        pos = cuts[-1] + step_size
    return cuts


def stabilize_both_axes(col_cuts: list[int], row_cuts: list[int],
                         w: int, h: int, cfg: PixelSnapConfig) -> tuple[list[int], list[int]]:
    def uniform(size: int, n: int) -> list[int]:
        return [int(round(i * size / n)) for i in range(n + 1)]

    def stabilize(cuts: list[int], size: int) -> list[int]:
        return cuts if len(cuts) >= cfg.min_cuts_per_axis + 1 else uniform(size, cfg.fallback_target_segments)

    col_cuts = stabilize(col_cuts, w)
    row_cuts = stabilize(row_cuts, h)

    if len(col_cuts) > 1 and len(row_cuts) > 1:
        xs = (col_cuts[-1] - col_cuts[0]) / (len(col_cuts) - 1)
        ys = (row_cuts[-1] - row_cuts[0]) / (len(row_cuts) - 1)
        ratio = max(xs, ys) / (min(xs, ys) + 1e-9)
        if ratio > cfg.max_step_ratio:
            avg = (xs + ys) / 2
            col_cuts = uniform(w, max(1, round(w / avg)))
            row_cuts = uniform(h, max(1, round(h / avg)))

    return col_cuts, row_cuts


def resample_vectorized(img_array: np.ndarray,
                        col_cuts: list[int], row_cuts: list[int]) -> np.ndarray:
    """numpy完全ベクトル化リサンプリング（Pythonループなし）。"""
    n_cols = len(col_cuts) - 1
    n_rows = len(row_cuts) - 1
    h, w = img_array.shape[:2]

    row_assign = np.zeros(h, dtype=np.int32)
    for r in range(n_rows):
        row_assign[row_cuts[r]:row_cuts[r + 1]] = r

    col_assign = np.zeros(w, dtype=np.int32)
    for c in range(n_cols):
        col_assign[col_cuts[c]:col_cuts[c + 1]] = c

    cell_idx = row_assign[:, None] * n_cols + col_assign[None, :]

    n_cells = n_rows * n_cols
    flat_idx = cell_idx.ravel()
    flat_px = img_array[:, :, :3].reshape(-1, 3).astype(np.float32)

    sums = np.zeros((n_cells, 3), dtype=np.float32)
    counts = np.zeros(n_cells, dtype=np.int32)
    np.add.at(sums, flat_idx, flat_px)
    np.add.at(counts, flat_idx, 1)

    result = (sums / np.maximum(counts, 1)[:, None]).clip(0, 255).astype(np.uint8)
    return result.reshape(n_rows, n_cols, 3)


def pixel_snap(image: Image.Image, cfg: PixelSnapConfig | None = None) -> Image.Image:
    if cfg is None:
        cfg = PixelSnapConfig()

    img_rgb = image.convert("RGB")

    # 大きすぎる画像はリサイズ（速度向上）
    iw, ih = img_rgb.size
    if max(iw, ih) > cfg.max_input_size:
        scale = cfg.max_input_size / max(iw, ih)
        img_rgb = img_rgb.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)

    img_array = _quantize_pil(img_rgb, cfg.k_colors)
    h, w = img_array.shape[:2]

    # dot_count（fallback_target_segments）を列数として直接使用する。
    # AI生成画像には既存のピクセルグリッドがないため、エッジ自動検出は使わない。
    # 縦ドット数は元画像の縦横比から計算し、ピクセルを常に正方形に保つ。
    n_cols = max(cfg.min_cuts_per_axis, cfg.fallback_target_segments)
    n_rows = max(1, round(n_cols * h / w))
    print(f"[pixel_snap] fallback_target_segments={cfg.fallback_target_segments}, n_cols={n_cols}, n_rows={n_rows}, img={w}×{h}")
    col_cuts = [int(round(i * w / n_cols)) for i in range(n_cols + 1)]
    row_cuts = [int(round(i * h / n_rows)) for i in range(n_rows + 1)]

    pixel_art_array = resample_vectorized(img_array, col_cuts, row_cuts)

    out_h, out_w = pixel_art_array.shape[:2]
    result = Image.fromarray(pixel_art_array, "RGB")
    if cfg.output_scale > 1:
        result = result.resize(
            (out_w * cfg.output_scale, out_h * cfg.output_scale), Image.NEAREST
        )
    return result


def process_image(input_path: Path, output_dir: Path,
                  cfg: PixelSnapConfig | None = None) -> Path:
    if cfg is None:
        cfg = PixelSnapConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_snapped.png"
    print(f"[Step 2] ピクセルスナップ: {input_path.name}")
    result = pixel_snap(Image.open(input_path), cfg)
    result.save(output_path, "PNG")
    print(f"  完了: {output_path} ({result.size})")
    return output_path
