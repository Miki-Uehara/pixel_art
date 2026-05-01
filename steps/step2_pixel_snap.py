"""
Step 2: Pixel Art Snapper アルゴリズム実装
元実装: https://github.com/Hugo-Dz/spritefusion-pixel-snapper (Rust/WASM)
Pythonポート参照: https://github.com/x0x0b/ComfyUI-spritefusion-pixel-snapper
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class PixelSnapConfig:
    k_colors: int = 16               # カラーパレット数
    k_seed: int = 42                 # K-Means乱数シード
    max_kmeans_iterations: int = 15  # K-Means最大反復数
    peak_threshold_multiplier: float = 0.2   # グリッド検出しきい値
    peak_distance_filter: int = 4    # ピーク間最小距離(px)
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64  # 検出失敗時のフォールバック分割数
    max_step_ratio: float = 1.8
    output_scale: int = 1            # 出力アップスケール倍率


def _kmeans_plus_plus_init(pixels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(pixels)
    centers = [pixels[rng.integers(n)].astype(np.float64)]
    for _ in range(k - 1):
        dists = np.min(
            [np.sum((pixels - c) ** 2, axis=1) for c in centers], axis=0
        ).astype(np.float64)
        probs = dists / dists.sum()
        centers.append(pixels[rng.choice(n, p=probs)].astype(np.float64))
    return np.array(centers)


def quantize_image(pixels: np.ndarray, cfg: PixelSnapConfig) -> np.ndarray:
    """K-Meansでカラーパレットに量子化する。pixels shape: (N, 3) float64 [0,1]"""
    rng = np.random.default_rng(cfg.k_seed)
    centers = _kmeans_plus_plus_init(pixels, cfg.k_colors, rng)

    for _ in range(cfg.max_kmeans_iterations):
        dists = np.sum((pixels[:, None] - centers[None]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(cfg.k_colors)
        ])
        if np.max(np.linalg.norm(new_centers - centers, axis=1)) < 0.01:
            break
        centers = new_centers

    dists = np.sum((pixels[:, None] - centers[None]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    return centers[labels]


def compute_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """水平・垂直方向のエッジ強度プロファイルを計算する。"""
    col_profile = np.sum(np.abs(gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)), axis=0)
    row_profile = np.sum(np.abs(gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64)), axis=1)
    return col_profile, row_profile


def estimate_step_size(profile: np.ndarray, cfg: PixelSnapConfig) -> float | None:
    """プロファイルからグリッドのステップサイズを推定する。"""
    threshold = profile.max() * cfg.peak_threshold_multiplier
    peaks = [i for i in range(1, len(profile) - 1)
             if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > threshold]

    # ピーク間距離でフィルタリング
    filtered = [peaks[0]] if peaks else []
    for p in peaks[1:]:
        if p - filtered[-1] >= cfg.peak_distance_filter:
            filtered.append(p)

    if len(filtered) < 2:
        return None

    diffs = np.diff(filtered)
    return float(np.median(diffs))


def walk(profile: np.ndarray, step_size: float, cfg: PixelSnapConfig) -> list[int]:
    """グリッド切り位置を探索する。"""
    cuts = [0]
    pos = step_size
    while pos < len(profile):
        window = max(cfg.walker_min_search_window, step_size * cfg.walker_search_window_ratio)
        lo = max(0, int(pos - window))
        hi = min(len(profile) - 1, int(pos + window))
        segment = profile[lo:hi + 1]
        mean_val = segment.mean()
        local_max_idx = np.argmax(segment)
        local_max_val = segment[local_max_idx]
        if local_max_val > mean_val * cfg.walker_strength_threshold:
            cuts.append(lo + local_max_idx)
        else:
            cuts.append(int(round(pos)))
        pos = cuts[-1] + step_size
    return cuts


def stabilize_both_axes(
    col_cuts: list[int], row_cuts: list[int], w: int, h: int, cfg: PixelSnapConfig
) -> tuple[list[int], list[int]]:
    """両軸のグリッドを安定化させる。"""
    def uniform_cuts(size: int, n: int) -> list[int]:
        return [int(round(i * size / n)) for i in range(n + 1)]

    def _stabilize(cuts: list[int], size: int) -> list[int]:
        if len(cuts) < cfg.min_cuts_per_axis + 1:
            return uniform_cuts(size, cfg.fallback_target_segments)
        return cuts

    col_cuts = _stabilize(col_cuts, w)
    row_cuts = _stabilize(row_cuts, h)

    if len(col_cuts) > 1 and len(row_cuts) > 1:
        x_step = (col_cuts[-1] - col_cuts[0]) / (len(col_cuts) - 1)
        y_step = (row_cuts[-1] - row_cuts[0]) / (len(row_cuts) - 1)
        ratio = max(x_step, y_step) / (min(x_step, y_step) + 1e-9)
        if ratio > cfg.max_step_ratio:
            avg_step = (x_step + y_step) / 2
            n_x = max(1, int(round(w / avg_step)))
            n_y = max(1, int(round(h / avg_step)))
            col_cuts = uniform_cuts(w, n_x)
            row_cuts = uniform_cuts(h, n_y)

    return col_cuts, row_cuts


def resample(img_array: np.ndarray, col_cuts: list[int], row_cuts: list[int]) -> np.ndarray:
    """グリッドセルごとに最頻色でリサンプリング（ダウンスケール）する。"""
    n_cols = len(col_cuts) - 1
    n_rows = len(row_cuts) - 1
    out = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)

    for r in range(n_rows):
        r0, r1 = row_cuts[r], row_cuts[r + 1]
        for c in range(n_cols):
            c0, c1 = col_cuts[c], col_cuts[c + 1]
            cell = img_array[r0:r1, c0:c1].reshape(-1, 3)
            if len(cell) == 0:
                continue
            # 最頻色を選択
            unique, counts = np.unique(cell, axis=0, return_counts=True)
            out[r, c] = unique[np.argmax(counts)]

    return out


def pixel_snap(image: Image.Image, cfg: PixelSnapConfig | None = None) -> Image.Image:
    """
    ピクセルアートスナッパーのメイン処理。
    入力: PIL Image (RGB)
    出力: ピクセルアート化されたPIL Image (RGB)
    """
    if cfg is None:
        cfg = PixelSnapConfig()

    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]

    # グレースケール変換（ITU-R BT.601）
    gray = (0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]).astype(np.uint8)

    # カラー量子化
    pixels_f = img_array.reshape(-1, 3).astype(np.float64) / 255.0
    quantized_f = quantize_image(pixels_f, cfg)
    quantized = (quantized_f * 255).clip(0, 255).astype(np.uint8).reshape(h, w, 3)

    # エッジプロファイル計算
    col_profile, row_profile = compute_profiles(gray)

    # グリッドステップ推定
    col_step = estimate_step_size(col_profile, cfg)
    row_step = estimate_step_size(row_profile, cfg)

    # グリッド探索
    if col_step:
        col_cuts = walk(col_profile, col_step, cfg)
    else:
        n = max(cfg.min_cuts_per_axis, w // cfg.fallback_target_segments)
        col_cuts = [int(round(i * w / n)) for i in range(n + 1)]

    if row_step:
        row_cuts = walk(row_profile, row_step, cfg)
    else:
        n = max(cfg.min_cuts_per_axis, h // cfg.fallback_target_segments)
        row_cuts = [int(round(i * h / n)) for i in range(n + 1)]

    col_cuts, row_cuts = stabilize_both_axes(col_cuts, row_cuts, w, h, cfg)

    # リサンプリング
    pixel_art_array = resample(quantized, col_cuts, row_cuts)

    # アップスケール
    out_h, out_w = pixel_art_array.shape[:2]
    result = Image.fromarray(pixel_art_array, "RGB")
    if cfg.output_scale > 1:
        result = result.resize(
            (out_w * cfg.output_scale, out_h * cfg.output_scale),
            Image.NEAREST
        )

    return result


def process_image(
    input_path: Path,
    output_dir: Path,
    cfg: PixelSnapConfig | None = None,
) -> Path:
    """ファイルパスを受け取ってピクセルスナップ処理を行い保存する。"""
    if cfg is None:
        cfg = PixelSnapConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_path = output_dir / f"{stem}_snapped.png"

    print(f"[Step 2] ピクセルスナップ処理中: {input_path.name}")
    image = Image.open(input_path)
    result = pixel_snap(image, cfg)
    result.save(output_path, "PNG")
    print(f"  完了: {output_path} ({result.size})")
    return output_path
