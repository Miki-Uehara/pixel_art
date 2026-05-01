"""
Step 3: 白背景を透過処理（高速版）
- scipy.ndimage.label で連結成分分析（Python BFS → C実装に置き換え）
- binary_fill_holes でシルエット内部の白を保護
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label as ndlabel, binary_fill_holes


def remove_white_background(image: Image.Image, tolerance: int = 15) -> Image.Image:
    img_rgba = image.convert("RGBA")
    pixels = np.array(img_rgba, dtype=np.uint8)
    h, w = pixels.shape[:2]

    # 白判定マップ（ベクトル化）
    is_white = np.all(pixels[:, :, :3].astype(np.int16) >= (255 - tolerance), axis=2)

    # 連結成分ラベリング（scipy C実装、Python BFSより大幅高速）
    labeled, _ = ndlabel(is_white)

    # 外周に接するラベル = 背景
    border = set()
    border.update(np.unique(labeled[0, :]))
    border.update(np.unique(labeled[-1, :]))
    border.update(np.unique(labeled[:, 0]))
    border.update(np.unique(labeled[:, -1]))
    border.discard(0)

    bg_mask = np.isin(labeled, list(border))

    # シルエット内部を fill_holes で保護
    foreground_filled = binary_fill_holes(~bg_mask)
    final_bg = ~foreground_filled

    pixels[final_bg, 3] = 0
    return Image.fromarray(pixels, "RGBA")


def process_image(input_path: Path, output_dir: Path, tolerance: int = 15) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_nobg.png"
    print(f"[Step 3] 白背景除去: {input_path.name}")
    result = remove_white_background(Image.open(input_path), tolerance=tolerance)
    result.save(output_path, "PNG")
    print(f"  完了: {output_path}")
    return output_path
