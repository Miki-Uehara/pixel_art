"""
Step 3: 白背景を透過処理
- キャラクター内部の白色要素は保持する（flood fill で外側のみ除去）
"""

from __future__ import annotations
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image


def _is_white(pixel: tuple[int, int, int], tolerance: int = 15) -> bool:
    r, g, b = pixel
    return r >= 255 - tolerance and g >= 255 - tolerance and b >= 255 - tolerance


def remove_white_background(
    image: Image.Image,
    tolerance: int = 15,
) -> Image.Image:
    """
    画像の外周から flood fill で白領域を検出し、透過にする。
    内側（キャラクター内部）の白は保持する。

    Args:
        image: 入力画像（RGB or RGBA）
        tolerance: 白判定の許容誤差（0–255）

    Returns:
        RGBA画像（背景が透過済み）
    """
    img_rgba = image.convert("RGBA")
    pixels = np.array(img_rgba)
    h, w = pixels.shape[:2]

    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=bool)  # True = 背景として除去

    queue: deque[tuple[int, int]] = deque()

    # 四辺のピクセルをシードとして追加
    for x in range(w):
        for y in [0, h - 1]:
            r, g, b = pixels[y, x, :3]
            if not visited[y, x] and _is_white((r, g, b), tolerance):
                queue.append((y, x))
                visited[y, x] = True
                mask[y, x] = True

    for y in range(h):
        for x in [0, w - 1]:
            r, g, b = pixels[y, x, :3]
            if not visited[y, x] and _is_white((r, g, b), tolerance):
                queue.append((y, x))
                visited[y, x] = True
                mask[y, x] = True

    # 4近傍 BFS
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                r, g, b = pixels[ny, nx, :3]
                if _is_white((r, g, b), tolerance):
                    visited[ny, nx] = True
                    mask[ny, nx] = True
                    queue.append((ny, nx))

    # マスク適用：背景白を透過に
    pixels[mask, 3] = 0

    return Image.fromarray(pixels, "RGBA")


def process_image(
    input_path: Path,
    output_dir: Path,
    tolerance: int = 15,
) -> Path:
    """ファイルパスを受け取って白背景除去を行い保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_path = output_dir / f"{stem}_nobg.png"

    print(f"[Step 3] 白背景除去中: {input_path.name}  (tolerance={tolerance})")
    image = Image.open(input_path)
    result = remove_white_background(image, tolerance=tolerance)
    result.save(output_path, "PNG")
    print(f"  完了: {output_path}")
    return output_path
