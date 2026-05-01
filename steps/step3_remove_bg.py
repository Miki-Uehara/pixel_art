"""
Step 3: 白背景を透過処理
- 外周flood fillでキャラクター外側の背景を検出
- binary_fill_holes でシルエット内部を保護（洋服の白など）
"""

from __future__ import annotations
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes


def remove_white_background(
    image: Image.Image,
    tolerance: int = 15,
) -> Image.Image:
    """
    外周flood fill → シルエット内部をfill_holesで保護 → 外側のみ透過。

    内側に閉じられた白（服の白い部分など）は透過しない。
    """
    img_rgba = image.convert("RGBA")
    pixels = np.array(img_rgba, dtype=np.uint8)
    h, w = pixels.shape[:2]

    # 白判定マップ
    rgb = pixels[:, :, :3].astype(np.int16)
    is_white = np.all(rgb >= (255 - tolerance), axis=2)

    # 外周flood fill：外側の背景白だけを検出
    bg_mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def _seed(y: int, x: int) -> None:
        if not visited[y, x] and is_white[y, x]:
            visited[y, x] = True
            bg_mask[y, x] = True
            queue.append((y, x))

    for x in range(w):
        _seed(0, x)
        _seed(h - 1, x)
    for y in range(1, h - 1):
        _seed(y, 0)
        _seed(y, w - 1)

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_white[ny, nx]:
                visited[ny, nx] = True
                bg_mask[ny, nx] = True
                queue.append((ny, nx))

    # キャラクターのシルエット = 背景でない部分
    # binary_fill_holes でシルエット内部の「穴」（内側の白）を埋める
    # → 外側から続いていない白はすべて「内側」として保護される
    foreground = ~bg_mask
    foreground_filled = binary_fill_holes(foreground)

    # 透過対象 = 元の背景 かつ シルエットの外側
    final_bg = ~foreground_filled

    pixels[final_bg, 3] = 0
    return Image.fromarray(pixels, "RGBA")


def process_image(
    input_path: Path,
    output_dir: Path,
    tolerance: int = 15,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_path = output_dir / f"{stem}_nobg.png"
    print(f"[Step 3] 白背景除去中: {input_path.name}  (tolerance={tolerance})")
    image = Image.open(input_path)
    result = remove_white_background(image, tolerance=tolerance)
    result.save(output_path, "PNG")
    print(f"  完了: {output_path}")
    return output_path
