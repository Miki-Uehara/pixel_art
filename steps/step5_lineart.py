"""線画ツール: 輝度→透明度変換 & 下塗り"""

from PIL import Image
import numpy as np


def extract_lineart(img: Image.Image, threshold: int = 128) -> Image.Image:
    """
    輝度を透明度に変換して線画を抽出する。
    threshold=0  : スムーズ変換（輝度をそのまま透明度に）
    threshold>0  : 2値化（threshold より暗いピクセルのみ不透明）
    """
    gray = img.convert("RGB").convert("L")
    arr = np.array(gray, dtype=np.uint8)

    if threshold > 0:
        alpha = np.where(arr < threshold, 255, 0).astype(np.uint8)
    else:
        alpha = (255 - arr).astype(np.uint8)

    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.putalpha(Image.fromarray(alpha, "L"))
    return result


def _get_background_mask(alpha_arr: np.ndarray) -> np.ndarray:
    """画像境界から連結する透明領域を背景としてマークする。"""
    transparent = alpha_arr < 128

    try:
        from scipy.ndimage import label
        labeled, _ = label(transparent)
        h, w = transparent.shape
        border_labels: set = set()
        border_labels.update(labeled[0, :].tolist())
        border_labels.update(labeled[-1, :].tolist())
        border_labels.update(labeled[:, 0].tolist())
        border_labels.update(labeled[:, -1].tolist())
        border_labels.discard(0)
        return np.isin(labeled, list(border_labels))

    except ImportError:
        # scipy が無い場合の BFS フォールバック
        from collections import deque
        h, w = transparent.shape
        visited = np.zeros_like(transparent, dtype=bool)
        queue: deque = deque()

        for x in range(w):
            for y in [0, h - 1]:
                if transparent[y, x] and not visited[y, x]:
                    visited[y, x] = True
                    queue.append((y, x))
        for y in range(1, h - 1):
            for x in [0, w - 1]:
                if transparent[y, x] and not visited[y, x]:
                    visited[y, x] = True
                    queue.append((y, x))

        while queue:
            cy, cx = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and transparent[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        return visited


def base_coat(lineart_rgba: Image.Image, color_hex: str = "#ff88cc") -> Image.Image:
    """
    線画の内側（キャラクター領域）を指定色で下塗りする。
    線画の線は変更せず、背景は透過のまま保つ。
    """
    arr = np.array(lineart_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]

    bg_mask = _get_background_mask(alpha)
    interior = (alpha < 128) & ~bg_mask

    hex_str = color_hex.lstrip("#")
    r_val = int(hex_str[0:2], 16)
    g_val = int(hex_str[2:4], 16)
    b_val = int(hex_str[4:6], 16)

    result = arr.copy()
    result[interior, 0] = r_val
    result[interior, 1] = g_val
    result[interior, 2] = b_val
    result[interior, 3] = 255

    return Image.fromarray(result, "RGBA")
