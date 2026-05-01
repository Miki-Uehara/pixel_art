"""線画 + 背景色付き画像を使ったキャラ領域抽出（クリック修正対応）"""

from PIL import Image
import numpy as np


def detect_bg_color(img_bg_painted: Image.Image) -> tuple:
    """背景色塗り画像の四隅から背景色を推定（中央値）。"""
    arr = np.array(img_bg_painted.convert("RGB"))
    h, w = arr.shape[:2]
    pad = max(1, min(h, w) // 50)
    samples = np.concatenate([
        arr[:pad, :pad].reshape(-1, 3),
        arr[:pad, -pad:].reshape(-1, 3),
        arr[-pad:, :pad].reshape(-1, 3),
        arr[-pad:, -pad:].reshape(-1, 3),
    ], axis=0)
    return tuple(int(v) for v in np.median(samples, axis=0))


def line_mask_from_lineart(lineart_img: Image.Image, line_threshold: int = 80) -> np.ndarray:
    """線画から「線のあるピクセル」のboolマスクを返す。
    RGBA画像でアルファが意味を持てばアルファを、そうでなければ輝度（黒い=線）を使う。"""
    img = lineart_img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    if int(alpha.min()) == int(alpha.max()):
        # アルファ情報なし → 黒線on白背景として輝度から推定
        gray = np.array(img.convert("L"))
        line_strength = 255 - gray  # 暗いほど線
    else:
        line_strength = alpha
    return line_strength >= int(line_threshold)


def build_regions(is_line: np.ndarray):
    """線画で区切られた非線領域をラベリング。 (labels, num) を返す。"""
    non_line = ~is_line
    try:
        from scipy.ndimage import label
        labels, num = label(non_line)
        return labels.astype(np.int32), int(num)
    except ImportError:
        # BFS フォールバック
        h, w = non_line.shape
        labels = np.zeros((h, w), dtype=np.int32)
        cur = 0
        from collections import deque
        for sy in range(h):
            for sx in range(w):
                if non_line[sy, sx] and labels[sy, sx] == 0:
                    cur += 1
                    labels[sy, sx] = cur
                    q = deque([(sy, sx)])
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and non_line[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = cur
                                q.append((ny, nx))
        return labels, cur


def initial_classify(labels: np.ndarray, num_regions: int,
                     img_orig: Image.Image,
                     img_bg_painted: Image.Image,
                     diff_threshold: int = 40) -> np.ndarray:
    """各領域について「キャラ(True) or 背景(False)」を判定。
    ②大元 と ③背景色塗り画像 のピクセル差分が大きい部分=背景塗り変えられた部分。
    領域内のそうしたピクセル比率が過半数なら背景判定。"""
    arr_o = np.array(img_orig.convert("RGB")).astype(np.int32)
    arr_b = np.array(img_bg_painted.convert("RGB")).astype(np.int32)
    diff = np.abs(arr_o - arr_b).max(axis=2)
    is_bg_pixel = diff >= int(diff_threshold)

    flat_labels = labels.ravel()
    flat_bg = is_bg_pixel.ravel().astype(np.int64)

    total = np.bincount(flat_labels, minlength=num_regions + 1)
    bg_count = np.bincount(flat_labels, weights=flat_bg, minlength=num_regions + 1)

    region_is_char = np.ones(num_regions + 1, dtype=bool)
    valid = total > 0
    ratio = np.zeros_like(bg_count, dtype=np.float64)
    ratio[valid] = bg_count[valid] / total[valid]
    region_is_char[valid] = ratio[valid] < 0.5
    region_is_char[0] = True
    return region_is_char


def absorb_enclosed_islands(labels: np.ndarray, num_regions: int,
                             region_is_char: np.ndarray,
                             max_island_size: int = 2000) -> np.ndarray:
    """画像境界に接していない背景領域=線画に囲まれた閉じ島を、キャラ判定に変換する。
    境界に接する領域（真の外側背景）は絶対に触らない。
    max_island_size: この値以下のサイズの閉じ島のみ吸収（0なら無効、-1なら全サイズ）。
    """
    if max_island_size == 0:
        return region_is_char

    h, w = labels.shape
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)

    sizes = np.bincount(labels.ravel(), minlength=num_regions + 1)

    new_region = region_is_char.copy()
    for rid in range(1, num_regions + 1):
        if new_region[rid]:
            continue  # 既にキャラ
        if rid in border_labels:
            continue  # 境界に接している＝真の外側
        if max_island_size > 0 and sizes[rid] > max_island_size:
            continue  # 大きすぎる閉じ島は意図的なホールかもなので残す
        new_region[rid] = True  # 線画内の閉じ島 → キャラに吸収
    return new_region


def make_mask(labels: np.ndarray, region_is_char: np.ndarray,
              is_line: np.ndarray) -> np.ndarray:
    """最終的なキャラマスク（True=キャラ）。線画自身もキャラに含める。"""
    char_pixel = region_is_char[labels]
    return char_pixel | is_line


def render_basecoat(is_line: np.ndarray, region_is_char: np.ndarray,
                    labels: np.ndarray,
                    fill_color=(255, 136, 204),
                    line_color=(20, 20, 20)) -> Image.Image:
    """下塗りRGBA画像を生成。
    - 線画ピクセル: 線色で不透明
    - キャラ領域: 塗り色で不透明
    - 背景領域: 透明
    """
    h, w = is_line.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    char_pixel = region_is_char[labels]
    only_char = char_pixel & ~is_line

    fc = np.asarray(fill_color, dtype=np.uint8)
    lc = np.asarray(line_color, dtype=np.uint8)

    rgba[only_char, 0] = fc[0]
    rgba[only_char, 1] = fc[1]
    rgba[only_char, 2] = fc[2]
    rgba[only_char, 3] = 255

    rgba[is_line, 0] = lc[0]
    rgba[is_line, 1] = lc[1]
    rgba[is_line, 2] = lc[2]
    rgba[is_line, 3] = 255

    return Image.fromarray(rgba, "RGBA")


def toggle_region_at(region_is_char: np.ndarray, labels: np.ndarray,
                     x: int, y: int) -> np.ndarray:
    """(x,y) をクリック → その連結領域のキャラ/背景を反転。"""
    h, w = labels.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    rid = int(labels[y, x])
    if rid == 0:
        return region_is_char  # 線画上クリックは無視
    new = region_is_char.copy()
    new[rid] = not bool(new[rid])
    return new


def apply_mask_as_alpha(img_orig: Image.Image, mask: np.ndarray) -> Image.Image:
    """大元画像にマスクをアルファとして適用（背景透過版）。"""
    base = img_orig.convert("RGBA")
    w, h = base.size
    if mask.shape != (h, w):
        m_img = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize((w, h), Image.NEAREST)
        mask_full = np.array(m_img) >= 128
    else:
        mask_full = mask
    arr = np.array(base)
    arr[~mask_full, 3] = 0
    return Image.fromarray(arr, "RGBA")


def pixelize_with_mask(img_orig: Image.Image, mask: np.ndarray,
                       pixel_snap_fn, dots: int, scale: int, colors: int) -> Image.Image:
    """大元をピクセル化し、マスクをダウンサンプルしてアルファとして合成。
    pixel_snap_fn は (PIL.Image, dots, scale, colors) -> PIL.Image (RGB) を返す関数。"""
    snapped = pixel_snap_fn(img_orig, dots, scale, colors).convert("RGBA")
    sw, sh = snapped.size

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    iw, ih = mask_img.size
    dw = int(dots)
    dh = max(4, round(dw * ih / iw))
    small = mask_img.resize((dw, dh), Image.BILINEAR)
    small_arr = np.array(small)
    small_bin = np.where(small_arr >= 128, 255, 0).astype(np.uint8)
    big = Image.fromarray(small_bin, "L").resize((sw, sh), Image.NEAREST)

    arr = np.array(snapped)
    alpha_arr = np.array(big)
    arr[:, :, 3] = alpha_arr
    return Image.fromarray(arr, "RGBA")
