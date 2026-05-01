#!/usr/bin/env python3
"""
ピクセルアート生成システム - Web UI
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image, ImageEnhance
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from steps.step2_pixel_snap import pixel_snap, PixelSnapConfig
from steps.step3_remove_bg import remove_white_background

FINAL_DIR = Path(__file__).parent / "output" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ─── カラー調整 ────────────────────────────────────────

def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """(H,W,3) float32 [0,1] → HSV (H,W,3) float32"""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 1e-6, delta / maxc, 0.0)
    h = np.zeros_like(r)
    nz = delta > 1e-6
    mr = nz & (maxc == r)
    mg = nz & (maxc == g)
    mb = nz & (maxc == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4.0
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=2)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """(H,W,3) float32 HSV → RGB float32"""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h6 = h * 6.0
    i = h6.astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    for idx, (rv, gv, bv) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                          (p, q, v), (t, p, v), (v, p, q)]):
        m = i == idx
        rgb[m, 0] = rv[m]; rgb[m, 1] = gv[m]; rgb[m, 2] = bv[m]
    return rgb


def adjust_colors(
    image: Image.Image,
    saturation: float = 1.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    hue_shift: float = 0.0,
) -> Image.Image:
    """彩度・明度・コントラスト・色相シフトを適用する。"""
    img = image.convert("RGB")

    # 彩度
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    # 明度
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    # コントラスト
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    # 色相シフト
    if hue_shift != 0.0:
        arr = np.array(img).astype(np.float32) / 255.0
        hsv = _rgb_to_hsv(arr)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 360.0) % 1.0
        arr2 = _hsv_to_rgb(hsv)
        img = Image.fromarray((arr2 * 255).clip(0, 255).astype(np.uint8), "RGB")

    return img


# ─── ユーティリティ ────────────────────────────────────

def checkerboard_bg(w: int, h: int, tile: int = 16) -> np.ndarray:
    xs = np.arange(w) // tile
    ys = np.arange(h) // tile
    grid = (xs[None, :] + ys[:, None]) % 2
    v = np.where(grid, 255, 200).astype(np.uint8)
    return np.stack([v, v, v], axis=2)


def composite_on_checker(img_rgba: Image.Image) -> Image.Image:
    arr = np.array(img_rgba.convert("RGBA"))
    h, w = arr.shape[:2]
    board = checkerboard_bg(w, h)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)
    result = (rgb * alpha + board * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGB")


def to_pil(image) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


def calc_size_label(image, dot_count: int, display_scale: int) -> str:
    if image is None:
        return "画像をアップロードするとサイズが表示されます"
    img = to_pil(image)
    iw, ih = img.size
    dot_h = max(4, round(dot_count * ih / iw))
    out_w, out_h = dot_count * display_scale, dot_h * display_scale
    return (
        f"入力: **{iw}×{ih}px**　→　"
        f"ドット数: **{dot_count}×{dot_h}**　→　"
        f"出力: **{out_w}×{out_h}px**（×{display_scale} 表示）"
    )


# ─── スナップ処理 ──────────────────────────────────────

def run_snap(img: Image.Image, dot_count: int, display_scale: int, k_colors: int) -> Image.Image:
    iw, ih = img.size
    cfg = PixelSnapConfig(
        k_colors=int(k_colors),
        output_scale=int(display_scale),
        fallback_target_segments=int(dot_count),
        min_cuts_per_axis=4,
        peak_threshold_multiplier=0.2,
    )
    return pixel_snap(img, cfg)


def make_final(snapped: Image.Image, color_params: dict, do_remove_bg: bool, bg_tolerance: int):
    adjusted = adjust_colors(snapped, **color_params)
    if do_remove_bg:
        rgba = remove_white_background(adjusted, tolerance=int(bg_tolerance))
    else:
        rgba = adjusted.convert("RGBA")
    return rgba


def save_png(rgba: Image.Image, prefix: str = "pixel_art") -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = FINAL_DIR / f"{prefix}_{date_str}.png"
    rgba.save(path, "PNG")
    return str(path)


# ─── Gradio ハンドラ ───────────────────────────────────

def on_snap(image, dot_count, display_scale, k_colors,
            saturation, brightness, contrast, hue_shift,
            do_remove_bg, bg_tolerance):
    """変換実行：スナップ→カラー調整→背景透過"""
    if image is None:
        return None, None, "画像をアップロードしてください", gr.update(visible=False), None

    img = to_pil(image)
    snapped = run_snap(img, int(dot_count), int(display_scale), int(k_colors))

    color_params = dict(
        saturation=float(saturation), brightness=float(brightness),
        contrast=float(contrast), hue_shift=float(hue_shift),
    )
    rgba = make_final(snapped, color_params, do_remove_bg, int(bg_tolerance))
    preview = composite_on_checker(rgba)

    iw, ih = img.size
    dot_h = max(4, round(int(dot_count) * ih / iw))
    out_w, out_h = int(dot_count) * int(display_scale), dot_h * int(display_scale)
    info = (
        f"ドット数: {int(dot_count)}×{dot_h}　出力: {out_w}×{out_h}px\n"
        f"カラー: {int(k_colors)}色　背景透過: {'ON' if do_remove_bg else 'OFF'}"
    )
    path = save_png(rgba)
    return snapped, preview, info, gr.update(visible=True, value=path), snapped


def on_color_change(snapped_state, saturation, brightness, contrast, hue_shift,
                    do_remove_bg, bg_tolerance):
    """カラースライダー変更時：スナップ結果に色調整だけ再適用（高速）"""
    if snapped_state is None:
        return None
    snapped = to_pil(snapped_state)
    color_params = dict(
        saturation=float(saturation), brightness=float(brightness),
        contrast=float(contrast), hue_shift=float(hue_shift),
    )
    rgba = make_final(snapped, color_params, do_remove_bg, int(bg_tolerance))
    return composite_on_checker(rgba)


def on_color_save(snapped_state, saturation, brightness, contrast, hue_shift,
                  do_remove_bg, bg_tolerance):
    if snapped_state is None:
        return gr.update(visible=False), "先に変換実行してください"
    snapped = to_pil(snapped_state)
    color_params = dict(
        saturation=float(saturation), brightness=float(brightness),
        contrast=float(contrast), hue_shift=float(hue_shift),
    )
    rgba = make_final(snapped, color_params, do_remove_bg, int(bg_tolerance))
    path = save_png(rgba)
    return gr.update(visible=True, value=path), f"保存: {Path(path).name}"


def on_compare(image, k_colors, display_scale,
               saturation, brightness, contrast, hue_shift,
               do_remove_bg, bg_tolerance):
    if image is None:
        return [], "画像をアップロードしてください"
    img = to_pil(image)
    color_params = dict(
        saturation=float(saturation), brightness=float(brightness),
        contrast=float(contrast), hue_shift=float(hue_shift),
    )
    results = []
    for dot in [64, 96, 128, 192, 256]:
        snapped = run_snap(img, dot, int(display_scale), int(k_colors))
        rgba = make_final(snapped, color_params, do_remove_bg, int(bg_tolerance))
        preview = composite_on_checker(rgba)
        iw, ih = img.size
        dot_h = max(4, round(dot * ih / iw))
        out_w, out_h = dot * int(display_scale), dot_h * int(display_scale)
        results.append((preview, f"{dot}ドット → {out_w}×{out_h}px"))
    return results, f"5パターン生成完了（カラー{int(k_colors)}色 / ×{int(display_scale)}表示）"


def on_compare_save(image, save_dot, display_scale, k_colors,
                    saturation, brightness, contrast, hue_shift,
                    do_remove_bg, bg_tolerance):
    if image is None:
        return gr.update(visible=False), "画像がありません"
    img = to_pil(image)
    snapped = run_snap(img, int(save_dot), int(display_scale), int(k_colors))
    color_params = dict(
        saturation=float(saturation), brightness=float(brightness),
        contrast=float(contrast), hue_shift=float(hue_shift),
    )
    rgba = make_final(snapped, color_params, do_remove_bg, int(bg_tolerance))
    path = save_png(rgba, prefix=f"pixel_{int(save_dot)}dot")
    return gr.update(visible=True, value=path), f"保存: {Path(path).name}"


# ─── UI 定義 ───────────────────────────────────────────

CSS = """
footer { display: none !important; }
.size-label { background:#eef3ff; border-left:4px solid #4a90e2;
              padding:6px 12px; border-radius:4px; font-size:.95em; }
"""

with gr.Blocks(title="ピクセルアート生成システム") as demo:

    snapped_state = gr.State(None)

    gr.Markdown("# ピクセルアート生成システム")

    # ── 共通：画像 + 基本設定 ──────────────────────────
    with gr.Row():
        input_image = gr.Image(type="pil", label="入力画像", height=240, scale=1)
        with gr.Column(scale=3):
            size_label = gr.Markdown("画像をアップロードするとサイズが表示されます",
                                     elem_classes="size-label")
            with gr.Row():
                dot_count    = gr.Slider(32, 512, value=64, step=8,
                                label="ドット数（横）— 少ない→粗い / 多い→細かい")
                display_scale = gr.Slider(1, 16, value=1, step=1,
                                label="表示倍率 — 1ドットを何px表示するか")
            k_colors = gr.Slider(4, 64, value=16, step=4,
                        label="カラーパレット数 — 少ない→よりドット絵らしく")

            with gr.Row():
                do_remove_bg = gr.Checkbox(value=True, label="白背景を透過する")
                bg_tolerance = gr.Slider(0, 60, value=15, step=1,
                               label="背景白の許容誤差（大きい→広く除去）")

    for comp in [input_image, dot_count, display_scale]:
        comp.change(fn=calc_size_label,
                    inputs=[input_image, dot_count, display_scale],
                    outputs=size_label)

    # ── カラー調整（共通スライダー） ───────────────────
    with gr.Accordion("カラー調整（スライダーで色味をリアルタイム変更）", open=False):
        gr.Markdown("「変換実行」後にスライダーを動かすと即座にプレビューが更新されます。")
        with gr.Row():
            saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.05,
                          label="彩度 — 1.0=変化なし / 低い→グレー / 高い→鮮やか")
            brightness = gr.Slider(0.2, 2.0, value=1.0, step=0.05,
                          label="明度 — 1.0=変化なし / 低い→暗く / 高い→明るく")
        with gr.Row():
            contrast   = gr.Slider(0.2, 3.0, value=1.0, step=0.05,
                          label="コントラスト — 1.0=変化なし / 低い→フラット / 高い→メリハリ")
            hue_shift  = gr.Slider(-180, 180, value=0, step=5,
                          label="色相シフト（度）— 0=変化なし / ±で全体の色相を回転")

    gr.Markdown("---")

    color_inputs = [saturation, brightness, contrast, hue_shift, do_remove_bg, bg_tolerance]

    # ── タブ ──────────────────────────────────────────
    with gr.Tabs():

        # タブ1：1枚変換
        with gr.TabItem("1枚変換"):
            run_btn = gr.Button("変換実行", variant="primary", size="lg")

            with gr.Row():
                out_snap  = gr.Image(type="pil", label="ピクセルスナップ結果",
                                     height=340, image_mode="RGB")
                out_final = gr.Image(type="pil",
                                     label="透過プレビュー（チェッカー=透過部分）",
                                     height=340, image_mode="RGB")

            info_box = gr.Textbox(label="処理情報", lines=2, interactive=False)
            dl_btn   = gr.DownloadButton("PNGをダウンロード", variant="secondary", visible=False)
            with gr.Row():
                save_color_btn = gr.Button("現在のカラー設定で保存", variant="secondary")
                dl_btn2 = gr.DownloadButton("保存済みPNGをダウンロード",
                                            variant="secondary", visible=False)
            save_info = gr.Textbox(label="", lines=1, interactive=False)

            run_btn.click(
                fn=on_snap,
                inputs=[input_image, dot_count, display_scale, k_colors] + color_inputs,
                outputs=[out_snap, out_final, info_box, dl_btn, snapped_state],
            )

            # カラースライダー変更 → プレビューのみ更新（高速）
            for sl in color_inputs:
                sl.change(
                    fn=on_color_change,
                    inputs=[snapped_state] + color_inputs,
                    outputs=out_final,
                )

            save_color_btn.click(
                fn=on_color_save,
                inputs=[snapped_state] + color_inputs,
                outputs=[dl_btn2, save_info],
            )

        # タブ2：5パターン比較
        with gr.TabItem("5パターン比較（64 / 96 / 128 / 192 / 256ドット）"):
            gr.Markdown(
                "ドット数を **64 / 96 / 128 / 192 / 256** の5段階で自動生成。"
                "カラー調整も反映されます。"
            )
            compare_btn  = gr.Button("5パターン生成", variant="primary", size="lg")
            compare_info = gr.Textbox(label="", lines=1, interactive=False)

            gallery = gr.Gallery(label="比較結果（クリックで拡大）",
                                 columns=5, height=380, object_fit="contain")

            gr.Markdown("#### 気に入ったものを選んで保存")
            with gr.Row():
                save_dot = gr.Dropdown(choices=[64, 96, 128, 192, 256], value=128,
                                       label="保存するドット数", type="value")
                comp_save_btn = gr.Button("選択を保存", variant="secondary")
            comp_dl   = gr.DownloadButton("PNGをダウンロード", variant="secondary", visible=False)
            comp_info = gr.Textbox(label="", lines=1, interactive=False)

            compare_btn.click(
                fn=on_compare,
                inputs=[input_image, k_colors, display_scale] + color_inputs,
                outputs=[gallery, compare_info],
            )
            comp_save_btn.click(
                fn=on_compare_save,
                inputs=[input_image, save_dot, display_scale, k_colors] + color_inputs,
                outputs=[comp_dl, comp_info],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(),
    )
