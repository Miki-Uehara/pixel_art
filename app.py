#!/usr/bin/env python3
"""
ピクセルアート生成システム - Game UI
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
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    s = np.where(maxc > 1e-6, delta / maxc, 0.0)
    h = np.zeros_like(r)
    nz = delta > 1e-6
    mr, mg, mb = nz & (maxc == r), nz & (maxc == g), nz & (maxc == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4.0
    return np.stack([(h / 6.0) % 1.0, s, maxc], axis=2)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    h6 = h * 6.0
    i = h6.astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p, q, t = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    for idx, (rv, gv, bv) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m = i == idx
        rgb[m,0]=rv[m]; rgb[m,1]=gv[m]; rgb[m,2]=bv[m]
    return rgb


def adjust_colors(image: Image.Image, saturation=1.0, brightness=1.0,
                  contrast=1.0, hue_shift=0.0) -> Image.Image:
    img = image.convert("RGB")
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if hue_shift != 0.0:
        arr = np.array(img).astype(np.float32) / 255.0
        hsv = _rgb_to_hsv(arr)
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift / 360.0) % 1.0
        img = Image.fromarray((_hsv_to_rgb(hsv)*255).clip(0,255).astype(np.uint8), "RGB")
    return img


# ─── ユーティリティ ────────────────────────────────────

def checkerboard(w: int, h: int, tile: int = 12) -> np.ndarray:
    xs, ys = np.arange(w)//tile, np.arange(h)//tile
    v = np.where((xs[None,:]+ys[:,None])%2, 255, 190).astype(np.uint8)
    return np.stack([v,v,v], axis=2)


def composite_checker(img_rgba: Image.Image) -> Image.Image:
    arr = np.array(img_rgba.convert("RGBA"))
    h, w = arr.shape[:2]
    board = checkerboard(w, h)
    alpha = arr[:,:,3:4].astype(np.float32)/255.0
    rgb = (arr[:,:,:3].astype(np.float32)*alpha + board*(1-alpha)).clip(0,255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def to_pil(img) -> Image.Image:
    return Image.fromarray(img) if isinstance(img, np.ndarray) else img


def size_label(image, dot_count: int, display_scale: int) -> str:
    if image is None:
        return "► 画像をロードするとサイズが表示されます"
    img = to_pil(image)
    iw, ih = img.size
    dh = max(4, round(dot_count * ih / iw))
    return (f"INPUT  {iw}×{ih}px"
            f"  ──►  DOTS  {dot_count}×{dh}"
            f"  ──►  OUTPUT  {dot_count*display_scale}×{dh*display_scale}px  (×{display_scale})")


def run_snap(img: Image.Image, dot_count, display_scale, k_colors) -> Image.Image:
    cfg = PixelSnapConfig(
        k_colors=int(k_colors), output_scale=int(display_scale),
        fallback_target_segments=int(dot_count), min_cuts_per_axis=4,
    )
    return pixel_snap(img, cfg)


def make_final(snapped, saturation, brightness, contrast, hue_shift, do_bg, tol) -> Image.Image:
    adj = adjust_colors(snapped, float(saturation), float(brightness), float(contrast), float(hue_shift))
    return remove_white_background(adj, int(tol)) if do_bg else adj.convert("RGBA")


def save_png(rgba: Image.Image, prefix="pixel_art") -> str:
    p = FINAL_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    rgba.save(p, "PNG")
    return str(p)


# ─── ハンドラ ──────────────────────────────────────────

def on_snap(image, dot_count, display_scale, k_colors,
            saturation, brightness, contrast, hue_shift, do_bg, tol):
    if image is None:
        return None, None, "[ IMAGE NOT LOADED ]", gr.update(visible=False), None
    img = to_pil(image)
    snapped = run_snap(img, dot_count, display_scale, k_colors)
    rgba = make_final(snapped, saturation, brightness, contrast, hue_shift, do_bg, tol)
    preview = composite_checker(rgba)
    iw, ih = img.size
    dh = max(4, round(int(dot_count)*ih/iw))
    info = (f"DOTS: {int(dot_count)}×{dh}  |  "
            f"OUTPUT: {int(dot_count)*int(display_scale)}×{dh*int(display_scale)}px  |  "
            f"COLORS: {int(k_colors)}  |  BG ERASE: {'ON' if do_bg else 'OFF'}")
    path = save_png(rgba)
    return snapped, preview, info, gr.update(visible=True, value=path), snapped


def on_color(snapped_state, saturation, brightness, contrast, hue_shift, do_bg, tol):
    if snapped_state is None:
        return None
    rgba = make_final(to_pil(snapped_state), saturation, brightness, contrast, hue_shift, do_bg, tol)
    return composite_checker(rgba)


def on_color_save(snapped_state, saturation, brightness, contrast, hue_shift, do_bg, tol):
    if snapped_state is None:
        return gr.update(visible=False), "[ NO DATA — RUN FIRST ]"
    rgba = make_final(to_pil(snapped_state), saturation, brightness, contrast, hue_shift, do_bg, tol)
    path = save_png(rgba)
    return gr.update(visible=True, value=path), f"SAVED ► {Path(path).name}"


def on_compare(image, k_colors, display_scale, saturation, brightness, contrast, hue_shift, do_bg, tol):
    if image is None:
        return [], "[ IMAGE NOT LOADED ]"
    img = to_pil(image)
    results = []
    for dot in [64, 96, 128, 192, 256]:
        snapped = run_snap(img, dot, int(display_scale), k_colors)
        rgba = make_final(snapped, saturation, brightness, contrast, hue_shift, do_bg, tol)
        iw, ih = img.size
        dh = max(4, round(dot*ih/iw))
        results.append((composite_checker(rgba),
                         f"{dot}dots → {dot*int(display_scale)}×{dh*int(display_scale)}px"))
    return results, f"5 PATTERNS READY  /  {int(k_colors)} COLORS  /  ×{int(display_scale)} SCALE"


def on_compare_save(image, save_dot, display_scale, k_colors,
                    saturation, brightness, contrast, hue_shift, do_bg, tol):
    if image is None:
        return gr.update(visible=False), "[ NO IMAGE ]"
    rgba = make_final(run_snap(to_pil(image), int(save_dot), int(display_scale), k_colors),
                      saturation, brightness, contrast, hue_shift, do_bg, tol)
    path = save_png(rgba, prefix=f"pixel_{int(save_dot)}dot")
    return gr.update(visible=True, value=path), f"SAVED ► {Path(path).name}"


# ─── CSS：ゲームRPGテーマ ──────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323:wght@400&display=swap');

/* ── ベース ── */
body, .gradio-container, .main {
    background: #080818 !important;
    color: #c8d8ff !important;
}
.gradio-container {
    background:
        radial-gradient(ellipse 80% 40% at 50% 0%, #1a1060 0%, transparent 70%),
        radial-gradient(ellipse 60% 30% at 80% 100%, #0a2040 0%, transparent 60%),
        #080818 !important;
}

/* ── タイトル ── */
h1 {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 18px !important;
    color: #00e5ff !important;
    text-shadow: 0 0 8px #00aaff, 0 0 20px #0055ff88 !important;
    text-align: center !important;
    letter-spacing: 3px !important;
    padding: 16px 0 8px !important;
}
h2, h3 {
    font-family: 'VT323', monospace !important;
    color: #88bbff !important;
    letter-spacing: 2px !important;
}

/* ── パネル ── */
.gr-panel, .gr-box, .panel-box {
    background: #0c0c28 !important;
    border: 1px solid #2244aa !important;
    box-shadow: 0 0 12px #1133aa33, inset 0 0 30px #00001488 !important;
    border-radius: 4px !important;
}

/* ── ラベル ── */
label span, .gr-form label {
    font-family: 'VT323', monospace !important;
    font-size: 15px !important;
    color: #6699dd !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── スライダー ── */
input[type=range] { accent-color: #00ccff !important; }

/* ── テキストボックス ── */
textarea, input[type=text], .gr-textbox textarea {
    font-family: 'VT323', monospace !important;
    font-size: 15px !important;
    background: #060614 !important;
    border: 1px solid #224488 !important;
    color: #88ccff !important;
    border-radius: 2px !important;
}

/* ── ボタン（プライマリ）── */
button.primary, .gr-button-primary {
    font-family: 'VT323', monospace !important;
    font-size: 20px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    background: linear-gradient(180deg, #3311bb 0%, #220099 50%, #110077 100%) !important;
    border: 2px solid #6644ff !important;
    box-shadow: 4px 4px 0 #110044, 0 0 16px #4422ff44 !important;
    color: #ddaaff !important;
    border-radius: 2px !important;
    transition: all 0.08s !important;
    padding: 10px 28px !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: linear-gradient(180deg, #5533dd 0%, #3311bb 100%) !important;
    color: #ffffff !important;
    box-shadow: 4px 4px 0 #110044, 0 0 24px #6644ff88 !important;
}
button.primary:active, .gr-button-primary:active {
    transform: translate(4px, 4px) !important;
    box-shadow: 0 0 8px #4422ff44 !important;
}

/* ── ボタン（セカンダリ）── */
button.secondary, .gr-button-secondary {
    font-family: 'VT323', monospace !important;
    font-size: 17px !important;
    letter-spacing: 2px !important;
    background: #0a0a22 !important;
    border: 2px solid #334477 !important;
    box-shadow: 3px 3px 0 #000010 !important;
    color: #6699cc !important;
    border-radius: 2px !important;
    transition: all 0.08s !important;
}
button.secondary:hover {
    border-color: #5577aa !important;
    color: #aaccff !important;
}
button.secondary:active {
    transform: translate(3px, 3px) !important;
    box-shadow: none !important;
}

/* ── タブ ── */
.tab-nav button {
    font-family: 'VT323', monospace !important;
    font-size: 17px !important;
    letter-spacing: 2px !important;
    color: #5577aa !important;
    background: #0a0a20 !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
}

/* ── チェックボックス ── */
input[type=checkbox] { accent-color: #00ccff !important; }

/* ── アコーディオン ── */
.gr-accordion > .label-wrap {
    font-family: 'VT323', monospace !important;
    font-size: 17px !important;
    color: #55aadd !important;
    letter-spacing: 2px !important;
    background: #0c0c28 !important;
    border: 1px solid #223366 !important;
    padding: 8px 14px !important;
}

/* ── サイズ情報 ── */
.size-info p {
    font-family: 'VT323', monospace !important;
    font-size: 17px !important;
    color: #ffcc44 !important;
    background: #0c0c1c !important;
    border: 1px solid #554400 !important;
    border-left: 4px solid #ffaa00 !important;
    padding: 6px 12px !important;
    border-radius: 2px !important;
    letter-spacing: 1px !important;
}

/* ── ギャラリー ── */
.gallery-item { border: 1px solid #224488 !important; }

/* ── 画像ウィジェット ── */
.image-container {
    border: 1px solid #1a2a4a !important;
    background: #06060f !important;
}

/* ── footer非表示 ── */
footer { display: none !important; }
"""

# ─── UI ───────────────────────────────────────────────

with gr.Blocks(title="PIXEL ART GENERATOR") as demo:

    snapped_state = gr.State(None)

    gr.Markdown("# ◈  PIXEL ART  GENERATOR  ◈")
    gr.Markdown(
        "<center style='font-family:VT323,monospace;color:#445588;"
        "letter-spacing:2px;font-size:16px'>"
        "AI ILLUSTRATION  ──►  PIXEL ART  ──►  TRANSPARENT PNG"
        "</center>"
    )

    # ── INPUT + 基本設定 ───────────────────────────────
    gr.Markdown("### ▸ LOAD & SETTINGS")
    with gr.Row():
        input_image = gr.Image(type="pil", label="◈ INPUT FILE", height=230, scale=1)
        with gr.Column(scale=3):
            size_info = gr.Markdown("► 画像をロードするとサイズが表示されます",
                                    elem_classes="size-info")
            with gr.Row():
                dot_count     = gr.Slider(32, 512, value=64, step=8,
                                  label="◈ DOTS  ( less = chunky pixel / more = fine detail )")
                display_scale = gr.Slider(1, 16, value=1, step=1,
                                  label="◈ SCALE  ( px per dot )")
            k_colors = gr.Slider(4, 64, value=16, step=4,
                         label="◈ PALETTE  ( colors — less = more retro )")
            with gr.Row():
                do_remove_bg = gr.Checkbox(value=True, label="◈ BG ERASE  白背景を透過")
                bg_tolerance = gr.Slider(0, 60, value=15, step=1,
                               label="◈ BG TOLERANCE  ( wider = remove more )")

    for comp in [input_image, dot_count, display_scale]:
        comp.change(fn=size_label,
                    inputs=[input_image, dot_count, display_scale],
                    outputs=size_info)

    # ── カラーチューン（アコーディオン）──────────────
    with gr.Accordion("▸ COLOR TUNE  ( adjust after conversion )", open=False):
        gr.Markdown(
            "<span style='font-family:VT323,monospace;color:#446688;"
            "font-size:14px;letter-spacing:1px'>"
            "変換後にスライダーを動かすと即座にプレビューが更新されます</span>"
        )
        with gr.Row():
            saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.05,
                           label="SATURATION  彩度  ( 1.0 = original )")
            brightness = gr.Slider(0.2, 2.0, value=1.0, step=0.05,
                           label="BRIGHTNESS  明度  ( 1.0 = original )")
        with gr.Row():
            contrast   = gr.Slider(0.2, 3.0, value=1.0, step=0.05,
                           label="CONTRAST  コントラスト  ( 1.0 = original )")
            hue_shift  = gr.Slider(-180, 180, value=0, step=5,
                           label="HUE SHIFT  色相回転  ( deg )")

    color_inputs = [saturation, brightness, contrast, hue_shift, do_remove_bg, bg_tolerance]

    gr.Markdown("---")

    # ── タブ ──────────────────────────────────────────
    with gr.Tabs():

        # タブ1: 1枚変換
        with gr.TabItem("▸ CONVERT"):
            run_btn = gr.Button("▶  EXECUTE  変換実行", variant="primary", size="lg")

            with gr.Row():
                out_snap  = gr.Image(type="pil", label="◈ PIXEL SNAP",
                                     height=340, image_mode="RGB")
                out_final = gr.Image(type="pil", label="◈ FINAL  ( checker = transparent )",
                                     height=340, image_mode="RGB")

            info_box = gr.Textbox(label="◈ STATUS", lines=2, interactive=False)

            with gr.Row():
                dl_btn         = gr.DownloadButton("▼ DOWNLOAD  PNG", variant="primary", visible=False)
                save_color_btn = gr.Button("◈ SAVE WITH COLOR TUNE", variant="secondary")
                dl_btn2        = gr.DownloadButton("▼ DOWNLOAD  COLOR TUNED",
                                                    variant="secondary", visible=False)
            save_info = gr.Textbox(label="", lines=1, interactive=False)

            run_btn.click(
                fn=on_snap,
                inputs=[input_image, dot_count, display_scale, k_colors] + color_inputs,
                outputs=[out_snap, out_final, info_box, dl_btn, snapped_state],
            )
            for sl in color_inputs:
                sl.change(fn=on_color,
                          inputs=[snapped_state] + color_inputs,
                          outputs=out_final)
            save_color_btn.click(
                fn=on_color_save,
                inputs=[snapped_state] + color_inputs,
                outputs=[dl_btn2, save_info],
            )

        # タブ2: 5パターン比較
        with gr.TabItem("▸ COMPARE  5 PATTERNS"):
            gr.Markdown(
                "<span style='font-family:VT323,monospace;color:#446688;font-size:16px'>"
                "DOTS: 64 / 96 / 128 / 192 / 256  ──  自動で5パターン生成"
                "</span>"
            )
            compare_btn  = gr.Button("▶  GENERATE  5 PATTERNS", variant="primary", size="lg")
            compare_info = gr.Textbox(label="◈ STATUS", lines=1, interactive=False)

            gallery = gr.Gallery(label="◈ PATTERN GALLERY  ( click to zoom )",
                                 columns=5, height=360, object_fit="contain")

            gr.Markdown(
                "<span style='font-family:VT323,monospace;color:#446688;font-size:16px'>"
                "▸ SELECT & SAVE"
                "</span>"
            )
            with gr.Row():
                save_dot      = gr.Dropdown(choices=[64, 96, 128, 192, 256], value=128,
                                            label="◈ DOTS TO SAVE", type="value")
                comp_save_btn = gr.Button("◈ SAVE SELECTED", variant="secondary")
            comp_dl   = gr.DownloadButton("▼ DOWNLOAD", variant="secondary", visible=False)
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
        theme=gr.themes.Base(),
    )
