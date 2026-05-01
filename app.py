#!/usr/bin/env python3
"""
ピクセルアート生成システム - Web UI
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image
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


# ─── ユーティリティ ──────────────────────────────────────

def checkerboard_bg(width: int, height: int, tile: int = 16) -> np.ndarray:
    """透過プレビュー用チェッカーボード背景を生成する。"""
    xs = np.arange(width) // tile
    ys = np.arange(height) // tile
    grid = (xs[None, :] + ys[:, None]) % 2
    board = np.where(grid[:, :, None], 255, 204).astype(np.uint8)
    return np.stack([board[:, :, 0], board[:, :, 0], board[:, :, 0]], axis=2)


def composite_on_checker(img_rgba: Image.Image, tile: int = 16) -> Image.Image:
    """RGBA画像をチェッカーボード上に合成してRGB画像として返す（透過が見えるように）。"""
    arr = np.array(img_rgba.convert("RGBA"))
    h, w = arr.shape[:2]
    board = checkerboard_bg(w, h, tile)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)
    result = (rgb * alpha + board * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGB")


def pil_or_array(image) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


def calc_size_info(input_img, dot_count: int, display_scale: int) -> str:
    if input_img is None:
        return "画像をアップロードすると出力サイズが表示されます"
    img = pil_or_array(input_img)
    w, h = img.size
    dot_w = dot_count
    dot_h = int(dot_count * h / w) if w > 0 else dot_count
    out_w = dot_w * display_scale
    out_h = dot_h * display_scale
    return (
        f"入力: {w} × {h} px  →  "
        f"ドット数: {dot_w} × {dot_h}  →  "
        f"出力: {out_w} × {out_h} px（×{display_scale} 表示）"
    )


# ─── 処理関数 ──────────────────────────────────────────

def run_snap(img: Image.Image, dot_count: int, display_scale: int, k_colors: int) -> Image.Image:
    w, h = img.size
    dot_h = max(4, int(dot_count * h / w))
    cfg = PixelSnapConfig(
        k_colors=int(k_colors),
        output_scale=int(display_scale),
        fallback_target_segments=int(dot_count),
        min_cuts_per_axis=4,
        peak_threshold_multiplier=0.2,
    )
    return pixel_snap(img, cfg)


def process_single(image, dot_count, display_scale, k_colors, do_remove_bg, bg_tolerance):
    """メイン処理：1枚変換。"""
    if image is None:
        return None, None, "画像をアップロードしてください。", gr.update(visible=False)

    img = pil_or_array(image)
    snapped = run_snap(img, int(dot_count), int(display_scale), int(k_colors))

    if do_remove_bg:
        final_rgba = remove_white_background(snapped, tolerance=int(bg_tolerance))
    else:
        final_rgba = snapped.convert("RGBA")

    preview = composite_on_checker(final_rgba)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = FINAL_DIR / f"pixel_art_{date_str}.png"
    final_rgba.save(save_path, "PNG")

    dot_h = max(4, int(int(dot_count) * img.size[1] / img.size[0]))
    out_w = int(dot_count) * int(display_scale)
    out_h = dot_h * int(display_scale)
    info = (
        f"入力: {img.size[0]}×{img.size[1]}px  →  "
        f"ドット: {int(dot_count)}×{dot_h}  →  "
        f"出力: {out_w}×{out_h}px\n"
        f"カラー: {int(k_colors)}色  |  背景透過: {'ON' if do_remove_bg else 'OFF'}\n"
        f"保存: {save_path.name}"
    )
    return snapped, preview, info, gr.update(visible=True, value=str(save_path))


def process_compare(image, k_colors, display_scale, do_remove_bg, bg_tolerance):
    """5パターン比較：ドット数を変えて並べる。"""
    if image is None:
        return [], "画像をアップロードしてください。"

    img = pil_or_array(image)
    dot_presets = [16, 32, 48, 64, 96]
    results = []

    for dot_count in dot_presets:
        snapped = run_snap(img, dot_count, int(display_scale), int(k_colors))
        if do_remove_bg:
            final_rgba = remove_white_background(snapped, tolerance=int(bg_tolerance))
        else:
            final_rgba = snapped.convert("RGBA")
        preview = composite_on_checker(final_rgba)

        dot_h = max(4, int(dot_count * img.size[1] / img.size[0]))
        out_w = dot_count * int(display_scale)
        out_h = dot_h * int(display_scale)
        label = f"{dot_count}ドット → {out_w}×{out_h}px"
        results.append((preview, label))

    return results, f"5パターン生成完了（カラー{int(k_colors)}色 / ×{int(display_scale)}表示）"


def save_compare_item(image, dot_count, display_scale, k_colors, do_remove_bg, bg_tolerance):
    """比較結果から選んで保存。"""
    if image is None:
        return gr.update(visible=False), "画像がありません"
    img = pil_or_array(image)
    snapped = run_snap(img, int(dot_count), int(display_scale), int(k_colors))
    if do_remove_bg:
        final_rgba = remove_white_background(snapped, tolerance=int(bg_tolerance))
    else:
        final_rgba = snapped.convert("RGBA")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = FINAL_DIR / f"pixel_art_{date_str}.png"
    final_rgba.save(save_path, "PNG")
    return gr.update(visible=True, value=str(save_path)), f"保存完了: {save_path.name}"


# ─── UI ──────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.size-info { background: #f0f4ff; border-radius: 8px; padding: 8px 12px;
             font-size: 0.95em; border-left: 3px solid #4a90d9; }
"""

with gr.Blocks(title="ピクセルアート生成システム") as demo:

    gr.Markdown("# ピクセルアート生成システム")

    # ── 共通設定（タブ間で共有） ──
    with gr.Row():
        input_image = gr.Image(type="pil", label="入力画像", height=260, scale=1)
        with gr.Column(scale=2):
            gr.Markdown("### 基本設定")

            size_info_box = gr.Markdown(
                "画像をアップロードすると出力サイズが表示されます",
                elem_classes="size-info",
            )

            with gr.Row():
                dot_count = gr.Slider(8, 128, value=64, step=4,
                    label="ドット数（横）- 少ない→粗いドット絵 / 多い→細かい")
                display_scale = gr.Slider(1, 8, value=1, step=1,
                    label="表示倍率 - 1ドットを何px表示するか")
            k_colors = gr.Slider(4, 64, value=16, step=4,
                label="カラーパレット数 - 少ない→よりドット絵らしく")

            with gr.Row():
                do_remove_bg = gr.Checkbox(value=True, label="白背景を透過する")
                bg_tolerance = gr.Slider(0, 60, value=15, step=1,
                    label="背景白の許容誤差（大きい→より広く除去）")

    # サイズ情報をリアルタイム更新
    for comp in [input_image, dot_count, display_scale]:
        comp.change(
            fn=calc_size_info,
            inputs=[input_image, dot_count, display_scale],
            outputs=size_info_box,
        )

    gr.Markdown("---")

    # ── タブ ──
    with gr.Tabs():

        # タブ1: 1枚変換
        with gr.TabItem("1枚変換"):
            run_btn = gr.Button("変換実行", variant="primary", size="lg")

            with gr.Row():
                out_snap = gr.Image(type="pil", label="Step2：ピクセルスナップ",
                                    height=320, image_mode="RGB")
                out_final = gr.Image(type="pil", label="Step3：透過プレビュー（チェッカー背景）",
                                     height=320, image_mode="RGB")

            info_box = gr.Textbox(label="処理情報", lines=3, interactive=False)
            dl_btn = gr.DownloadButton("PNGをダウンロード", variant="secondary", visible=False)

            run_btn.click(
                fn=process_single,
                inputs=[input_image, dot_count, display_scale, k_colors, do_remove_bg, bg_tolerance],
                outputs=[out_snap, out_final, info_box, dl_btn],
            )

        # タブ2: 5パターン比較
        with gr.TabItem("5パターン比較"):
            gr.Markdown(
                "ドット数を **16 / 32 / 48 / 64 / 96** の5段階で自動生成します。"
                "気に入ったものを選んで保存できます。"
            )
            compare_btn = gr.Button("5パターン生成", variant="primary", size="lg")
            compare_info = gr.Textbox(label="", lines=1, interactive=False)

            gallery = gr.Gallery(
                label="比較結果（クリックで拡大）",
                columns=5,
                height=400,
                object_fit="contain",
            )

            gr.Markdown("#### 気に入ったパターンを選んで保存")
            with gr.Row():
                save_dot = gr.Dropdown(
                    choices=[16, 32, 48, 64, 96],
                    value=64,
                    label="保存するドット数",
                    type="value",
                )
                save_btn = gr.Button("選択パターンを保存", variant="secondary")

            save_info = gr.Textbox(label="保存情報", lines=1, interactive=False)
            dl_btn2 = gr.DownloadButton("PNGをダウンロード", variant="secondary", visible=False)

            compare_btn.click(
                fn=process_compare,
                inputs=[input_image, k_colors, display_scale, do_remove_bg, bg_tolerance],
                outputs=[gallery, compare_info],
            )
            save_btn.click(
                fn=save_compare_item,
                inputs=[input_image, save_dot, display_scale, k_colors, do_remove_bg, bg_tolerance],
                outputs=[dl_btn2, save_info],
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
