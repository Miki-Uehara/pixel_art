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


def process(
    image,
    k_colors,
    output_scale,
    peak_threshold,
    fallback_segments,
    min_cuts,
    do_remove_bg,
    bg_tolerance,
):
    if image is None:
        return None, None, gr.update(value="画像をアップロードしてください。"), gr.update(visible=False)

    pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image

    # Step 2: ピクセルスナップ
    cfg = PixelSnapConfig(
        k_colors=int(k_colors),
        output_scale=int(output_scale),
        peak_threshold_multiplier=float(peak_threshold),
        min_cuts_per_axis=int(min_cuts),
        fallback_target_segments=int(fallback_segments),
    )
    snapped = pixel_snap(pil_image, cfg)
    grid_w = snapped.size[0] // int(output_scale)
    grid_h = snapped.size[1] // int(output_scale)
    info_snap = f"ピクセルグリッド: {grid_w} × {grid_h} マス  |  パレット: {int(k_colors)} 色  |  出力サイズ: {snapped.size[0]} × {snapped.size[1]} px"

    # Step 3: 背景透過
    if do_remove_bg:
        result = remove_white_background(snapped, tolerance=int(bg_tolerance))
        info_bg = f"背景透過: ON（許容誤差 {int(bg_tolerance)}）"
    else:
        result = snapped.convert("RGBA")
        info_bg = "背景透過: OFF"

    # 保存
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = FINAL_DIR / f"pixel_art_{date_str}.png"
    result.save(save_path, "PNG")

    info = f"{info_snap}\n{info_bg}\n保存: {save_path}"
    return snapped, result, gr.update(value=info), gr.update(visible=True, value=str(save_path))


CSS = """
footer { display: none !important; }
.title { text-align: center; }
"""

with gr.Blocks(title="ピクセルアート生成システム") as demo:
    gr.Markdown("# ピクセルアート生成システム", elem_classes="title")
    gr.Markdown("画像をアップロードし、パラメータを調整して「**変換実行**」を押してください。")

    with gr.Row():
        # 左カラム: 入力・設定
        with gr.Column(scale=1, min_width=320):

            input_image = gr.Image(
                type="pil",
                label="入力画像",
                height=280,
            )

            with gr.Accordion("ピクセルアート設定", open=True):
                k_colors = gr.Slider(4, 64, value=16, step=4,
                    label="カラーパレット数（少ない→よりドット絵らしく）")
                output_scale = gr.Slider(1, 8, value=1, step=1,
                    label="出力スケール（大きい→ドットを拡大表示）")
                peak_threshold = gr.Slider(0.05, 1.0, value=0.2, step=0.05,
                    label="グリッド検出感度（高い→大きめのマス）")
                fallback_segments = gr.Slider(8, 256, value=64, step=8,
                    label="グリッド分割数（少ない→粗いドット）")
                min_cuts = gr.Slider(2, 32, value=4, step=1,
                    label="最小グリッド線数")

            with gr.Accordion("背景透過設定", open=True):
                do_remove_bg = gr.Checkbox(value=True, label="白背景を透過する")
                bg_tolerance = gr.Slider(0, 60, value=15, step=1,
                    label="白判定の許容誤差（大きい→広く除去）")
                do_remove_bg.change(
                    fn=lambda v: gr.update(interactive=v),
                    inputs=do_remove_bg,
                    outputs=bg_tolerance,
                )

            run_btn = gr.Button("変換実行", variant="primary", size="lg")

        # 右カラム: 出力
        with gr.Column(scale=2, min_width=480):
            with gr.Row():
                out_snap = gr.Image(type="pil", label="Step 2：ピクセルスナップ",
                                    height=320, image_mode="RGB")
                out_final = gr.Image(type="pil", label="Step 3：背景透過（最終）",
                                     height=320, image_mode="RGBA")

            info_box = gr.Textbox(label="処理情報", lines=3, interactive=False)
            download_btn = gr.DownloadButton(
                label="PNGをダウンロード",
                variant="secondary",
                visible=False,
            )

    run_btn.click(
        fn=process,
        inputs=[input_image, k_colors, output_scale, peak_threshold,
                fallback_segments, min_cuts, do_remove_bg, bg_tolerance],
        outputs=[out_snap, out_final, info_box, download_btn],
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
