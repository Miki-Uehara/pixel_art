#!/usr/bin/env python3
"""ピクセルアート生成システム - スマート抽出専用UI"""

import os, sys
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image, ImageEnhance
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

_env = Path(__file__).parent / ".env"
if _env.exists():
    for ln in _env.read_text(encoding="utf-8").splitlines():
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from steps.step2_pixel_snap import pixel_snap, PixelSnapConfig
from steps.step5_lineart import extract_lineart
from steps.step6_smart_extract import (
    line_mask_from_lineart, build_regions,
    initial_classify, absorb_enclosed_islands, make_mask,
    render_basecoat, toggle_region_at, pixelize_with_mask,
    extract_outer_edge_line, detect_dominant_line_color,
)

OUTPUT_ROOT = Path(__file__).parent / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def save_versioned(img: Image.Image, type_name: str) -> str:
    """output/YYYYMMDD/YYYYMMDD_pixelart_NNN_<type>.png の連番で保存。"""
    today = datetime.now().strftime("%Y%m%d")
    out_dir = OUTPUT_ROOT / today
    out_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in out_dir.glob(f"{today}_pixelart_*_{type_name}.png"):
        parts = p.stem.split("_")
        if len(parts) >= 4 and parts[2].isdigit():
            nums.append(int(parts[2]))
    n = (max(nums) + 1) if nums else 1
    path = out_dir / f"{today}_pixelart_{n:03d}_{type_name}.png"
    img.save(path, "PNG")
    return str(path)


def on_black(rgba: Image.Image) -> Image.Image:
    """RGBAを真っ黒背景に合成したRGB画像を返す。"""
    base = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(base, rgba.convert("RGBA")).convert("RGB")


# ─── カラー調整 ────────────────────────────────────────

def _rgb_to_hsv(a):
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    mx=np.maximum(np.maximum(r,g),b); mn=np.minimum(np.minimum(r,g),b); d=mx-mn
    s=np.where(mx>1e-6,d/mx,0.)
    h=np.zeros_like(r); nz=d>1e-6
    mr,mg,mb=nz&(mx==r),nz&(mx==g),nz&(mx==b)
    h[mr]=((g[mr]-b[mr])/d[mr])%6.; h[mg]=(b[mg]-r[mg])/d[mg]+2.; h[mb]=(r[mb]-g[mb])/d[mb]+4.
    return np.stack([(h/6.)%1.,s,mx],axis=2)

def _hsv_to_rgb(hsv):
    h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]; h6=h*6.; i=h6.astype(np.int32)%6; f=h6-np.floor(h6)
    p,q,t=v*(1-s),v*(1-s*f),v*(1-s*(1-f))
    rgb=np.zeros((*h.shape,3),dtype=np.float32)
    for idx,(rv,gv,bv) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m=i==idx; rgb[m,0]=rv[m]; rgb[m,1]=gv[m]; rgb[m,2]=bv[m]
    return rgb

def adjust_colors(img, sat=1., bri=1., con=1., hue=0.):
    im = img.convert("RGB")
    if sat!=1.: im=ImageEnhance.Color(im).enhance(sat)
    if bri!=1.: im=ImageEnhance.Brightness(im).enhance(bri)
    if con!=1.: im=ImageEnhance.Contrast(im).enhance(con)
    if hue!=0.:
        a=np.array(im).astype(np.float32)/255.
        hsv=_rgb_to_hsv(a); hsv[:,:,0]=(hsv[:,:,0]+hue/360.)%1.
        im=Image.fromarray((_hsv_to_rgb(hsv)*255).clip(0,255).astype(np.uint8),"RGB")
    return im


# ─── ユーティリティ ─────────────────────────────────────

def checker(w,h,tile=14):
    xs,ys=np.arange(w)//tile,np.arange(h)//tile
    v=np.where((xs[None,:]+ys[:,None])%2,240,200).astype(np.uint8)
    return np.stack([v,v,v],axis=2)

def on_checker(rgba):
    a=np.array(rgba.convert("RGBA")); h,w=a.shape[:2]; b=checker(w,h)
    al=a[:,:,3:4].astype(np.float32)/255.
    return Image.fromarray((a[:,:,:3]*al+b*(1-al)).clip(0,255).astype(np.uint8),"RGB")

def as_pil(x): return Image.fromarray(x) if isinstance(x,np.ndarray) else x

def do_snap(img,dots,scale,colors):
    cfg=PixelSnapConfig(k_colors=int(colors),output_scale=int(scale),
                        fallback_target_segments=int(dots),min_cuts_per_axis=4)
    return pixel_snap(as_pil(img),cfg)

def size_text(img, dots, scale):
    if img is None:
        return "⬆ ②大元の画像をアップロードしてね！"
    iw, ih = as_pil(img).size
    dh = max(4, round(int(dots) * ih / iw))
    return (f"📥 入力 {iw}×{ih}px 　🔲 ドット数 {int(dots)}×{dh} 　"
            f"📤 出力 {int(dots)*int(scale)}×{dh*int(scale)}px （×{int(scale)} 表示）")

def _hex_to_rgb(hex_color: str) -> tuple:
    s = (hex_color or "#141414").strip()
    if s.startswith("#"):
        h = s.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        try:
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (20, 20, 20)
    if s.lower().startswith(("rgb(", "rgba(")):
        import re
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            return tuple(max(0, min(255, int(float(nums[i])))) for i in range(3))
    return (20, 20, 20)

# ─── スマート抽出ハンドラ ────────────────────────────────

def _ensure_same_size(img_a, img_b, img_c):
    base = as_pil(img_b).convert("RGB")
    w, h = base.size
    a = as_pil(img_a).convert("RGBA")
    if a.size != (w, h):
        a = a.resize((w, h), Image.LANCZOS)
    c = as_pil(img_c).convert("RGB")
    if c.size != (w, h):
        c = c.resize((w, h), Image.LANCZOS)
    return a, base, c

def h_smart_build(img_line, img_orig, img_bgpaint, line_thr, diff_thr, island_max, fill_color):
    if img_line is None or img_orig is None or img_bgpaint is None:
        return (None, None, None, None, None,
                "⚠ 3枚すべて（線画／大元／背景色塗り）をアップロードしてください",
                None)
    a_raw, b, c = _ensure_same_size(img_line, img_orig, img_bgpaint)
    a_lineart = extract_lineart(a_raw, int(line_thr))
    is_line = line_mask_from_lineart(a_lineart, 128)
    labels, num = build_regions(is_line)
    region_is_char = initial_classify(labels, num, b, c, int(diff_thr))
    before_char = int(region_is_char[1:].sum())
    region_is_char = absorb_enclosed_islands(labels, num, region_is_char, int(island_max))
    absorbed = int(region_is_char[1:].sum()) - before_char
    fc = _hex_to_rgb(fill_color or "#ff88cc")
    basecoat = render_basecoat(is_line, region_is_char, labels, fill_color=fc)
    preview = on_checker(basecoat)
    mask = make_mask(labels, region_is_char, is_line)
    line_px = int(is_line.sum())
    char_px = int(mask.sum())
    bg_px = mask.size - char_px
    info = (f"✅ 下塗り生成完了！  線画しきい値 {int(line_thr)} / 差分しきい値 {int(diff_thr)} / "
            f"領域数 {num}（閉じ島 {absorbed} 個を自動吸収）/ "
            f"線 {line_px}px / キャラ {char_px}px / 背景 {bg_px}px")
    return (labels, is_line, region_is_char, b, preview, info,
            a_lineart)

def h_smart_click(labels_st, is_line_st, region_st, orig_st, fill_color, evt: gr.SelectData):
    if labels_st is None or region_st is None or orig_st is None:
        return None, region_st, "⚠ 先に「下塗り生成」を実行してください"
    x, y = int(evt.index[0]), int(evt.index[1])
    new_region = toggle_region_at(region_st, labels_st, x, y)
    fc = _hex_to_rgb(fill_color or "#ff88cc")
    basecoat = render_basecoat(is_line_st, new_region, labels_st, fill_color=fc)
    preview = on_checker(basecoat)
    mask = make_mask(labels_st, new_region, is_line_st)
    return preview, new_region, f"🖱 ({x},{y}) の領域を反転  /  キャラ {int(mask.sum())}px / 背景 {mask.size - int(mask.sum())}px"

def h_smart_reset(labels_st, is_line_st, orig_st, img_bgpaint, diff_thr, island_max, fill_color):
    if labels_st is None or img_bgpaint is None or orig_st is None:
        return None, None, "⚠ まず下塗り生成を実行してください"
    base = as_pil(orig_st).convert("RGB")
    ow, oh = base.size
    c = as_pil(img_bgpaint).convert("RGB")
    if c.size != (ow, oh):
        c = c.resize((ow, oh), Image.LANCZOS)
    num = int(labels_st.max())
    region_is_char = initial_classify(labels_st, num, base, c, int(diff_thr))
    before_char = int(region_is_char[1:].sum())
    region_is_char = absorb_enclosed_islands(labels_st, num, region_is_char, int(island_max))
    absorbed = int(region_is_char[1:].sum()) - before_char
    fc = _hex_to_rgb(fill_color or "#ff88cc")
    basecoat = render_basecoat(is_line_st, region_is_char, labels_st, fill_color=fc)
    preview = on_checker(basecoat)
    mask = make_mask(labels_st, region_is_char, is_line_st)
    return preview, region_is_char, f"♻ 再判定しました（閉じ島 {absorbed} 個吸収）  /  キャラ {int(mask.sum())}px / 背景 {mask.size - int(mask.sum())}px"

def _pixelize_line_alpha(is_line: np.ndarray, dots: int, scale: int) -> np.ndarray:
    """線画(boolマスク)を指定ドット数でピクセル化したアルファ配列(uint8)を返す。"""
    h, w = is_line.shape
    dw = int(dots)
    dh = max(4, round(dw * h / w))
    sw, sh = dw * int(scale), dh * int(scale)
    line_img = Image.fromarray((is_line.astype(np.uint8) * 255), "L")
    small = line_img.resize((dw, dh), Image.BILINEAR)
    small_bin = np.where(np.array(small) >= 128, 255, 0).astype(np.uint8)
    return np.array(Image.fromarray(small_bin, "L").resize((sw, sh), Image.NEAREST))

def _colorize_edge(edge_alpha: np.ndarray, color: tuple, opacity_pct: float) -> Image.Image:
    h, w = edge_alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color[0], color[1], color[2]
    rgba[..., 3] = (edge_alpha.astype(np.float32) * (float(opacity_pct) / 100.0)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")

def _composite_line_over(pixchar: Image.Image, line_img: Image.Image) -> Image.Image:
    base = pixchar.convert("RGBA")
    if line_img.size != base.size:
        line_img = line_img.resize(base.size, Image.NEAREST)
    return Image.alpha_composite(base, line_img)

def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def h_smart_finalize(labels_st, is_line_st, region_st, orig_st,
                     dots, scale, colors, sat, bri, con, hue, edge_thick,
                     line_alpha):
    if labels_st is None or region_st is None or orig_st is None:
        return (None, None, None, "⚠ 先に「下塗り生成」を実行してください",
                gr.update(), None, None, None)
    mask = make_mask(labels_st, region_st, is_line_st)
    base = as_pil(orig_st)
    adjusted = adjust_colors(base, float(sat), float(bri), float(con), float(hue))
    rgba = pixelize_with_mask(adjusted, mask, do_snap, int(dots), int(scale), int(colors))
    prev = on_checker(rgba)

    outer_edge = extract_outer_edge_line(is_line_st, region_st, labels_st, int(edge_thick))
    edge_alpha = _pixelize_line_alpha(outer_edge, int(dots), int(scale))

    dom = detect_dominant_line_color(base, is_line_st)
    dom_hex = _rgb_to_hex(dom)
    line_img = _colorize_edge(edge_alpha, dom, float(line_alpha))
    line_prev = on_checker(line_img)

    composite = _composite_line_over(rgba, line_img)
    comp_prev = on_black(composite)

    iw, ih = rgba.size
    info = (f"✅ ピクセル化＆透過完了！  {iw}×{ih}px  /  {int(colors)}色  /  ×{int(scale)}表示  /  "
            f"検出された線色: {dom_hex}")
    return (line_prev, prev, comp_prev, info,
            gr.update(value=dom_hex),
            edge_alpha, rgba, line_img, composite)


def h_recolor_line(edge_alpha_st, pixchar_st, line_color, line_alpha):
    if edge_alpha_st is None or pixchar_st is None:
        return None, None, None, None
    color = _hex_to_rgb(line_color or "#141414")
    line_img = _colorize_edge(edge_alpha_st, color, float(line_alpha))
    line_prev = on_checker(line_img)
    composite = _composite_line_over(pixchar_st, line_img)
    comp_prev = on_black(composite)
    return line_prev, comp_prev, line_img, composite


def h_save(pil_img, type_name):
    if pil_img is None:
        return gr.update(value="⚠ まだ画像が生成されていません", visible=True)
    path = save_versioned(pil_img, type_name)
    return gr.update(value=f"✅ 保存しました：`{path}`", visible=True)




# ─── CSS ──────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DotGothic16&family=M+PLUS+Rounded+1c:wght@500;700;800&display=swap');

body, .gradio-container, .main {
    background:
        radial-gradient(ellipse 120% 60% at 20% 0%,  #ffe0f8 0%, transparent 55%),
        radial-gradient(ellipse 100% 50% at 80% 100%, #e0f0ff 0%, transparent 55%),
        radial-gradient(ellipse 80%  40% at 50% 50%,  #f0e8ff 0%, transparent 60%),
        #fdf6ff !important;
    background-attachment: fixed !important;
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
}

h1 {
    font-family: 'DotGothic16', monospace !important;
    font-size: 26px !important;
    color: #cc33aa !important;
    text-align: center !important;
    letter-spacing: 4px !important;
    text-shadow: 3px 3px 0 #ffbbee, 6px 6px 0 #ffddff44 !important;
    padding: 18px 0 6px !important;
    margin: 0 !important;
}

.subtitle p {
    font-family: 'DotGothic16', monospace !important;
    font-size: 12px !important;
    color: #aa66cc !important;
    text-align: center !important;
    letter-spacing: 3px !important;
    margin-bottom: 16px !important;
}

.section-head p {
    font-family: 'DotGothic16', monospace !important;
    font-size: 15px !important;
    color: #8833cc !important;
    background: linear-gradient(90deg, #f5e8ff, #ffeeff88) !important;
    border-left: 5px solid #cc55ee !important;
    padding: 6px 14px !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 6px 0 !important;
    letter-spacing: 2px !important;
}

.size-bar p {
    font-family: 'DotGothic16', monospace !important;
    font-size: 14px !important;
    color: #664488 !important;
    background: linear-gradient(90deg, #fff0ff, #f8f0ff) !important;
    border: 2px solid #ddaaff !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    text-align: center !important;
    letter-spacing: 1px !important;
    box-shadow: 2px 2px 0 #e8ccff !important;
    margin: 6px 0 12px !important;
}

.gr-group, .gr-box {
    background: rgba(255,255,255,0.80) !important;
    border: 2px solid #e8d0ff !important;
    border-radius: 14px !important;
    box-shadow: 4px 4px 0 #ddc8f8, 0 0 20px #cc88ff18 !important;
}

label > span, .gr-form > label > span {
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #7733bb !important;
    letter-spacing: 0.5px !important;
}

input[type=range] { accent-color: #cc44bb !important; height: 5px !important; cursor: pointer !important; }
input[type=checkbox] { accent-color: #cc44bb !important; width: 18px !important; height: 18px !important; }

textarea, .gr-textbox textarea {
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
    font-size: 14px !important;
    background: #fdf8ff !important;
    border: 2px solid #e0c8ff !important;
    border-radius: 10px !important;
    color: #553388 !important;
}

.status-box textarea {
    font-family: 'DotGothic16', monospace !important;
    font-size: 13px !important;
    color: #664499 !important;
    background: #fef8ff !important;
    border: 2px dashed #cc99ee !important;
    border-radius: 10px !important;
}

button.primary {
    font-family: 'DotGothic16', monospace !important;
    font-size: 20px !important;
    letter-spacing: 4px !important;
    background: linear-gradient(180deg, #ff88cc 0%, #ee44aa 50%, #cc2299 100%) !important;
    border: 3px solid #ffbbee !important;
    box-shadow: 5px 5px 0 #aa1177, 0 0 20px #ff66cc44 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    transition: all 0.07s !important;
    padding: 14px 40px !important;
    text-shadow: 1px 1px 0 #aa2288 !important;
}
button.primary:hover { background: linear-gradient(180deg, #ffaadd 0%, #ff55bb 100%) !important; }
button.primary:active { transform: translate(5px, 5px) !important; box-shadow: 0 0 0 #aa1177 !important; }

button.secondary {
    font-family: 'DotGothic16', monospace !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    background: linear-gradient(180deg, #aaddff 0%, #77bbff 100%) !important;
    border: 2px solid #bbeeff !important;
    box-shadow: 4px 4px 0 #4499cc !important;
    color: #114477 !important;
    border-radius: 10px !important;
    transition: all 0.07s !important;
    padding: 10px 20px !important;
}
button.secondary:hover { background: linear-gradient(180deg, #cceeff 0%, #99ccff 100%) !important; }
button.secondary:active { transform: translate(4px, 4px) !important; box-shadow: 0 0 0 #4499cc !important; }

.gr-accordion > .label-wrap {
    font-family: 'DotGothic16', monospace !important;
    font-size: 15px !important;
    color: #9944cc !important;
    background: linear-gradient(90deg, #f8eeff, #fff5ff) !important;
    border: 2px solid #e0c8ff !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    letter-spacing: 2px !important;
}

.image-container {
    border: 2px solid #e8d0ff !important;
    border-radius: 12px !important;
    background: #fdf8ff !important;
}

footer { display: none !important; }
"""

# ─── UI ───────────────────────────────────────────────

with gr.Blocks(title="ピクセルアートジェネレーター") as demo:

    sm_labels_state = gr.State(None)
    sm_isline_state = gr.State(None)
    sm_region_state = gr.State(None)
    sm_orig_state   = gr.State(None)
    sm_edge_alpha_state = gr.State(None)
    sm_pixchar_state = gr.State(None)
    sm_lineart_img_state = gr.State(None)
    sm_line_img_state    = gr.State(None)
    sm_comp_img_state    = gr.State(None)

    gr.Markdown("# ✨ スマート抽出 ピクセルアート ✨", elem_classes="title")
    gr.Markdown("線画＋背景色画像で正確な下塗りを作って、大元画像をピクセル化＆透過",
                elem_classes="subtitle")

    # ══════════════════════════════════════════════════
    # 1. 画像アップロード
    # ══════════════════════════════════════════════════
    gr.Markdown("📁 **STEP 1　画像を3枚アップロード**", elem_classes="section-head")
    with gr.Row():
        sm_line_in = gr.Image(type="pil", label="① 線画用画像（白背景つきでOK・自動で線画抽出します）", height=260)
        sm_orig_in = gr.Image(type="pil", label="② 大元の白背景キャラ（最終ソース）", height=260)
        sm_bg_in   = gr.Image(type="pil", label="③ 背景だけ色を変えた画像", height=260)

    # ══════════════════════════════════════════════════
    # 2. 下塗り生成
    # ══════════════════════════════════════════════════
    gr.Markdown("🖌 **STEP 2　下塗りを自動生成して微修正**", elem_classes="section-head")
    with gr.Row():
        sm_line_thr = gr.Slider(10, 220, value=150, step=1,
            label="線画抽出しきい値（輝度この値より暗い＝線）",
            info="低い：濃い線のみ　高い：薄い線も拾う（標準150）")
        sm_bg_tol = gr.Slider(5, 250, value=150, step=1,
            label="②と③の差分しきい値（背景判定）",
            info="②と③でこの値以上色が変わったピクセル＝背景。高い：大幅変化のみ背景（標準150）")
    with gr.Row():
        sm_island_max = gr.Slider(0, 20000, value=2000, step=100,
            label="閉じ島の自動吸収サイズ（px）",
            info="線画に囲まれてる背景の島で、このサイズ以下なら自動でキャラ判定。0=無効  -1=無制限")
        sm_fill = gr.ColorPicker(value="#ff88cc",
            label="🎨 下塗りの色（プレビュー用）")

    with gr.Row():
        sm_build_btn = gr.Button("🖌 下塗り生成", variant="primary")
        sm_reset_btn = gr.Button("♻ 現在の差分しきい値で再判定", variant="secondary")

    gr.Markdown(
        "🖱 **下の画像をクリックで、その連結領域を「キャラ ↔ 背景」反転できます。**\n"
        "塗り残しの島は1クリックで埋まり、はみ出した塗り部分も1クリックで透過に戻ります。",
        elem_classes="subtitle"
    )
    sm_overlay = gr.Image(type="pil",
        label="🎯 下塗りプレビュー（チェック柄=透過。クリックで領域トグル）",
        height=460, interactive=False)
    sm_info = gr.Textbox(label="📋 ステータス", lines=2, interactive=False,
                         elem_classes="status-box")
    with gr.Row():
        sm_lineart_save = gr.Button("💾 線画抽出後の画像を保存", variant="secondary")
    sm_lineart_save_status = gr.Markdown(visible=False, elem_classes="status-box")

    # ══════════════════════════════════════════════════
    # 3. ピクセル化＆透過
    # ══════════════════════════════════════════════════
    gr.Markdown("✨ **STEP 3　ピクセル化＆背景透過の設定**", elem_classes="section-head")

    size_bar = gr.Markdown("⬆ ②大元の画像をアップロードしてね！", elem_classes="size-bar")

    with gr.Row():
        dot_count = gr.Slider(32, 512, value=200, step=8,
            label="ドット数（横）",
            info="少ない：粗くてレトロ　多い：細かくてなめらか")
        display_scale = gr.Slider(1, 16, value=6, step=1,
            label="表示倍率",
            info="1ドットを何pxで表示するか")
    k_colors = gr.Slider(4, 64, value=16, step=4,
        label="カラーパレット数",
        info="少ない：よりドット絵らしく　多い：原画に近い色味")
    sm_edge_thick = gr.Slider(1, 30, value=10, step=1,
        label="外淵の太さ（px・元解像度基準）",
        info="ピクセル化前の元画像で何px分の縁取りを残すか。大きいほど太い縁取りに")

    with gr.Accordion("🌈 カラー調整（彩度・明度・コントラスト・色相）", open=False):
        with gr.Row():
            saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.05,
                label="彩度", info="1.0 = 変化なし")
            brightness = gr.Slider(0.2, 2.0, value=1.0, step=0.05,
                label="明度", info="1.0 = 変化なし")
        with gr.Row():
            contrast = gr.Slider(0.2, 3.0, value=1.0, step=0.05,
                label="コントラスト", info="1.0 = 変化なし")
            hue_shift = gr.Slider(-180, 180, value=0, step=5,
                label="色相シフト（度）", info="0 = 変化なし")

    sm_final_btn = gr.Button("🎨 ピクセル化＆透過を実行！", variant="primary", size="lg")
    with gr.Row():
        sm_lineart_pix = gr.Image(type="pil",
            label="✏ 外淵線画（馴染ませた色・チェック柄=透過）", height=380)
        sm_final_prev = gr.Image(type="pil",
            label="✅ ピクセルアート透過プレビュー（チェック柄）", height=380)

    gr.Markdown("🎨 **線画の色を調整**（実行後、自動で元イラストの主要線色を検出）",
                elem_classes="section-head")
    with gr.Row():
        sm_line_color = gr.ColorPicker(value="#141414",
            label="線画カラー（自動検出後に変更可）")
        sm_line_alpha = gr.Slider(0, 100, value=100, step=1,
            label="線画の不透明度 (%)", info="100=くっきり / 下げるほど透けて馴染む")

    sm_final_info = gr.Textbox(label="📋 ステータス", lines=1, interactive=False,
                               elem_classes="status-box")

    sm_composite_prev = gr.Image(type="pil",
        label="🌟 最終合成（線画＋透過キャラ・背景=黒）", height=460)

    with gr.Row():
        sm_lineart_pix_save = gr.Button("💾 外淵線画PNGを保存", variant="secondary")
        sm_save = gr.Button("💾 ピクセルアート透過PNGを保存", variant="secondary")
        sm_composite_save = gr.Button("💾 最終合成PNGを保存", variant="secondary")
    sm_save_status = gr.Markdown(visible=False, elem_classes="status-box")

    # ── ハンドラ配線 ────────────────────────────
    for comp in [sm_orig_in, dot_count, display_scale]:
        comp.change(fn=size_text,
                    inputs=[sm_orig_in, dot_count, display_scale],
                    outputs=size_bar)

    sm_build_btn.click(
        fn=h_smart_build,
        inputs=[sm_line_in, sm_orig_in, sm_bg_in, sm_line_thr, sm_bg_tol, sm_island_max, sm_fill],
        outputs=[sm_labels_state, sm_isline_state, sm_region_state,
                 sm_orig_state, sm_overlay, sm_info, sm_lineart_img_state],
    )
    sm_reset_btn.click(
        fn=h_smart_reset,
        inputs=[sm_labels_state, sm_isline_state, sm_orig_state, sm_bg_in, sm_bg_tol, sm_island_max, sm_fill],
        outputs=[sm_overlay, sm_region_state, sm_info],
    )
    sm_overlay.select(
        fn=h_smart_click,
        inputs=[sm_labels_state, sm_isline_state, sm_region_state, sm_orig_state, sm_fill],
        outputs=[sm_overlay, sm_region_state, sm_info],
    )
    sm_final_btn.click(
        fn=h_smart_finalize,
        inputs=[sm_labels_state, sm_isline_state, sm_region_state, sm_orig_state,
                dot_count, display_scale, k_colors,
                saturation, brightness, contrast, hue_shift, sm_edge_thick,
                sm_line_alpha],
        outputs=[sm_lineart_pix, sm_final_prev, sm_composite_prev, sm_final_info,
                 sm_line_color, sm_edge_alpha_state, sm_pixchar_state,
                 sm_line_img_state, sm_comp_img_state],
    )

    for comp in [sm_line_color, sm_line_alpha]:
        comp.change(
            fn=h_recolor_line,
            inputs=[sm_edge_alpha_state, sm_pixchar_state, sm_line_color, sm_line_alpha],
            outputs=[sm_lineart_pix, sm_composite_prev, sm_line_img_state, sm_comp_img_state],
        )

    sm_lineart_save.click(
        fn=lambda i: h_save(i, "lineart"),
        inputs=[sm_lineart_img_state], outputs=[sm_lineart_save_status],
    )
    sm_lineart_pix_save.click(
        fn=lambda i: h_save(i, "outerline"),
        inputs=[sm_line_img_state], outputs=[sm_save_status],
    )
    sm_save.click(
        fn=lambda i: h_save(i, "character"),
        inputs=[sm_pixchar_state], outputs=[sm_save_status],
    )
    sm_composite_save.click(
        fn=lambda i: h_save(i, "composite"),
        inputs=[sm_comp_img_state], outputs=[sm_save_status],
    )



if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.pink,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.purple,
            font=gr.themes.GoogleFont("M PLUS Rounded 1c"),
        ),
    )
