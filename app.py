#!/usr/bin/env python3
"""ピクセルアート生成システム - ポップゲームUI"""

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
from steps.step3_remove_bg import remove_white_background, remove_color_background
from steps.step5_lineart import extract_lineart, base_coat
from steps.step6_smart_extract import (
    detect_bg_color, line_mask_from_lineart, build_regions,
    initial_classify, make_mask, render_overlay, toggle_region_at,
    pixelize_with_mask,
)

FINAL_DIR = Path(__file__).parent / "output" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


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

def size_text(img,dots,scale):
    if img is None: return "⬆ 画像をアップロードしてね！"
    iw,ih=as_pil(img).size; dh=max(4,round(int(dots)*ih/iw))
    return (f"📥 入力  {iw}×{ih}px　　"
            f"🔲 ドット数  {int(dots)}×{dh}　　"
            f"📤 出力  {int(dots)*int(scale)}×{dh*int(scale)}px　（×{int(scale)} 表示）")

def do_snap(img,dots,scale,colors):
    cfg=PixelSnapConfig(k_colors=int(colors),output_scale=int(scale),
                        fallback_target_segments=int(dots),min_cuts_per_axis=4)
    return pixel_snap(as_pil(img),cfg)

def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def do_final(snapped,sat,bri,con,hue,bg,bg_mode,custom_color,tol):
    adj=adjust_colors(snapped,float(sat),float(bri),float(con),float(hue))
    if not bg:
        return adj.convert("RGBA")
    if bg_mode == "カスタム色":
        rgb = _hex_to_rgb(custom_color or "#ffffff")
        return remove_color_background(adj, rgb, int(tol))
    return remove_white_background(adj, int(tol))

def save(rgba,prefix="pixel_art"):
    p=FINAL_DIR/f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    rgba.save(p,"PNG"); return str(p)


# ─── ハンドラ ──────────────────────────────────────────

def h_snap(img,dots,scale,colors,sat,bri,con,hue,bg,bg_mode,custom_color,tol):
    if img is None:
        return None,None,"⚠ 画像がまだロードされていません！",gr.update(visible=False),None
    snapped=do_snap(img,dots,scale,colors)
    rgba=do_final(snapped,sat,bri,con,hue,bg,bg_mode,custom_color,tol)
    prev=on_checker(rgba)
    iw,ih=as_pil(img).size; dh=max(4,round(int(dots)*ih/iw))
    info=(f"✅ 変換完了！  ドット数 {int(dots)}×{dh}  /  "
          f"出力 {int(dots)*int(scale)}×{dh*int(scale)}px  /  "
          f"{int(colors)}色  /  背景透過 {'ON' if bg else 'OFF'}")
    path=save(rgba)
    return snapped,prev,info,snapped,gr.update(visible=True,value=path)

def h_color(st,sat,bri,con,hue,bg,bg_mode,custom_color,tol):
    if st is None: return None
    return on_checker(do_final(as_pil(st),sat,bri,con,hue,bg,bg_mode,custom_color,tol))

def h_color_save(st,sat,bri,con,hue,bg,bg_mode,custom_color,tol):
    if st is None: return gr.update(visible=False),"⚠ 先に変換実行してください"
    path=save(do_final(as_pil(st),sat,bri,con,hue,bg,bg_mode,custom_color,tol))
    return gr.update(visible=True,value=path),f"💾 保存完了！  {Path(path).name}"

def h_compare(img,colors,scale,sat,bri,con,hue,bg,bg_mode,custom_color,tol):
    if img is None: return [],"⚠ 画像がありません"
    pil=as_pil(img); iw,ih=pil.size; res=[]
    for d in [64,96,128,192,256]:
        snapped=do_snap(img,d,scale,colors)
        rgba=do_final(snapped,sat,bri,con,hue,bg,bg_mode,custom_color,tol)
        dh=max(4,round(d*ih/iw))
        res.append((on_checker(rgba),f"{d}ドット → {d*int(scale)}×{dh*int(scale)}px"))
    return res,f"✨ 5パターン生成完了！  {int(colors)}色 / ×{int(scale)}表示"

def h_compare_save(img,dot,scale,colors,sat,bri,con,hue,bg,bg_mode,custom_color,tol):
    if img is None: return gr.update(visible=False),"⚠ 画像がありません"
    rgba=do_final(do_snap(img,int(dot),scale,colors),sat,bri,con,hue,bg,bg_mode,custom_color,tol)
    path=save(rgba,prefix=f"pixel_{int(dot)}dot")
    return gr.update(visible=True,value=path),f"💾 保存完了！  {Path(path).name}"

def h_pick_color(img, evt: gr.SelectData):
    if img is None:
        return "#ffffff"
    pil = as_pil(img).convert("RGB")
    w, h = pil.size
    x = min(max(0, evt.index[0]), w - 1)
    y = min(max(0, evt.index[1]), h - 1)
    r, g, b = pil.getpixel((x, y))
    return f"#{r:02x}{g:02x}{b:02x}"

def h_lineart_extract(img, thr):
    if img is None:
        return None, "⚠ 画像をアップロードしてください", gr.update(visible=False), None
    result = extract_lineart(as_pil(img), int(thr))
    prev = on_checker(result)
    path = save(result, "lineart")
    iw, ih = result.size
    return prev, f"✅ 線画抽出完了！  {iw}×{ih}px", gr.update(visible=True, value=path), result

def h_base_coat(lineart_st, color):
    if lineart_st is None:
        return None, "⚠ 先に線画を抽出してください", gr.update(visible=False)
    result = base_coat(as_pil(lineart_st), color)
    prev = on_checker(result)
    path = save(result, "basecoat")
    return prev, f"✅ 下塗り完了！  塗り色: {color}", gr.update(visible=True, value=path)


# ─── スマート抽出タブ用 ──────────────────────────────────

def _ensure_same_size(img_a, img_b, img_c):
    """3枚の画像を ②大元 のサイズに揃える（線画と背景色画像をリサイズ）。"""
    base = as_pil(img_b).convert("RGB")
    w, h = base.size
    a = as_pil(img_a).convert("RGBA")
    if a.size != (w, h):
        a = a.resize((w, h), Image.LANCZOS)
    c = as_pil(img_c).convert("RGB")
    if c.size != (w, h):
        c = c.resize((w, h), Image.LANCZOS)
    return a, base, c

def h_smart_build(img_line, img_orig, img_bgpaint, line_thr, bg_tol):
    if img_line is None or img_orig is None or img_bgpaint is None:
        return (None, None, None, None, None,
                "⚠ 3枚すべて（線画／大元／背景色塗り）をアップロードしてください")
    a, b, c = _ensure_same_size(img_line, img_orig, img_bgpaint)
    bg_color = detect_bg_color(c)
    is_line = line_mask_from_lineart(a, int(line_thr))
    labels, num = build_regions(is_line)
    region_is_char = initial_classify(labels, num, c, bg_color, int(bg_tol))
    mask = make_mask(labels, region_is_char, is_line)
    overlay = render_overlay(b, mask)
    info = (f"✅ 自動マスク生成完了！  領域数 {num}  /  "
            f"検出背景色 RGB{bg_color}  /  キャラ画素 {int(mask.sum())}px")
    return labels, is_line, region_is_char, b, overlay, info

def h_smart_click(labels_st, is_line_st, region_st, orig_st, evt: gr.SelectData):
    if labels_st is None or region_st is None or orig_st is None:
        return None, region_st, "⚠ 先に「自動マスク生成」を実行してください"
    x, y = int(evt.index[0]), int(evt.index[1])
    new_region = toggle_region_at(region_st, labels_st, x, y)
    mask = make_mask(labels_st, new_region, is_line_st)
    overlay = render_overlay(as_pil(orig_st), mask)
    return overlay, new_region, f"🖱 ({x},{y}) の領域を反転  /  キャラ画素 {int(mask.sum())}px"

def h_smart_reset(labels_st, is_line_st, orig_st, img_bgpaint, bg_tol):
    if labels_st is None or img_bgpaint is None or orig_st is None:
        return None, None, "⚠ まず自動マスク生成を実行してください"
    c = as_pil(img_bgpaint).convert("RGB")
    ow, oh = as_pil(orig_st).size
    if c.size != (ow, oh):
        c = c.resize((ow, oh), Image.LANCZOS)
    num = int(labels_st.max())
    bg_color = detect_bg_color(c)
    region_is_char = initial_classify(labels_st, num, c, bg_color, int(bg_tol))
    mask = make_mask(labels_st, region_is_char, is_line_st)
    overlay = render_overlay(as_pil(orig_st), mask)
    return overlay, region_is_char, f"♻ リセットしました  /  キャラ画素 {int(mask.sum())}px"

def h_smart_finalize(labels_st, is_line_st, region_st, orig_st,
                     dots, scale, colors, sat, bri, con, hue):
    if labels_st is None or region_st is None or orig_st is None:
        return None, None, "⚠ 先に「自動マスク生成」を実行してください", gr.update(visible=False)
    mask = make_mask(labels_st, region_st, is_line_st)
    base = as_pil(orig_st)
    adjusted = adjust_colors(base, float(sat), float(bri), float(con), float(hue))
    rgba = pixelize_with_mask(adjusted, mask, do_snap, int(dots), int(scale), int(colors))
    prev = on_checker(rgba)
    path = save(rgba, "smart_pixel")
    iw, ih = rgba.size
    info = (f"✅ ピクセル化＆透過完了！  {iw}×{ih}px  /  "
            f"{int(colors)}色  /  ×{int(scale)}表示")
    return rgba, prev, info, gr.update(visible=True, value=path)


# ─── CSS ──────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DotGothic16&family=M+PLUS+Rounded+1c:wght@500;700;800&display=swap');

/* ══ ベース ══ */
body, .gradio-container, .main {
    background:
        radial-gradient(ellipse 120% 60% at 20% 0%,  #ffe0f8 0%, transparent 55%),
        radial-gradient(ellipse 100% 50% at 80% 100%, #e0f0ff 0%, transparent 55%),
        radial-gradient(ellipse 80%  40% at 50% 50%,  #f0e8ff 0%, transparent 60%),
        #fdf6ff !important;
    background-attachment: fixed !important;
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
}

/* ══ タイトル ══ */
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

/* ══ サブタイトル ══ */
.subtitle p {
    font-family: 'DotGothic16', monospace !important;
    font-size: 12px !important;
    color: #aa66cc !important;
    text-align: center !important;
    letter-spacing: 3px !important;
    margin-bottom: 16px !important;
}

/* ══ セクション見出し ══ */
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

/* ══ サイズ情報バー ══ */
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
}

/* ══ パネル ══ */
.gr-group, .gr-box {
    background: rgba(255,255,255,0.80) !important;
    border: 2px solid #e8d0ff !important;
    border-radius: 14px !important;
    box-shadow: 4px 4px 0 #ddc8f8, 0 0 20px #cc88ff18 !important;
}

/* ══ ラベル ══ */
label > span, .gr-form > label > span {
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #7733bb !important;
    letter-spacing: 0.5px !important;
}

/* ══ スライダー ══ */
input[type=range] {
    accent-color: #cc44bb !important;
    height: 5px !important;
    cursor: pointer !important;
}

/* ══ チェックボックス ══ */
input[type=checkbox] {
    accent-color: #cc44bb !important;
    width: 18px !important;
    height: 18px !important;
}

/* ══ テキストエリア ══ */
textarea, .gr-textbox textarea {
    font-family: 'M PLUS Rounded 1c', sans-serif !important;
    font-size: 14px !important;
    background: #fdf8ff !important;
    border: 2px solid #e0c8ff !important;
    border-radius: 10px !important;
    color: #553388 !important;
}

/* ══ ステータスボックス ══ */
.status-box textarea {
    font-family: 'DotGothic16', monospace !important;
    font-size: 13px !important;
    color: #664499 !important;
    background: #fef8ff !important;
    border: 2px dashed #cc99ee !important;
    border-radius: 10px !important;
}

/* ══ プライマリボタン（変換実行） ══ */
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
button.primary:hover {
    background: linear-gradient(180deg, #ffaadd 0%, #ff55bb 100%) !important;
    box-shadow: 5px 5px 0 #aa1177, 0 0 30px #ff66cc66 !important;
}
button.primary:active {
    transform: translate(5px, 5px) !important;
    box-shadow: 0 0 0 #aa1177 !important;
}

/* ══ セカンダリボタン ══ */
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
button.secondary:hover {
    background: linear-gradient(180deg, #cceeff 0%, #99ccff 100%) !important;
}
button.secondary:active {
    transform: translate(4px, 4px) !important;
    box-shadow: 0 0 0 #4499cc !important;
}

/* ══ タブ ══ */
.tab-nav { border-bottom: 3px solid #e0c0ff !important; }
.tab-nav button {
    font-family: 'DotGothic16', monospace !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    color: #9966cc !important;
    background: transparent !important;
    padding: 10px 22px !important;
    border-radius: 10px 10px 0 0 !important;
    transition: all 0.15s !important;
}
.tab-nav button:hover { background: #f5e8ff !important; color: #6633aa !important; }
.tab-nav button.selected {
    color: #cc33aa !important;
    background: linear-gradient(180deg,#ffe0f8,#fff5ff) !important;
    border-bottom: 3px solid #cc33aa !important;
    font-weight: bold !important;
}

/* ══ アコーディオン ══ */
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

/* ══ ギャラリー ══ */
.gallery-item {
    border: 3px solid #e0c8ff !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 3px 3px 0 #ddb8ff !important;
}
.gallery-item:hover { border-color: #cc55ee !important; }

/* ══ 画像ウィジェット枠 ══ */
.image-container {
    border: 2px solid #e8d0ff !important;
    border-radius: 12px !important;
    background: #fdf8ff !important;
}

footer { display: none !important; }
"""

# ─── UI ───────────────────────────────────────────────

with gr.Blocks(title="ピクセルアートジェネレーター") as demo:

    snapped_state   = gr.State(None)
    lineart_state   = gr.State(None)
    sm_labels_state = gr.State(None)
    sm_isline_state = gr.State(None)
    sm_region_state = gr.State(None)
    sm_orig_state   = gr.State(None)

    gr.Markdown("# ✨ ピクセルアート ジェネレーター ✨", elem_classes="title")
    gr.Markdown("AIイラスト → ピクセルアート → 透過PNG  の自動変換ツール",
                elem_classes="subtitle")

    # ══════════════════════════════════════════════════
    # 画像 ＋ 基本設定
    # ══════════════════════════════════════════════════
    gr.Markdown("📁 **画像をロードして設定しよう**", elem_classes="section-head")

    with gr.Row(equal_height=False):

        # 左：画像アップロード
        with gr.Column(scale=1, min_width=280):
            input_image = gr.Image(type="pil", label="📷 入力画像", height=430)

        # 右：設定エリア
        with gr.Column(scale=2):

            size_bar = gr.Markdown("⬆ 画像をアップロードしてね！", elem_classes="size-bar")

            # ── ドット・スケール設定 ──
            gr.Markdown("🔲 **ドット設定**", elem_classes="section-head")
            with gr.Row():
                dot_count = gr.Slider(32, 512, value=64, step=8,
                    label="ドット数（横）",
                    info="少ない：粗くてレトロ　多い：細かくてなめらか")
                display_scale = gr.Slider(1, 16, value=1, step=1,
                    label="表示倍率",
                    info="1ドットを何pxで表示するか。大きいとドットがくっきり")

            # ── カラー設定 ──
            gr.Markdown("🎨 **カラー設定**", elem_classes="section-head")
            k_colors = gr.Slider(4, 64, value=16, step=4,
                label="カラーパレット数",
                info="少ない：よりドット絵らしく　多い：原画に近い色味")

            # ── 背景透過設定 ──
            gr.Markdown("✂ **背景透過設定**", elem_classes="section-head")
            with gr.Row():
                do_remove_bg = gr.Checkbox(value=True, label="背景を透過する")
                bg_tolerance = gr.Slider(0, 60, value=15, step=1,
                    label="色の許容誤差",
                    info="0：指定色のみ除去　60：近似色も広く除去")
            bg_mode = gr.Radio(
                choices=["白背景", "カスタム色"],
                value="白背景",
                label="透過する背景の色",
            )
            with gr.Row(visible=False) as custom_color_row:
                custom_color = gr.ColorPicker(
                    value="#ffffff",
                    label="🎨 背景色（入力画像をクリックしてピック）",
                )
                gr.Markdown(
                    "← 左の入力画像をクリックすると\nその場所の色が自動でセットされます",
                    elem_classes="subtitle",
                )

    # サイズ情報リアルタイム更新
    for comp in [input_image, dot_count, display_scale]:
        comp.change(fn=size_text,
                    inputs=[input_image, dot_count, display_scale],
                    outputs=size_bar)

    # 入力画像クリック → カスタム色ピック
    input_image.select(fn=h_pick_color, inputs=[input_image], outputs=[custom_color])

    # bg_mode 切り替え → カスタム色ピッカーの表示/非表示
    bg_mode.change(
        fn=lambda m: gr.update(visible=(m == "カスタム色")),
        inputs=[bg_mode],
        outputs=[custom_color_row],
    )

    # ══════════════════════════════════════════════════
    # カラー調整（アコーディオン）
    # ══════════════════════════════════════════════════
    with gr.Accordion("🌈 カラー調整 — 変換後にリアルタイムで色味を変えられます", open=False):
        gr.Markdown("変換実行後にスライダーを動かすと、すぐにプレビューに反映されます。",
                    elem_classes="subtitle")
        with gr.Row():
            saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.05,
                label="彩度",
                info="低い：グレーっぽく　高い：鮮やかに　（1.0 = 変化なし）")
            brightness = gr.Slider(0.2, 2.0, value=1.0, step=0.05,
                label="明度",
                info="低い：暗く　高い：明るく　（1.0 = 変化なし）")
        with gr.Row():
            contrast = gr.Slider(0.2, 3.0, value=1.0, step=0.05,
                label="コントラスト",
                info="低い：フラット　高い：メリハリ　（1.0 = 変化なし）")
            hue_shift = gr.Slider(-180, 180, value=0, step=5,
                label="色相シフト（度）",
                info="0 = 変化なし　±180で色が反転")

    color_inputs = [saturation, brightness, contrast, hue_shift, do_remove_bg, bg_mode, custom_color, bg_tolerance]

    # ══════════════════════════════════════════════════
    # タブ
    # ══════════════════════════════════════════════════
    with gr.Tabs():

        # ── タブ1：1枚変換 ──
        with gr.TabItem("🎮 変換する"):

            run_btn = gr.Button("✨  変換実行！", variant="primary", size="lg")

            with gr.Row():
                out_snap = gr.Image(type="pil",
                    label="🔲 ピクセルスナップ結果",
                    height=360, image_mode="RGB",
                    )
                out_final = gr.Image(type="pil",
                    label="✅ 完成！（チェック柄 = 透過部分）",
                    height=360, image_mode="RGB",
                    )

            info_box = gr.Textbox(label="📋 ステータス", lines=2,
                                  interactive=False, elem_classes="status-box")
            dl_final = gr.DownloadButton("📥 透過PNG をダウンロード",
                variant="secondary", visible=False)

            run_btn.click(
                fn=h_snap,
                inputs=[input_image, dot_count, display_scale, k_colors] + color_inputs,
                outputs=[out_snap, out_final, info_box, snapped_state, dl_final],
            )
            for sl in color_inputs:
                sl.change(fn=h_color,
                          inputs=[snapped_state] + color_inputs,
                          outputs=out_final)

        # ── タブ2：線画ツール ──
        with gr.TabItem("✏ 線画ツール"):

            gr.Markdown("### STEP 1　線画抽出（輝度 → 透明度変換）", elem_classes="section-head")
            gr.Markdown(
                "白・明るい部分 → 透明　/　黒・暗い部分 → 不透明（線のみを残す）",
                elem_classes="subtitle"
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    la_input = gr.Image(type="pil", label="📷 線画入力", height=380)
                    la_threshold = gr.Slider(0, 255, value=128, step=1,
                        label="閾値（2値化の境界）",
                        info="0=スムーズ変換　128=標準2値化　255=ほぼ全域不透明")
                    la_btn = gr.Button("✨  線画抽出！", variant="primary")

                with gr.Column(scale=1, min_width=260):
                    la_preview = gr.Image(type="pil",
                        label="✅ 抽出結果（チェック柄 = 透過）", height=380,
                        )
                    la_dl = gr.DownloadButton("📥 線画を透過PNG でダウンロード",
                        variant="secondary", visible=False)
                    la_info = gr.Textbox(label="📋 ステータス", lines=1,
                        interactive=False, elem_classes="status-box")

            gr.Markdown("---")
            gr.Markdown("### STEP 2　下塗り（キャラクターの内側を1色で塗る）",
                        elem_classes="section-head")
            gr.Markdown(
                "STEP 1 で抽出した線画を使って下塗り。線は変えず・背景は透過のまま・キャラだけ塗る。",
                elem_classes="subtitle"
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    bc_color = gr.ColorPicker(value="#ff88cc", label="🎨 塗り色を選択")
                    bc_btn = gr.Button("🖌  下塗り実行！", variant="primary")

                with gr.Column(scale=1, min_width=260):
                    bc_preview = gr.Image(type="pil",
                        label="✅ 下塗り結果（チェック柄 = 透過）", height=380,
                        )
                    bc_dl = gr.DownloadButton("📥 下塗り画像を透過PNG でダウンロード",
                        variant="secondary", visible=False)
                    bc_info = gr.Textbox(label="📋 ステータス", lines=1,
                        interactive=False, elem_classes="status-box")

            la_btn.click(
                fn=h_lineart_extract,
                inputs=[la_input, la_threshold],
                outputs=[la_preview, la_info, la_dl, lineart_state],
            )
            bc_btn.click(
                fn=h_base_coat,
                inputs=[lineart_state, bc_color],
                outputs=[bc_preview, bc_info, bc_dl],
            )

        # ── タブ：スマート抽出＆ピクセル化 ──
        with gr.TabItem("🪄 スマート抽出"):
            gr.Markdown(
                "**3枚の画像** をアップロードして、線画と背景色塗り画像を手がかりに "
                "キャラ領域を自動判別＆クリックで微修正 → 大元画像をピクセル化＆透過します。",
                elem_classes="subtitle"
            )

            gr.Markdown("📁 **3枚アップロード（同じサイズ推奨／自動リサイズあり）**",
                        elem_classes="section-head")
            with gr.Row():
                sm_line_in = gr.Image(type="pil", label="① 線画", height=260)
                sm_orig_in = gr.Image(type="pil", label="② 大元の白背景キャラ（最終ソース）", height=260)
                sm_bg_in   = gr.Image(type="pil", label="③ 背景だけ色を変えた画像", height=260)

            with gr.Row():
                sm_line_thr = gr.Slider(10, 200, value=80, step=5,
                    label="線画の線判定しきい値",
                    info="低い：細い線も拾う　高い：濃い線のみ")
                sm_bg_tol = gr.Slider(0, 80, value=30, step=1,
                    label="背景色の許容誤差",
                    info="③の四隅から背景色を自動検出。値を上げるとより広く背景判定")

            with gr.Row():
                sm_build_btn = gr.Button("🔍 自動マスク生成", variant="primary")
                sm_reset_btn = gr.Button("♻ 現在の許容誤差で再判定", variant="secondary")

            gr.Markdown(
                "🖱 **下の画像をクリックすると、その連結領域を「キャラ ↔ 背景」反転できます。**\n"
                "緑がかった部分が「背景」と判定された領域。塗り残しの島は1クリックで埋まり、"
                "背景にはみ出した部分も1クリックで除去できます。",
                elem_classes="subtitle"
            )
            sm_overlay = gr.Image(type="pil", label="🎯 マスクプレビュー（クリックで領域トグル）",
                                  height=460, interactive=False)
            sm_info = gr.Textbox(label="📋 ステータス", lines=2, interactive=False,
                                 elem_classes="status-box")

            gr.Markdown("---")
            gr.Markdown("✨ **マスクが整ったらピクセル化＆透過**", elem_classes="section-head")
            gr.Markdown(
                "上部の「ドット設定／カラー設定／カラー調整」がそのまま使われます。"
                "（背景透過チェックは無視 — マスクで透過します）",
                elem_classes="subtitle"
            )
            sm_final_btn = gr.Button("🎨 ピクセル化＆透過を実行！", variant="primary", size="lg")
            with gr.Row():
                sm_final_raw = gr.Image(type="pil", label="✅ 透過PNG（生）", height=380)
                sm_final_prev = gr.Image(type="pil", label="✅ 透過プレビュー（チェック柄）", height=380)
            sm_final_info = gr.Textbox(label="📋 ステータス", lines=1, interactive=False,
                                       elem_classes="status-box")
            sm_dl = gr.DownloadButton("📥 透過PNG をダウンロード",
                                      variant="secondary", visible=False)

            sm_build_btn.click(
                fn=h_smart_build,
                inputs=[sm_line_in, sm_orig_in, sm_bg_in, sm_line_thr, sm_bg_tol],
                outputs=[sm_labels_state, sm_isline_state, sm_region_state,
                         sm_orig_state, sm_overlay, sm_info],
            )
            sm_reset_btn.click(
                fn=h_smart_reset,
                inputs=[sm_labels_state, sm_isline_state, sm_orig_state, sm_bg_in, sm_bg_tol],
                outputs=[sm_overlay, sm_region_state, sm_info],
            )
            sm_overlay.select(
                fn=h_smart_click,
                inputs=[sm_labels_state, sm_isline_state, sm_region_state, sm_orig_state],
                outputs=[sm_overlay, sm_region_state, sm_info],
            )
            sm_final_btn.click(
                fn=h_smart_finalize,
                inputs=[sm_labels_state, sm_isline_state, sm_region_state, sm_orig_state,
                        dot_count, display_scale, k_colors,
                        saturation, brightness, contrast, hue_shift],
                outputs=[sm_final_raw, sm_final_prev, sm_final_info, sm_dl],
            )

        # ── タブ3：5パターン比較 ──
        with gr.TabItem("🔍 5パターン比較"):
            gr.Markdown(
                "**64 / 96 / 128 / 192 / 256 ドット** の5種類を一気に生成して比較できます。\n"
                "気に入ったパターンを選んで保存しましょう！",
                elem_classes="subtitle"
            )
            compare_btn = gr.Button("✨  5パターンを生成する！", variant="primary", size="lg")
            compare_info = gr.Textbox(label="📋 ステータス", lines=1,
                                      interactive=False, elem_classes="status-box")

            gallery = gr.Gallery(
                label="🖼 比較ギャラリー（クリックで拡大）",
                columns=5, height=380, object_fit="contain"
            )

            gr.Markdown("---")
            gr.Markdown("💾 **気に入ったパターンを保存**", elem_classes="section-head")
            with gr.Row():
                save_dot = gr.Dropdown(
                    choices=[64, 96, 128, 192, 256], value=128,
                    label="保存するドット数を選択", type="value"
                )
                comp_save_btn = gr.Button("💾 選択したパターンを保存", variant="secondary")
            comp_dl   = gr.DownloadButton("📥 透過PNG をダウンロード", variant="secondary", visible=False)
            comp_info = gr.Textbox(label="", lines=1, interactive=False, elem_classes="status-box")

            compare_btn.click(
                fn=h_compare,
                inputs=[input_image, k_colors, display_scale] + color_inputs,
                outputs=[gallery, compare_info],
            )
            comp_save_btn.click(
                fn=h_compare_save,
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
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.pink,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.purple,
            font=gr.themes.GoogleFont("M PLUS Rounded 1c"),
        ),
    )
