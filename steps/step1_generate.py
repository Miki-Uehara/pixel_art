"""
Step 1: Gemini APIでピクセルアート風イラストを生成
モデル: gemini-3.1-flash-image-preview（最新の無料枠対応画像生成モデル）
"""

import os
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
import io


# Geminiの最新画像生成モデル
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

PIXEL_ART_SYSTEM_PROMPT = """You are a pixel art illustrator.
Generate a clean pixel art style illustration with:
- Clear pixel grid (16x16 or 32x32 sprite style)
- Limited color palette (16-32 colors)
- White background
- No anti-aliasing
- Sharp, clean edges
"""


def generate_image(
    prompt: str,
    api_key: str | None = None,
    output_dir: Path = Path("output/step1_generated"),
    filename_prefix: str = "generated",
) -> Path:
    """
    Gemini APIでピクセルアート風イラストを生成し、ファイルに保存する。
    Returns: 保存したPNGファイルのパス
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY が設定されていません。環境変数か引数で渡してください。")

    client = genai.Client(api_key=key)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_prompt = f"{PIXEL_ART_SYSTEM_PROMPT}\n\n{prompt}"

    print(f"[Step 1] Gemini ({GEMINI_IMAGE_MODEL}) で画像生成中...")
    print(f"  プロンプト: {prompt}")

    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{filename_prefix}_{timestamp}.png"

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            image_data = part.inline_data.data
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path, "PNG")
            print(f"  生成完了: {output_path} ({image.size})")
            return output_path

    raise RuntimeError("Gemini APIから画像を取得できませんでした。レスポンスに画像が含まれていません。")
