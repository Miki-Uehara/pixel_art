#!/usr/bin/env python3
"""
ピクセルアート生成パイプライン
  Step 1: Gemini API でイラスト生成
  Step 2: Pixel Art Snapper でピクセルアート化
  Step 3: 白背景を透過（内部の白は保持）
  Step 4: ファイル名＋日付付きで保存

使い方:
  python generate_pixel_art.py "a cute cat character" --name cat_sprite
  python generate_pixel_art.py "a fantasy sword" --name sword --colors 8
  python generate_pixel_art.py --skip-generate --input path/to/image.png --name sword
"""

import argparse
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# .env ファイルを読み込む
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from steps.step1_generate import generate_image
from steps.step2_pixel_snap import process_image as pixel_snap_image, PixelSnapConfig
from steps.step3_remove_bg import process_image as remove_bg
from steps.step4_save import save_final

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "output" / "work"
FINAL_DIR = BASE_DIR / "output" / "final"


def run_pipeline(
    prompt: str | None,
    name: str,
    api_key: str | None,
    skip_generate: bool = False,
    input_image: Path | None = None,
    k_colors: int = 16,
    output_scale: int = 1,
    bg_tolerance: int = 15,
) -> Path:
    print("=" * 50)
    print("ピクセルアート生成パイプライン 開始")
    print("=" * 50)

    # Step 1: 画像生成
    if skip_generate:
        if input_image is None:
            raise ValueError("--skip-generate の場合は --input でファイルを指定してください")
        step1_path = input_image
        print(f"[Step 1] スキップ: {step1_path}")
    else:
        if not prompt:
            raise ValueError("プロンプトを指定してください")
        step1_path = generate_image(
            prompt=prompt,
            api_key=api_key,
            output_dir=WORK_DIR / "step1",
            filename_prefix=name,
        )

    # Step 2: ピクセルスナップ
    cfg = PixelSnapConfig(k_colors=k_colors, output_scale=output_scale)
    step2_path = pixel_snap_image(
        input_path=step1_path,
        output_dir=WORK_DIR / "step2",
        cfg=cfg,
    )

    # Step 3: 白背景除去
    step3_path = remove_bg(
        input_path=step2_path,
        output_dir=WORK_DIR / "step3",
        tolerance=bg_tolerance,
    )

    # Step 4: 最終保存
    final_path = save_final(
        input_path=step3_path,
        output_dir=FINAL_DIR,
        base_name=name,
    )

    print()
    print("=" * 50)
    print(f"完了！ → {final_path}")
    print("=" * 50)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIイラスト → ピクセルアート生成パイプライン"
    )
    parser.add_argument("prompt", nargs="?", help="生成プロンプト（英語推奨）")
    parser.add_argument("--name", "-n", default="pixel_art", help="出力ファイルのベース名")
    parser.add_argument("--api-key", help="Gemini API key（未指定時は GEMINI_API_KEY 環境変数を使用）")
    parser.add_argument("--skip-generate", action="store_true", help="Step 1をスキップして既存画像を使用")
    parser.add_argument("--input", "-i", type=Path, help="--skip-generate 時の入力画像パス")
    parser.add_argument("--colors", "-c", type=int, default=16, help="ピクセルアートのカラーパレット数 (デフォルト: 16)")
    parser.add_argument("--scale", "-s", type=int, default=1, help="出力アップスケール倍率 (デフォルト: 1)")
    parser.add_argument("--bg-tolerance", type=int, default=15, help="白背景判定の許容誤差 0-255 (デフォルト: 15)")

    args = parser.parse_args()

    run_pipeline(
        prompt=args.prompt,
        name=args.name,
        api_key=args.api_key,
        skip_generate=args.skip_generate,
        input_image=args.input,
        k_colors=args.colors,
        output_scale=args.scale,
        bg_tolerance=args.bg_tolerance,
    )


if __name__ == "__main__":
    main()
