"""
Step 4: ファイル名・出力日付を付けて最終保存
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from PIL import Image
import shutil


def save_final(
    input_path: Path,
    output_dir: Path,
    base_name: str,
) -> Path:
    """
    ファイル名 + 出力日付（YYYYMMDD）を付けて output_dir に保存する。

    例: character_20260501.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"{base_name}_{date_str}.png"

    # 同名ファイルが存在する場合は連番を付与
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{base_name}_{date_str}_{counter:02d}.png"
        counter += 1

    shutil.copy2(input_path, output_path)
    print(f"[Step 4] 最終保存完了: {output_path}")
    return output_path
