"""Slide Parameters Store: versioned persistence + rollback.

Provides:
- SlideParamsStore: file-based versioned parameter storage with diff and rollback

Storage structure:
    workspace/.memory/params/
    ├── slide_01/
    │   ├── v1.json
    │   ├── v2.json
    │   └── latest.json
    └── slide_02/
        └── ...
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SlideParamsStore:
    """幻灯片参数持久化存储 + 版本管理"""

    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, slide_id: str, result: Any) -> int:
        """保存参数快照，返回版本号

        Args:
            slide_id: 幻灯片 ID
            result: 任何具有 params 和 confidence 属性的对象（如 ExtractionResult）
        """
        slide_dir = self.store_dir / slide_id
        slide_dir.mkdir(exist_ok=True)

        # 确定版本号
        existing = sorted(slide_dir.glob("v*.json"))
        version = len(existing) + 1

        # 保存
        data = {
            "version": version,
            "slide_id": slide_id,
            "params": result.params,
            "confidence": result.confidence,
            "timestamp": datetime.now().isoformat(),
        }
        (slide_dir / f"v{version}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 更新latest
        (slide_dir / "latest.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return version

    def get_latest(self, slide_id: str) -> dict | None:
        """获取最新版本参数"""
        path = self.store_dir / slide_id / "latest.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def get_version(self, slide_id: str, version: int) -> dict | None:
        """获取指定版本参数"""
        path = self.store_dir / slide_id / f"v{version}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def get_version_count(self, slide_id: str) -> int:
        """获取版本数量"""
        slide_dir = self.store_dir / slide_id
        if not slide_dir.exists():
            return 0
        return len(list(slide_dir.glob("v*.json")))

    def diff(self, slide_id: str, v1: int, v2: int) -> dict:
        """对比两个版本的参数差异"""
        data1 = self.get_version(slide_id, v1)
        data2 = self.get_version(slide_id, v2)
        if not data1 or not data2:
            return {}

        diffs = {}
        all_params = set(
            list(data1["params"].keys()) + list(data2["params"].keys())
        )
        for param in sorted(all_params):
            b = data1["params"].get(param)
            a = data2["params"].get(param)
            if b != a:
                delta = None
                if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                    delta = a - b
                diffs[param] = {"before": b, "after": a, "delta": delta}
        return diffs

    def rollback(self, slide_id: str, to_version: int) -> bool:
        """回滚到指定版本"""
        data = self.get_version(slide_id, to_version)
        if not data:
            return False
        latest_path = self.store_dir / slide_id / "latest.json"
        latest_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True

    def list_slides(self) -> list[str]:
        """列出所有有参数快照的slide ID"""
        if not self.store_dir.exists():
            return []
        return [
            d.name
            for d in self.store_dir.iterdir()
            if d.is_dir() and list(d.glob("v*.json"))
        ]
