"""Rollback Manager — single-slide HTML rollback with parameter version restore."""

from __future__ import annotations

import logging
from pathlib import Path

from ..store.params_store import SlideParamsStore

logger = logging.getLogger(__name__)


class RollbackManager:
    """回滚管理器：支持单页级别的PPT回滚（含磁盘持久化）"""

    CHECKPOINT_DIR_NAME = ".rollback"

    def __init__(self, params_store: SlideParamsStore):
        self.params_store = params_store
        self._html_checkpoints: dict[str, str] = {}  # slide_id → HTML content
        self._checkpoint_dir: Path | None = None

    def set_checkpoint_dir(self, workspace: Path) -> None:
        """设置磁盘持久化目录（修改开始前调用）"""
        self._checkpoint_dir = workspace / self.CHECKPOINT_DIR_NAME
        self._checkpoint_dir.mkdir(exist_ok=True)

    def create_checkpoint(self, slide_id: str, html_content: str) -> None:
        """创建回滚点（修改前调用），同时写入磁盘"""
        self._html_checkpoints[slide_id] = html_content
        if self._checkpoint_dir:
            try:
                (self._checkpoint_dir / f"{slide_id}.html").write_text(
                    html_content, encoding="utf-8"
                )
            except Exception as e:
                logger.debug("Checkpoint disk write failed (non-fatal): %s", e)

    def load_from_disk(self, workspace: Path) -> int:
        """从磁盘加载检查点（session 恢复时调用），返回加载数量"""
        self.set_checkpoint_dir(workspace)
        loaded = 0
        if self._checkpoint_dir and self._checkpoint_dir.is_dir():
            for f in self._checkpoint_dir.glob("*.html"):
                slide_id = f.stem
                if slide_id not in self._html_checkpoints:
                    try:
                        self._html_checkpoints[slide_id] = f.read_text(encoding="utf-8")
                        loaded += 1
                    except Exception as e:
                        logger.debug("Checkpoint disk read failed: %s", e)
        return loaded

    async def rollback(self, slide_id: str, slide_path: Path) -> bool:
        """回滚到上一个检查点"""
        if slide_id not in self._html_checkpoints:
            return False

        # 恢复HTML
        slide_path.write_text(self._html_checkpoints[slide_id], encoding="utf-8")

        # 恢复参数版本
        version = self.params_store.get_version_count(slide_id)
        if version >= 2:
            self.params_store.rollback(slide_id, version - 1)

        del self._html_checkpoints[slide_id]
        # 删除磁盘持久化文件
        if self._checkpoint_dir:
            try:
                (self._checkpoint_dir / f"{slide_id}.html").unlink(missing_ok=True)
            except Exception:
                pass
        return True

    def has_checkpoint(self, slide_id: str) -> bool:
        """检查是否有回滚点"""
        return slide_id in self._html_checkpoints

    def clear_checkpoint(self, slide_id: str) -> None:
        """修改成功后清除检查点"""
        self._html_checkpoints.pop(slide_id, None)
        if self._checkpoint_dir:
            try:
                (self._checkpoint_dir / f"{slide_id}.html").unlink(missing_ok=True)
            except Exception:
                pass

    def clear_all(self) -> None:
        """清除所有检查点"""
        self._html_checkpoints.clear()
        if self._checkpoint_dir and self._checkpoint_dir.is_dir():
            for f in self._checkpoint_dir.glob("*.html"):
                try:
                    f.unlink()
                except Exception:
                    pass
