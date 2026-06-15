"""JobHistory — WM 中的轮次历史摘要管理。"""

from ..core.models import RoundSummary

FULL_SUMMARY_WINDOW = 5
MAX_TOTAL_SUMMARIES = 20


class JobHistory:
    """管理 Job 内的 RoundSummary 滚动窗口。"""

    def __init__(self):
        self._summaries: list[RoundSummary] = []

    def add_summary(self, summary: RoundSummary) -> None:
        """添加 RoundSummary。"""
        self._summaries.append(summary)
        if len(self._summaries) > MAX_TOTAL_SUMMARIES:
            self._summaries.pop(0)

    def build_context_prompt(self) -> str:
        """构建历史摘要 prompt"""
        if not self._summaries:
            return ""

        lines = ["<round_history>"]

        # 早期轮次压缩为一行
        if len(self._summaries) > FULL_SUMMARY_WINDOW:
            early_count = len(self._summaries) - FULL_SUMMARY_WINDOW
            for i in range(early_count):
                s = self._summaries[i]
                req_preview = s.user_request[:50] + "..." if len(s.user_request) > 50 else s.user_request
                slides_str = ", ".join(s.slides_modified) if s.slides_modified else "无"
                lines.append(f"第{s.round_index}轮: {req_preview} → {s.outcome} (修改: {slides_str})")

            lines.append("")  # 空行分隔

        # 最近 FULL_SUMMARY_WINDOW 条完整输出
        recent_start = max(0, len(self._summaries) - FULL_SUMMARY_WINDOW)
        for i in range(recent_start, len(self._summaries)):
            lines.append(self._summaries[i].to_prompt_text())
            lines.append("")

        lines.append("</round_history>")
        return "\n".join(lines)

    def release(self) -> None:
        """释放所有数据"""
        self._summaries.clear()
