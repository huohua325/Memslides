"""Implementation details."""

from ..core.models import RoundSummary

FULL_SUMMARY_WINDOW = 5
MAX_TOTAL_SUMMARIES = 20


class JobHistory:
    """Implementation details."""

    def __init__(self):
        self._summaries: list[RoundSummary] = []

    def add_summary(self, summary: RoundSummary) -> None:
        """Implementation details."""
        self._summaries.append(summary)
        if len(self._summaries) > MAX_TOTAL_SUMMARIES:
            self._summaries.pop(0)

    def build_context_prompt(self) -> str:
        """Implementation details."""
        if not self._summaries:
            return ""

        lines = ["<round_history>"]


        if len(self._summaries) > FULL_SUMMARY_WINDOW:
            early_count = len(self._summaries) - FULL_SUMMARY_WINDOW
            for i in range(early_count):
                s = self._summaries[i]
                req_preview = s.user_request[:50] + "..." if len(s.user_request) > 50 else s.user_request
                slides_str = ", ".join(s.slides_modified) if s.slides_modified else "无"
                lines.append(f"第{s.round_index}轮: {req_preview} → {s.outcome} (修改: {slides_str})")

            lines.append("")


        recent_start = max(0, len(self._summaries) - FULL_SUMMARY_WINDOW)
        for i in range(recent_start, len(self._summaries)):
            lines.append(self._summaries[i].to_prompt_text())
            lines.append("")

        lines.append("</round_history>")
        return "\n".join(lines)

    def release(self) -> None:
        """Implementation details."""
        self._summaries.clear()
