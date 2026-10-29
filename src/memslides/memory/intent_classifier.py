"""IntentClassifier — LLM 意图分类器

用于 modify 流程路由：判断用户消息的目标幻灯片和操作类型。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .prompts import build_intent_classification_prompt

logger = logging.getLogger(__name__)

LLMCallable = Any  # async callable(prompt: str) -> str


class IntentClassifier:
    """LLM 意图分类器"""

    def __init__(
        self,
        llm: LLMCallable,
        artifact_writer: Any = None,
        **_kwargs: Any,
    ):
        self.llm = llm
        self._artifact_writer = artifact_writer
        self._turn_id = 0

    def set_turn_id(self, turn_id: int) -> None:
        self._turn_id = turn_id

    async def classify(
        self,
        user_message: str,
        context_messages: list[dict] | None = None,
        current_slide: str = "",
    ) -> dict:
        """分类用户消息意图

        Returns:
            {"intent_type": str, "confidence": float, "target_slide": str, "modification_description": str}
        """
        prompt = build_intent_classification_prompt(
            user_message, context_messages, current_slide
        )
        result_text = await self._call_llm(prompt)

        try:
            intent_result = json.loads(result_text)
        except json.JSONDecodeError:
            intent_result = {
                "intent_type": "chat",
                "confidence": 0.3,
                "target_slide": "",
                "modification_description": user_message,
            }

        if self._artifact_writer and hasattr(self._artifact_writer, "write_intent_classification"):
            try:
                self._artifact_writer.write_intent_classification(
                    turn_id=self._turn_id,
                    user_message=user_message,
                    prompt=prompt,
                    response=result_text,
                    intent_result=intent_result,
                )
            except Exception as e:
                logger.debug("Intent artifact write failed: %s", e)

        return intent_result

    async def _call_llm(self, prompt: str) -> str:
        if callable(self.llm) and not hasattr(self.llm, "run"):
            return await self.llm(prompt)
        response = await self.llm.run(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
