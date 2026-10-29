from memslides.agents.agent import Agent
from memslides.utils.constants import (
    FORCE_FINALIZE_MSG,
    MAX_MODIFY_ITERATIONS,
)
from memslides.utils.log import info, warning
from memslides.utils.typings import ChatMessage, InputRequest, Role


class RevisionEditor(Agent):
    """Agent specialized for multi-turn slide revisions.

    Uses RevisionEditor.yaml config which includes asset tools (image_caption,
    image_generation) and search tools, providing a broader toolset than
    the deck designer for revision tasks.
    """

    async def loop(self, req: InputRequest, **kwargs):
        _iter = 0
        outcome = None
        while True:
            _iter += 1
            if _iter > MAX_MODIFY_ITERATIONS:
                warning(
                    f"RevisionEditor.loop() exceeded max iterations ({MAX_MODIFY_ITERATIONS})"
                )
                self.chat_history.append(
                    ChatMessage(role=Role.USER, content=FORCE_FINALIZE_MSG["text"])
                )
                agent_message = await self.action(
                    prompt=req.designagent_prompt,
                )
                yield agent_message
                if agent_message.tool_calls:
                    outcome = await self.execute(agent_message.tool_calls)
                break

            agent_message = await self.action(
                prompt=req.designagent_prompt,
            )
            yield agent_message
            if not agent_message.tool_calls:
                break

            outcome = await self.execute(agent_message.tool_calls)

            if isinstance(outcome, list):
                for item in outcome:
                    yield item
            else:
                break

        if outcome is not None:
            if isinstance(outcome, list):
                for item in outcome:
                    yield item
            else:
                yield outcome

    async def finish(self, result: str):
        info(f"RevisionEditor agent finished with result: {result}")
