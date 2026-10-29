from memslides.agents.agent import Agent
from memslides.utils.constants import (
    FORCE_FINALIZE_MSG,
    MAX_AGENT_ITERATIONS,
)
from memslides.utils.log import warning
from memslides.utils.typings import ChatMessage, InputRequest, Role


class TemplatePlanner(Agent):
    async def loop(self, req: InputRequest, markdown_file: str):
        _iter = 0
        outcome = None
        while True:
            _iter += 1
            if _iter > MAX_AGENT_ITERATIONS:
                warning(
                    f"TemplatePlanner.loop() exceeded max iterations ({MAX_AGENT_ITERATIONS})"
                )
                self.chat_history.append(
                    ChatMessage(role=Role.USER, content=FORCE_FINALIZE_MSG["text"])
                )
                agent_message = await self.action(
                    markdown_file=markdown_file, prompt=req.template_planner_prompt
                )
                yield agent_message
                if agent_message.tool_calls:
                    outcome = await self.execute(agent_message.tool_calls)
                break

            agent_message = await self.action(
                markdown_file=markdown_file, prompt=req.template_planner_prompt
            )
            yield agent_message
            if not agent_message.tool_calls:
                break

            outcome = await self.execute(self.chat_history[-1].tool_calls)

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
