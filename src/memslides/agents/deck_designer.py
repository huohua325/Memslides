from memslides.agents.agent import Agent
from memslides.utils.constants import (
    FORCE_FINALIZE_MSG,
    MAX_AGENT_ITERATIONS,
)
from memslides.utils.log import warning
from memslides.utils.typings import ChatMessage, InputRequest, Role
from memslides.runtime.deck_execution_state import render_deck_progress_prompt


class DeckDesigner(Agent):
    async def loop(self, req: InputRequest, markdown_file: str):
        (self.workspace / "slides").mkdir(exist_ok=True)
        _iter = 0
        outcome = None
        initial_progress_prompt = render_deck_progress_prompt(self.workspace)
        if initial_progress_prompt:
            self.chat_history.append(
                ChatMessage(role=Role.USER, content=initial_progress_prompt)
            )
        while True:
            _iter += 1
            if _iter > MAX_AGENT_ITERATIONS:
                warning(
                    f"DeckDesigner.loop() exceeded max iterations ({MAX_AGENT_ITERATIONS})"
                )
                self.chat_history.append(
                    ChatMessage(role=Role.USER, content=FORCE_FINALIZE_MSG["text"])
                )
                agent_message = await self.action(
                    markdown_file=markdown_file, prompt=req.designagent_prompt
                )
                yield agent_message
                if agent_message.tool_calls:
                    outcome = await self.execute(agent_message.tool_calls)
                break

            agent_message = await self.action(
                markdown_file=markdown_file, prompt=req.designagent_prompt
            )
            yield agent_message
            if not agent_message.tool_calls:
                break

            outcome = await self.execute(self.chat_history[-1].tool_calls)

            if isinstance(outcome, list):
                for item in outcome:
                    yield item
                progress_prompt = render_deck_progress_prompt(self.workspace)
                if progress_prompt:
                    self.chat_history.append(
                        ChatMessage(role=Role.USER, content=progress_prompt)
                    )
            else:
                break

        if outcome is not None:
            if isinstance(outcome, list):
                for item in outcome:
                    yield item
            else:
                yield outcome
