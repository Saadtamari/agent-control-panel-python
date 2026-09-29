from __future__ import annotations

from typing import Literal

AgentStatus = Literal["idle", "running", "paused", "stopped"]


class AgentPausedError(Exception):
    """Raised by ``wrap_agent_run`` while the agent is paused or stopped.

    Catch it to skip work without treating it as an LLM failure.
    """

    def __init__(self, status: AgentStatus) -> None:
        super().__init__(f"Agent is {status}; skipping execution")
        self.status: AgentStatus = status
