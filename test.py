"""
Quick smoke test: connect to the Data Cleaning environment and run
reset + a single explore step on titanic_easy.

Usage:
    ENV_URL=https://huggingface.co/spaces/shivshri/openenv_dataclean python test.py
"""

from __future__ import annotations

import asyncio
import os

from client import DataCleaningClient
from models import ExploreAction, DoneAction


ENV_URL = (
    os.getenv("ENV_BASE_URL")
    or os.getenv("ENV_URL")
    or "https://shivshri-openenv-dataclean.hf.space"
)


async def main():
    async with DataCleaningClient(base_url=ENV_URL) as client:
        # Reset
        result = await client.reset(task_id="titanic_easy")
        obs = result.observation
        print(f"Reset OK — reward={result.reward}")
        print(f"  Task: {obs.task_description[:120]}")
        print(f"  Errors: {len(obs.constraint_status or {})} tracked")

        # Explore step
        result = await client.step(ExploreAction(query="df.head()"))
        print(f"\nExplore step — reward={result.reward}, done={result.done}")
        print(f"  Explore result: {(result.observation.explore_result or '')[:200]}")

        # Done step
        result = await client.step(DoneAction())
        print(f"\nDone step — reward={result.reward}, done={result.done}")


if __name__ == "__main__":
    asyncio.run(main())
