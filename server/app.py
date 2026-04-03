"""FastAPI application for the Data Cleaning OpenEnv environment."""

from __future__ import annotations

import sys
import os

# Ensure root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ActionWrapper, DataCleaningObservation
from server.environment import DataCleaningEnvironment
from openenv.core import create_app

app = create_app(
    env=DataCleaningEnvironment,
    action_cls=ActionWrapper,
    observation_cls=DataCleaningObservation,
    env_name="data_cleaning_env",
)
