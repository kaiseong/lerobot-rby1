#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Register in-tree robot configs that are not imported elsewhere before CLI parsing.
# Import only the config module to avoid pulling in robot runtime dependencies here.
from .rby1.config_rby1 import Rby1Config

__all__ = ["RobotConfig", "Robot", "make_robot_from_config", "Rby1Config"]
