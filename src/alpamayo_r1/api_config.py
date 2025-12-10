# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration management for the API server."""

import os
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for the API server."""

    model_name: str = os.getenv("ALPAMAYO_MODEL_NAME", "nvidia/Alpamayo-R1-10B")
    device: str = os.getenv("ALPAMAYO_DEVICE", "cuda")
    host: str = os.getenv("ALPAMAYO_HOST", "0.0.0.0")
    port: int = int(os.getenv("ALPAMAYO_PORT", "8001"))
    default_temperature: float = float(os.getenv("ALPAMAYO_TEMPERATURE", "0.6"))
    default_top_p: float = float(os.getenv("ALPAMAYO_TOP_P", "0.98"))
    default_num_traj_samples: int = int(os.getenv("ALPAMAYO_NUM_TRAJ_SAMPLES", "1"))
    default_max_generation_length: int = int(os.getenv("ALPAMAYO_MAX_GEN_LENGTH", "256"))
    dtype: str = os.getenv("ALPAMAYO_DTYPE", "bfloat16")


# Global configuration instance
config = APIConfig()
