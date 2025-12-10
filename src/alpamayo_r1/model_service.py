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

"""Model service for loading and managing the AlpamayoR1 model."""

from typing import Any

import torch

from alpamayo_r1.api_config import config
from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class ModelService:
    """Singleton service for managing the AlpamayoR1 model."""

    _instance: "ModelService | None" = None
    _model: AlpamayoR1 | None = None
    _processor: Any | None = None
    _device: str | None = None
    _dtype: torch.dtype | None = None

    def __new__(cls) -> "ModelService":
        """Create or return existing singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model service (only runs once due to singleton)."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load the model and processor."""
        # Determine dtype
        if config.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif config.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._dtype = dtype
        self._device = config.device

        print(f"Loading model {config.model_name} on {self._device} with dtype {dtype}...")
        self._model = AlpamayoR1.from_pretrained(config.model_name, dtype=dtype).to(self._device)
        print("Model loaded successfully.")

        print("Initializing processor...")
        self._processor = helper.get_processor(self._model.tokenizer)
        print("Processor initialized.")

    @property
    def model(self) -> AlpamayoR1:
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        return self._model

    @property
    def processor(self) -> Any:
        """Get the processor."""
        if self._processor is None:
            raise RuntimeError("Processor not initialized. Call _load_model() first.")
        return self._processor

    @property
    def device(self) -> str:
        """Get the device."""
        if self._device is None:
            raise RuntimeError("Device not set. Call _load_model() first.")
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype."""
        if self._dtype is None:
            raise RuntimeError("Dtype not set. Call _load_model() first.")
        return self._dtype

    def infer(
        self,
        model_inputs: dict[str, Any],
        top_p: float = 0.98,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Run inference on the model.

        Args:
            model_inputs: Dictionary containing tokenized_data, ego_history_xyz, ego_history_rot
            top_p: Top-p sampling parameter
            temperature: Sampling temperature
            num_traj_samples: Number of trajectory samples
            max_generation_length: Maximum generation length

        Returns:
            Tuple of (pred_xyz, pred_rot, extra) where extra contains CoT and other text outputs
        """
        # Move inputs to device
        model_inputs = helper.to_device(model_inputs, self.device)

        # Run inference
        with torch.autocast(self.device, dtype=self.dtype):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=num_traj_samples,
                max_generation_length=max_generation_length,
                return_extra=True,
            )

        return pred_xyz, pred_rot, extra


# Global singleton instance
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Get the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
