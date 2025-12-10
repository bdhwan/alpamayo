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

"""Pydantic models for OpenAI-compatible API requests and responses."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class ImageURL(BaseModel):
    """Image URL model for OpenAI format."""

    url: str = Field(..., description="Base64 encoded image data URL")


class TextContent(BaseModel):
    """Text content model."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content model."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class ChatMessage(BaseModel):
    """Chat message model matching OpenAI format."""

    role: Literal["system", "user", "assistant"]
    content: Union[str, list[Union[TextContent, ImageContent]]]


class TrajectoryHistory(BaseModel):
    """Trajectory history data."""

    ego_history_xyz: list[list[list[float]]] = Field(
        ..., description="Ego history xyz coordinates [batch, traj_group, steps, 3]"
    )
    ego_history_rot: list[list[list[list[float]]]] = Field(
        ..., description="Ego history rotation matrices [batch, traj_group, steps, 3, 3]"
    )


class ChatCompletionRequest(BaseModel):
    """Chat completion request model matching OpenAI format."""

    model: str = Field(default="alpamayo-r1", description="Model name")
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.98, ge=0.0, le=1.0, description="Top-p sampling parameter")
    num_traj_samples: int = Field(default=1, ge=1, description="Number of trajectory samples")
    max_generation_length: int = Field(default=256, ge=1, description="Maximum generation length")
    stream: bool = Field(default=False, description="Whether to stream the response")
    # Custom fields for trajectory history
    trajectory_history: TrajectoryHistory | None = Field(
        default=None, description="Optional trajectory history data"
    )
    clip_id: str | None = Field(default=None, description="Optional clip ID to load from dataset")
    t0_us: int | None = Field(default=None, description="Optional timestamp in microseconds")


class TrajectoryPrediction(BaseModel):
    """Trajectory prediction data."""

    xyz: list[list[list[float]]] = Field(
        ..., description="Predicted xyz coordinates [batch, traj_set, traj_sample, steps, 3]"
    )
    rotation: list[list[list[list[list[float]]]]] = Field(
        ...,
        description="Predicted rotation matrices [batch, traj_set, traj_sample, steps, 3, 3]",
    )


class ChatMessageResponse(BaseModel):
    """Chat message response model."""

    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    """Chat choice model with extended fields."""

    index: int
    message: ChatMessageResponse
    finish_reason: str | None = None
    # Extended fields for Alpamayo-specific data
    chain_of_thought: str | None = Field(default=None, description="Chain-of-thought reasoning")
    meta_action: str | None = Field(default=None, description="Meta action description")
    answer: str | None = Field(default=None, description="Answer text")
    trajectory: TrajectoryPrediction | None = Field(
        default=None, description="Trajectory predictions"
    )


class Usage(BaseModel):
    """Token usage model."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Chat completion response model matching OpenAI format."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage
