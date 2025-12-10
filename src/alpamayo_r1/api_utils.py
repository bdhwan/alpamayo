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

"""Utility functions for converting between OpenAI format and model inputs/outputs."""

import base64
import io
from typing import Any

import numpy as np
import torch
from PIL import Image

from alpamayo_r1.api_models import ChatMessage, TrajectoryHistory
from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


def extract_images_from_messages(messages: list[ChatMessage]) -> list[torch.Tensor]:
    """Extract images from OpenAI message format and convert to torch tensors.

    Args:
        messages: List of chat messages in OpenAI format

    Returns:
        List of image tensors of shape (C, H, W)
    """
    images = []
    for message in messages:
        if isinstance(message.content, list):
            for content_item in message.content:
                # Handle both Pydantic models and dicts
                image_url = None
                if isinstance(content_item, dict):
                    content_type = content_item.get("type", "")
                    if content_type == "image_url":
                        image_url = content_item.get("image_url", {}).get("url", "")
                elif hasattr(content_item, "type"):
                    content_type = content_item.type
                    if content_type == "image_url":
                        image_url = content_item.image_url.url

                if image_url:
                    try:
                        # Extract base64 data
                        if image_url.startswith("data:image"):
                            # Format: data:image/png;base64,<base64_data>
                            if "," not in image_url:
                                raise ValueError(
                                    f"Invalid data URL format. Expected 'data:image/...;base64,<data>', "
                                    f"got: {image_url[:100]}..."
                                )
                            base64_data = image_url.split(",", 1)[1]
                        else:
                            # Assume it's raw base64
                            base64_data = image_url

                        if not base64_data:
                            raise ValueError("Empty base64 data")

                        # Decode base64 to image
                        try:
                            image_bytes = base64.b64decode(base64_data, validate=True)
                        except Exception as e:
                            raise ValueError(
                                f"Failed to decode base64 image data: {str(e)}. "
                                f"Image URL prefix: {image_url[:50] if len(image_url) > 50 else image_url}, "
                                f"Base64 data length: {len(base64_data)}"
                            ) from e

                        if len(image_bytes) == 0:
                            raise ValueError("Decoded image bytes are empty")

                        # Open image with PIL
                        try:
                            image = Image.open(io.BytesIO(image_bytes))
                            # Verify image was loaded correctly
                            image.verify()
                            # Reopen after verify (verify closes the image)
                            image = Image.open(io.BytesIO(image_bytes))
                        except Exception as e:
                            raise ValueError(
                                f"Failed to open image: {str(e)}. "
                                f"Image bytes length: {len(image_bytes)}, "
                                f"First 20 bytes (hex): {image_bytes[:20].hex() if len(image_bytes) >= 20 else 'too short'}"
                            ) from e

                        # Convert to RGB if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        # Convert PIL Image to torch tensor (H, W, C) -> (C, H, W)
                        image_array = np.array(image)
                        if image_array.size == 0:
                            raise ValueError("Image array is empty after conversion")
                        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
                        images.append(image_tensor)
                    except Exception as e:
                        raise ValueError(
                            f"Error processing image from message (index {len(images)}): {str(e)}"
                        ) from e

    return images


def prepare_model_inputs(
    messages: list[ChatMessage],
    trajectory_history: TrajectoryHistory | None = None,
    clip_id: str | None = None,
    t0_us: int | None = None,
    processor: Any = None,
) -> dict[str, Any]:
    """Prepare model inputs from OpenAI message format.

    Args:
        messages: List of chat messages in OpenAI format
        trajectory_history: Optional trajectory history data
        clip_id: Optional clip ID to load from dataset
        t0_us: Optional timestamp in microseconds
        processor: Model processor

    Returns:
        Dictionary with tokenized_data, ego_history_xyz, ego_history_rot
    """
    # Extract images from messages
    image_tensors = extract_images_from_messages(messages)

    if not image_tensors:
        raise ValueError("No images found in messages. At least one image is required.")

    # Stack images: (N, C, H, W) where N is number of images
    frames = torch.stack(image_tensors, dim=0)

    # Create messages using helper function
    model_messages = helper.create_message(frames)

    # Apply chat template
    if processor is None:
        raise ValueError("Processor is required")

    inputs = processor.apply_chat_template(
        model_messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Handle trajectory history
    if clip_id is not None:
        # Load from dataset
        if t0_us is None:
            t0_us = 5_100_000  # Default timestamp
        data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
    elif trajectory_history is not None:
        # Use provided trajectory history
        ego_history_xyz = torch.tensor(
            trajectory_history.ego_history_xyz, dtype=torch.float32
        )
        ego_history_rot = torch.tensor(
            trajectory_history.ego_history_rot, dtype=torch.float32
        )
    else:
        # Create zero history (may not work well, but allow it)
        # Default shape: (1, 1, 16, 3) for xyz and (1, 1, 16, 3, 3) for rot
        num_history_steps = 16
        ego_history_xyz = torch.zeros(1, 1, num_history_steps, 3, dtype=torch.float32)
        ego_history_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ego_history_rot = ego_history_rot.repeat(1, 1, num_history_steps, 1, 1)

    return {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def format_model_outputs(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    extra: dict[str, Any],
    batch_idx: int = 0,
    traj_set_idx: int = 0,
    traj_sample_idx: int = 0,
) -> dict[str, Any]:
    """Format model outputs for API response.

    Args:
        pred_xyz: Predicted xyz coordinates [batch, traj_set, traj_sample, steps, 3]
        pred_rot: Predicted rotation matrices [batch, traj_set, traj_sample, steps, 3, 3]
        extra: Extra outputs containing CoT and other text
        batch_idx: Batch index to extract
        traj_set_idx: Trajectory set index to extract
        traj_sample_idx: Trajectory sample index to extract

    Returns:
        Dictionary with formatted outputs
    """
    # Extract text outputs
    # The extra dict contains numpy arrays with shape [B, ns, nj] where each element is a string
    cot = None
    meta_action = None
    answer = None

    if "cot" in extra:
        cot_data = extra["cot"]
        if isinstance(cot_data, np.ndarray) and cot_data.ndim == 3:
            if (
                cot_data.shape[0] > batch_idx
                and cot_data.shape[1] > traj_set_idx
                and cot_data.shape[2] > traj_sample_idx
            ):
                cot = str(cot_data[batch_idx, traj_set_idx, traj_sample_idx])
        elif isinstance(cot_data, (list, np.ndarray)):
            # Handle flattened or different shapes
            try:
                cot = str(cot_data[batch_idx][traj_set_idx][traj_sample_idx])
            except (IndexError, TypeError):
                pass

    if "meta_action" in extra:
        meta_action_data = extra["meta_action"]
        if isinstance(meta_action_data, np.ndarray) and meta_action_data.ndim == 3:
            if (
                meta_action_data.shape[0] > batch_idx
                and meta_action_data.shape[1] > traj_set_idx
                and meta_action_data.shape[2] > traj_sample_idx
            ):
                meta_action = str(meta_action_data[batch_idx, traj_set_idx, traj_sample_idx])
        elif isinstance(meta_action_data, (list, np.ndarray)):
            try:
                meta_action = str(meta_action_data[batch_idx][traj_set_idx][traj_sample_idx])
            except (IndexError, TypeError):
                pass

    if "answer" in extra:
        answer_data = extra["answer"]
        if isinstance(answer_data, np.ndarray) and answer_data.ndim == 3:
            if (
                answer_data.shape[0] > batch_idx
                and answer_data.shape[1] > traj_set_idx
                and answer_data.shape[2] > traj_sample_idx
            ):
                answer = str(answer_data[batch_idx, traj_set_idx, traj_sample_idx])
        elif isinstance(answer_data, (list, np.ndarray)):
            try:
                answer = str(answer_data[batch_idx][traj_set_idx][traj_sample_idx])
            except (IndexError, TypeError):
                pass

    # Combine text outputs for message content
    content_parts = []
    if cot:
        content_parts.append(cot)
    if meta_action:
        content_parts.append(f"Meta Action: {meta_action}")
    if answer:
        content_parts.append(f"Answer: {answer}")

    content = "\n\n".join(content_parts) if content_parts else "No text output generated."

    # Extract trajectory predictions
    pred_xyz_np = pred_xyz.cpu().numpy()
    pred_rot_np = pred_rot.cpu().numpy()

    trajectory = {
        "xyz": pred_xyz_np.tolist(),
        "rotation": pred_rot_np.tolist(),
    }

    return {
        "content": content,
        "chain_of_thought": cot,
        "meta_action": meta_action,
        "answer": answer,
        "trajectory": trajectory,
    }
