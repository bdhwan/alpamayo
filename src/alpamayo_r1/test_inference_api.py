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

# API call example that replicates test_inference.py functionality:
# This script loads a dataset, converts images to files, calls the API,
# and optionally computes minADE. It demonstrates how to use the API
# to perform the same inference as test_inference.py.
#
# Alternative simpler approach using inference_client.py:
#   from alpamayo.client.inference_client import AlpamayoClient
#   client = AlpamayoClient(base_url="http://localhost:8001")
#   result = client.inference(
#       image_paths=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
#       clip_id="030c760c-ae38-49aa-9ad8-f5650a545d26",
#       t0_us=5_100_000,
#       temperature=0.6,
#       top_p=0.98,
#       num_traj_samples=1,
#       max_generation_length=256,
#   )

import base64
import tempfile
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


def tensor_to_image_file(tensor: torch.Tensor, output_path: Path) -> None:
    """Convert a torch tensor image to a file.

    Args:
        tensor: Image tensor of shape (C, H, W) with values in [0, 255] (uint8)
        output_path: Path to save the image file
    """
    # Convert to numpy and ensure correct shape
    if tensor.dim() == 3:
        # (C, H, W) -> (H, W, C)
        img_array = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"Expected 3D tensor (C, H, W), got shape {tensor.shape}")

    # Ensure uint8 and valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Convert to PIL Image and save
    img = Image.fromarray(img_array)
    img.save(output_path)


def main():
    """Main function demonstrating API call equivalent to test_inference.py."""
    # Same parameters as test_inference.py
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    t0_us = 5_100_000
    api_url = "http://localhost:8001"

    # Same inference parameters as test_inference.py
    temperature = 0.6
    top_p = 0.98
    num_traj_samples = 1
    max_generation_length = 256

    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    print("Dataset loaded.")

    # Extract images: data["image_frames"] is (N_cameras, num_frames, 3, H, W)
    # We need to flatten to (N_cameras * num_frames, 3, H, W) like test_inference.py does
    image_frames = data["image_frames"].flatten(0, 1)  # (N_cameras * num_frames, 3, H, W)

    # Save images to temporary files
    print(f"Converting {len(image_frames)} images to temporary files...")
    temp_dir = Path(tempfile.mkdtemp(prefix="alpamayo_api_test_"))
    image_paths = []
    for i, frame in enumerate(image_frames):
        img_path = temp_dir / f"frame_{i:03d}.jpg"
        tensor_to_image_file(frame, img_path)
        image_paths.append(str(img_path))
    print(f"Images saved to temporary directory: {temp_dir}")

    # Prepare API request
    # Encode images as base64 data URLs
    image_contents = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        image_contents.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

    payload = {
        "model": "alpamayo-r1",
        "messages": [
            {
                "role": "user",
                "content": image_contents,
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "num_traj_samples": num_traj_samples,
        "max_generation_length": max_generation_length,
        "clip_id": clip_id,
        "t0_us": t0_us,
    }

    # Call API
    print(f"\nCalling API at {api_url}/v1/chat/completions...")
    try:
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        print("API call successful!")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return 1

    # Print results
    print("\n" + "=" * 80)
    print("API RESPONSE:")
    print("=" * 80)

    if result.get("choices"):
        choice = result["choices"][0]
        if choice.get("chain_of_thought"):
            print("\nChain-of-Causation (per trajectory):")
            print(choice["chain_of_thought"])

        if choice.get("trajectory"):
            traj = choice["trajectory"]
            print(f"\nTrajectory predictions received:")
            print(f"  - xyz shape: {len(traj['xyz'])} batches, "
                  f"{len(traj['xyz'][0])} traj_sets, "
                  f"{len(traj['xyz'][0][0])} traj_samples, "
                  f"{len(traj['xyz'][0][0][0])} steps")
            print(f"  - rotation shape: {len(traj['rotation'])} batches, "
                  f"{len(traj['rotation'][0])} traj_sets, "
                  f"{len(traj['rotation'][0][0])} traj_samples, "
                  f"{len(traj['rotation'][0][0][0])} steps")

            # Compute minADE if ground truth is available
            try:
                # Convert API response to numpy arrays
                pred_xyz = np.array(traj["xyz"])  # [batch, traj_set, traj_sample, steps, 3]
                # Extract xy coordinates: [traj_sample, 2, steps]
                pred_xy = pred_xyz[0, 0, :, :, :2].transpose(0, 2, 1)  # [traj_sample, steps, 2] -> [traj_sample, 2, steps]

                # Get ground truth
                gt_xyz = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]  # [steps, 2]
                gt_xy = gt_xyz.T  # [2, steps]

                # Compute minADE
                diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
                min_ade = diff.min()
                print(f"\nminADE: {min_ade:.4f} meters")
                print(
                    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
                    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
                    "variance in minADE is expected."
                )
            except Exception as e:
                print(f"\nNote: Could not compute minADE: {e}")

    # Cleanup temporary files
    print(f"\nCleaning up temporary directory: {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir)

    return 0


if __name__ == "__main__":
    exit(main())
