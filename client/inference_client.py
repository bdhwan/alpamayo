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

"""Python client for Alpamayo R1 inference API."""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image


class AlpamayoClient:
    """Client for interacting with Alpamayo R1 API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: int = 300,
        verbose: bool = False,
    ):
        """Initialize the client.

        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
            verbose: Whether to print verbose error messages
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.verbose = verbose

    def encode_image(self, image_path: str | Path) -> str:
        """Encode an image file to base64 data URL.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded data URL string
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Verify it's actually an image file before encoding
        try:
            with Image.open(image_path) as img:
                img.verify()
            # Reopen after verify (verify closes the image)
            with Image.open(image_path) as img:
                # Get actual format
                actual_format = img.format
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file {image_path}: {str(e)}") from e

        # Determine image format from extension
        ext = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        # Read and encode image
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            if len(image_bytes) == 0:
                raise ValueError(f"Image file is empty: {image_path}")
            image_data = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read/encode image {image_path}: {str(e)}") from e

        return f"data:{mime_type};base64,{image_data}"

    def get_recent_images(
        self,
        folder_path: str | Path,
        n: int = 4,
        sort_by: str = "mtime",
    ) -> list[Path]:
        """Get the n most recently created/modified image files from a folder.

        Args:
            folder_path: Path to the folder containing images
            n: Number of recent images to return (default: 4)
            sort_by: Sort by 'mtime' (modification time) or 'ctime' (creation time)
                (default: 'mtime')

        Returns:
            List of Path objects for the n most recent image files, sorted by
            most recent first

        Raises:
            FileNotFoundError: If the folder doesn't exist
            ValueError: If n is less than 1 or sort_by is invalid
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if sort_by not in ("mtime", "ctime"):
            raise ValueError(f"sort_by must be 'mtime' or 'ctime', got {sort_by}")

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        # Collect all image files with their timestamps
        image_files = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                try:
                    if sort_by == "mtime":
                        timestamp = file_path.stat().st_mtime
                    else:  # ctime
                        timestamp = file_path.stat().st_ctime
                    image_files.append((timestamp, file_path))
                except OSError:
                    # Skip files that can't be accessed
                    continue

        # Sort by timestamp (most recent first) and return top n
        image_files.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in image_files[:n]]

    def inference(
        self,
        image_paths: list[str | Path],
        clip_id: str | None = None,
        t0_us: int | None = None,
        trajectory_history: dict[str, Any] | None = None,
        temperature: float = 0.6,
        top_p: float = 0.98,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
        model: str = "alpamayo-r1",
    ) -> dict[str, Any]:
        """Run inference on images.

        Args:
            image_paths: List of paths to image files
            clip_id: Optional clip ID to load trajectory history from dataset
            t0_us: Optional timestamp in microseconds (used with clip_id)
            trajectory_history: Optional trajectory history data dict with
                'ego_history_xyz' and 'ego_history_rot' keys
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_traj_samples: Number of trajectory samples
            max_generation_length: Maximum generation length
            model: Model name

        Returns:
            API response as dictionary
        """
        if not image_paths:
            raise ValueError("At least one image path is required")

        # Encode all images
        image_contents = []
        for img_path in image_paths:
            image_url = self.encode_image(img_path)
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )

        # Build request payload
        payload = {
            "model": model,
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
        }

        # Add trajectory history options
        if clip_id is not None:
            payload["clip_id"] = clip_id
            if t0_us is not None:
                payload["t0_us"] = t0_us
        elif trajectory_history is not None:
            payload["trajectory_history"] = trajectory_history

        # Make request
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Extract detailed error message from response
            error_detail = None
            if hasattr(e.response, "text"):
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get("detail", error_response.get("message", str(e)))
                except (ValueError, KeyError):
                    error_detail = e.response.text or str(e)
            else:
                error_detail = str(e)

            error_msg = f"API request failed with status {e.response.status_code}: {error_detail}"
            if self.verbose:
                error_msg += f"\nFull response: {e.response.text}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            if self.verbose:
                error_msg += f"\nRequest URL: {self.api_url}"
                error_msg += f"\nPayload keys: {list(payload.keys())}"
            raise RuntimeError(error_msg) from e

    def health_check(self) -> dict[str, Any]:
        """Check if the API server is healthy.

        Returns:
            Health check response
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Health check failed: {e}") from e


def main():
    """Command-line interface for the inference client."""
    parser = argparse.ArgumentParser(
        description="Alpamayo R1 Inference API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using clip_id with specific image files (recommended)
  python inference_client.py image1.jpg image2.jpg image3.jpg image4.jpg \
    --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" --t0-us 5100000

  # Using folder with n most recent images
  python inference_client.py --folder /path/to/images \
    --num-images 4 --clip-id "030c760c-ae38-49aa-9ad8-f5650a545d26" --t0-us 5100000

  # Using custom API URL
  python inference_client.py image1.jpg --clip-id "xxx" \
    --api-url "http://localhost:8001"

  # Save response to file
  python inference_client.py --folder /path/to/images --num-images 4 \
    --clip-id "xxx" --output result.json --pretty
        """,
    )

    parser.add_argument(
        "images",
        nargs="*",
        help="Paths to image files or a folder path (supports multiple images or one folder)",
    )
    parser.add_argument(
        "--folder",
        help="Path to folder containing images (will use n most recent images)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of recent images to use when --folder is specified (default: 4)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["mtime", "ctime"],
        default="mtime",
        help="Sort images by modification time (mtime) or creation time (ctime) (default: mtime)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8001",
        help="Base URL of the API server (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--clip-id",
        help="Clip ID to load trajectory history from dataset",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        help="Timestamp in microseconds (used with --clip-id)",
    )
    parser.add_argument(
        "--trajectory-history",
        help="Path to JSON file containing trajectory history",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.98,
        help="Top-p sampling parameter (default: 0.98)",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        help="Number of trajectory samples (default: 1)",
    )
    parser.add_argument(
        "--max-gen-length",
        type=int,
        default=256,
        help="Maximum generation length (default: 256)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save the response JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print the JSON response",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose error messages including full server response",
    )

    args = parser.parse_args()

    # Initialize client
    client = AlpamayoClient(base_url=args.api_url, verbose=args.verbose)

    # Check health
    try:
        health = client.health_check()
        print(f"✓ API server is healthy: {health}")
    except Exception as e:
        print(f"⚠ Warning: Health check failed: {e}")
        print("Continuing anyway...")

    # Determine image paths
    image_paths = []
    if args.folder:
        # Use folder mode - get recent images from folder
        print(f"Loading {args.num_images} most recent images from folder: {args.folder}")
        try:
            image_paths = client.get_recent_images(
                folder_path=args.folder,
                n=args.num_images,
                sort_by=args.sort_by,
            )
            if not image_paths:
                print(f"⚠ Warning: No image files found in folder: {args.folder}")
                return 1
            print(f"Found {len(image_paths)} image(s): {[str(p.name) for p in image_paths]}")
        except Exception as e:
            print(f"✗ Error loading images from folder: {e}")
            return 1
    elif args.images:
        # Use provided image paths
        image_paths = args.images
    else:
        print("✗ Error: Either provide image file paths or use --folder option")
        parser.print_help()
        return 1

    # Load trajectory history if provided
    trajectory_history = None
    if args.trajectory_history:
        with open(args.trajectory_history, "r") as f:
            trajectory_history = json.load(f)

    # Run inference
    try:
        print(f"\nRunning inference on {len(image_paths)} image(s)...")
        result = client.inference(
            image_paths=image_paths,
            clip_id=args.clip_id,
            t0_us=args.t0_us,
            trajectory_history=trajectory_history,
            temperature=args.temperature,
            top_p=args.top_p,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_gen_length,
        )

        # Print results
        if args.pretty:
            print("\n" + "=" * 80)
            print("RESPONSE:")
            print("=" * 80)
            print(json.dumps(result, indent=2))
        else:
            print("\nResponse received successfully!")
            if result.get("choices"):
                choice = result["choices"][0]
                if choice.get("chain_of_thought"):
                    print(f"\nChain of Thought:\n{choice['chain_of_thought']}")
                if choice.get("trajectory"):
                    traj = choice["trajectory"]
                    print(f"\nTrajectory shape: xyz={len(traj['xyz'])}, rotation={len(traj['rotation'])}")

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2 if args.pretty else None)
            print(f"\n✓ Response saved to {args.output}")

    except RuntimeError as e:
        print(f"\n✗ API Error: {e}")
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
