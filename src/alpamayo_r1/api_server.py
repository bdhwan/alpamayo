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

"""FastAPI server for OpenAI-compatible chat completion API."""

import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from alpamayo_r1.api_config import config
from alpamayo_r1.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessageResponse,
    TrajectoryPrediction,
    Usage,
)
from alpamayo_r1.api_utils import format_model_outputs, prepare_model_inputs
from alpamayo_r1.model_service import get_model_service

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Load model
    print("Starting API server...")
    model_service = get_model_service()
    print(f"Model service initialized: {model_service.model}")
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down API server...")


app = FastAPI(
    title="Alpamayo R1 API",
    description="OpenAI-compatible chat completion API for Alpamayo R1 model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Chat completions endpoint matching OpenAI format.

    Args:
        request: Chat completion request

    Returns:
        Chat completion response
    """
    try:
        logger.info(f"Received chat completion request: model={request.model}, "
                   f"num_images={len(request.messages)}, "
                   f"clip_id={request.clip_id}, "
                   f"num_traj_samples={request.num_traj_samples}")

        # Get model service
        logger.debug("Getting model service...")
        model_service = get_model_service()
        logger.debug("Model service retrieved successfully")

        # Prepare model inputs
        logger.info("Preparing model inputs...")
        try:
            model_inputs = prepare_model_inputs(
                messages=request.messages,
                trajectory_history=request.trajectory_history,
                clip_id=request.clip_id,
                t0_us=request.t0_us,
                processor=model_service.processor,
            )
            logger.info(f"Model inputs prepared: "
                       f"input_shape={model_inputs['tokenized_data']['input_ids'].shape}, "
                       f"history_shape={model_inputs['ego_history_xyz'].shape}")
        except Exception as e:
            logger.error(f"Error preparing model inputs: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400,
                detail=f"Failed to prepare model inputs: {str(e)}"
            )

        # Run inference
        logger.info("Running model inference...")
        try:
            pred_xyz, pred_rot, extra = model_service.infer(
                model_inputs=model_inputs,
                top_p=request.top_p,
                temperature=request.temperature,
                num_traj_samples=request.num_traj_samples,
                max_generation_length=request.max_generation_length,
            )
            logger.info(f"Inference completed: "
                       f"pred_xyz_shape={pred_xyz.shape}, "
                       f"pred_rot_shape={pred_rot.shape}, "
                       f"extra_keys={list(extra.keys()) if extra else None}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Model inference failed: {str(e)}"
            )

        # Format outputs
        logger.debug("Formatting model outputs...")
        try:
            formatted_outputs = format_model_outputs(
                pred_xyz=pred_xyz,
                pred_rot=pred_rot,
                extra=extra,
                batch_idx=0,
                traj_set_idx=0,
                traj_sample_idx=0,
            )
            logger.debug("Model outputs formatted successfully")
        except Exception as e:
            logger.error(f"Error formatting model outputs: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to format model outputs: {str(e)}"
            )

        # Create response
        logger.debug("Creating response object...")
        try:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())

            # Create trajectory prediction object
            trajectory_pred = None
            if formatted_outputs["trajectory"]:
                trajectory_pred = TrajectoryPrediction(
                    xyz=formatted_outputs["trajectory"]["xyz"],
                    rotation=formatted_outputs["trajectory"]["rotation"],
                )

            choice = ChatChoice(
                index=0,
                message=ChatMessageResponse(
                    role="assistant",
                    content=formatted_outputs["content"],
                ),
                finish_reason="stop",
                chain_of_thought=formatted_outputs["chain_of_thought"],
                meta_action=formatted_outputs["meta_action"],
                answer=formatted_outputs["answer"],
                trajectory=trajectory_pred,
            )

            # Estimate token usage (rough approximation)
            input_tokens = model_inputs["tokenized_data"]["input_ids"].numel()
            output_tokens = len(formatted_outputs["content"].split()) * 1.3

            usage = Usage(
                prompt_tokens=int(input_tokens),
                completion_tokens=int(output_tokens),
                total_tokens=int(input_tokens + output_tokens),
            )

            response = ChatCompletionResponse(
                id=response_id,
                created=created,
                model=request.model,
                choices=[choice],
                usage=usage,
            )

            logger.info(f"Response created successfully: id={response_id}")
            return response
        except Exception as e:
            logger.error(f"Error creating response: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create response: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "alpamayo_r1.api_server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
