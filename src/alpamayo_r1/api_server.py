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

import time
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
        # Get model service
        model_service = get_model_service()

        # Prepare model inputs
        model_inputs = prepare_model_inputs(
            messages=request.messages,
            trajectory_history=request.trajectory_history,
            clip_id=request.clip_id,
            t0_us=request.t0_us,
            processor=model_service.processor,
        )

        # Run inference
        pred_xyz, pred_rot, extra = model_service.infer(
            model_inputs=model_inputs,
            top_p=request.top_p,
            temperature=request.temperature,
            num_traj_samples=request.num_traj_samples,
            max_generation_length=request.max_generation_length,
        )

        # Format outputs
        # For now, return the first trajectory sample
        formatted_outputs = format_model_outputs(
            pred_xyz=pred_xyz,
            pred_rot=pred_rot,
            extra=extra,
            batch_idx=0,
            traj_set_idx=0,
            traj_sample_idx=0,
        )

        # Create response
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
        # Count tokens in input and output
        input_tokens = model_inputs["tokenized_data"]["input_ids"].numel()
        output_tokens = len(formatted_outputs["content"].split()) * 1.3  # Rough estimate

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

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "alpamayo_r1.api_server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
