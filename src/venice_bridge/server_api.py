"""
Venice Bridge OpenAI-Compatible API Server
Allows standard OpenAI API clients to use Venice AI through the bridge.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from venice_bridge.client import VeniceClient, VeniceError

logger = logging.getLogger("venice_bridge.api")
app = FastAPI(title="Venice AI OpenAI-Compatible API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = VeniceClient()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v4-1-flash-uncensored"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    web_enabled: Optional[bool] = True
    web_scrape_enabled: Optional[bool] = True


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "chroma"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"


def _format_content(content: Any) -> str:
    """Extracts text content even if passed in multimodal OpenAI parts format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[Imagem anexada]")
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


@app.get("/v1/models")
async def list_models():
    """Lists available models formatted for OpenAI client compatibility."""
    try:
        data = client.list_models()
        chat_models = data.get("agentModelOptions", [])
        models_data = []

        # Always ensure primary models are top
        seen = set()
        primary_defaults = [
            ("venice-uncensored-1-2", "Venice Uncensored 1.2"),
            ("deepseek-v4-1-flash-uncensored", "DeepSeek V4.1 Flash Uncensored"),
        ]
        for mid, mname in primary_defaults:
            models_data.append({
                "id": mid,
                "object": "model",
                "created": 1700000000,
                "owned_by": "venice",
            })
            seen.add(mid)

        for m in chat_models:
            mid = m.get("id")
            if mid and mid not in seen:
                models_data.append({
                    "id": mid,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "venice",
                })
                seen.add(mid)

        return {"object": "list", "data": models_data}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {
            "object": "list",
            "data": [
                {"id": "venice-uncensored-1-2", "object": "model", "created": 1700000000, "owned_by": "venice"},
                {"id": "deepseek-v4-1-flash-uncensored", "object": "model", "created": 1700000000, "owned_by": "venice"},
            ]
        }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    system_prompt = ""
    user_prompt = ""
    history_messages = []

    for msg in req.messages:
        text_content = _format_content(msg.content)
        if msg.role == "system":
            system_prompt = text_content
        elif msg.role in ("user", "assistant"):
            history_messages.append({"role": msg.role, "content": text_content})

    if history_messages and history_messages[-1]["role"] == "user":
        user_prompt = history_messages[-1]["content"]

    try:
        res = client.chat(
            prompt=user_prompt or "Hello",
            model=req.model,
            messages=history_messages,
            system_prompt=system_prompt,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
            web_enabled=True if req.web_enabled is None else req.web_enabled,
            web_scrape_enabled=True if req.web_scrape_enabled is None else req.web_scrape_enabled,
        )

        reply_text = res["text"]
        cmpl_id = res.get("meta", {}).get("completion_id", f"chatcmpl-{int(time.time()*1000)}")
        created_time = int(time.time())

        if req.stream:
            async def event_generator():
                # Single chunk delta for stream compatibility
                chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": reply_text},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return {
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created_time,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_prompt.split()),
                "completion_tokens": len(reply_text.split()),
                "total_tokens": len(user_prompt.split()) + len(reply_text.split())
            }
        }
    except VeniceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/v1/images/generations")
async def images_generations(req: ImageGenerationRequest):
    """OpenAI-compatible image generation endpoint."""
    width, height = 1024, 1024
    if req.size:
        try:
            parts = req.size.lower().split("x")
            width, height = int(parts[0]), int(parts[1])
        except Exception:
            pass

    try:
        res = client.generate_image(
            prompt=req.prompt,
            model=req.model or "chroma",
            width=width,
            height=height,
        )
        return {
            "created": int(time.time()),
            "data": [
                {
                    "b64_json": res["base64"],
                    "revised_prompt": req.prompt,
                }
            ],
        }
    except VeniceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def main():
    """Runs the FastAPI server with uvicorn on localhost:8000."""
    import uvicorn
    uvicorn.run("venice_bridge.server_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
