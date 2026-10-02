import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Atlas Cloud")


NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {
        "name": "Atlas Cloud",
        "status": "online"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    if not NVIDIA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_API_KEY is not configured"
        )

    if not NVIDIA_MODEL:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_MODEL is not configured"
        )

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Atlas, a fast and helpful voice "
                    "assistant running on an ESP32. "
                    "Keep responses concise and natural."
                )
            },
            {
                "role": "user",
                "content": request.message
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:

            response = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        return {
            "response": answer
        }

    except httpx.RequestError as error:

        raise HTTPException(
            status_code=502,
            detail=f"NVIDIA connection failed: {error}"
        )
