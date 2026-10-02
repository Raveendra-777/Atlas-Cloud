import os
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Atlas Cloud")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL")


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
        "status": "ok",
        "nvidia_key_present": bool(NVIDIA_API_KEY),
        "nvidia_model": NVIDIA_MODEL,
        "nvidia_base_url": NVIDIA_BASE_URL
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    print("=== ATLAS CHAT REQUEST ===")
    print("Message:", request.message)
    print("NVIDIA key present:", bool(NVIDIA_API_KEY))
    print("NVIDIA model:", NVIDIA_MODEL)
    print("NVIDIA URL:", NVIDIA_BASE_URL)

    if not NVIDIA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_API_KEY is missing in Render"
        )

    if not NVIDIA_MODEL:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_MODEL is missing in Render"
        )

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Atlas, a fast and helpful voice assistant. "
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

        print("Sending request to NVIDIA...")

        async with httpx.AsyncClient(timeout=30.0) as client:

            response = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )

        print("NVIDIA status:", response.status_code)
        print("NVIDIA response:", response.text)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        print("Atlas answer:", answer)

        return {
            "response": answer
        }

    except httpx.RequestError as error:

        print("NVIDIA connection error:", error)

        raise HTTPException(
            status_code=502,
            detail=f"NVIDIA connection failed: {error}"
        )

    except Exception as error:

        print("Atlas internal error:", error)

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
