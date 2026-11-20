"""FastAPI entry point for the RiderLink LLM SOS dispatcher."""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.dispatcher import generate_chat_reply, generate_dispatch
from app.models import ChatRequest, ChatResponse, CrashTelemetry, SOSDispatchResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RiderLink SOS Dispatcher", version="1.0.0")

dashboard_origins = [
    origin.strip()
    for origin in os.getenv(
        "DASHBOARD_ORIGINS",
        "http://127.0.0.1:8001,http://localhost:8001",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=dashboard_origins,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/api/v1/sos-dispatch", response_model=SOSDispatchResponse)
async def sos_dispatch(telemetry: CrashTelemetry) -> SOSDispatchResponse:
    try:
        dispatch_text = await generate_dispatch(telemetry)
        return SOSDispatchResponse(dispatch_text=dispatch_text)
    except Exception as exc:
        logger.exception("SOS dispatch generation failed")
        raise HTTPException(status_code=502, detail="SOS dispatch generation failed") from exc


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        reply = await generate_chat_reply(request.messages)
        return ChatResponse(reply=reply)
    except Exception as exc:
        logger.exception("Chat response generation failed")
        raise HTTPException(status_code=502, detail="Chat response generation failed") from exc
