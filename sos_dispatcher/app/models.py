"""Validated request and response models for the SOS dispatcher."""

from datetime import datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CrashTelemetry(BaseModel):
    """Crash event forwarded by a RiderLink internet gateway."""

    model_config = ConfigDict(extra="forbid")

    rider_id: str = Field(min_length=1, max_length=128)
    bike_model: str = Field(default="Triumph Speed 400", min_length=1, max_length=128)
    speed_kmh: float = Field(ge=0, le=500)
    impact_g_force: float = Field(ge=0, le=100)
    pre_crash_lean_angle: float = Field(ge=-180, le=180, description="Degrees")
    coordinates: tuple[float, float] = Field(description="Latitude and longitude in decimal degrees")
    tumbles_detected: bool
    time_of_impact: datetime

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, value: tuple[float, float]) -> tuple[float, float]:
        latitude, longitude = value
        if not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return value


class SOSDispatchResponse(BaseModel):
    dispatch_text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)


class ChatResponse(BaseModel):
    reply: str
