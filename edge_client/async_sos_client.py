"""Example 100 Hz edge monitor that dispatches crash telemetry in the background.

Replace ``read_imu_sample`` with the device-specific nonblocking sensor read.
The mock crash is emitted once so a sustained threshold does not create duplicate SOS calls.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)
DISPATCH_URL = os.getenv("SOS_DISPATCH_URL", "http://localhost:8000/api/v1/sos-dispatch")


async def trigger_llm_dispatch(crash_data: dict[str, Any]) -> None:
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(DISPATCH_URL, json=crash_data) as response:
                response.raise_for_status()
                result: dict[str, Any] = await response.json()
                logger.warning("SOS dispatch response: %s", result.get("dispatch_text", ""))
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.exception("Could not send crash telemetry to SOS dispatcher: %s", exc)


async def read_imu_sample(sample_index: int) -> dict[str, float]:
    """Mock sensor sample. Replace with the actual IMU driver integration."""
    if sample_index == 25:
        return {"speed_kmh": 48.0, "impact_g_force": 4.2, "lean_angle": 52.0}
    return {"speed_kmh": 35.0, "impact_g_force": 0.2, "lean_angle": 8.0}


async def monitor_imu() -> None:
    sample_index = 0
    dispatch_tasks: set[asyncio.Task[None]] = set()
    crash_dispatched = False
    try:
        while True:
            sample = await read_imu_sample(sample_index)
            if sample["impact_g_force"] > 3.0 and not crash_dispatched:
                crash_dispatched = True
                crash_data = {
                    "rider_id": os.getenv("RIDER_ID", "demo-rider"),
                    "bike_model": "Triumph Speed 400",
                    "speed_kmh": sample["speed_kmh"],
                    "impact_g_force": sample["impact_g_force"],
                    "pre_crash_lean_angle": sample["lean_angle"],
                    "coordinates": [12.9716, 77.5946],
                    "tumbles_detected": False,
                    "time_of_impact": datetime.now(timezone.utc).isoformat(),
                }
                task = asyncio.create_task(trigger_llm_dispatch(crash_data))
                dispatch_tasks.add(task)
                task.add_done_callback(dispatch_tasks.discard)
                logger.critical("Crash threshold crossed; dispatch request running in background")

            sample_index += 1
            await asyncio.sleep(0.01)  # 100 Hz polling cadence
    finally:
        if dispatch_tasks:
            await asyncio.gather(*dispatch_tasks, return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor_imu())
