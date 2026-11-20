# RiderLink SOS dispatcher

FastAPI service and example asyncio edge client for forwarding a detected crash to an LLM-generated responder broadcast.

## Run the service

```sh
cd sos_dispatcher
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service uses Gemini through LangChain. Set `GEMINI_MODEL` to select a supported model. Supply `GOOGLE_API_KEY` through a Kubernetes Secret; do not bake credentials into the image. The request schema is strict about extra fields and validates latitude/longitude. `coordinates` is `[latitude, longitude]`; `time_of_impact` must be an ISO 8601 datetime.

The simulator dashboard at `http://127.0.0.1:8001` posts its debounced crash event to this API. Local dashboard origins are allowed by default; set `DASHBOARD_ORIGINS` to a comma-separated list of exact trusted origins when deploying. The simulator uses a demo rider ID and example Bengaluru coordinates, and the LLM prompt labels those events as simulations.

The dashboard's Sarathi panel includes a user-initiated `tel:112` link labeled Indian Medical Services. India's Ministry of Home Affairs identifies 112 as the nationwide emergency number and routes health emergencies through its Emergency Response Support System. The Flutter profile's personal emergency contact remains a separate saved contact in the mobile app.

## Container

```sh
docker build -t riderlink-sos-dispatcher:local .
docker run --rm -p 8000:8000 -e GOOGLE_API_KEY="$GOOGLE_API_KEY" riderlink-sos-dispatcher:local
```

## Edge client

Install `aiohttp` (`python -m pip install aiohttp`) and run `python edge_client/async_sos_client.py` from the repository root. Configure `SOS_DISPATCH_URL` and `RIDER_ID` for the gateway. Replace `read_imu_sample` and the example GPS values with actual sensor and gateway data. The threshold decision stays in the local loop; HTTP is scheduled as a background task.

## API

`POST /api/v1/sos-dispatch` returns `{"dispatch_text":"..."}`. `POST /api/v1/chat` accepts up to 20 `{role, content}` messages and returns `{"reply":"..."}` for the dashboard chatbot. Missing model credentials or LLM failures return HTTP 502. Generated text is a responder aid and does not replace deterministic crash detection or professional medical assessment.
