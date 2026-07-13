"""
AURORA - Backend Server
Listens to audio, analyzes it, and streams the result to the browser (visual
engine) over WebSocket.

Run:
    pip install -r requirements.txt          # core (simulation works out of the box)
    pip install -r requirements-mic.txt      # optional: real microphone (PyAudio)
    python main.py
    -> Browser: http://localhost:5000

Engineering notes:
  * Microphone reads are BLOCKING (PyAudio), so they run in a worker thread via
    run_in_executor - the asyncio event loop is never blocked.
  * A session seed is generated once per run: the shader field stays stable for
    the whole session, while each analyzed instant still gets its own moment ID.
  * WebSocket connections are origin-checked (cross-site hijacking guard) and
    capped at MAX_CLIENTS. The server binds to localhost unless ALLOW_LAN=True.
  * Every packet carries a server timestamp so the client can display measured
    pipeline latency (valid on localhost, where clocks are identical).
"""

import asyncio
import json
import secrets
import time
from pathlib import Path

import numpy as np
from aiohttp import web

from config import (
    HOST, PORT, COLORS, EMOTION_COLORS, MAX_CLIENTS, EXTRA_ALLOWED_ORIGINS,
    SAMPLE_RATE, CHUNK_SIZE, USE_REAL_MIC, SILENCE_THRESHOLD, LOG_INTERVAL,
)
from audio_engine import EmotionAnalyzer

analyzer = EmotionAnalyzer()
connected_clients: set = set()
SESSION_SEED = secrets.randbelow(100000)   # fixed for the whole session


# ---------------------------------------------------------------------- #
# Audio sources
# ---------------------------------------------------------------------- #
def open_microphone():
    """Open the real microphone (PyAudio). Returns (None, None) on failure."""
    try:
        import pyaudio
    except ImportError:
        print(f"{COLORS['YELLOW']}[Aurora] PyAudio not installed "
              f"(pip install -r requirements-mic.txt) -> simulation.{COLORS['RESET']}")
        return None, None
    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK_SIZE,
        )
        print(f"{COLORS['GREEN']}[Aurora] Microphone opened.{COLORS['RESET']}")
        return stream, pa
    except Exception as exc:
        print(f"{COLORS['RED']}[Aurora] Microphone error: {exc} -> simulation.{COLORS['RESET']}")
        return None, None


def read_chunk_blocking(stream) -> np.ndarray:
    """Blocking PyAudio read - executed in a worker thread, never on the loop."""
    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    return np.frombuffer(raw, dtype=np.float32)


def simulated_chunk(t0: float) -> np.ndarray:
    """When there is no microphone, generate a living demo: a slowly drifting
    pitch plus noise, so anyone cloning the repo sees the visuals breathe."""
    t = np.linspace(0, CHUNK_SIZE / SAMPLE_RATE, CHUNK_SIZE, endpoint=False)
    freq = 220.0 + 180.0 * (0.5 + 0.5 * np.sin(t0 * 0.35))       # drifting note
    amp = 0.18 + 0.12 * (0.5 + 0.5 * np.sin(t0 * 0.9))           # breathing level
    signal = amp * np.sin(2 * np.pi * freq * t)
    signal += 0.4 * amp * np.sin(2 * np.pi * freq * 2 * t)       # harmonic
    signal += np.random.normal(0, 0.03, CHUNK_SIZE)             # ambient noise
    return signal.astype(np.float32)


# ---------------------------------------------------------------------- #
# Analysis loop
# ---------------------------------------------------------------------- #
async def audio_processor():
    loop = asyncio.get_running_loop()
    stream, pa = (open_microphone() if USE_REAL_MIC else (None, None))
    live = stream is not None
    print(f"{COLORS['CYAN']}[Aurora] Analysis started "
          f"({'MICROPHONE' if live else 'SIMULATION'}). Session seed: #{SESSION_SEED:05d}{COLORS['RESET']}")

    t0 = 0.0
    last_log = 0.0
    try:
        while True:
            if live:
                try:
                    # blocking read happens in a worker thread
                    audio = await loop.run_in_executor(None, read_chunk_blocking, stream)
                except Exception as exc:
                    print(f"{COLORS['RED']}[Aurora] Read error: {exc}{COLORS['RESET']}")
                    audio = np.zeros(CHUNK_SIZE, dtype=np.float32)
            else:
                await asyncio.sleep(0.08)
                audio = simulated_chunk(t0)
                t0 += 0.08

            vol = float(np.sqrt(np.mean(audio ** 2)))
            if vol > SILENCE_THRESHOLD:
                try:
                    r = analyzer.analyze(audio)
                except Exception as exc:
                    # analysis must never kill the loop; skip this chunk
                    print(f"{COLORS['RED']}[Aurora] Analysis error: {exc}{COLORS['RESET']}")
                    continue

                now = time.time()
                if now - last_log >= LOG_INTERVAL:
                    last_log = now
                    color = EMOTION_COLORS.get(r["emotion"], COLORS["WHITE"])
                    print(f"{color}NOTE: {r['note']:<2} | PITCH: {r['pitch_hz']:>7.1f} Hz | "
                          f"CENTROID: {r['centroid']:>7.0f} | RMS: {r['rms']:.3f} | "
                          f"EMOTION: {r['emotion']:<8} | ID: #{r['moment_id']:05d}{COLORS['RESET']}")

                packet = {
                    "rms": r["rms"],
                    "rms_norm": r["rms_norm"],
                    "centroid": r["centroid"],
                    "pitch": r["pitch_hz"],       # real pitch (Hz)
                    "note": r["note"],
                    "emotion": r["emotion"],
                    "confidence": r["confidence"],
                    "moment_id": r["moment_id"],
                    "session_seed": SESSION_SEED,
                    "ts": now * 1000.0,           # server timestamp (ms) for latency HUD
                }
                await broadcast(json.dumps(packet))
    except asyncio.CancelledError:
        pass
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if pa:
            pa.terminate()


async def broadcast(message: str):
    for client in list(connected_clients):
        if client.closed:
            connected_clients.discard(client)
            continue
        try:
            await client.send_str(message)
        except Exception:
            connected_clients.discard(client)
            try:
                await client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------- #
# HTTP + WebSocket
# ---------------------------------------------------------------------- #
def origin_allowed(request) -> bool:
    """Cross-site WebSocket hijacking guard: only same-host origins (or an
    explicit allowlist) may open the socket. No Origin header = non-browser
    client (e.g. curl/tests) - allowed, it has no cross-site ambient authority."""
    origin = request.headers.get("Origin")
    if origin is None:
        return True
    allowed = {
        f"http://{request.host}",
        f"https://{request.host}",
        f"http://localhost:{PORT}",
        f"http://127.0.0.1:{PORT}",
    } | EXTRA_ALLOWED_ORIGINS
    return origin in allowed


async def websocket_handler(request):
    if not origin_allowed(request):
        raise web.HTTPForbidden(text="Origin not allowed")
    if len(connected_clients) >= MAX_CLIENTS:
        raise web.HTTPTooManyRequests(text="Viewer limit reached")

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    connected_clients.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        connected_clients.discard(ws)
    return ws


async def index_handler(request):
    return web.FileResponse(Path(__file__).parent / "public" / "index.html")


async def start_server():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/public", path=Path(__file__).parent / "public", name="public")

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()

    bar = COLORS["MAGENTA"] + "=" * 44 + COLORS["RESET"]
    print(bar)
    print(f"{COLORS['GREEN']}   AURORA STARTED!  (http://localhost:{PORT}){COLORS['RESET']}")
    print(f"{COLORS['CYAN']}   Bound to {HOST} | viewer cap: {MAX_CLIENTS}{COLORS['RESET']}")
    print(bar)

    asyncio.create_task(audio_processor())
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nAurora stopped.")
