"""
AURORA - Configuration
Single source of truth. The server and engine read every setting from here.
"""

# --- Server ---
# Default binds to localhost only (safe). Set ALLOW_LAN = True to let devices
# on your local network (tablet/phone) view the stream.
ALLOW_LAN = False
HOST = "0.0.0.0" if ALLOW_LAN else "127.0.0.1"
PORT = 5000             # index.html derives the WebSocket URL automatically

# Security limits
MAX_CLIENTS = 8         # cap concurrent WebSocket viewers (basic DoS guard)
# Origins allowed to open the WebSocket (cross-site hijacking guard).
# Same-host origins are always allowed; add extras here if you serve elsewhere.
EXTRA_ALLOWED_ORIGINS: set = set()

# --- Audio capture ---
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048       # ~46 ms window @ 44.1 kHz -> low latency

# --- Run mode ---
# True  : Real microphone (for piano / live performance)
# False : Simulation (no mic; anyone who clones the repo still sees it running)
USE_REAL_MIC = True

# --- Analysis feature switches ---
ENABLE_PITCH = True     # real pitch (Hz) via librosa.yin
ENABLE_MFCC = True      # timbre characteristics
SILENCE_THRESHOLD = 0.01  # RMS below this counts as silence and is skipped

# Piano frequency range (yin fmin/fmax) - C2 .. C7
PITCH_FMIN = 65.0
PITCH_FMAX = 2093.0

# --- Console logging ---
LOG_INTERVAL = 0.5      # print at most one analysis line per this many seconds

# --- Console colors ---
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
}

# Emotion -> console color (for readable logs)
EMOTION_COLORS = {
    "calm": COLORS["BLUE"],
    "happy": COLORS["YELLOW"],
    "excited": COLORS["MAGENTA"],
    "sad": COLORS["CYAN"],
    "angry": COLORS["RED"],
    "neutral": COLORS["WHITE"],
}
