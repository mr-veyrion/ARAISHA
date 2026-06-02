import queue
import numpy as np
import sounddevice as sd
import torch
import time
import nemo.collections.asr as nemo_asr
from collections import deque

# -------- CONFIG --------
MODEL_PATH = "./models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
SAMPLE_RATE = 16000
CHUNK_MS = 100

SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 2.2
MIN_UTTERANCE_DURATION = 2.5

PRE_ROLL_SEC = 0.8
POST_ROLL_SEC = 0.5
MAX_WINDOW_SEC = 20

# 🔹 Turn-repair config
MERGE_WINDOW_SEC = 2.5
TAIL_KEEP_SEC = 0.2
# ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = nemo_asr.models.ASRModel.restore_from(
    restore_path=MODEL_PATH,
    map_location=device
)
model.eval()
model.to(device)

chunk_size = int(SAMPLE_RATE * CHUNK_MS / 1000)
audio_queue = queue.Queue()

def mic_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy().flatten())

# Sliding audio buffers (for VAD + continuity only)
audio_buffer = deque(maxlen=int(SAMPLE_RATE * MAX_WINDOW_SEC))
pre_roll_buffer = deque(maxlen=int(SAMPLE_RATE * PRE_ROLL_SEC))

# 🔑 Turn audio buffer (FULL utterance, never truncated)
turn_audio_buffer = []

# Turn timing state
speech_start_time = None
last_voice_time = None

# 🔹 Pending (soft-finalized) turn state
pending_time = None
pending_audio_tail = None

# 🔹 NEW: stitching state
pending_segments = []
last_asr_text = ""

print("\n🎤 Speak naturally. Long pause ends your turn.\n")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=chunk_size,
    callback=mic_callback,
):
    while True:
        chunk = audio_queue.get()

        # -------------------------------
        # ALWAYS CAPTURE AUDIO
        # -------------------------------
        turn_audio_buffer.append(chunk)

        audio_buffer.extend(chunk)
        pre_roll_buffer.extend(chunk)

        energy = np.sqrt(np.mean(chunk ** 2))
        now = time.time()

        # -------------------------------
        # SPEECH DETECTION (TIMING ONLY)
        # -------------------------------
        if energy > SILENCE_THRESHOLD:
            last_voice_time = now
            if speech_start_time is None:
                speech_start_time = now

            # 🔹 User resumed speaking while a turn was pending
            if pending_segments and pending_time is not None:
                if (now - pending_time) <= MERGE_WINDOW_SEC:
                    # Merge continuation: restore tail audio
                    audio_buffer.clear()
                    pre_roll_buffer.clear()
                    audio_buffer.extend(pending_audio_tail)
                    pre_roll_buffer.extend(pending_audio_tail)
                else:
                    # Pending expired → commit
                    final_text = " ".join(pending_segments).strip()
                    print(f"\n🧑 USER: {final_text}")
                    # llm.generate(final_text)

                    pending_segments.clear()
                    last_asr_text = ""

                pending_time = None
                pending_audio_tail = None

        # -------------------------------
        # TURN END DETECTION
        # -------------------------------
        if (
            speech_start_time
            and last_voice_time
            and (now - last_voice_time) >= SILENCE_DURATION
            and (now - speech_start_time) >= MIN_UTTERANCE_DURATION
        ):
            # Build FINAL audio for ASR
            buffer_np = np.concatenate(
                [
                    np.array(pre_roll_buffer, dtype=np.float32),
                    np.concatenate(turn_audio_buffer).astype(np.float32),
                ]
            )

            with torch.no_grad():
                hyp = model.transcribe([buffer_np])[0]

            current_text = hyp.text.strip()

            # 🔹 STITCHING LOGIC (KEY FIX)
            if current_text:
                if current_text.startswith(last_asr_text):
                    delta = current_text[len(last_asr_text):].strip()
                else:
                    # RNNT revision or reset
                    delta = current_text

                if delta:
                    pending_segments.append(delta)

                last_asr_text = current_text
                pending_time = now

                # Keep small tail for possible continuation
                tail_len = int(SAMPLE_RATE * TAIL_KEEP_SEC)
                pending_audio_tail = list(audio_buffer)[-tail_len:]

            # Reset speech timers
            speech_start_time = None
            last_voice_time = None

            # 🔑 Clear full-turn buffer (no truncation, no leak)
            turn_audio_buffer.clear()

            # Reset buffers but keep tail
            audio_buffer.clear()
            pre_roll_buffer.clear()

            if pending_audio_tail:
                audio_buffer.extend(pending_audio_tail)
                pre_roll_buffer.extend(pending_audio_tail)

            print("\n🎤 Listening...\n")

        # -------------------------------
        # FINAL COMMIT (NO CONTINUATION)
        # -------------------------------
        if pending_segments and pending_time:
            if (now - pending_time) > MERGE_WINDOW_SEC:
                final_text = " ".join(pending_segments).strip()
                print(f"\n🧑 USER: {final_text}")
                # llm.generate(final_text)

                pending_segments.clear()
                last_asr_text = ""
                pending_time = None
                pending_audio_tail = None

                print("\n🎤 Listening...\n")
