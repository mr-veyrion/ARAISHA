"""
ASR Wrapper - Standalone voice input for integrated_system.py

This module loads the NeMo ASR model directly and provides a clean interface
for voice input. It does NOT import from asr.py to avoid executing its main loop.
"""
from __future__ import annotations

import queue
import time
import numpy as np
import threading
from typing import Optional, Callable
from collections import deque

try:
    import sounddevice as sd
    import torch
    import nemo.collections.asr as nemo_asr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None
    torch = None
    nemo_asr = None

# ============================================================================
# CONFIG (same as asr.py but defined here to avoid import side effects)
# ============================================================================
MODEL_PATH = "./models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
SAMPLE_RATE = 16000
CHUNK_MS = 100

SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 2.2
MIN_UTTERANCE_DURATION = 2.5

PRE_ROLL_SEC = 0.8
POST_ROLL_SEC = 0.5
MAX_WINDOW_SEC = 20

# Turn-repair config
MERGE_WINDOW_SEC = 2.5
TAIL_KEEP_SEC = 0.2


class ASRInput:
    """
    Voice input handler that yields complete utterances.
    
    Usage:
        asr = ASRInput()
        asr.start()
        text = asr.get_next_utterance()  # Blocks until user finishes speaking
        asr.stop()
    """
    
    _model = None  # Class-level model singleton
    _model_lock = threading.Lock()
    
    @classmethod
    def _get_model(cls):
        """Lazy-load ASR model (singleton)."""
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    if not AUDIO_AVAILABLE:
                        raise RuntimeError("Audio dependencies not available")
                    
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"🎤 Loading ASR model on {device}...")
                    
                    cls._model = nemo_asr.models.ASRModel.restore_from(
                        restore_path=MODEL_PATH,
                        map_location=device
                    )
                    cls._model.eval()
                    cls._model.to(device)
                    print("🎤 ASR model loaded!")
        return cls._model
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Initialize ASR input handler.
        
        Args:
            callback: Optional function called with each transcription (for streaming UI)
        """
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio dependencies (sounddevice, torch, nemo) not available")
        
        self.callback = callback
        self.chunk_size = int(SAMPLE_RATE * CHUNK_MS / 1000)
        
        # Audio state
        self.audio_queue: queue.Queue = queue.Queue()
        self.audio_buffer: deque = deque(maxlen=int(SAMPLE_RATE * MAX_WINDOW_SEC))
        self.pre_roll_buffer: deque = deque(maxlen=int(SAMPLE_RATE * PRE_ROLL_SEC))
        self.turn_audio_buffer: list = []
        
        # Timing state
        self.speech_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None
        
        # Pending turn state
        self.pending_time: Optional[float] = None
        self.pending_audio_tail: Optional[list] = None
        self.pending_segments: list = []
        self.last_asr_text: str = ""
        
        # Control
        self.stream: Optional[sd.InputStream] = None
        self.running: bool = False
    
    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone callback - puts audio chunks in queue."""
        if self.running:
            self.audio_queue.put(indata.copy().flatten())
    
    def start(self):
        """Start microphone stream."""
        if self.running:
            return
        
        # Ensure model is loaded
        self._get_model()
        
        self.running = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=self.chunk_size,
            callback=self._mic_callback,
        )
        self.stream.start()
        print("🎤 Voice input started...")
    
    def stop(self):
        """Stop microphone stream."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("🎤 Voice input stopped.")
    
    def _reset_buffers(self):
        """Reset all audio buffers for new turn."""
        self.audio_buffer.clear()
        self.pre_roll_buffer.clear()
        self.turn_audio_buffer.clear()
        self.speech_start_time = None
        self.last_voice_time = None
        self.pending_segments.clear()
        self.last_asr_text = ""
        self.pending_time = None
        self.pending_audio_tail = None
    
    def get_next_utterance(self) -> str:
        """
        Block until user finishes speaking and return transcription.
        
        Returns:
            Complete transcribed utterance from user
        """
        if not self.running:
            self.start()
        
        # Only reset timing state, NOT audio buffers (allows continuous listening)
        self.speech_start_time = None
        self.last_voice_time = None
        # Keep pending_segments, last_asr_text for turn stitching
        
        model = self._get_model()
        
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)  # Faster polling
            except queue.Empty:
                continue
            
            # Always capture audio
            self.turn_audio_buffer.append(chunk)
            self.audio_buffer.extend(chunk)
            self.pre_roll_buffer.extend(chunk)
            
            energy = np.sqrt(np.mean(chunk ** 2))
            now = time.time()
            
            # Speech detection (timing only)
            if energy > SILENCE_THRESHOLD:
                self.last_voice_time = now
                if self.speech_start_time is None:
                    self.speech_start_time = now
                
                # User resumed speaking while a turn was pending
                if self.pending_segments and self.pending_time is not None:
                    if (now - self.pending_time) <= MERGE_WINDOW_SEC:
                        # Merge continuation: restore tail audio
                        self.audio_buffer.clear()
                        self.pre_roll_buffer.clear()
                        if self.pending_audio_tail:
                            self.audio_buffer.extend(self.pending_audio_tail)
                            self.pre_roll_buffer.extend(self.pending_audio_tail)
                    else:
                        # Pending expired → commit
                        final_text = " ".join(self.pending_segments).strip()
                        if final_text:
                            return final_text
                        self.pending_segments.clear()
                        self.last_asr_text = ""
                    
                    self.pending_time = None
                    self.pending_audio_tail = None
            
            # Turn end detection
            if (
                self.speech_start_time
                and self.last_voice_time
                and (now - self.last_voice_time) >= SILENCE_DURATION
                and (now - self.speech_start_time) >= MIN_UTTERANCE_DURATION
            ):
                # Build audio for ASR
                buffer_np = np.concatenate([
                    np.array(self.pre_roll_buffer, dtype=np.float32),
                    np.concatenate(self.turn_audio_buffer).astype(np.float32),
                ])
                
                with torch.no_grad():
                    hyp = model.transcribe([buffer_np])[0]
                
                current_text = hyp.text.strip()
                
                # Stitching logic
                if current_text:
                    if current_text.startswith(self.last_asr_text):
                        delta = current_text[len(self.last_asr_text):].strip()
                    else:
                        delta = current_text
                    
                    if delta:
                        self.pending_segments.append(delta)
                    
                    self.last_asr_text = current_text
                    self.pending_time = now
                    
                    # Keep small tail for possible continuation
                    tail_len = int(SAMPLE_RATE * TAIL_KEEP_SEC)
                    self.pending_audio_tail = list(self.audio_buffer)[-tail_len:]
                
                # Reset speech timers
                self.speech_start_time = None
                self.last_voice_time = None
                self.turn_audio_buffer.clear()
                self.audio_buffer.clear()
                self.pre_roll_buffer.clear()
                
                if self.pending_audio_tail:
                    self.audio_buffer.extend(self.pending_audio_tail)
                    self.pre_roll_buffer.extend(self.pending_audio_tail)
            
            # Final commit (no continuation)
            if self.pending_segments and self.pending_time:
                if (now - self.pending_time) > MERGE_WINDOW_SEC:
                    final_text = " ".join(self.pending_segments).strip()
                    self.pending_segments.clear()
                    self.last_asr_text = ""
                    self.pending_time = None
                    self.pending_audio_tail = None
                    
                    if final_text:
                        return final_text


# Singleton instance for easy import
_asr_instance: Optional[ASRInput] = None


def get_asr() -> ASRInput:
    """Get or create the singleton ASR instance."""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASRInput()
    return _asr_instance


def voice_input() -> str:
    """
    Simple function to get voice input.
    Blocks until user finishes speaking.
    
    Returns:
        Transcribed text from user's speech
    """
    asr = get_asr()
    return asr.get_next_utterance()
