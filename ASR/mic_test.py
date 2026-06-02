import sounddevice as sd
import numpy as np

# List all audio devices
print("Available audio devices:")
devices = sd.query_devices()
print(devices)

print("\n\nDefault input device:")
print(sd.default.device)

# Try to record a test
print("\n\nTesting audio input for 2 seconds...")
try:
    duration = 2
    samplerate = 16000
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print(f"✓ Successfully recorded {len(audio)} samples")
    print(f"  Volume level: {np.sqrt(np.mean(audio ** 2)):.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
