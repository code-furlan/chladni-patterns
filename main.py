import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Audio parameters
CHUNK = 4096  # Size of audio chunk
RATE = 44100  # Sampling rate (Hz)

# Simulation parameters
RES = 300  # Grid resolution for the plate
MAX_MODE = 15  # Maximum mode number (m, n up to this)
GAMMA = 0.5  # Damping factor for resonance width
K = 0.05  # Scaling factor to map real frequency (Hz) to simulation frequency space
# Adjust K as needed; e.g., for a 440 Hz tone to excite modes around f_sim=22, K=0.05 works

# Create grid for the plate (0 to 1 normalized)
x = np.linspace(0, 1, RES)
y = np.linspace(0, 1, RES)
X, Y = np.meshgrid(x, y)

def compute_displacement(f_sim):
    """
    Compute the displacement field for the given simulation frequency using superposition of modes.
    This simulates the vibration patterns on a square Chladni plate.
    """
    Z = np.zeros((RES, RES))
    for m in range(1, MAX_MODE + 1):
        for n in range(1, MAX_MODE + 1):
            f_mn = (m**2 + n**2)  # Proportional to actual eigenfrequency
            denom = (f_sim - f_mn)**2 + GAMMA**2
            if denom < 1e-6:
                denom = 1e-6
            amp = 1.0 / denom
            mode = np.sin(np.pi * m * X) * np.sin(np.pi * n * Y)
            Z += amp * mode
    return Z

# Initialize PyAudio and open microphone stream (default input)
# Note: This uses your microphone. To visualize system sound output, you may need to set up
# audio loopback (e.g., using software like VB-Audio on Windows/Mac, or PulseAudio on Linux)
# and select the loopback device. For simplicity, play sound near your mic or configure accordingly.
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 8))
Z_init = compute_displacement(0)
plot_data = np.abs(Z_init)**0.2  # Enhance contrast for visualization
im = ax.imshow(plot_data, cmap='Blues_r', origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=np.max(plot_data))
ax.set_title('Chladni Pattern (Waiting for sound...)')
ax.axis('off')  # Hide axes for a cleaner look

def update(frame):
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
    except IOError:
        return im,  # Skip frame on overflow
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)  # Convert to float for FFT
    # Compute FFT
    fft = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
    mag = np.abs(fft)
    # Find peak frequency (ignore DC component)
    if np.max(mag) < 1000:  # Threshold to ignore silence/noise
        ax.set_title('Chladni Pattern (No significant sound detected)')
        return im,
    peak_idx = np.argmax(mag[1:]) + 1
    peak_freq = freqs[peak_idx]
    f_sim = peak_freq * K
    # Compute new pattern
    Z = compute_displacement(f_sim)
    plot_data = np.abs(Z)**0.2  # Low exponent for high contrast (nodal lines clear)
    im.set_data(plot_data)
    im.set_clim(0, np.max(plot_data))  # Autoscale colors
    ax.set_title(f'Chladni Pattern at {peak_freq:.0f} Hz (sim f={f_sim:.1f})')
    return im,

# Animate (update every 50ms)
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

plt.show()

# Clean up on close
stream.stop_stream()
stream.close()
p.terminate()