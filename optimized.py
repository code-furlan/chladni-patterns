import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Audio parameters
CHUNK = 1024  # Reduced chunk size for faster processing
RATE = 44100  # Sampling rate (Hz)

# Simulation parameters
RES = 100  # Reduced grid resolution
MAX_MODE = 8  # Reduced max mode number
GAMMA = 0.5  # Damping factor
K = 0.05  # Frequency scaling factor

# Create grid
x = np.linspace(0, 1, RES)
y = np.linspace(0, 1, RES)
X, Y = np.meshgrid(x, y)

# Precompute mode shapes to save time
mode_shapes = {}
for m in range(1, MAX_MODE + 1):
    for n in range(1, MAX_MODE + 1):
        mode_shapes[(m, n)] = np.sin(np.pi * m * X) * np.sin(np.pi * n * Y)

def compute_displacement(f_sim):
    """Compute displacement using precomputed modes."""
    Z = np.zeros((RES, RES))
    for m in range(1, MAX_MODE + 1):
        for n in range(1, MAX_MODE + 1):
            f_mn = (m**2 + n**2)
            denom = (f_sim - f_mn)**2 + GAMMA**2
            if denom < 1e-6:
                denom = 1e-6
            amp = 1.0 / denom
            Z += amp * mode_shapes[(m, n)]
    return Z

# Initialize PyAudio and microphone stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))
Z_init = compute_displacement(0)
plot_data = np.abs(Z_init)**0.2
im = ax.imshow(plot_data, cmap='Blues_r', origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=np.max(plot_data))
ax.set_title('Chladni Pattern (Waiting for sound...)')
ax.axis('off')

def update(frame):
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
    except IOError:
        return im,
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    fft = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
    mag = np.abs(fft)
    if np.max(mag) < 1000:
        ax.set_title('Chladni Pattern (No significant sound detected)')
        return im,
    peak_idx = np.argmax(mag[1:]) + 1
    peak_freq = freqs[peak_idx]
    f_sim = peak_freq * K
    Z = compute_displacement(f_sim)
    plot_data = np.abs(Z)**0.2
    im.set_data(plot_data)
    im.set_clim(0, np.max(plot_data))
    ax.set_title(f'Chladni Pattern at {peak_freq:.0f} Hz (sim f={f_sim:.1f})')
    return im,

# Animate with longer interval for smoother performance
ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

plt.show()

# Clean up
stream.stop_stream()
stream.close()
p.terminate()