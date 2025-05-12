import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(f, t, Sxx, title="Spectrogram", vmin=-60, vmax=-20):
    """
    Plot a spectrogram given a frequency and time array and a 2D power array.

    Parameters
    ----------
    f : array_like
        Frequency array
    t : array_like
        Time array
    Sxx : array_like
        2D power array
    title : str
        Plot title
    vmin : float
        Minimum power value
    vmax : float
        Maximum power value

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar(label='Power [dB]')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.show()