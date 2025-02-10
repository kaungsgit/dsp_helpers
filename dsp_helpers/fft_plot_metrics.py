import numpy as np
import matplotlib.pyplot as plt


def nonlinear_block(signal, harmonics_db):
    """
    Creates harmonic distortion using polynomial nonlinearity.

    Parameters:
    signal (numpy array): Input signal to be distorted.
    harmonics_db (list): Harmonic distortion levels in dBc for HD2-HD5.

    Returns:
    numpy array: Distorted signal.
    """
    hd_linear = 10 ** (np.array(harmonics_db) / 20)  # Convert dBc to linear scale

    # Solve for polynomial coefficients using harmonic balance equations
    a5 = 16 * hd_linear[3]  # HD5 coefficient (x^5 term)
    a4 = 8 * hd_linear[2]  # HD4 coefficient (x^4 term)
    a3 = 4 * hd_linear[1] - (5 / 4) * a5  # HD3 coefficient (x^3 term)
    a2 = 2 * hd_linear[0] - a4  # HD2 coefficient (x^2 term)

    # Polynomial: y = x + a2*x² + a3*x³ + a4*x⁴ + a5*x⁵
    return np.polyval([a5, a4, a3, a2, 1, 0], signal)  # [x⁵,x⁴,x³,x²,x,const]


def get_harmonic_bin(h, f0, freq_res, N):
    """
    Returns the bin index corresponding to the h-th harmonic frequency.
    Handles wrap-around past Nyquist.
    """
    return int(round(h * f0 / freq_res)) % (N // 2)


def plot_fft_metrics(
    signal, Fs, f0, f0_bin_range, y_limits=None, harmonics=[2, 3, 4, 5]
):
    """
    Plots FFT metrics of a given signal and calculates various performance metrics.

    Parameters:
    signal (numpy array): Input signal.
    Fs (float): Sampling frequency.
    f0 (float): Fundamental frequency.
    f0_bin_range (int): Range of bins around the fundamental frequency to consider.
    y_limits (tuple, optional): Y-axis limits for the plot.
    harmonics (list, optional): List of harmonic orders to calculate metrics for. Default is [2,3,4,5].

    Returns:
    None
    """
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(N, 1 / Fs)

    # FFT magnitude (normalize by N/2 for correct scaling)
    fft_magnitude = np.abs(fft_result) / (N / 2)

    # Find fundamental bin
    fund_bin = np.argmin(np.abs(fft_freq - f0))
    freq_res = Fs / N

    # Calculate fundamental power using bin_range
    half_window = f0_bin_range // 2
    start = max(0, fund_bin - half_window)
    end = min(N // 2, fund_bin + half_window)
    positive_fund_bins = np.arange(start, end + 1)
    negative_fund_bins = (N - positive_fund_bins) % N
    all_fund_bins = np.unique(np.concatenate([positive_fund_bins, negative_fund_bins]))
    fund_power = np.sum(fft_magnitude[all_fund_bins] ** 2)  # Power from magnitude

    # Calculate harmonic powers
    def total_power(h):
        bin_num = get_harmonic_bin(h, f0, freq_res, N)
        return fft_magnitude[bin_num] ** 2 + fft_magnitude[N - bin_num] ** 2

    h_powers = [total_power(h) + 1e-10 for h in harmonics]

    # Noise calculation (exclude DC, fundamental, harmonics)
    exclude_bins = (
        [0]
        + all_fund_bins.tolist()
        + [get_harmonic_bin(h, f0, freq_res, N) for h in harmonics]
        + [N - get_harmonic_bin(h, f0, freq_res, N) for h in harmonics]
    )
    noise_power = np.sum(np.delete(fft_magnitude**2, exclude_bins))

    # Metrics calculation
    snr = 10 * np.log10(fund_power / noise_power)
    sndr = 10 * np.log10(fund_power / (noise_power + np.sum(h_powers)))
    hd = [10 * np.log10(hp / fund_power) for hp in h_powers]
    sfdr = 10 * np.log10(fund_power / np.max(h_powers))
    nsd = 10 * np.log10(noise_power / (Fs / 2))
    enob = (sndr - 1.76) / 6.02

    # Plot setup (FFT magnitude in dBFS)
    plt.figure(figsize=(10, 6))
    freqs = fft_freq[: N // 2 + 1] / 1e6
    fft_db = 20 * np.log10(fft_magnitude[: N // 2 + 1] + 1e-20)  # Avoid log(0)
    sorted_indices = np.argsort(freqs)
    plt.plot(freqs[sorted_indices], fft_db[sorted_indices])
    plt.xlim(0, Fs / 2e6)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dBFS)")
    if y_limits:
        plt.ylim(y_limits)

    # Add blue band around fundamental tone
    plt.axvspan(freqs[start], freqs[end], color="blue", alpha=0.3)

    # Add colored bands around harmonics and label them
    colors = ["red", "green", "purple", "orange"]
    for i, h in enumerate(harmonics):
        harmonic_bin = get_harmonic_bin(h, f0, freq_res, N)
        start = max(0, harmonic_bin - half_window)
        end = min(N // 2, harmonic_bin + half_window)
        plt.axvspan(freqs[start], freqs[end], color=colors[i], alpha=0.3)
        plt.text(
            (freqs[start] + freqs[end]) / 2,
            y_limits[1] if y_limits else plt.ylim()[1],
            f"H{h}",
            color=colors[i],
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

    # Annotation box
    text = (
        f"SNR: {snr:.1f} dB\n"
        f"SNDR: {sndr:.1f} dB\n"
        f"ENOB: {enob:.2f} bits\n"
        f"Ain: {10*np.log10(fund_power/2):.1f} dBFS\n"
        f"HD2: {hd[0]:.1f} dBc\nHD3: {hd[1]:.1f} dBc\n"
        f"HD4: {hd[2]:.1f} dBc\nHD5: {hd[3]:.1f} dBc\n"
        f"SFDR: {sfdr:.1f} dBc\nNSD: {nsd:.1f} dBFS/Hz\n"
    )
    plt.annotate(
        text,
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round", fc="white"),
        ha="right",
        va="bottom",
    )  # Align text to bottom-right

    plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    """
    Example usage of the nonlinear_block and plot_fft_metrics functions.
    Generates a distorted sine wave with added noise and plots its FFT metrics.
    """
    # Example usage
    Fs = 50e6  # Sampling frequency
    N = 1024  # Number of samples
    M = 41  # Number of cycles for coherence
    f0 = M * Fs / N  # Coherent frequency (≈2.002 MHz)

    # Generate clean sine wave
    t = np.arange(N) / Fs
    signal = np.sin(2 * np.pi * f0 * t)

    # Generate noisy sine wave with Gaussian noise
    noise_std = 1e-4  # Standard deviation of the noise

    # Apply nonlinear distortion
    distorted_signal = nonlinear_block(signal, [-65, -70, -73, -78])
    # distorted_signal = signal

    distorted_signal += np.random.normal(0, noise_std, N)  # Add noise

    # Analyze and plot
    plot_fft_metrics(distorted_signal, Fs, f0, f0_bin_range=5)

    plt.show()
