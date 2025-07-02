import numpy as np
import pycbc
from pycbc.filter.matchedfilter import matched_filter
import matplotlib.pyplot as plt

flow = 5
delta_f = 1 / 32
flen = int(4096 / (2 * delta_f)) + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

def scale_signal(signal_ts, num):

    noise = generate_noise()

    template, delta_t_eccentric, start_time_eccentric = inject_signal_with_peak_in_window(
                                            signal_ts=signal_ts,
                                            noise_ts=noise,
                                            peak_window=(2.0, 2.2))
    
    data = pycbc.types.TimeSeries(np.array(template) + np.array(noise), delta_t=delta_t_eccentric, epoch=start_time_eccentric)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t_eccentric, epoch=start_time_eccentric)

    snr = matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=flow)

    peak_snr = abs(snr).numpy().max()

    # print(f"SNR before scaling: {peak_snr}")

    if peak_snr < 10:
        snr_desired = np.random.uniform(10, 20)
        print(f"Scaling signal to achieve SNR of {snr_desired:.2f} for sample {num}")
        scale_factor = snr_desired / peak_snr
        template *= scale_factor
        data = pycbc.types.TimeSeries(np.array(template) + np.array(noise), delta_t=delta_t_eccentric, epoch=start_time_eccentric)
        snr = matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=flow)

        peak_snr = abs(snr).numpy().max()

        # print(f"SNR after scaling: {peak_snr}")


    # plt.plot(snr.sample_times, abs(snr), label='SNR')
    # plt.xlabel('Time (s)')
    # plt.ylabel('SNR')
    # plt.title(f'SNR Timeseries')
    # plt.legend()
    # plt.show()

    # plt.plot(data.sample_times, data, label='Eccentric Signal + Noise')
    # plt.plot(template.sample_times, template, label='Eccentric Signal', linestyle='--')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title(f'Eccentric Signal + Noise')
    # plt.legend()
    # plt.show()

    return data, peak_snr

def inject_signal_with_peak_in_window(signal_ts, noise_ts, peak_window=(2.0, 2.2)):
    """
    Zero-pads and aligns the signal so that its peak occurs within the last `peak_window` seconds of the noise.

    Parameters:
    ----------
    signal_ts : pycbc.types.TimeSeries
        The time-domain eccentric waveform.
    noise_ts : pycbc.types.TimeSeries
        The time-domain noise waveform of the same sampling rate.
    peak_window : tuple (float, float)
        Time window (in seconds) before the end of the noise where the signal peak should be injected.

    Returns:
    -------
    padded_signal : np.ndarray
        The zero-padded signal aligned with the desired peak location.
    injection_index : int
        The index in the array where the peak was injected.
    """
    # Convert to numpy arrays
    signal = np.array(signal_ts)
    noise = np.array(noise_ts)

    # Sampling info
    delta_t = noise_ts.delta_t
    N = len(noise_ts)
    duration = N * delta_t

    # Step 1: Find peak index in the signal
    peak_index = np.argmax(np.abs(signal))

    # Step 2: Choose target time for the peak
    min_offset, max_offset = peak_window
    t_peak = np.random.uniform(duration - max_offset, duration - min_offset)
    target_index = int(t_peak / delta_t)

    # Step 3: Calculate start and end indices for injection
    start_index = target_index - peak_index
    end_index = start_index + len(signal)

    # Step 4: Handle truncation if signal goes out of bounds
    padded_signal = np.zeros(N)

    if start_index < 0:
        signal = signal[-start_index:]  # Trim the front
        start_index = 0
        end_index = start_index + len(signal)

    if end_index > N:
        signal = signal[:N - start_index]  # Trim the end
        end_index = start_index + len(signal)

    # Step 5: Insert the signal into the padded array
    padded_signal[start_index:end_index] = signal

    return padded_signal, delta_t, signal_ts.start_time

def generate_noise():
    # flow = 5
    # delta_f = 1 / 32
    # flen = int(4096 / (2 * delta_f)) + 1
    # psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    delta_t = 1.0 / 4096
    tsamples = int(32 / delta_t)
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=None)

    return noise