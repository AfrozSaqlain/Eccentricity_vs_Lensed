import numpy as np
import matplotlib.pyplot as plt

import gwmat
import pycbc
import pycbc.psd
from gwmat import point_lens
from pycbc.types import FrequencySeries
from pycbc.detector.ground import Detector
from pycbc.filter.matchedfilter import matched_filter
from pycbc.waveform import get_td_waveform, taper_timeseries

flow = 5
delta_f = 1 / 32
flen = int(4096 / (2 * delta_f)) + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

DELTA_T = 1.0 / 4096
F_LOWER = 5.0
DETECTOR_NAME = 'H1'
detector = Detector(DETECTOR_NAME)
REFERENCE_DISTANCE = 300

def compute_lensed_waveform(sp, sc, m_lens, y_lens):
    """Apply lensing amplification to the unlensed waveform."""
    sp_freq = sp.to_frequencyseries(delta_f=sp.delta_f)
    sc_freq = sc.to_frequencyseries(delta_f=sc.delta_f)
    freqs = sp_freq.sample_frequencies

    Ffs = np.vectorize(lambda f: gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens))(freqs)
    t_delay = point_lens.time_delay(ml=m_lens, y=y_lens)

    sp_lens = FrequencySeries(np.conj(Ffs) * sp_freq.numpy(), delta_f=sp_freq.delta_f).cyclic_time_shift(-(0.1 + t_delay))
    sc_lens = FrequencySeries(np.conj(Ffs) * sc_freq.numpy(), delta_f=sc_freq.delta_f).cyclic_time_shift(-(0.1 + t_delay))

    return sp_lens.to_timeseries(delta_t=sp_lens.delta_t), sc_lens.to_timeseries(delta_t=sc_lens.delta_t), t_delay

def injnect_noise_with_SNR(gw_signal, params, mass1, mass2, m_lens, y_lens, snr_desired, num, gw_signal_type='eccentric'):
    
    noise = generate_noise()

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=gw_signal,
                                            noise_ts=noise,
                                            peak_window=(2.0, 2.2))
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)

    ########### Add seed to matched_filter ###########
    
    snr = pycbc.filter.matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=flow)

    peak_snr_0 = abs(snr).numpy().max()

    distance = REFERENCE_DISTANCE * (peak_snr_0 / snr_desired) 

    if gw_signal_type == 'eccentric':
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 ecc=params['eccentricity'], coa_phase=params['coa_phase'], f_lower=F_LOWER)
        
        gw_signal = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
    
    elif gw_signal_type == 'quasi_circular':
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_LOWER)
        
        gw_signal = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        
    else: 
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_LOWER) 
        
        hp_lens, hc_lens, t_delay = compute_lensed_waveform(hp, hc, m_lens, y_lens)

        gw_signal = taper_timeseries(detector.project_wave(hp_lens, hc_lens, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=gw_signal,
                                            noise_ts=noise,
                                            peak_window=(2.0, 2.2))
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)
    snr = pycbc.filter.matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=flow)

    peak_snr = abs(snr).numpy().max()

    print(f"SNR before scaling: {peak_snr_0:.2f} | SNR after scaling: {peak_snr:.2f} for sample {num} {gw_signal_type} | Distance: {distance:.2f}")

    return data, peak_snr, distance

def scale_signal(signal_ts, num=None):

    noise = generate_noise()

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=signal_ts,
                                            noise_ts=noise,
                                            peak_window=(2.0, 2.2))
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)

    snr = matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=flow)

    peak_snr = abs(snr).numpy().max()

    # print(f"SNR before scaling: {peak_snr}")

    if peak_snr < 10:
        snr_desired = np.random.uniform(10, 20)
        # print(f"Scaling signal to achieve SNR of {snr_desired:.2f} for sample {num}")
        scale_factor = snr_desired / peak_snr
        template *= scale_factor
        data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
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

def add_noise(signal_ts, num=None):

    noise = generate_noise()

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=signal_ts,
                                            noise_ts=noise,
                                            peak_window=(2.0, 2.2))
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)

    snr = matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=flow)

    peak_snr = abs(snr).numpy().max()

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
    signal = np.asarray(signal_ts)
    noise = np.asarray(noise_ts)

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

    delta_t = 1.0 / 4096
    tsamples = int(32 / delta_t)
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=None)

    return noise