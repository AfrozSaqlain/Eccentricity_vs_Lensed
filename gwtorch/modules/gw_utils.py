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

# ------------ Predifined Constant ---------------

flow = 5
delta_f = 1 / 32
flen = int(4096 / (2 * delta_f)) + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

DELTA_T = 1.0 / 4096
F_LOWER = 5.0
DETECTOR_NAME = 'H1'
detector = Detector(DETECTOR_NAME)
REFERENCE_DISTANCE = 300

# ------------------------------------------------

def generate_noise(time_duration=32, seed=None):
    """
    Generate a Gaussian noise realization using the O4 PSD.

    This function generates simulated Gaussian noise from the Advanced LIGO O4
    design sensitivity (aLIGOZeroDetHighPower) power spectral density (PSD).
    The generated noise is sampled at 4096 Hz over a specified duration.

    Parameters
    ----------
    time_duration : float, optional
        Duration of the time series in seconds. Default is 32 seconds.
    seed : int or None, optional
        Seed for the random number generator, used to ensure reproducibility.
        If None, the noise will be randomly generated without a fixed seed.

    Returns
    -------
    noise : pycbc.types.TimeSeries
        A PyCBC TimeSeries object containing the generated noise realization.
    """
    delta_t = 1.0 / 4096
    tsamples = int(time_duration / delta_t)
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=seed)

    return noise


def compute_lensed_waveform(sp, sc, m_lens, y_lens):
    """
    Apply gravitational lensing amplification to a gravitational waveform using a point mass lens model.

    This function modifies the plus (`sp`) and cross (`sc`) polarizations of a gravitational wave
    waveform based on the lensing amplification due to a redshifted point mass lens.
    It also computes the time delay introduced by the lensing.

    Parameters
    ----------
    sp : pycbc.types.TimeSeries
        Plus polarization of the gravitational waveform.
    sc : pycbc.types.TimeSeries
        Cross polarization of the gravitational waveform.
    m_lens : float
        Redshifted mass of the lens in solar masses.
    y_lens : float
        Dimensionless impact parameter (source-lens-observer alignment).

    Returns
    -------
    sp_lensed : pycbc.types.TimeSeries
        Lensed plus-polarized gravitational waveform.
    sc_lensed : pycbc.types.TimeSeries
        Lensed cross-polarized gravitational waveform.
    t_delay : float
        Time delay (in seconds) between the lensed and unlensed waveforms due to lensing.
    """

    sp_freq = sp.to_frequencyseries(delta_f=sp.delta_f)
    sc_freq = sc.to_frequencyseries(delta_f=sc.delta_f)
    freqs = sp_freq.sample_frequencies

    Ffs = np.vectorize(lambda f: gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens))(freqs)
    t_delay = point_lens.time_delay(ml=m_lens, y=y_lens)

    sp_lens = FrequencySeries(np.conj(Ffs) * sp_freq.numpy(), delta_f=sp_freq.delta_f).cyclic_time_shift(-(0.1 + t_delay))
    sc_lens = FrequencySeries(np.conj(Ffs) * sc_freq.numpy(), delta_f=sc_freq.delta_f).cyclic_time_shift(-(0.1 + t_delay))

    return sp_lens.to_timeseries(delta_t=sp_lens.delta_t), sc_lens.to_timeseries(delta_t=sc_lens.delta_t), t_delay

def inject_noise_with_target_SNR(gw_signal, params, mass1, mass2, m_lens, y_lens, snr_desired, num, gw_signal_type='eccentric'):
    """
    Inject a gravitational wave signal into Gaussian noise and scale it to achieve a desired SNR.

    This function takes a gravitational wave (GW) signal and injects it into 
    simulated Gaussian noise generated from the O4 power spectral density (PSD). 
    It rescales the signal to achieve a target signal-to-noise ratio (SNR) by 
    adjusting the luminosity distance. Supports eccentric, quasi-circular, and 
    lensed waveforms.

    Parameters
    ----------
    gw_signal : pycbc.types.TimeSeries
        Initial unscaled gravitational wave signal to be injected.
    params : dict
        Dictionary containing intrinsic and extrinsic waveform parameters, such as
        `spin1z`, `spin2z`, `eccentricity`, `coa_phase`, `ra`, `dec`, and `polarization`.
    mass1 : float
        Mass of the primary compact object (in solar masses).
    mass2 : float
        Mass of the secondary compact object (in solar masses).
    m_lens : float
        Redshifted mass of the point mass lens (in solar masses). Used only for lensed waveforms.
    y_lens : float
        Dimensionless impact parameter for lensing. Used only for lensed waveforms.
    snr_desired : float
        Target SNR to scale the signal to.
    num : int
        Sample index, used for logging/debugging purposes.
    gw_signal_type : str, optional
        Type of waveform to generate. Options:
        - 'eccentric' (default): Eccentric waveform using input parameters.
        - 'quasi_circular': Circular waveform with zero eccentricity.
        - 'lensed': Point-mass lensed waveform.

    Returns
    -------
    data : pycbc.types.TimeSeries
        The noise + injected GW signal time series (scaled to the desired SNR).
    peak_snr : float
        The peak matched-filter SNR after rescaling.
    distance : float
        The luminosity distance (in Mpc) used to scale the waveform to the target SNR.
    """

    SEED = np.random.uniform(20, 200)
    
    noise = generate_noise(seed = int(SEED))

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

def add_noise(signal_ts):

    SEED = np.random.uniform(20, 200)

    noise = generate_noise(seed = int(SEED))

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

    Parameters
    ----------
    signal_ts : pycbc.types.TimeSeries
        The time-domain eccentric waveform.
    noise_ts : pycbc.types.TimeSeries
        The time-domain noise waveform of the same sampling rate.
    peak_window : tuple (float, float)
        Time window (in seconds) before the end of the noise where the signal peak should be injected.

    Returns
    ----------
    padded_signal : np.ndarray
        The zero-padded signal aligned with the desired peak location.
    injection_index : int
        The index in the array where the peak was injected.
    """
    # Convert to numpy arrays
    signal = np.asarray(signal_ts)

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

def save_qtransform(ts, path):
    """
    Compute and save the Q-transform spectrogram of a time series.

    This function computes the Q-transform of the given PyCBC time series object and saves 
    the resulting spectrogram as an image file. The Q-transform provides a time-frequency 
    representation of the signal and is useful for visualizing transient features such as 
    gravitational wave signals.

    Parameters
    ----------
    ts : pycbc.types.TimeSeries
        The input time series to analyze, typically containing gravitational wave strain data.
    path : str
        The file path where the generated Q-transform image will be saved.

    Returns
    -------
    None
        The function saves the figure to disk and does not return anything.
    """
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(ts.q_transform(logf=True, norm='mean', frange=(5, 512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.yscale('log')
    plt.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()























def scale_signal(signal_ts, num=None):

    SEED = np.random.uniform(20, 200)

    noise = generate_noise(SEED)

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