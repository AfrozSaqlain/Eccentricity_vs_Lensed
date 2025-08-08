import numpy as np
import matplotlib.pyplot as plt

import gc
import gwmat
import pycbc
import bilby
import pycbc.psd
from gwmat import point_lens
from gwpy.timeseries import TimeSeries
from pycbc.types import FrequencySeries
from bilby.gw.prior import BBHPriorDict
from pycbc.filter.matchedfilter import matched_filter
from pycbc.waveform import get_td_waveform, taper_timeseries
from matplotlib.colors import Normalize

from gwtorch.modules.constants import *

# ------------ Predifined Constant ---------------

delta_f = 1 / 32
flen = int(4096 / (2 * delta_f)) + 1
einstein_psd = pycbc.psd.EinsteinTelescopeP1600143(flen, delta_f, F_LOWER)
aligo_psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
cosmic_psd = pycbc.psd.CosmicExplorerP1600143(flen, delta_f, F_LOWER)

# ------------------------------------------------

def generate_prior(num_samples, snr_type = "powerlaw"):
    """
    Generate a list of sampled prior parameter dictionaries for binary black hole (BBH) gravitational wave signals.

    This function defines prior distributions for various BBH parameters including component spins, 
    masses, eccentricity, lensing parameters, and signal-to-noise ratio (SNR). It samples from 
    these distributions to generate synthetic population samples, which are useful for simulations, 
    training machine learning models, or population studies.

    Parameters
    ----------
    num_samples : int
        The number of prior samples to generate.
    
    snr_type : str, optional
        Type of SNR prior to use. Options are:
        - "PowerLaw" (default): Samples SNR from a power-law distribution with α = -4 in [15, 500].
        - "Uniform: Samples SNR from a uniform distribution in [10, 100].

    Returns
    -------
    list of dict
        A list containing `num_samples` dictionaries, each representing a sample of GW parameters 
        drawn from the defined priors. Each dictionary includes:
            - chirp_mass
            - mass_ratio
            - spin1z, spin2z
            - eccentricity
            - coa_phase
            - polarization
            - Log_Mlz (log lens mass)
            - yl (scaled impact parameter)
            - snr

    Notes
    -----
    - Some default parameters from `BBHPriorDict` such as spins (a_1, a_2), orientation angles, 
      distance, and phase are explicitly removed.
    - The prior on `chirp_mass` and `mass_ratio` uses component-based uniform priors provided by Bilby.
    """
    priors = BBHPriorDict()

    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_12', 'phi_jl', 'luminosity_distance', 'psi', 'phase']:
        priors.pop(key, None)

    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
    priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=0.0, maximum=0.9)
    priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=0.0, maximum=0.9)
    priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.1, maximum=0.6)
    priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
    priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")
    priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum=2, maximum=5)
    priors['yl'] = bilby.core.prior.PowerLaw(alpha=1, minimum=0.01, maximum=1.0)
    if snr_type == "powerlaw":
        priors['snr'] = bilby.core.prior.PowerLaw(minimum=10, maximum=500, alpha=-4)
    else:
        priors['snr'] = bilby.core.prior.Uniform(minimum=10, maximum=100)

    parameters = priors.sample(num_samples)
    return [
        {key: parameters[key][i] for key in parameters}
        for i in range(num_samples)
    ]

def generate_waveform(num ,params, peak_window=(2.0, 2.2), detector=default_detector, display=False):
    """
    Generate eccentric, unlensed, and lensed gravitational waveforms with injected noise.

    This function constructs three types of gravitational waveforms using the `TEOBResumS` approximant:
    1. An eccentric waveform (`ecc_noisy`),
    2. A quasi-circular (unlensed) waveform (`unlens_noisy`),
    3. A lensed version of the quasi-circular waveform (`lens_noisy`),
    each projected onto a detector frame, tapered, and injected into Gaussian noise to achieve a 
    target signal-to-noise ratio (SNR).

    Parameters
    ----------
    num : int
        Sample index used for logging and file-specific identification.
    
    params : dict
        Dictionary of source and lens parameters. Expected keys include:
        - 'chirp_mass': Chirp mass of the binary system (in solar masses).
        - 'mass_ratio': Mass ratio (m2/m1), where 0 < q <= 1.
        - 'spin1z', 'spin2z': Dimensionless aligned spin components of the two bodies.
        - 'eccentricity': Initial eccentricity of the binary.
        - 'coa_phase': Coalescence phase (in radians).
        - 'polarization': Polarization angle (in radians).
        - 'ra', 'dec': Right ascension and declination of the source (in radians).
        - 'Log_Mlz': Log10 of the lens mass in solar masses.
        - 'yl': Dimensionless lensing impact parameter (source-lens alignment).
        - 'snr': Target signal-to-noise ratio for all waveform types (float).

    peak_window : tuple (float, float)
        Time window (in seconds) before the end of the noise where the signal peak should be injected.

    display " str, optional
        Whether to display the TimeSeries data or not. Default is False.

    Returns
    -------
    tuple or None
        Returns a tuple of three `TimeSeries` objects:
        - ecc_noisy : Eccentric waveform injected into noise.
        - lens_noisy : Lensed quasi-circular waveform injected into noise.
        - unlens_noisy : Unlensed quasi-circular waveform injected into noise.

        Returns `None` if an error occurs during waveform generation or projection.

    Notes
    -----
    - Waveforms are generated using the `get_td_waveform` function from `lalsimulation` via PyCBC.
    - All waveforms are projected onto the detector using `detector.project_wave(...)`.
    - Tapering is applied at both ends to reduce spectral leakage.
    - `inject_noise_with_target_SNR` scales each waveform to match the specified SNR.
    - The returned waveforms are cropped from the left to remove early-time artifacts (24 samples).
    - The input dictionary is mutated by popping the 'snr' key internally.

    Exceptions
    ----------
    If any step fails (e.g., waveform generation, projection, or noise injection), the function prints
    an error message including the sample number and returns `None`.
    """
    try:
        mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
            params['chirp_mass'], params['mass_ratio'])

        m_lens = 10 ** params["Log_Mlz"]
        y_lens = params["yl"]

        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=REFERENCE_DISTANCE, delta_t=DELTA_T,
                                 ecc=params['eccentricity'], coa_phase=params['coa_phase'], f_lower=F_LOWER)

        sp, sc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=REFERENCE_DISTANCE, delta_t=DELTA_T,
                                 ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_LOWER)

        sp_lensed, sc_lensed, t_delay = compute_lensed_waveform(sp, sc, m_lens, y_lens)

        ecc = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        unlensed = taper_timeseries(detector.project_wave(sp, sc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        lensed = taper_timeseries(detector.project_wave(sp_lensed, sc_lensed, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

        target_snr = params['snr']

        ecc_noisy, snr_e, distance_e, ecc_padded = inject_noise_with_target_SNR(ecc, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='eccentric', peak_window=peak_window, detector=detector)
        lens_noisy, snr_l, distance_l, lens_padded = inject_noise_with_target_SNR(lensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='lensed', peak_window=peak_window, detector=detector)
        unlens_noisy, snr_u, distance_u, unlens_padded = inject_noise_with_target_SNR(unlensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='quasi_circular', peak_window=peak_window, detector=detector)

        ecc_noisy = ecc_noisy.crop(left=24, right=0)
        lens_noisy = lens_noisy.crop(left=24, right=0)
        unlens_noisy = unlens_noisy.crop(left=24, right=0)

        # ecc_padded.start_time = unlens_padded.start_time
        # lens_padded.start_time = unlens_padded.start_time
        # ecc_noisy.start_time = unlens_padded.start_time
        # lens_noisy.start_time = ecc_noisy.start_time
        # unlens_noisy.start_time = ecc_noisy.start_time

        if display:

            fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            axs[0].plot(ecc_noisy.sample_times, ecc_noisy, label="Eccentric (noisy)", linestyle='-', color='C0')
            axs[0].plot(ecc_padded.sample_times, ecc_padded, label="Eccentric (padded)", linestyle='-', color='C1')
            axs[0].set_ylabel("Strain")
            axs[0].set_title("Eccentric Waveform")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(lens_noisy.sample_times, lens_noisy, label="Lensed (noisy)", linestyle='-', color='C2')
            axs[1].plot(lens_padded.sample_times, lens_padded, label="Lensed (padded)", linestyle='-', color='C3')
            axs[1].set_ylabel("Strain")
            axs[1].set_title("Lensed Waveform")
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(unlens_noisy.sample_times, unlens_noisy, label="Unlensed (noisy)", linestyle='-', color='C4')
            axs[2].plot(unlens_padded.sample_times, unlens_padded, label="Unlensed (padded)", linestyle='-', color='C5')
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("Strain")
            axs[2].set_title("Unlensed Waveform")
            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout()
            plt.show()

        ecc_dict = {'ecc_signal': ecc_noisy, 'snr_e': snr_e, 'distance_e': distance_e}
        lens_dict = {'lensed_signal': lens_noisy, 'snr_l': snr_l, 'distance_l': distance_l}
        unlens_dict = {'unlensed_signal': unlens_noisy, 'snr_u': snr_u, 'distance_u': distance_u}

        del hp, hc, sp, sc, sp_lensed, sc_lensed
        del ecc, unlensed, lensed, ecc_noisy, lens_noisy, unlens_noisy, ecc_padded, lens_padded, unlens_padded
        gc.collect()

        return ecc_dict, lens_dict, unlens_dict, mass1, mass2, m_lens, y_lens
        
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_noise(time_duration=32, PSD_TYPE='aLIGO', seed=None):
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

    if PSD_TYPE == 'aLIGO':
        noise = pycbc.noise.noise_from_psd(tsamples, delta_t, aligo_psd, seed=seed)
    elif PSD_TYPE == 'ET':
        noise = pycbc.noise.noise_from_psd(tsamples, delta_t, einstein_psd, seed=seed)
    elif PSD_TYPE == 'CE':
        noise = pycbc.noise.noise_from_psd(tsamples, delta_t, cosmic_psd, seed=seed)

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

    # Ffs = np.vectorize(lambda f: gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens))(freqs)
    Ffs = [gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens) for f in freqs]
    t_delay = point_lens.time_delay(ml=m_lens, y=y_lens)

    sp_lens = FrequencySeries(np.conj(Ffs) * sp_freq.numpy(), delta_f=sp_freq.delta_f).cyclic_time_shift(-(t_delay - 0.1))
    sc_lens = FrequencySeries(np.conj(Ffs) * sc_freq.numpy(), delta_f=sc_freq.delta_f).cyclic_time_shift(-(t_delay - 0.1))

    return sp_lens.to_timeseries(delta_t=sp_lens.delta_t), sc_lens.to_timeseries(delta_t=sc_lens.delta_t), t_delay

def get_psd_type(detector=default_detector):
    """
    Get the type of PSD used in the module.

    Returns
    -------
    str
        The type of PSD used, which is 'aLIGO' by default.
    """
    if detector.name == 'E1' or detector.name == 'E2' or detector.name == 'E3':
        return 'ET', einstein_psd
    elif detector.name == 'C1':
        return 'CE', cosmic_psd
    elif detector.name == 'L1' or detector.name == 'H1' or detector.name == 'V1' or detector.name == 'A1':
        return 'aLIGO', aligo_psd

def inject_noise_with_target_SNR(gw_signal, params, mass1, mass2, m_lens, y_lens, snr_desired, num, gw_signal_type='eccentric', peak_window=(2.0, 2.2), detector=default_detector, SEED = None):
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
    peak_window : tuple (float, float)
        Time window (in seconds) before the end of the noise where the signal peak should be injected.
    detector : pycbc.detector.Detector, optional
        The detector object used to project the waveform onto the detector frame. Default is `default_detector`.

    Returns
    -------
    data : pycbc.types.TimeSeries
        The noise + injected GW signal time series (scaled to the desired SNR).
    peak_snr : float
        The peak matched-filter SNR after rescaling.
    distance : float
        The luminosity distance (in Mpc) used to scale the waveform to the target SNR.
    """
    PSD_TYPE, psd = get_psd_type(detector)

    noise = generate_noise(PSD_TYPE=PSD_TYPE, seed = int(SEED)) if SEED is not None else generate_noise(PSD_TYPE=PSD_TYPE)

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=gw_signal,
                                            noise_ts=noise,
                                            peak_window=peak_window)
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)

    ########### Add seed to matched_filter ###########
    
    snr = pycbc.filter.matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=F_LOWER)

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
                                            peak_window=peak_window)
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)
    snr = pycbc.filter.matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=F_LOWER)

    peak_snr = abs(snr).numpy().max()

    # print(f"SNR before scaling: {peak_snr_0:.2f} | SNR after scaling: {peak_snr:.2f} for sample {num} {gw_signal_type} | Distance: {distance:.2f}")

    return data, peak_snr, distance, template

def add_noise(signal_ts, detector=default_detector, SEED = None):

    PSD_TYPE, psd = get_psd_type(detector)

    noise = generate_noise(PSD_TYPE=PSD_TYPE, seed = int(SEED)) if SEED is not None else generate_noise(PSD_TYPE=PSD_TYPE)

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=signal_ts,
                                            noise_ts=noise,
                                            peak_window=(2.0, 2.2))
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)

    snr = matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=F_LOWER)

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


def display_qtransform(ts):
    """
    Compute and save the Q-transform spectrogram of a time series.

    This function computes the Q-transform of the given PyCBC time series object and displays 
    the resulting spectrogram as an image. The Q-transform provides a time-frequency 
    representation of the signal and is useful for visualizing transient features such as 
    gravitational wave signals.

    Parameters
    ----------
    ts : pycbc.types.TimeSeries
        The input time series to analyze, typically containing gravitational wave strain data.

    Returns
    -------
    None
        The function displays the figure does not return anything.
    """
    ts = TimeSeries.from_pycbc(ts)

    plt.figure(figsize=(12, 8), frameon=False)
    plt.pcolormesh(ts.q_transform(logf=True, norm='mean', frange=(5, 512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.yscale('log')
    plt.show()


# def save_qtransform_with_checks(waveform_dict, params, path):
#     if "eccentric" in str(path):
#         stacked_q = []
#         for key in waveform_dict:
#             mass1 = waveform_dict[key]['mass1']
#             mass2 = waveform_dict[key]['mass2']
#             m_lens = waveform_dict[key]['m_lens']
#             y_lens = waveform_dict[key]['y_lens']

#             distance_e = waveform_dict[key]['eccentric']['distance_e']
#             signal_snr = waveform_dict[key]['eccentric']['snr_e']
#             ts = waveform_dict[key]['eccentric']['ecc_signal']

#             # Regenerate same signal with eccentricity and non-eccentricity for lnB check
#             hp_, _ = get_td_waveform(
#                 approximant='teobresums', mass1=mass1, mass2=mass2,
#                 lambda1=0, lambda2=0,
#                 spin1z=params['spin1z'], spin2z=params['spin2z'],
#                 distance=distance_e, delta_t=DELTA_T,
#                 ecc=params['eccentricity'], coa_phase=params['coa_phase'], f_lower=F_LOWER
#             )

#             sp_, _ = get_td_waveform(
#                 approximant='teobresums', mass1=mass1, mass2=mass2,
#                 lambda1=0, lambda2=0,
#                 spin1z=params['spin1z'], spin2z=params['spin2z'],
#                 distance=distance_e, delta_t=DELTA_T,
#                 ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_LOWER
#             )

#             tlen = max(len(hp_), len(sp_))
#             hp_.resize(tlen)
#             sp_.resize(tlen)
#             sp_ = pycbc.types.TimeSeries(sp_.data, delta_t=hp_.delta_t, epoch=hp_.start_time)

#             duration_ = hp_.duration
#             delta_f_ = 1 / duration_
#             flen_ = tlen // 2 + 1
#             psd = pycbc.psd.EinsteinTelescopeP1600143(flen_, delta_f_, F_LOWER)

#             m = match(hp_, sp_, psd=psd, low_frequency_cutoff=5)[0]
#             ln_B = 0.5 * (1 - m**2) * signal_snr**2

#             if ln_B >= np.log(10):
#                 ts = TimeSeries.from_pycbc(ts)
#                 q = ts.q_transform(logf=True, norm='median', frange=(F_LOWER, 512), whiten=True, qrange=(4, 64))
#                 q_data = np.abs(q.data)
#                 stacked_q.append(q_data)

#         if len(stacked_q) != 3:
#             print(f"Only {len(stacked_q)} detectors passed the ln_B threshold. Not saving plot.")
#             return False

#         # Normalize each to [0,1] and stack into 3-channel (RGB-like)
#         normed = [Normalize()(q) for q in stacked_q]
#         rgb_image = np.stack(normed, axis=-1)  # Shape: (time, freq, 3)

#         plt.figure(figsize=(12, 8))
#         plt.imshow(rgb_image, aspect='auto', origin='lower')
#         plt.axis('off')
#         plt.savefig(path, bbox_inches='tight', pad_inches=0)
#         plt.close()

#         return True

        
#     elif "lensed" in str(path) and "unlensed" not in str(path):
#         hp_, _ = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
#                                  lambda1=0, lambda2=0,
#                                  spin1z=params['spin1z'], spin2z=params['spin2z'],
#                                  distance=distance_l, delta_t=DELTA_T,
#                                  ecc=0, coa_phase=params['coa_phase'], f_lower=F_LOWER)
        
#         sp_, _, _ = compute_lensed_waveform(hp_, _, m_lens, y_lens)
        
#         # Ensure both waveforms have the same length and delta_t
#         tlen = max(len(hp_), len(sp_))
#         hp_.resize(tlen)
#         sp_.resize(tlen)
        
#         # Force both waveforms to have the same delta_t by recreating sp_
#         # This ensures consistent frequency resolution for PSD matching
#         sp_ = pycbc.types.TimeSeries(sp_.data, delta_t=hp_.delta_t, epoch=hp_.start_time)

#         duration_ = hp_.duration
#         delta_f_ = 1 / duration_
#         flen_ = tlen // 2 + 1
#         psd = pycbc.psd.EinsteinTelescopeP1600143(flen_, delta_f_, F_LOWER)

#         match = pycbc.filter.match(hp_, sp_, psd=psd, low_frequency_cutoff=5)[0]
#         ln_B = 0.5 * (1 - match ** 2) * signal_snr ** 2

#         if ln_B < np.log(10):
#             # sample_number = str(path).split("_")[1].split(".")[0]
#             # print(rf"Skipping lensed sample {sample_number} as $\ln B$ is {ln_B}")
#             return False

#     ts = TimeSeries.from_pycbc(ts)

#     plt.figure(figsize=(12, 8))
#     plt.pcolormesh(ts.q_transform(logf=True, norm='median', frange=(F_LOWER, 512), whiten=True, qrange=(4, 64)))
#     plt.axis('off')
#     plt.yscale('log')
#     plt.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
#     plt.close()

#     return True






def _get_category_parts(path_str):
    """Return ('eccentric' | 'lensed' | 'unlensed', True/False)"""
    if "eccentric" in path_str:
        return "eccentric"
    if "lensed"    in path_str and "unlensed" not in path_str:
        return "lensed"
    return "unlensed"

def save_qtransform_with_checks(waveform_dict, params, path):
    """
    Saves an RGB‑stacked Q‑transform image:
      R = detector‑0 spectrogram
      G = detector‑1 spectrogram
      B = detector‑2 spectrogram
    Category is inferred from the filename: eccentric / lensed / unlensed.
    Returns True on success, False otherwise.
    """
    category = _get_category_parts(str(path))
    stacked_q = []

    # --- iterate over detectors ------------------------------------------------
    for key in waveform_dict:
        wf = waveform_dict[key]

        mass1, mass2  = wf['mass1'],  wf['mass2']
        m_lens, y_lens = wf['m_lens'], wf['y_lens']

        # pick fields that differ by category
        if category == "eccentric":
            distance   = wf['eccentric']['distance_e']
            signal_snr = wf['eccentric']['snr_e']
            ts_raw     = wf['eccentric']['ecc_signal']
        elif category == "lensed":
            distance   = wf['lensed']['distance_l']
            signal_snr = wf['lensed']['snr_l']
            ts_raw     = wf['lensed']['lensed_signal']
        else:  # unlensed
            distance   = wf['unlensed']['distance_u']
            signal_snr = wf['unlensed']['snr_u']
            ts_raw     = wf['unlensed']['unlensed_signal']

        # --- Bayes factor / match test where applicable ------------------------
        compute_q = True  # default for unlensed case
        if category in ("eccentric", "lensed"):
            # reference (un‑lensed, zero‑ecc) waveform
            hp_, _ = get_td_waveform(
                approximant='teobresums',
                mass1=mass1, mass2=mass2,
                lambda1=0, lambda2=0,
                spin1z=params['spin1z'], spin2z=params['spin2z'],
                distance=distance, delta_t=DELTA_T,
                ecc = 0 if category=="lensed" else params['eccentricity'],
                coa_phase=params['coa_phase'], f_lower=F_LOWER
            )

            # comparison waveform
            if category == "eccentric":
                sp_, _ = get_td_waveform(
                    approximant='teobresums',
                    mass1=mass1, mass2=mass2,
                    lambda1=0, lambda2=0,
                    spin1z=params['spin1z'], spin2z=params['spin2z'],
                    distance=distance, delta_t=DELTA_T,
                    ecc   = 0.0,  # quasi-circular reference
                    coa_phase=params['coa_phase'], f_lower=F_LOWER
                )
            else:  # lensed
                sp_, _, _ = compute_lensed_waveform(hp_, _, m_lens, y_lens)

            # standardise length / dt
            tlen = max(len(hp_), len(sp_))
            hp_.resize(tlen)
            sp_.resize(tlen)
            sp_ = pycbc.types.TimeSeries(sp_.data,
                                         delta_t=hp_.delta_t,
                                         epoch=hp_.start_time)

            # PSD & ln B
            duration = hp_.duration
            flen     = tlen // 2 + 1

            if key.name == 'E1' or key.name == 'E2' or key.name == 'E3':
                psd = pycbc.psd.EinsteinTelescopeP1600143(flen, 1/duration, F_LOWER)
            elif key.name == 'C1':
                psd = pycbc.psd.CosmicExplorerP1600143(flen, 1/duration, F_LOWER)
            elif key.name == 'L1' or key.name == 'H1' or key.name == 'V1' or key.name == 'A1':
                psd = pycbc.psd.aLIGOZeroDetHighPower(flen, 1/duration, F_LOWER)

            m = pycbc.filter.match(hp_, sp_, psd=psd, low_frequency_cutoff=F_LOWER)[0]
            ln_B = 0.5 * (1 - m**2) * signal_snr**2
            compute_q = ln_B >= np.log(10)

        # --- Q‑transform & stack ------------------------------------------------
        if compute_q:
            ts = TimeSeries.from_pycbc(ts_raw)
            q  = ts.q_transform(logf=True, norm='median',
                                frange=(F_LOWER, 512),
                                whiten=True, qrange=(4, 64))
            stacked_q.append(np.abs(q.data))
        else:
            print(f"{category}: {key.name} failed ln_B test – skipping Q-transform.")
            return False

    # # need exactly 3 detectors
    # if len(stacked_q) != len(DETECTOR_LIST):
    #     print(f"{category}: only {len(stacked_q)} detectors passed – skipping save.")
    #     return False

    # Find the maximum shape across all detectors
    max_shape = (
        max(q.shape[0] for q in stacked_q),
        max(q.shape[1] for q in stacked_q)
    )
    
    # Pad all arrays to the same shape, then normalize
    padded_q = [pad_to_shape(q, max_shape) for q in stacked_q]
    normed = [Normalize()(q) for q in padded_q]
    
    # # Now stack should work since all arrays have the same shape
    # rgb_img = np.stack(normed, axis=0).transpose(2, 1, 0)

    # plt.figure(figsize=(12, 8))
    # plt.imshow(rgb_img, aspect='auto', origin='lower')
    # plt.axis('off')
    # plt.savefig(path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    # return True

    stacked = np.stack(normed, axis=0)              # (C, H, W)
    stacked_img = np.transpose(stacked, (2, 1, 0))  # (H, W, C)

    # Save as .npy or .tiff
    out_path = str(path)
    if out_path.endswith('.png'):
        out_path = out_path.replace('.png', '.npy')
    np.save(out_path, np.asarray(stacked_img))
    
    return True

def pad_to_shape(arr, target_shape):
    pad_y = target_shape[0] - arr.shape[0]
    pad_x = target_shape[1] - arr.shape[1]
    pad_y1, pad_y2 = pad_y // 2, pad_y - pad_y // 2
    pad_x1, pad_x2 = pad_x // 2, pad_x - pad_x // 2
    return np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant')