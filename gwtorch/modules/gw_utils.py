import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

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

from gwtorch.modules.constants import *

# ------------ Predifined Constant ---------------

delta_f = 1 / 32
flen = int(4096 / (2 * delta_f)) + 1
einstein_psd = pycbc.psd.EinsteinTelescopeP1600143(flen, delta_f, F_START)
aligo_psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, F_START)
cosmic_psd = pycbc.psd.CosmicExplorerP1600143(flen, delta_f, F_START)

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

    priors['mass_1'].minimum = 10
    priors['mass_2'].minimum = 10
    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=10, maximum=100)
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
    priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=-0.9, maximum=0.9)
    priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=-0.9, maximum=0.9)
    priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.01, maximum=0.6)
    priors['anomaly'] = bilby.core.prior.Uniform(name='anomaly', minimum=0.0, maximum=2*np.pi, boundary="periodic")
    priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
    priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")
    priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum=2, maximum=5)
    priors['yl'] = bilby.core.prior.PowerLaw(alpha=1, minimum=0.01, maximum=1.0)
    if snr_type == "powerlaw":
        priors['snr'] = bilby.core.prior.PowerLaw(minimum=10, maximum=500, alpha=-4)
    else:
        priors['snr'] = bilby.core.prior.Uniform(minimum=15, maximum=100)

    parameters = priors.sample(num_samples)
    return [
        {key: parameters[key][i] for key in parameters}
        for i in range(num_samples)
    ]

bounds = [
    (10, 100), # Chirp Mass
    (0.1, 1), # Mass Ratio
    (-0.9, 0.9), # spin1z
    (-0.9, 0.9), # spin2z
    (0, 2*np.pi), # coa_phase
]

def objective_function(params_array, args):
    """
    Compute the mismatch (1 - match) between a target gravitational-wave signal 
    and a template waveform generated with given binary parameters.

    This function constructs a template waveform using the TEOBResumS approximant,
    aligns it with the target signal, and evaluates the noise-weighted match 
    under the detector’s power spectral density (PSD). The returned value is 
    the mismatch, suitable for use as a minimization objective.

    Parameters
    ----------
    params_array : array-like of float
        Model parameters in the order:
        [chirp_mass, mass_ratio, spin1z, spin2z, coa_phase].
        - chirp_mass : float
            The chirp mass of the binary (in solar masses).
        - mass_ratio : float
            The mass ratio q = m2/m1 (with m1 ≥ m2).
        - spin1z : float
            Dimensionless aligned spin component of the primary mass.
        - spin2z : float
            Dimensionless aligned spin component of the secondary mass.
        - coa_phase : float
            Coalescence phase in radians.

    args : tuple
        Additional arguments (signal, distance, detector_name):
        - signal : pycbc.types.TimeSeries
            The target gravitational-wave signal time series.
        - distance : float
            Luminosity distance to the source (in Mpc).
        - detector_name : str
            Name of the detector ('E1', 'E2', 'E3', 'C1', 'L1', 'H1', 'V1', or 'A1'),
            which determines the PSD model to use.

    Returns
    -------
    float
        The mismatch (1 - match_value) between the signal and the template.
        A smaller value indicates a better match. If an error occurs, returns 1.0.

    Notes
    -----
    - The template waveform is generated using the `teobresums` approximant 
      with zero tidal deformabilities and zero eccentricity.
    - The PSD is chosen based on the detector:
        * 'E1', 'E2', 'E3' → Einstein Telescope PSD (ET-D)
        * 'C1'             → Cosmic Explorer PSD
        * 'L1', 'H1', 'V1', 'A1' → aLIGO Zero-Detuned High Power PSD
    - Both the signal and template are resized to the maximum of their lengths
      before computing the match.
    - The objective function is designed for use in parameter estimation 
      or optimization routines, where minimizing the mismatch corresponds 
      to maximizing the similarity between the template and the target signal.
    """
    chirp_mass, mass_ratio, spin1z, spin2z, coa_phase = params_array
    signal, distance, detector_name = args
    try:
        mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass=chirp_mass, mass_ratio=mass_ratio)

        template_hp, template_hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                                    lambda1=0, lambda2=0,
                                                    spin1z=spin1z, spin2z=spin2z,
                                                    distance=distance, delta_t=DELTA_T,
                                                    eccentricity=0.0, coa_phase=coa_phase, f_lower=F_START)
        
        signal_copy = signal.copy()
        temp_copy = template_hp.copy()

        target_len = max(len(signal_copy), len(temp_copy))
        signal_copy.resize(target_len)
        temp_copy.resize(target_len)

        signal_copy = pycbc.types.TimeSeries(signal_copy.data,
                                             delta_t=temp_copy.delta_t,
                                             epoch=temp_copy.start_time)

        duration = temp_copy.duration
        flen     = target_len // 2 + 1

        if detector_name == 'E1' or detector_name == 'E2' or detector_name == 'E3':
            psd = pycbc.psd.EinsteinTelescopeP1600143(flen, 1/duration, F_START)
        elif detector_name == 'C1':
            psd = pycbc.psd.CosmicExplorerP1600143(flen, 1/duration, F_START)
        elif detector_name == 'L1' or detector_name == 'H1' or detector_name == 'V1' or detector_name == 'A1':
            psd = pycbc.psd.aLIGOZeroDetHighPower(flen, 1/duration, F_START)
        
        match_value, _ = pycbc.filter.match(signal_copy, temp_copy, psd=psd, low_frequency_cutoff=F_LOWER)

        return 1.0 - match_value
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1.0

def generate_waveform(num, params, peak_window=(2.0, 2.2), detector=default_detector,
                      display=False, compute_lnB_based_on_match=False, 
                      compute_lnB_based_on_FF=False, SEED=None):
    """
    Generate eccentric, unlensed, and lensed gravitational waveforms with injected noise.

    This function constructs three types of gravitational waveforms using the 
    `TEOBResumS` approximant:
    1. An eccentric waveform (`ecc_noisy`),
    2. A quasi-circular (unlensed) waveform (`unlens_noisy`),
    3. A lensed version of the quasi-circular waveform (`lens_noisy`).

    Each waveform is projected onto a detector, tapered to reduce spectral leakage,
    and injected into Gaussian noise to achieve a specified target signal-to-noise ratio (SNR).
    Optionally, waveform matches or fitting factors can be computed for Bayesian evidence estimation.

    Parameters
    ----------
    num : int
        Sample index used for logging and identification.

    params : dict
        Dictionary of source and lens parameters. Expected keys include:
        - 'chirp_mass' : float
            Chirp mass of the binary system (solar masses).
        - 'mass_ratio' : float
            Mass ratio q = m2/m1 (0 < q ≤ 1).
        - 'spin1z', 'spin2z' : float
            Dimensionless aligned spin components of the two bodies.
        - 'eccentricity' : float
            Initial eccentricity of the binary.
        - 'coa_phase' : float
            Coalescence phase (radians).
        - 'polarization' : float
            Polarization angle (radians).
        - 'ra', 'dec' : float
            Right ascension and declination of the source (radians).
        - 'Log_Mlz' : float
            Log10 of the lens mass (solar masses).
        - 'yl' : float
            Dimensionless lensing impact parameter.
        - 'snr' : float
            Target signal-to-noise ratio for noise injection.

    peak_window : tuple of (float, float), optional
        Time window (in seconds, relative to the end of the timeseries) in which 
        the signal peak is injected. Default is (2.0, 2.2).

    detector : Detector object, optional
        Detector instance (e.g., `bilby.gw.detector.Interferometer`).
        Determines antenna response and PSD. Default is `default_detector`.

    display : bool, optional
        If True, plots noisy and padded versions of the three waveforms.
        Default is False.

    compute_lnB_based_on_match : bool, optional
        If True, computes match values (overlap integrals) between waveforms
        to build evidence-related quantities. Default is True.

    compute_lnB_based_on_FF : bool, optional
        If True, computes fitting factors by performing parameter optimization 
        using `scipy.optimize.differential_evolution`. Default is False.

    SEED : dict, optional
        Dictionary of random seeds for reproducible noise injection, with keys:
        - 'ecc_seed', 'lens_seed', 'unlens_seed'.

    Returns
    -------
    tuple or None
        On success, returns:
        - ecc_dict : dict
            Contains noisy eccentric waveform, SNR, distance, and optionally match/FF.
        - lens_dict : dict
            Contains noisy lensed waveform, SNR, distance, and optionally match/FF.
        - unlens_dict : dict
            Contains noisy unlensed waveform, SNR, distance.
        - mass1 : float
            Primary mass (solar masses).
        - mass2 : float
            Secondary mass (solar masses).
        - m_lens : float
            Lens mass (solar masses).
        - y_lens : float
            Lensing impact parameter.

        Returns `None` if waveform generation or noise injection fails.

    Notes
    -----
    - Waveforms are generated using `pycbc.waveform.get_td_waveform` with the
      `teobresums` approximant, zero tidal deformabilities, and specified eccentricity.
    - The lensing effect is computed using `compute_lensed_waveform`.
    - Noise injection and SNR scaling are performed by `inject_noise_with_target_SNR`.
    - Waveforms are cropped from the left (24 samples) to remove startup artifacts.
    - PSDs are selected based on the detector name:
        * 'E1', 'E2', 'E3' → Einstein Telescope
        * 'C1' → Cosmic Explorer
        * 'L1', 'H1', 'V1', 'A1' → aLIGO Zero-Detuned High Power
    - Setting `compute_lnB_based_on_FF=True` can be computationally expensive.
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
                                 eccentricity=params['eccentricity'], mean_per_ano=params['anomaly'], coa_phase=params['coa_phase'], f_lower=F_START)

        sp, sc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=REFERENCE_DISTANCE, delta_t=DELTA_T,
                                 eccentricity=0.0, coa_phase=params['coa_phase'], f_lower=F_START)

        sp_lensed, sc_lensed, t_delay = compute_lensed_waveform(sp, sc, m_lens, y_lens)

        ecc = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        unlensed = taper_timeseries(detector.project_wave(sp, sc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        lensed = taper_timeseries(detector.project_wave(sp_lensed, sc_lensed, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

        target_snr = params['snr']

        ECC_SEED = SEED['ecc_seed']
        LENS_SEED = SEED['lens_seed']
        UNLENS_SEED = SEED['unlens_seed']

        ecc_noisy, snr_e, distance_e, ecc_padded = inject_noise_with_target_SNR(ecc, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='eccentric', peak_window=peak_window, detector=detector, SEED=ECC_SEED)
        lens_noisy, snr_l, distance_l, lens_padded = inject_noise_with_target_SNR(lensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='lensed', peak_window=peak_window, detector=detector, SEED=LENS_SEED)
        unlens_noisy, snr_u, distance_u, unlens_padded = inject_noise_with_target_SNR(unlensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='quasi_circular', peak_window=peak_window, detector=detector, SEED=UNLENS_SEED)
        
        ecc_noisy = ecc_noisy.crop(left=24, right=0)
        lens_noisy = lens_noisy.crop(left=24, right=0)
        unlens_noisy = unlens_noisy.crop(left=24, right=0)

        if display:

            fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            axs[0].plot(ecc_noisy.sample_times, ecc_noisy, label="Eccentric (noisy)", linestyle='-', color='C0')
            axs[0].plot(ecc_padded.sample_times, ecc_padded, label="Eccentric (padded)", linestyle='-', color='C1')
            # axs[0].plot(ecc.sample_times, ecc)
            axs[0].set_ylabel("Strain")
            axs[0].set_title("Eccentric Waveform")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(lens_noisy.sample_times, lens_noisy, label="Lensed (noisy)", linestyle='-', color='C2')
            axs[1].plot(lens_padded.sample_times, lens_padded, label="Lensed (padded)", linestyle='-', color='C3')
            # axs[1].plot(lensed.sample_times, lensed)
            axs[1].set_ylabel("Strain")
            axs[1].set_title("Lensed Waveform")
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(unlens_noisy.sample_times, unlens_noisy, label="Unlensed (noisy)", linestyle='-', color='C4')
            axs[2].plot(unlens_padded.sample_times, unlens_padded, label="Unlensed (padded)", linestyle='-', color='C5')
            # axs[2].plot(unlensed.sample_times, unlensed)
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("Strain")
            axs[2].set_title("Unlensed Waveform")
            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout()
            plt.show()

        if compute_lnB_based_on_match:
            tlen = max(len(hp), len(sp), len(sp_lensed))
            
            hp.resize(tlen)
            sp.resize(tlen)
            sp_lensed.resize(tlen)

            sp_lensed = pycbc.types.TimeSeries(sp_lensed.data,
                                         delta_t=hp.delta_t,
                                         epoch=hp.start_time)
            
            duration = hp.duration
            flen     = tlen // 2 + 1

            if detector.name == 'E1' or detector.name == 'E2' or detector.name == 'E3':
                psd = pycbc.psd.EinsteinTelescopeP1600143(flen, 1/duration, F_START)
            elif detector.name == 'C1':
                psd = pycbc.psd.CosmicExplorerP1600143(flen, 1/duration, F_START)
            elif detector.name == 'L1' or detector.name == 'H1' or detector.name == 'V1' or detector.name == 'A1':
                psd = pycbc.psd.aLIGOZeroDetHighPower(flen, 1/duration, F_START)

            ecc_m = pycbc.filter.match(hp, sp, psd=psd, low_frequency_cutoff=F_LOWER)[0]
            lens_m = pycbc.filter.match(sp_lensed, sp, psd=psd, low_frequency_cutoff=F_LOWER)[0]

            lens_dict = {'lensed_signal': lens_noisy, 'snr_l': snr_l, 'distance_l': distance_l, 'lens_m': lens_m}
            ecc_dict = {'ecc_signal': ecc_noisy, 'snr_e': snr_e, 'distance_e': distance_e, 'ecc_m': ecc_m}
            unlens_dict = {'unlensed_signal': unlens_noisy, 'snr_u': snr_u, 'distance_u': distance_u}
        
        elif compute_lnB_based_on_FF:
            ecc_args_for_optimizer = (hp, REFERENCE_DISTANCE, detector.name)
            ecc_result = differential_evolution(
                        func=objective_function,
                        bounds=bounds,
                        args=(ecc_args_for_optimizer,),
                        strategy='best1bin',
                        maxiter=30,
                        popsize=15,
                        disp=True,
                        workers=1,
                        updating="deferred"
                    )
            ecc_fitting_factor = 1.0 - ecc_result.fun
            
            lensed_args_for_optimizer = (sp_lensed, REFERENCE_DISTANCE, detector.name)
            lens_result = differential_evolution(
                        func=objective_function,
                        bounds=bounds,
                        args=(lensed_args_for_optimizer,),
                        strategy='best1bin',
                        maxiter=30,
                        popsize=15,
                        disp=True,
                        workers=1,
                        updating="deferred"
                    )
            lens_fitting_factor = 1.0 - lens_result.fun

            lens_dict = {'lensed_signal': lens_noisy, 'snr_l': snr_l, 'distance_l': distance_l, 'lens_FF': lens_fitting_factor}
            ecc_dict = {'ecc_signal': ecc_noisy, 'snr_e': snr_e, 'distance_e': distance_e, 'ecc_FF': ecc_fitting_factor}
            unlens_dict = {'unlensed_signal': unlens_noisy, 'snr_u': snr_u, 'distance_u': distance_u}

        else:
            lens_dict = {'lensed_signal': lens_noisy, 'snr_l': snr_l, 'distance_l': distance_l,}
            ecc_dict = {'ecc_signal': ecc_noisy, 'snr_e': snr_e, 'distance_e': distance_e}
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
    Generate Gaussian noise using detector-specific power spectral densities (PSDs).

    This function produces a simulated Gaussian noise time series based on the 
    specified detector sensitivity curve. The noise is sampled at 4096 Hz 
    and generated via the PyCBC `noise_from_psd` routine, which ensures that 
    the resulting noise realization follows the input PSD.

    Supported PSD types:
    - **aLIGO**: Advanced LIGO design sensitivity (O4; aLIGOZeroDetHighPower).
    - **ET**: Einstein Telescope projected sensitivity.
    - **CE**: Cosmic Explorer projected sensitivity.

    Parameters
    ----------
    time_duration : float, optional
        Duration of the time series in seconds. Default is 32 s.

    PSD_TYPE : str, optional
        Detector sensitivity curve to use. Must be one of:
        ``'aLIGO'`` (default), ``'ET'``, or ``'CE'``.

    seed : int or None, optional
        Random seed for reproducibility. If None (default), the noise realization
        will vary between runs.

    Returns
    -------
    noise : pycbc.types.TimeSeries
        A PyCBC `TimeSeries` object containing the generated Gaussian noise 
        realization sampled at 4096 Hz.

    Notes
    -----
    - The PSDs (`aligo_psd`, `einstein_psd`, `cosmic_psd`) must be preloaded 
      and passed to `pycbc.noise.noise_from_psd`.
    - The generated noise has zero mean and variance determined by the PSD.
    - Useful for injection studies, parameter estimation, and testing waveform recovery.
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
    Determine the PSD type and corresponding PSD object for a given detector.

    This function maps the input detector to the appropriate sensitivity curve 
    used in the module. Different ground-based interferometers are associated 
    with specific design sensitivity PSDs.

    Parameters
    ----------
    detector : bilby.gw.detector.Interferometer, optional
        Detector object (default is `default_detector`).

    Returns
    -------
    tuple
        A tuple (psd_type, psd_object), where:
        - psd_type : str
            Identifier of the PSD. One of:
            ``'ET'`` (Einstein Telescope),
            ``'CE'`` (Cosmic Explorer),
            ``'aLIGO'`` (Advanced LIGO design sensitivity).
        - psd_object : pycbc.types.FrequencySeries
            PSD object corresponding to the chosen detector.

    Notes
    -----
    - Einstein Telescope (detectors 'E1', 'E2', 'E3') → ``ET`` PSD.
    - Cosmic Explorer (detector 'C1') → ``CE`` PSD.
    - Advanced LIGO (detectors 'L1', 'H1', 'V1', 'A1') → ``aLIGO`` PSD.
    """
    if detector.name == 'E1' or detector.name == 'E2' or detector.name == 'E3':
        return 'ET', einstein_psd
    elif detector.name == 'C1':
        return 'CE', cosmic_psd
    elif detector.name == 'L1' or detector.name == 'H1' or detector.name == 'V1' or detector.name == 'A1':
        return 'aLIGO', aligo_psd

def inject_noise_with_target_SNR(gw_signal, params, mass1, mass2, m_lens, y_lens, snr_desired, num,
                                 gw_signal_type='eccentric', peak_window=(2.0, 2.2), 
                                 detector=default_detector, SEED=None):
    """
    Inject a gravitational wave signal into Gaussian noise and scale it to a target SNR.

    This function takes a gravitational wave (GW) signal, injects it into simulated 
    Gaussian noise drawn from a detector-specific power spectral density (PSD), and 
    rescales the signal amplitude (via luminosity distance) so that the resulting 
    matched-filter signal-to-noise ratio (SNR) matches the desired value. Supports 
    eccentric, quasi-circular, and point-mass lensed waveforms.

    Parameters
    ----------
    gw_signal : pycbc.types.TimeSeries
        Initial gravitational wave strain time series to be injected (before rescaling).

    params : dict
        Dictionary of intrinsic and extrinsic waveform parameters, including:
        - 'spin1z', 'spin2z' : float
            Dimensionless aligned spin components.
        - 'eccentricity' : float
            Orbital eccentricity at reference frequency.
        - 'coa_phase' : float
            Coalescence phase (radians).
        - 'ra', 'dec' : float
            Right ascension and declination (radians).
        - 'polarization' : float
            Polarization angle (radians).

    mass1 : float
        Mass of the primary object (in solar masses).

    mass2 : float
        Mass of the secondary object (in solar masses).

    m_lens : float
        Redshifted lens mass (in solar masses). Used only if `gw_signal_type='lensed'`.

    y_lens : float
        Dimensionless lensing impact parameter. Used only if `gw_signal_type='lensed'`.

    snr_desired : float
        Target matched-filter SNR to which the signal is rescaled.

    num : int
        Sample index, used for logging/debugging.

    gw_signal_type : {'eccentric', 'quasi_circular', 'lensed'}, optional
        Type of waveform injection:
        - ``'eccentric'`` : Generates an eccentric waveform (default).
        - ``'quasi_circular'`` : Generates a circular (ecc=0) waveform.
        - ``'lensed'`` : Generates a point-mass lensed waveform.

    peak_window : tuple of (float, float), optional
        Time window (in seconds, relative to the end of the timeseries) in which 
        the signal peak is injected. Default is (2.0, 2.2).

    detector : bilby.gw.detector.Interferometer, optional
        Detector object used to project the waveform and select the PSD. 
        Default is `default_detector`.

    SEED : int or None, optional
        Seed for Gaussian noise generation. If None (default), noise realizations 
        differ between runs.

    Returns
    -------
    data : pycbc.types.TimeSeries
        The time series containing the sum of noise and the injected, rescaled GW signal.

    peak_snr : float
        Measured matched-filter SNR of the injected signal after rescaling.

    distance : float
        Luminosity distance (in Mpc) to which the signal was scaled in order 
        to achieve the desired SNR.

    template : pycbc.types.TimeSeries
        The rescaled and projected GW waveform template used in the injection.

    Notes
    -----
    - Noise is generated using `generate_noise`, with PSD chosen via `get_psd_type`.
    - The scaling to the target SNR is achieved by adjusting the luminosity distance:
      ``distance = REFERENCE_DISTANCE * (peak_snr_initial / snr_desired)``.
    - Waveforms are generated with the `teobresums` approximant and tapered 
      with `TAPER_STARTEND` to reduce spectral artifacts.
    - For lensed injections, `compute_lensed_waveform` is used to apply 
      point-mass lensing magnification and time delays.
    - Injection is placed within `peak_window` seconds of the end of the 
      noise timeseries to avoid boundary effects.
    """
    PSD_TYPE, psd = get_psd_type(detector)

    noise = generate_noise(PSD_TYPE=PSD_TYPE, seed = int(SEED)) if SEED is not None else generate_noise(PSD_TYPE=PSD_TYPE)

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=gw_signal,
                                            noise_ts=noise,
                                            peak_window=peak_window)
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)
    
    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)
    
    snr = pycbc.filter.matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=F_LOWER)

    peak_snr_0 = abs(snr).numpy().max()

    distance = REFERENCE_DISTANCE * (peak_snr_0 / snr_desired) 

    if gw_signal_type == 'eccentric':
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 eccentricity=params['eccentricity'], mean_per_ano=params['anomaly'], coa_phase=params['coa_phase'], f_lower=F_START)
        
        gw_signal = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
    
    elif gw_signal_type == 'quasi_circular':
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 eccentricity=0.0, coa_phase=params['coa_phase'], f_lower=F_START)
        
        gw_signal = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        
    else: 
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 eccentricity=0.0, coa_phase=params['coa_phase'], f_lower=F_START) 
        
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
    Align a gravitational wave signal within a noise timeseries by placing its peak
    randomly inside a specified time window near the end of the noise.

    This function zero-pads the input signal so that its maximum amplitude occurs
    within the last `peak_window` seconds of the noise realization, ensuring 
    consistent placement for injection studies.

    Parameters
    ----------
    signal_ts : pycbc.types.TimeSeries
        Time-domain gravitational wave signal (strain).

    noise_ts : pycbc.types.TimeSeries
        Time-domain Gaussian noise with the same sampling rate as `signal_ts`.

    peak_window : tuple of (float, float), optional
        Time window (in seconds, relative to the end of the noise) in which the 
        signal peak will be placed. Default is (2.0, 2.2).

    Returns
    -------
    padded_signal : np.ndarray
        The zero-padded signal array, aligned so that its peak lies within the 
        chosen window inside the noise timeseries.

    delta_t : float
        Sampling interval (seconds) of the timeseries.

    start_time : lal.LIGOTimeGPS
        Start time of the output padded signal (same as `signal_ts.start_time`).

    Notes
    -----
    - The signal is truncated if it does not fully fit into the noise segment.
    - The peak location is chosen uniformly at random within the specified window.
    - This function does not perform the actual injection; it prepares the 
      signal alignment for later addition to the noise.
    """
    signal = np.asarray(signal_ts)

    delta_t = noise_ts.delta_t
    N = len(noise_ts)
    duration = N * delta_t

    peak_index = np.argmax(np.abs(signal))

    min_offset, max_offset = peak_window
    t_peak = np.random.uniform(duration - max_offset, duration - min_offset)
    target_index = int(t_peak / delta_t)

    start_index = target_index - peak_index
    end_index = start_index + len(signal)

    padded_signal = np.zeros(N)

    if start_index < 0:
        signal = signal[-start_index:]
        start_index = 0
        end_index = start_index + len(signal)

    if end_index > N:
        signal = signal[:N - start_index]
        end_index = start_index + len(signal)

    padded_signal[start_index:end_index] = signal

    return padded_signal, delta_t, signal_ts.start_time

def save_simple_qtransform(ts, file_name):
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
    file_name : str
        The file path where the generated Q-transform image will be saved.

    Returns
    -------
    None
        The function saves the figure to disk and does not return anything.
    """
    if not isinstance(ts, TimeSeries):
        ts = TimeSeries.from_pycbc(ts)

    qt = ts.q_transform(logf=True, norm='median', frange=(F_LOWER, 512), whiten=True, qrange=(4, 64))

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(qt)
    plt.axis('off')
    plt.yscale('log')
    plt.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()


def display_qtransform(ts, method='mean'):
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
    if not isinstance(ts, TimeSeries):
        ts = TimeSeries.from_pycbc(ts)

    qt = ts.q_transform(logf=True, norm=method, frange=(F_LOWER, 512), qrange=(4, 64), whiten=True)
    
    plt.figure(figsize=(12, 8), frameon=False)
    plt.pcolormesh(np.array(qt.data).T)
    plt.axis('off')
    plt.yscale('log')
    plt.show()

    return qt

# def _get_category_parts(path_str):
#     """Return ('eccentric' | 'lensed' | 'unlensed', True/False)"""
#     if "eccentric" in path_str or "-2" in path_str:
#         return "eccentric"
#     if "lensed" in path_str and "unlensed" not in path_str and "-1" not in path_str:
#         return "lensed"
#     if "-1" in path_str:
#         return "lensed"
#     return "unlensed"

def _get_category_parts(path_str):
    """Return ('eccentric' | 'lensed' | 'unlensed', True/False)"""
    if "eccentric" in path_str:
        return "eccentric"
    if "lensed" in path_str and "unlensed" not in path_str:
        return "lensed"
    if "-1" in path_str:
        return "lensed"
    return "unlensed"

def save_qtransform_with_checks(waveform_dict, path):
    """
    Compute and save multi-channel Q-transform spectrograms from waveform data.

    Each channel corresponds to one detector's strain timeseries. The function
    infers the waveform category (eccentric / lensed / unlensed) from the output 
    filename and selects the appropriate signal for the Q-transform. The resulting 
    spectrograms are zero-padded to match the largest shape among detectors, stacked 
    into a N-D array (C, H, W), and saved as a `.npy` file.

    Parameters
    ----------
    waveform_dict : dict
        Dictionary containing waveform data for one or more detectors. Expected keys
        include detector names (e.g., "H1", "L1") mapping to dictionaries with:
        - 'eccentric' → {'ecc_signal': TimeSeries}
        - 'lensed'    → {'lensed_signal': TimeSeries}
        - 'unlensed'  → {'unlensed_signal': TimeSeries}
    
    path : str or pathlib.Path
        Output file path. If it ends with `.png`, the extension will be replaced 
        with `.npy`.

    Returns
    -------
    success : bool
        Returns True if the Q-transform stack is successfully computed and saved,
        False otherwise.

    Notes
    -----
    - Uses GWpy's `TimeSeries.q_transform` for generating spectrograms.
    - Frequency range is fixed to `(F_LOWER, 512)` Hz with whitening and mean 
      normalization.
    - The output is stored as a numpy array of shape `(C, H, W)`, where:
        - C = number of detectors,
        - H = time bins,
        - W = frequency bins.
    - Padding is symmetric to center-align smaller spectrograms.
    """
    category = _get_category_parts(str(path))
    stacked_q = []

    for key in waveform_dict:
        wf = waveform_dict[key]

        if category == "eccentric":
            ts_raw     = wf['eccentric']['ecc_signal']
        elif category == "lensed":
            ts_raw     = wf['lensed']['lensed_signal']
        else:  # unlensed
            ts_raw     = wf['unlensed']['unlensed_signal']


        # --- Q‑transform & stack ------------------------------------------------

        ts = TimeSeries.from_pycbc(ts_raw)
        q  = ts.q_transform(logf=True, norm='mean',
                            frange=(F_LOWER, 512),
                            whiten=True, qrange=(4, 64))
        stacked_q.append(np.array(q.data).T)

    max_shape = (
        max(q.shape[0] for q in stacked_q),
        max(q.shape[1] for q in stacked_q)
    )
    
    padded_q = [pad_to_shape(q, max_shape) for q in stacked_q]
    # normed = [Normalize()(q) for q in padded_q]

    # stacked = np.stack(normed, axis=0)              # (C, H, W)
    stacked = np.stack(padded_q, axis=0)              # (C, H, W)

    out_path = str(path)
    if out_path.endswith('.png'):
        out_path = out_path.replace('.png', '.npy')
    np.save(out_path, np.asarray(stacked))
    
    return True

def pad_to_shape(arr, target_shape):
    """
    Zero-pad a 2D array symmetrically to match a target shape.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array to be padded.
    target_shape : tuple of (int, int)
        Desired output shape (rows, cols).

    Returns
    -------
    padded_arr : np.ndarray
        Padded 2D array with shape exactly equal to `target_shape`.

    Notes
    -----
    - Padding is applied equally on both sides (top/bottom, left/right).
    - If the difference in size is odd, the extra padding is placed on the 
      bottom/right side.
    """
    pad_y = target_shape[0] - arr.shape[0]
    pad_x = target_shape[1] - arr.shape[1]
    pad_y1, pad_y2 = pad_y // 2, pad_y - pad_y // 2
    pad_x1, pad_x2 = pad_x // 2, pad_x - pad_x // 2
    return np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant')