import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import argparse
from multiprocessing.pool import Pool
import bilby
import numpy as np
from pycbc.waveform import get_td_waveform, taper_timeseries
import pycbc.types
import pylab
import pycbc.noise
import pycbc.psd
from gwmat import point_lens
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import gwmat
import os
import sys
from pycbc.detector.ground import Detector
from pathlib import Path
from functools import partial
from pycbc.types import FrequencySeries
import importlib.resources as resources
from gwtorch.modules.gw_utils import scale_signal


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training data using Q-transform.")
    parser.add_argument('--num-samples', type=int, required=True, help='Number of waveform samples to generate')
    parser.add_argument('--path-name', type=str, default='data_for_evaluation', help='Path to save generated data')
    parser.add_argument('--models', type=str, required=True, choices=['Ecc_Qc', 'L_U', 'Qc_Ecc', 'U_L'], help='Either Eccentric vs Quasi-Circular or  Lensed vs Unlensed (Eccentric) waveforms')
    return parser.parse_args()


def setup_paths(base_path):
    ln_below_10 = Path(base_path) / 'ln_below_10'
    ln_bw_10_and_30 = Path(base_path) / 'ln_bw_10_and_30'
    ln_above_30 = Path(base_path) / 'ln_above_30'

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(ln_below_10, exist_ok=True)
    os.makedirs(ln_bw_10_and_30, exist_ok=True)
    os.makedirs(ln_above_30, exist_ok=True)

    return ln_below_10, ln_bw_10_and_30, ln_above_30


def define_priors(models):
    if models in ['L_U', 'U_L']:
        prior_file = resources.files('gwtorch.waveform_generation.Prior_Files') / 'ecc_lensed.prior'
    elif models in ['Ecc_Qc', 'Qc_Ecc']:
        prior_file = resources.files('gwtorch.waveform_generation.Prior_Files') / 'ecc_qc.prior'
    else:
        raise ValueError(f"Unsupported model: {models}")
    
    return bilby.gw.prior.PriorDict(filename=str(prior_file))



def save_qtransform_plot(signal, num, path, models):
    if models == 'Ecc_Qc':
        file_name = 'Eccentric'
    elif models == 'L_U':
        file_name = 'Lensed'
    elif models == 'Qc_Ecc':
        file_name = 'Quasi_Circular'
    elif models == 'U_L':
        file_name = 'Unlensed'

    noisy_gwpy_eccentric = TimeSeries.from_pycbc(signal)
    plt.figure(figsize=(12, 8), facecolor=None)
    plt.pcolormesh(noisy_gwpy_eccentric.q_transform(logf=True, norm='mean', frange=(5, 512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.yscale('log')
    plt.savefig(path / f'{file_name}_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()

def compute_lensed_waveform(sp, sc, m_lens, y_lens):
    """Apply lensing amplification to the unlensed waveform."""
    sp_freq = sp.to_frequencyseries(delta_f=sp.delta_f)
    sc_freq = sc.to_frequencyseries(delta_f=sc.delta_f)
    freqs = sp_freq.sample_frequencies

    Ffs = np.vectorize(lambda f: gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens))(freqs)
    t_delay = point_lens.time_delay(ml=m_lens, y=y_lens)

    sp_lens = FrequencySeries(np.conj(Ffs) * sp_freq.numpy(), delta_f=sp_freq.delta_f).cyclic_time_shift(-(0.1 + t_delay))
    sc_lens = FrequencySeries(np.conj(Ffs) * sc_freq.numpy(), delta_f=sc_freq.delta_f).cyclic_time_shift(-(0.1 + t_delay))

    return sp_lens.to_timeseries(delta_t=sp.delta_t), sc_lens.to_timeseries(delta_t=sc.delta_t), t_delay


def generate_waveform(parameters, models):
    mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        parameters['chirp_mass'], parameters['mass_ratio']
    )

    if models in ['L_U', 'U_L']:
        m_lens = 10 ** parameters.pop("Log_Mlz")
        y_lens = parameters.pop("yl")
    
    hp, hc = get_td_waveform(
        approximant='teobresums',
        mass1=mass1,
        mass2=mass2,
        lambda1=0,
        lambda2=0,
        spin1z=parameters['spin1z'],
        spin2z=parameters['spin2z'],
        distance=parameters['distance'],
        delta_t=1.0 / 4096,
        ecc=parameters['eccentricity'],
        coa_phase=parameters['coa_phase'],
        f_lower=5.0,
    )

    sp, sc = get_td_waveform(
        approximant='teobresums',
        mass1=mass1,
        mass2=mass2,
        lambda1=0,
        lambda2=0,
        spin1z=parameters['spin1z'],
        spin2z=parameters['spin2z'],
        distance=parameters['distance'],
        delta_t=1.0 / 4096,
        ecc=0,
        coa_phase=parameters['coa_phase'],
        f_lower=5.0,
    )

    if models in ['L_U', 'U_L']:
        sp, sc, t_delay = compute_lensed_waveform(sp, sc, m_lens, y_lens)

    if models in ['Qc_Ecc', 'L_U']:
        hp, hc, sp, sc = sp, sc, hp, hc

    return hp, hc, sp, sc


def project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num):
    detector = Detector('H1')
    signal = detector.project_wave(hp, hc, ra=parameters['ra'], dec=parameters['dec'], polarization=parameters['polarization'])
    signal = taper_timeseries(signal, tapermethod="TAPER_STARTEND", return_lal=False)
    noisy_signal, signal_snr = scale_signal(signal, num)
    noisy_signal = noisy_signal.crop(left=24, right=0)

    tlen = max(len(hp), len(sp))
    hp.resize(tlen)
    sp.resize(tlen)

    duration = hp.duration
    flow = 5
    delta_f = 1 / duration
    flen = tlen // 2 + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    match = pycbc.filter.match(hp, sp, psd=psd, low_frequency_cutoff=5)[0]
    ln_B = 0.5 * (1 - match ** 2) * signal_snr ** 2

    return ln_B, noisy_signal, match, signal_snr


def generate_additional_waveforms(num, models, indicator, match, signal_snr, samples, ln_below_10, ln_bw_10_and_30, ln_above_30):
    def adjust_distance_and_generate(ln_B_required, expected_condition, save_path, lower_bound, upper_bound, match=match, signal_snr=signal_snr, models=models):
        require_snr = np.sqrt(2 * ln_B_required / (1 - match ** 2))
        parameters = samples[num].copy()
        parameters['distance'] = parameters['distance'] * signal_snr / require_snr
        hp, hc, sp, sc = generate_waveform(parameters, models)
        ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)
        assert expected_condition(ln_B), f"ln_B should be between {lower_bound} and {upper_bound}, but was {ln_B}"
        save_qtransform_plot(noisy_signal, num, save_path, models)

    if indicator < 10:
        adjust_distance_and_generate(np.random.uniform(10, 30), lambda x: 10 < x < 30, ln_bw_10_and_30, 10, 30)
        adjust_distance_and_generate(np.random.uniform(30, 200), lambda x: x > 30, ln_above_30, 30, 200)
    elif 10 <= indicator < 30:
        adjust_distance_and_generate(np.random.uniform(1, 10), lambda x: x <= 10, ln_below_10, 1, 10)
        adjust_distance_and_generate(np.random.uniform(30, 200), lambda x: x > 30, ln_above_30, 30, 200)
    else:
        adjust_distance_and_generate(np.random.uniform(1, 10), lambda x: x <= 10, ln_below_10, 1, 10)
        adjust_distance_and_generate(np.random.uniform(10, 30), lambda x: 10 < x < 30, ln_bw_10_and_30, 10, 30)


def generate_training_qtransform(num, samples, models, ln_below_10, ln_bw_10_and_30, ln_above_30):
    parameters = samples[num].copy()
    hp, hc, sp, sc = generate_waveform(parameters, models)
    ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

    try:
        if ln_B <= 10:
            save_qtransform_plot(noisy_signal, num, ln_below_10, models)
        elif 10 < ln_B < 30:
            save_qtransform_plot(noisy_signal, num, ln_bw_10_and_30, models)
        else:
            save_qtransform_plot(noisy_signal, num, ln_above_30, models)

        generate_additional_waveforms(num, models, ln_B, match, signal_snr, samples, ln_below_10, ln_bw_10_and_30, ln_above_30)
    except Exception as e:
        print(f"Error generating Q-transform for sample {num}: {e}")
        return None

def task(i, samples, models, ln_below_10, ln_bw_10_and_30, ln_above_30):
    return generate_training_qtransform(i, samples, models, ln_below_10, ln_bw_10_and_30, ln_above_30)


def main():
    args = parse_args()
    num_samples = args.num_samples
    path_name = args.path_name
    models = args.models

    ln_below_10, ln_bw_10_and_30, ln_above_30 = setup_paths(path_name)

    priors = define_priors(models)
    parameters_list = priors.sample(num_samples)

    samples = [
        {key: parameters_list[key][i] for key in parameters_list}
        for i in range(num_samples)
    ]

    # Prepare the argument tuples for starmap
    task_args = [
        (i, samples, models, ln_below_10, ln_bw_10_and_30, ln_above_30)
        for i in range(num_samples)
    ]

    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(task, task_args)


if __name__ == "__main__":
    main()















































# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# import csv
# from multiprocessing.pool import Pool
# import bilby
# import numpy as np
# from pycbc.waveform import get_td_waveform, taper_timeseries
# import pycbc.types
# import pylab
# import pycbc.noise
# import pycbc.psd
# from gwmat import point_lens
# from gwpy.timeseries import TimeSeries
# import matplotlib.pyplot as plt
# import gwmat
# import os
# import sys
# from pycbc.detector.ground import Detector
# from pathlib import Path

# from gwtorch.modules.gw_utils import scale_signal

# num_processes = os.cpu_count()
# num_samples = int(sys.argv[1])
# path_name = 'data_for_testing'
# ln_below_10 = 'data_for_testing/ln_below_10'
# ln_bw_10_and_30 = 'data_for_testing/ln_bw_10_and_30'
# ln_above_30 = 'data_for_testing/ln_above_30'

# # sys.stdout = open("log.out", "w")
# # sys.stderr = open("error.err", "w")

# os.makedirs(path_name, exist_ok=True)
# os.makedirs(ln_below_10, exist_ok=True)
# os.makedirs(ln_above_30, exist_ok=True)
# os.makedirs(ln_bw_10_and_30, exist_ok=True)

# ln_above_30 = Path(ln_above_30)
# ln_below_10 = Path(ln_below_10)
# ln_bw_10_and_30 = Path(ln_bw_10_and_30)

# f_lower = 5.0       

# priors = bilby.core.prior.PriorDict()

# priors["mass1"] = bilby.core.prior.Constraint(name="mass1", minimum=10, maximum=100)
# priors["mass2"] = bilby.core.prior.Constraint(name="mass2", minimum=10, maximum=100)
# priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
# priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
# priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=0.0, maximum=0.9)
# priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=0.0, maximum=0.9)
# priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.1, maximum=0.6)
# priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
# priors['distance'] = bilby.core.prior.Uniform(name='distance', minimum=100, maximum=1000)
# priors['dec'] = bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)
# priors['ra'] = bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")
# priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")

# parameters_list = priors.sample(num_samples)

# samples = [
#     {key: parameters_list[key][i] for key in parameters_list}
#     for i in range(num_samples)
# ]

# print(f"Length of parameters_list: {len(samples)}")

# def save_qtransform_plot(signal, num, path):
#     noisy_gwpy_eccentric = TimeSeries.from_pycbc(signal)
#     plt.figure(figsize=(12,8), facecolor=None)
#     plt.pcolormesh(noisy_gwpy_eccentric.q_transform(logf=True, norm='mean', frange=(5,512), whiten=True, qrange=(4, 64)))
#     plt.axis('off')
#     plt.yscale('log')
#     plt.savefig(path / f'eccentric_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
#     plt.close()

# def generate_waveform(parameters):
#     mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
#         parameters['chirp_mass'], parameters['mass_ratio']
#     )

#     hp, hc = get_td_waveform(
#         approximant='teobresums',
#         mass1=mass1,
#         mass2=mass2,
#         lambda1=0,
#         lambda2=0,
#         spin1z=parameters['spin1z'],
#         spin2z=parameters['spin2z'],
#         distance=parameters['distance'],
#         delta_t=1.0 / 4096,
#         ecc=parameters['eccentricity'],
#         coa_phase=parameters['coa_phase'],
#         f_lower=f_lower,
#     )

#     sp, sc = get_td_waveform(
#         approximant='teobresums',
#         mass1=mass1,
#         mass2=mass2,
#         lambda1=0,
#         lambda2=0,
#         spin1z=parameters['spin1z'],
#         spin2z=parameters['spin2z'],
#         distance=parameters['distance'],
#         delta_t=1.0 / 4096,
#         ecc=0,
#         coa_phase=parameters['coa_phase'],
#         f_lower=f_lower,
#     )

#     return hp, hc, sp, sc

# def project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num):

#     ####---------------------Projecting on detector--------------------####

#     detector = Detector('H1')

#     eccentric_signal = detector.project_wave(hp, hc, ra = parameters['ra'], dec = parameters['dec'], polarization = parameters['polarization'])

#     ####-----------------------Eccentric Signal + Noise---------------------####

#     eccentric_signal = taper_timeseries(eccentric_signal, tapermethod="TAPER_STARTEND", return_lal=False)

#     noisy_signal, signal_snr = scale_signal(eccentric_signal, num)

#     ####-------Cropping the signal such that it has duration of 8s-------####

#     noisy_signal = noisy_signal.crop(left=24, right=0)

#     ####--------------ln Bayes Factor of Eccentric vs Quasi-Circular--------------####

#     tlen = max(len(hp), len(sp))
#     hp.resize(tlen)
#     sp.resize(tlen)

#     duration = hp.duration

#     flow = 5
#     delta_f = 1 / duration
#     flen = tlen // 2 + 1
#     psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

#     match = pycbc.filter.match(hp, sp, psd = psd, low_frequency_cutoff=5, high_frequency_cutoff=None)[0]

#     ln_B = 0.5 * (1 - match**2) * signal_snr**2

#     return ln_B, noisy_signal, match, signal_snr

# def generate_additional_waveforms(num, indicator, match, signal_snr):
#     if indicator < 10:
#         ln_B_required = np.random.uniform(10, 30)
#         require_snr = np.sqrt(2 * ln_B_required / (1 - match**2))

#         parameters = samples[num].copy()
#         parameters['distance'] = parameters['distance'] * signal_snr / require_snr

#         hp, hc, sp, sc = generate_waveform(parameters)

#         ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#         assert ln_B > 10 and ln_B < 30, "ln_B should be between 10 and 30"

#         save_qtransform_plot(noisy_signal, num, ln_bw_10_and_30)

#         ln_B_required = np.random.uniform(30, 200)
#         require_snr = np.sqrt(2 * ln_B_required / (1 - match**2))

#         parameters['distance'] = parameters['distance'] * signal_snr / require_snr

#         hp, hc, sp, sc = generate_waveform(parameters)

#         ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#         assert ln_B > 30, "ln_B should be between 10 and 30"

#         save_qtransform_plot(noisy_signal, num, ln_above_30)

#     elif indicator >= 10 and indicator < 30:
#         ln_B_required = np.random.uniform(1, 10)
#         require_snr = np.sqrt(2 * ln_B_required / (1 - match**2))

#         parameters = samples[num].copy()
#         parameters['distance'] = parameters['distance'] * signal_snr / require_snr

#         hp, hc, sp, sc = generate_waveform(parameters)

#         ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#         assert ln_B <= 10, "ln_B should be less than or equal to 10"

#         save_qtransform_plot(noisy_signal, num, ln_below_10)

#         ln_B_required = np.random.uniform(30, 200)
#         require_snr = np.sqrt(2 * ln_B_required / (1 - match**2))

#         parameters['distance'] = parameters['distance'] * signal_snr / require_snr

#         hp, hc, sp, sc = generate_waveform(parameters)

#         ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#         assert ln_B > 30, "ln_B should be between 10 and 30"

#         save_qtransform_plot(noisy_signal, num, ln_above_30)
#     else:
#         ln_B_required = np.random.uniform(1, 10)
#         require_snr = np.sqrt(2 * ln_B_required / (1 - match**2))

#         parameters = samples[num].copy()
#         parameters['distance'] = parameters['distance'] * signal_snr / require_snr

#         hp, hc, sp, sc = generate_waveform(parameters)

#         ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#         assert ln_B <= 10, "ln_B should be less than or equal to 10"

#         save_qtransform_plot(noisy_signal, num, ln_below_10)

#         ln_B_required = np.random.uniform(10, 30)
#         require_snr = np.sqrt(2 * ln_B_required / (1 - match**2))

#         parameters['distance'] = parameters['distance'] * signal_snr / require_snr

#         hp, hc, sp, sc = generate_waveform(parameters)

#         ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#         assert ln_B > 10 and ln_B < 30, "ln_B should be between 10 and 30"

#         save_qtransform_plot(noisy_signal, num, ln_bw_10_and_30)



# def generate_training_qtransform(num):
#     parameters = samples[num].copy()

#     hp, hc, sp, sc = generate_waveform(parameters)

#     ln_B, noisy_signal, match, signal_snr = project_signal_and_compute_ln_Bayes(hp, hc, sp, sc, parameters, num)

#     try:
#         if ln_B <= 10:
#             save_qtransform_plot(noisy_signal, num, ln_below_10)

#             generate_additional_waveforms(num, ln_B, match, signal_snr)
            
#         elif ln_B > 10 and ln_B < 30:
#             save_qtransform_plot(noisy_signal, num, ln_bw_10_and_30)

#             generate_additional_waveforms(num, ln_B, match, signal_snr)

#         else:
#             save_qtransform_plot(noisy_signal, num, ln_above_30)

#             generate_additional_waveforms(num, ln_B, match, signal_snr)

#     except Exception as e:
#         print(f"Error generating Q-transform for sample {num}: {e}")
#         return None


# num_range = list(range(int(num_samples)))

# with Pool(processes=num_processes) as pool:
#         results = pool.map(generate_training_qtransform, num_range)
