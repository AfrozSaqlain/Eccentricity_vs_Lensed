import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
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

from modules.gw_utils import scale_signal

num_processess = os.cpu_count()
num_samples = int(sys.argv[1])
path_name = str(sys.argv[2])

# sys.stdout = open("log.out", "w")
# sys.stderr = open("error.err", "w")

if path_name == 'test':
    num_samples //= 10

os.makedirs(f'../data/{path_name}', exist_ok=True)
os.makedirs('../results', exist_ok=True)

training_data_path = Path(f"../data/{path_name}")
results_dir = Path('../results')

f_lower = 5.0       

priors = bilby.core.prior.PriorDict()

priors["mass1"] = bilby.core.prior.Constraint(name="mass1", minimum=10, maximum=100)
priors["mass2"] = bilby.core.prior.Constraint(name="mass2", minimum=10, maximum=100)
priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=0.0, maximum=0.9)
priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=0.0, maximum=0.9)
priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.1, maximum=0.6)
priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
priors['distance'] = bilby.core.prior.Uniform(name='distance', minimum=100, maximum=1000)
priors['dec'] = bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)
priors['ra'] = bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")
priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")

priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum = 3, maximum = 5)
priors['yl'] = bilby.core.prior.PowerLaw(alpha = 1, minimum = 0.01, maximum = 1.0)

parameters_list = priors.sample(num_samples)

samples = [
    {key: parameters_list[key][i] for key in parameters_list}
    for i in range(num_samples)
]

print(f"Length of parameters_list: {len(samples)}")


def generate_training_qtransform(num):
    parameters = samples[num].copy()

    mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(parameters['chirp_mass'], parameters['mass_ratio'])

    m_lens = np.power(10., parameters.pop("Log_Mlz"))
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
        delta_t=1.0 / 4096 ,
        ecc=parameters['eccentricity'],
        coa_phase=parameters['coa_phase'],
        f_lower=5,
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
        delta_t=1.0 / 4096 ,
        ecc=0,
        coa_phase=parameters['coa_phase'],
        f_lower=5,
    )

    ####---------------------Generating Lensed Waveform--------------------####

    sp_freq = sp.to_frequencyseries(delta_f = sp.delta_f)
    sc_freq = sc.to_frequencyseries(delta_f = sc.delta_f)

    fs1 = sp_freq.sample_frequencies
    assert np.allclose(fs1, sc_freq.sample_frequencies), "Sample frequencies do not match!"

    Ffs_sp = np.vectorize(lambda f: gwmat.cythonized_point_lens.Ff_effective(f, ml=m_lens, y=y_lens))(fs1)
        
    time_Delay = point_lens.time_delay(ml=m_lens, y=y_lens)

    sp_lensed = pycbc.types.FrequencySeries(np.conj(Ffs_sp) * np.asarray(sp_freq), delta_f=sp_freq.delta_f).cyclic_time_shift(-1 * (0.1 + time_Delay))
    sc_lensed = pycbc.types.FrequencySeries(np.conj(Ffs_sp) * np.asarray(sc_freq), delta_f=sc_freq.delta_f).cyclic_time_shift(-1 * (0.1 + time_Delay))

    sp_lensed = sp_lensed.to_timeseries(delta_t=sp_lensed.delta_t)
    sc_lensed = sc_lensed.to_timeseries(delta_t=sc_lensed.delta_t)


    ####---------------------Projecting on detector--------------------####

    detector = Detector('H1')

    eccentric_signal = detector.project_wave(hp, hc, ra = parameters['ra'], dec = parameters['dec'], polarization = parameters['polarization'])

    lensed_signal = detector.project_wave(sp_lensed, sc_lensed, ra = parameters['ra'], dec = parameters['dec'], polarization = parameters['polarization'])

    unlensed_signal = detector.project_wave(sp, sc, ra = parameters['ra'], dec = parameters['dec'], polarization = parameters['polarization'])

    # plt.plot(eccentric_signal.sample_times, eccentric_signal, label='Eccentric Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Strain')
    # plt.title(f'Eccentric Signal')
    # plt.legend()
    # plt.savefig(training_data_path / f'Waveform_Eccentric_{num}.png')
    # plt.close()

    # plt.plot(lensed_signal.sample_times, lensed_signal, label='Lensed Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Strain')
    # plt.title(f'Lensed Signal')
    # plt.legend()
    # plt.savefig(training_data_path / f'Waveform_Lensed_{num}.png')
    # plt.close()

    # plt.plot(unlensed_signal.sample_times, unlensed_signal, label='Unlensed Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Strain')
    # plt.title(f'Unlensed Signal')
    # plt.legend()
    # plt.savefig(training_data_path / f'Waveform_Unlensed_{num}.png')
    # plt.close()

    ####-----------------------Eccentric Signal + Noise---------------------####

    eccentric_signal = taper_timeseries(eccentric_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    eccentric_noisy, eccentric_snr = scale_signal(eccentric_signal, num)

    ####-----------------------Lensed Signal + Noise---------------------####

    lensed_signal = taper_timeseries(lensed_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    lensed_noisy, lensed_snr = scale_signal(lensed_signal, num)

    ####-----------------------Unlensed Signal + Noise---------------------####

    unlensed_signal = taper_timeseries(unlensed_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    unlensed_noisy, unlensed_snr = scale_signal(unlensed_signal, num)

    ####-------Cropping the signal such that it has duration of 8s-------####

    eccentric_noisy = eccentric_noisy.crop(left=24, right=0)
    lensed_noisy = lensed_noisy.crop(left=24, right=0)
    unlensed_noisy = unlensed_noisy.crop(left=24, right=0)

    ####------------------------------------------------------------------####

    noisy_gwpy_eccentric = TimeSeries.from_pycbc(eccentric_noisy)
    noisy_gwpy_lensed = TimeSeries.from_pycbc(lensed_noisy)
    noisy_gwpy_unlensed = TimeSeries.from_pycbc(unlensed_noisy)

    ####------------------------------------------------------------------####
    try:
        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(noisy_gwpy_eccentric.q_transform(logf=True, norm='mean', frange=(5,512), whiten=True, qrange=(4, 64)))
        plt.axis('off')
        plt.yscale('log')
        plt.savefig(training_data_path / f'eccentric_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(noisy_gwpy_lensed.q_transform(logf=True, norm='mean', frange=(5,512), whiten=True, qrange=(4, 64)))
        plt.axis('off')
        plt.yscale('log')
        plt.savefig(training_data_path / f'lensed_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12,8), facecolor=None)
        plt.pcolormesh(noisy_gwpy_unlensed.q_transform(logf=True, norm='mean', frange=(5,512), whiten=True, qrange=(4, 64)))
        plt.axis('off')
        plt.yscale('log')
        plt.savefig(training_data_path / f'unlensed_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating Q-transform for sample {num}: {e}")
        return None
    
    # Create parameter dictionary for this sample
    # Get original parameters before they were modified
    original_params = samples[num].copy()
    
    # Add computed values
    param_dict = {
        'sample': num,
        'mass1': float(mass1),
        'mass2': float(mass2),
        'mass_ratio': float(original_params['mass_ratio']),
        'chirp_mass': float(original_params['chirp_mass']),
        'spin1z': float(original_params['spin1z']),
        'spin2z': float(original_params['spin2z']),
        'eccentricity': float(original_params['eccentricity']),
        'coa_phase': float(original_params['coa_phase']),
        'distance': float(original_params['distance']),
        'dec': float(original_params['dec']),
        'ra': float(original_params['ra']),
        'polarization': float(original_params['polarization']),
        'Log_Mlz': float(original_params['Log_Mlz']),
        'yl': float(original_params['yl']),
        'm_lens': float(m_lens),
        'y_lens': float(y_lens),
        'time_delay': float(time_Delay),
        'eccentric_snr': float(eccentric_snr),
        'lensed_snr': float(lensed_snr),
        'unlensed_snr': float(unlensed_snr)
    }
    
    return param_dict

num_range = list(range(int(num_samples)))

with Pool(processes=num_processess) as pool:
        results = pool.map(generate_training_qtransform, num_range)

# Filter out None results
valid_results = [entry for entry in results if entry is not None]

# Create SNR lookup table (backward compatibility)
snr_table = [
    {
        'sample': entry['sample'],
        'eccentric_snr': entry['eccentric_snr'],
        'lensed_snr': entry['lensed_snr'],
        'unlensed_snr': entry['unlensed_snr']
    }
    for entry in valid_results
]

# Save SNR lookup table
snr_csv_path = results_dir / f"{path_name}_data_snr_lookup_table.csv"
with open(snr_csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['sample', 'eccentric_snr', 'lensed_snr', 'unlensed_snr'])
    writer.writeheader()
    writer.writerows(snr_table)

print(f"SNR lookup table saved to {snr_csv_path}")

# Save parameters table
params_csv_path = results_dir / f"{path_name}_data_parameters.csv"
with open(params_csv_path, mode='w', newline='') as f:
    if valid_results:
        fieldnames = list(valid_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_results)

print(f"Parameters table saved to {params_csv_path}")
