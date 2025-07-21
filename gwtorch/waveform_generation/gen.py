import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool

import bilby
import pycbc
from gwpy.timeseries import TimeSeries
from pycbc.detector.ground import Detector
from pycbc.waveform import get_td_waveform, taper_timeseries

import gwmat
from gwmat import point_lens

from gwtorch.modules.gw_utils import injnect_noise_with_SNR, compute_lensed_waveform

# ---------------------- Constants ---------------------- #
DELTA_T = 1.0 / 4096
F_LOWER = 5.0
DETECTOR_NAME = 'H1'
detector = Detector(DETECTOR_NAME)
REFERENCE_DISTANCE = 300

# ---------------------- Utility Functions ---------------------- #

def generate_prior(num_samples):
    """Define and sample priors for GW parameters."""
    priors = bilby.core.prior.PriorDict()
    priors["mass1"] = bilby.core.prior.Constraint(name="mass1", minimum=10, maximum=100)
    priors["mass2"] = bilby.core.prior.Constraint(name="mass2", minimum=10, maximum=100)
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
    priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=0.0, maximum=0.9)
    priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=0.0, maximum=0.9)
    priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.1, maximum=0.6)
    priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
    priors['dec'] = bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)
    priors['ra'] = bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")
    priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")
    priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum=3, maximum=5)
    priors['yl'] = bilby.core.prior.PowerLaw(alpha=1, minimum=0.01, maximum=1.0)
    priors['snr'] = bilby.core.prior.Uniform(minimum=12, maximum=80)

    parameters = priors.sample(num_samples)
    return [
        {key: parameters[key][i] for key in parameters}
        for i in range(num_samples)
    ]

def save_qtransform(ts, path):
    """Save Q-transform of a time series."""
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(ts.q_transform(logf=True, norm='mean', frange=(5, 512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.yscale('log')
    plt.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

# ---------------------- Main Worker ---------------------- #

def generate_sample(args):
    num, samples, output_path = args
    try:
        params = samples[num].copy()
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

        detector = Detector(DETECTOR_NAME)
        ecc = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        unlensed = taper_timeseries(detector.project_wave(sp, sc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        lensed = taper_timeseries(detector.project_wave(sp_lensed, sc_lensed, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

        target_snr = params.pop('snr')
        
        ecc_noisy, snr_e, distance_e = injnect_noise_with_SNR(ecc, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='eccentric')
        lens_noisy, snr_l, distance_l = injnect_noise_with_SNR(lensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='lensed')
        unls_noisy, snr_u, distance_u = injnect_noise_with_SNR(unlensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='quasi_circular')

        ecc_noisy = ecc_noisy.crop(left=24, right=0)
        lens_noisy = lens_noisy.crop(left=24, right=0)
        unls_noisy = unls_noisy.crop(left=24, right=0)

        save_qtransform(TimeSeries.from_pycbc(ecc_noisy), output_path / f"eccentric_{num}.png")
        save_qtransform(TimeSeries.from_pycbc(lens_noisy), output_path / f"lensed_{num}.png")
        save_qtransform(TimeSeries.from_pycbc(unls_noisy), output_path / f"unlensed_{num}.png")

        return {
            'sample': num,
            'mass1': float(mass1), 'mass2': float(mass2),
            'chirp_mass': float(params['chirp_mass']),
            'mass_ratio': float(params['mass_ratio']),
            'spin1z': float(params['spin1z']), 'spin2z': float(params['spin2z']),
            'eccentricity': float(params['eccentricity']),
            'coa_phase': float(params['coa_phase']),
            'eccentric_distance': float(distance_e),
            'lensed_distance': float(distance_l),
            'unlensed_distance': float(distance_u),
            'ra': float(params['ra']), 'dec': float(params['dec']),
            'polarization': float(params['polarization']),
            'm_lens': float(m_lens), 'y_lens': float(y_lens),
            'Log_Mlz': float(params['Log_Mlz']),
            'yl': float(y_lens),
            'time_delay': float(t_delay),
            'eccentric_snr': float(snr_e),
            'lensed_snr': float(snr_l),
            'unlensed_snr': float(snr_u),
        }
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def save_csv(path, rows):
    """Write dictionary rows to CSV."""
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# ---------------------- Main Script ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate Q-transform training dataset.")
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--path-name', type=str, choices=['train', 'test', 'tmp'], required=True)
    args = parser.parse_args()

    num_samples = args.num_samples

    output_path = Path(f"./data/{args.path_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)

    samples = generate_prior(num_samples)

    print(f"Generating {len(samples)} samples...")

    args_list = [(i, samples, output_path) for i in range(num_samples)]

    with Pool(os.cpu_count()) as pool:
        results = pool.map(generate_sample, args_list)

    valid_results = [res for res in results if res is not None]

    save_csv(results_dir / f"{args.path_name}_data_parameters.csv", valid_results)
    save_csv(results_dir / f"{args.path_name}_data_snr_lookup_table.csv",
             [{'sample': r['sample'], 'eccentric_snr': r['eccentric_snr'], 'lensed_snr': r['lensed_snr'],
               'unlensed_snr': r['unlensed_snr'], 'eccentric_distance': r['eccentric_distance'],
               'lensed_distance': r['lensed_distance'], 'unlensed_distance': r['unlensed_distance']} for r in valid_results])
    
    print(f"Data generation complete! Generated {len(valid_results)} valid samples out of {num_samples} attempted.")

if __name__ == '__main__':
    main()