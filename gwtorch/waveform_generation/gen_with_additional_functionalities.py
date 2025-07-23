import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob

import bilby
import pycbc
from bilby.gw.prior import BBHPriorDict
from pycbc.waveform import get_td_waveform, taper_timeseries

from gwtorch.modules.gw_utils import inject_noise_with_target_SNR, compute_lensed_waveform, save_qtransform_with_checks
from gwtorch.modules.constants import *

import logging
bilby_logger = logging.getLogger("bilby")
bilby_logger.setLevel(logging.WARNING)

# ---------------------- Utility Functions ---------------------- #

def generate_prior(num_samples):
    """Define and sample priors for GW parameters."""
    priors = bilby.core.prior.PriorDict()
    priors = BBHPriorDict()

    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_12', 'phi_jl', 'luminosity_distance', 'psi', 'phase']:
        priors.pop(key, None)

    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
    priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=0.0, maximum=0.9)
    priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=0.0, maximum=0.9)
    priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.1, maximum=0.4)
    priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
    priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")
    priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum=1, maximum=5)
    priors['yl'] = bilby.core.prior.PowerLaw(alpha=1, minimum=0.01, maximum=1.0)
    # priors['snr'] = bilby.core.prior.Uniform(minimum=10, maximum=100)
    priors['snr'] = bilby.core.prior.PowerLaw(minimum=15, maximum=500, alpha=-4)

    parameters = priors.sample(num_samples)
    return [
        {key: parameters[key][i] for key in parameters}
        for i in range(num_samples)
    ]

# ---------------------- Main Worker ---------------------- #

def generate_sample(args):
    num, samples, jump, output_path = args
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

        ecc = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        unlensed = taper_timeseries(detector.project_wave(sp, sc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        lensed = taper_timeseries(detector.project_wave(sp_lensed, sc_lensed, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

        target_snr = params.pop('snr')

        ecc_noisy, snr_e, distance_e = inject_noise_with_target_SNR(ecc, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='eccentric')
        lens_noisy, snr_l, distance_l = inject_noise_with_target_SNR(lensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='lensed')
        unlens_noisy, snr_u, distance_u = inject_noise_with_target_SNR(unlensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='quasi_circular')

        ecc_noisy = ecc_noisy.crop(left=24, right=0)
        lens_noisy = lens_noisy.crop(left=24, right=0)
        unlens_noisy = unlens_noisy.crop(left=24, right=0)

        count = {
            'eccentric_saved': save_qtransform_with_checks(ecc_noisy, snr_e, params, mass1, mass2, distance_e, distance_l, distance_u, m_lens, y_lens, output_path / f"eccentric_{num + jump['ecc_files']}.png"),
            'lensed_saved': save_qtransform_with_checks(lens_noisy, snr_l, params, mass1, mass2, distance_e, distance_l, distance_u, m_lens, y_lens, output_path / f"lensed_{num + jump['lens_files']}.png"),
            'unlensed_saved': save_qtransform_with_checks(unlens_noisy, snr_u, params, mass1, mass2, distance_e, distance_l, distance_u, m_lens, y_lens, output_path / f"unlensed_{num + jump['unlens_files']}.png")
        }

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
            **count
        }
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# def save_csv(path, rows):
#     """Write dictionary rows to CSV."""
#     if not rows:
#         return
#     with open(path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)

def save_csv(path, rows):
    """Append rows to CSV if it exists, otherwise create and write header."""
    if not rows:
        return

    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ---------------------- Main Script ---------------------- #s

def main():
    parser = argparse.ArgumentParser(description="Generate Q-transform training dataset.")
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--path-name', type=str, choices=['train', 'test', 'tmp'], default=None,
                        help="Optional. If not provided, data will be stored directly in './results'")
    
    args = parser.parse_args()

    num_samples = args.num_samples

    jump = {"ecc_files": 0, "lens_files": 0, "unlens_files": 0}
    
    if args.path_name:
        directory = f"./data/{args.path_name}"
        ecc_pattern = os.path.expanduser(directory + '/eccentric_*')
        ecc_files = len([f for f in glob.glob(ecc_pattern) if os.path.isfile(f)])
        
        lens_pattern = os.path.expanduser(directory + '/lensed_*')
        lens_files = len([f for f in glob.glob(lens_pattern) if os.path.isfile(f)])
        
        unlens_pattern = os.path.expanduser(directory + '/unlensed_*')
        unlens_files = len([f for f in glob.glob(unlens_pattern) if os.path.isfile(f)])

        jump = {"ecc_files": ecc_files, "lens_files": lens_files, "unlens_files": unlens_files}
        
    # Setup output path
    if args.path_name is not None:
        output_path = Path(f"./data/{args.path_name}")
        output_path.mkdir(parents=True, exist_ok=True)
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True)
        save_prefix = f"{args.path_name}_"
    else:
        output_path = Path(f"./data/")
        output_path.mkdir(parents=True, exist_ok=True)
        results_dir = Path('./data/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        save_prefix = ""

    samples = generate_prior(num_samples)

    print(f"Generating {len(samples)} samples...")

    args_list = [(i, samples, jump, output_path) for i in range(num_samples)]

    with Pool(os.cpu_count()) as pool:
        results = pool.map(generate_sample, args_list)

    valid_results = [res for res in results if res is not None]

    save_csv(results_dir / f"{save_prefix}data_parameters.csv", valid_results)
    save_csv(results_dir / f"{save_prefix}data_snr_lookup_table.csv",
             [{'sample': r['sample'], 'eccentric_snr': r['eccentric_snr'], 'lensed_snr': r['lensed_snr'],
               'unlensed_snr': r['unlensed_snr'], 'eccentric_distance': r['eccentric_distance'],
               'lensed_distance': r['lensed_distance'], 'unlensed_distance': r['unlensed_distance']} for r in valid_results])

    n_eccentric = sum(1 for r in valid_results if r.get('eccentric_saved'))
    n_lensed = sum(1 for r in valid_results if r.get('lensed_saved'))
    n_unlensed = sum(1 for r in valid_results if r.get('unlensed_saved'))

    print(f"\nData generation complete!")
    print(f"Total valid samples: {len(valid_results)} / {num_samples}")
    print(f"Saved eccentric: {n_eccentric}, lensed: {n_lensed}, unlensed: {n_unlensed}")


if __name__ == '__main__':
    main()