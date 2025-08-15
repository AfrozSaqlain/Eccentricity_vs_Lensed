import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from gwtorch.modules.constants import *
from gwtorch.modules.gw_utils import generate_waveform, save_qtransform_with_checks, generate_prior
from gwtorch.modules.general_utils import save_csv

# ---------------------- Main Worker ---------------------- #

def generate_sample(args):
    num, params, jump, output_path = args
    peak_window = (2.0, 2.2)
    try:
        waveform_dict = {}
        for detector in detector_list:
            ecc_dict, lens_dict, unlens_dict, mass1, mass2, m_lens, y_lens = generate_waveform(num, params, peak_window, detector=detector)
            waveform_dict[detector] = {
                'eccentric': ecc_dict,
                'lensed': lens_dict,
                'unlensed': unlens_dict,
                'mass1': mass1,
                'mass2': mass2,
                'm_lens': m_lens,
                'y_lens': y_lens
            }

        count = {
            'eccentric_saved': save_qtransform_with_checks(waveform_dict, params, output_path / f"eccentric_{num + jump}.npy"),
            'lensed_saved': save_qtransform_with_checks(waveform_dict, params, output_path / f"lensed_{num + jump}.npy"),
            'unlensed_saved': save_qtransform_with_checks(waveform_dict, params, output_path / f"unlensed_{num + jump}.npy")
        }

        snr_e_network = np.sqrt(
            sum((waveform_dict[det]['eccentric']['snr_e'])**2 for det in detector_list)
        )
        snr_l_network = np.sqrt(
            sum((waveform_dict[det]['lensed']['snr_l'])**2 for det in detector_list)
        )
        snr_u_network = np.sqrt(
            sum((waveform_dict[det]['unlensed']['snr_u'])**2 for det in detector_list)
        )

        result = {
            'sample': num,
            'mass1': float(mass1), 'mass2': float(mass2),
            'chirp_mass': float(params['chirp_mass']),
            'mass_ratio': float(params['mass_ratio']),
            'spin1z': float(params['spin1z']), 'spin2z': float(params['spin2z']),
            'eccentricity': float(params['eccentricity']),
            'coa_phase': float(params['coa_phase']),
            'eccentric_distance': float(ecc_dict["distance_e"]),
            'lensed_distance': float(lens_dict["distance_l"]),
            'unlensed_distance': float(unlens_dict['distance_u']),
            'ra': float(params['ra']), 'dec': float(params['dec']),
            'polarization': float(params['polarization']),
            'm_lens': float(m_lens), 'y_lens': float(y_lens),
            'Log_Mlz': float(params['Log_Mlz']),
            'yl': float(y_lens),
            # 'time_delay': float(t_delay),
            'eccentric_snr': snr_e_network,
            'lensed_snr': snr_l_network,
            'unlensed_snr': snr_u_network,
            **count
        }

        del ecc_dict, lens_dict, unlens_dict, waveform_dict
        gc.collect()

        return result
    
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------- Main Script ---------------------- #s

def main():
    parser = argparse.ArgumentParser(description="Generate Q-transform training dataset.")
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument('--snr-prior', type=str, choices=['uniform', 'powerlaw'], default='powerlaw',
                        help='Defines the prior distribution for sampling in SNR. Default is PowerLaw.')
    
    args = parser.parse_args()

    num_samples = args.num_samples
    snr_prior = args.snr_prior

    jump = 0   
    
    directory = f"./data"

    if os.path.exists(directory):
    
        file_pattern = os.path.join(directory, 'unlensed_*')
        jump = len([f for f in glob.glob(file_pattern) if os.path.isfile(f)])

    output_path = Path(f"./data")
    results_dir = Path('./results')
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)

    samples = generate_prior(num_samples=num_samples, snr_type=snr_prior)

    print(f"Generating {len(samples)} samples in each class...")

    args_list = [(i, samples[i], jump, output_path) for i in range(num_samples)]

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(generate_sample, args_list), total=num_samples))

    valid_results = [res for res in results if res is not None]

    save_csv(results_dir / f"parameters.csv", valid_results)
    save_csv(results_dir / f"snr_lookup_table.csv",
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