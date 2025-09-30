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
from threadpoolctl import threadpool_limits

from gwtorch.modules.constants import *

from gwtorch.modules.general_utils import save_csv

from gwtorch.modules.gw_utils import (
    generate_prior,
    generate_waveform,
    save_qtransform_with_checks
)

def generate_sample(args):
    num, params, output_path, seed = args
    peak_window = (2.0, 2.2)
    try:
        rng = np.random.default_rng(seed)

        waveform_dict = {}
        for detector in detector_list:
            
            SEED = {
                'ecc_seed': rng.integers(2**32),
                'lens_seed': rng.integers(2**32),
                'unlens_seed': rng.integers(2**32)
            }

            ecc_dict, lens_dict, unlens_dict, mass1, mass2, m_lens, y_lens = generate_waveform(num, params, peak_window, detector=detector, compute_lnB_based_on_match=True, SEED=SEED)
            waveform_dict[detector] = {
                'eccentric': ecc_dict,
                'lensed': lens_dict,
                'unlensed': unlens_dict,
                'mass1': mass1,
                'mass2': mass2,
                'm_lens': m_lens,
                'y_lens': y_lens,
            }

        # --- Quasi-Circular Unlensed lnB ---
        snr_u_network = np.sqrt(
            sum((waveform_dict[det]['unlensed']['snr_u'])**2 for det in detector_list)
        )

        # --- Eccentric lnB ---
        rhosq_e = []
        match_e = []
        for det in detector_list:
            snr = waveform_dict[det]['eccentric']['snr_e']
            match_ecc  = waveform_dict[det]['eccentric']['ecc_m']
            rhosq_e.append(snr**2)
            match_e.append(match_ecc)
        rho_e_net_sq = np.sum(rhosq_e)
        FF_e_net = np.dot(rhosq_e, match_e) / rho_e_net_sq
        ecc_lnB = 0.5 * rho_e_net_sq * (1.0 - FF_e_net**2)

        # --- Lensed lnB ---
        rhosq_l = []
        match_l = []
        for det in detector_list:
            snr = waveform_dict[det]['lensed']['snr_l']
            match_lens  = waveform_dict[det]['lensed']['lens_m']
            rhosq_l.append(snr**2)
            match_l.append(match_lens)
        rho_l_net_sq = np.sum(rhosq_l)
        FF_l_net = np.dot(rhosq_l, match_l) / rho_l_net_sq
        lens_lnB = 0.5 * rho_l_net_sq * (1.0 - FF_l_net**2)

        count = {}

        count['eccentric_saved'] = save_qtransform_with_checks(waveform_dict, output_path / "eccentric" / f"{num}.npy")
        count["lensed_saved"] = save_qtransform_with_checks(waveform_dict, output_path / "lensed" / f"{num}.npy")
        count["unlensed_saved"] = save_qtransform_with_checks(waveform_dict, output_path / "unlensed" / f"{num}.npy")

        result = {
            'sample': num,
            'mass1': float(mass1), 'mass2': float(mass2),
            'spin1z': float(params['spin1z']), 'spin2z': float(params['spin2z']),
            'eccentricity': float(params['eccentricity']), 'anomaly': float(params['anomaly']),
            'coa_phase': float(params['coa_phase']),
            'ra': float(params['ra']), 'dec': float(params['dec']),
            'polarization': float(params['polarization']),
            'm_lens': float(m_lens), 'y_lens': float(y_lens),
            'eccentric_snr': np.sqrt(rho_e_net_sq), 'lensed_snr': np.sqrt(rho_l_net_sq), 'unlensed_snr': snr_u_network,
            'ecc_lnB': ecc_lnB, 'lens_lnB': lens_lnB, 
            **count
        }

        del ecc_dict, lens_dict, unlens_dict, waveform_dict, mass1, mass2, m_lens, y_lens, snr_u_network, ecc_lnB, lens_lnB, count

        return result
    
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return None

# ---------------------- Main Function ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate Q-transform training dataset.")
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument('--snr-prior', type=str, choices=['uniform', 'powerlaw'], default='powerlaw',
                        help='Defines the prior distribution for sampling in SNR. Default is PowerLaw.')
    parser.add_argument('--path-name', type=str, default='data',
                        help='Path to save the generated data. Default is "data".')
    parser.add_argument('--batch-size', type=int, default=150,
                        help='Batch size saves RAM usage by generating waveforms in batches. Default is 150.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for the master random number generator. If not provided, the seed will be None.')
    
    args = parser.parse_args()

    num_samples = args.num_samples
    snr_prior = args.snr_prior
    batch_size = args.batch_size

    n_eccentric, n_lensed, n_unlensed = 0, 0, 0

    directory = f"./{args.path_name}"

    init_jump = 0   

    if os.path.exists(directory):
        def get_existing_count(path):
            if not os.path.exists(path):
                return 0
            files = glob.glob(os.path.join(path, "*.npy"))
            ids = [os.path.basename(f).split('-')[0].split('.')[0] for f in files]
            return len(set(ids))

        eccentric_count = get_existing_count(os.path.join(directory, "eccentric"))
        lensed_count    = get_existing_count(os.path.join(directory, "lensed"))
        unlensed_count  = get_existing_count(os.path.join(directory, "unlensed"))

        init_jump = max(eccentric_count, lensed_count, unlensed_count)

    output_path = Path(directory)
    eccentric_path = output_path / Path("eccentric")
    lensed_path = output_path / Path("lensed")
    unlensed_path = output_path / Path("unlensed")
    results_dir = Path('./results')
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        eccentric_path.mkdir(parents=True, exist_ok=True)
        lensed_path.mkdir(parents=True, exist_ok=True)
        unlensed_path.mkdir(parents=True, exist_ok=True)
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)

    samples = generate_prior(num_samples=num_samples, snr_type=snr_prior)

    master_seed = args.seed
    master_rng = np.random.default_rng(master_seed)

    num_samples_to_generate = len(samples)
    task_seeds = master_rng.integers(low=0, high=2**32, size=num_samples_to_generate)

    print(f"Generating {num_samples_to_generate} samples in each class...")

    with threadpool_limits(limits=1, user_api="blas"):
        with Pool(args.num_workers, maxtasksperchild=1000) as pool:
            count = 0
            for i in range(0, num_samples_to_generate, batch_size):
                print(f"\nStarting batch {int(i / batch_size)}")

                batch_sample = samples[i:i+batch_size]
                batch_seeds = task_seeds[i:i+batch_size]

                jump = init_jump + (count) * batch_size
                print(f"Jumping {jump} files")

                args_list = [
                    (j + jump, batch_sample[j], output_path, batch_seeds[j])
                    for j in range(len(batch_sample))
                ]

                results = list(tqdm(pool.imap_unordered(generate_sample, args_list), total=len(batch_sample)))

                valid_results = [res for res in results if res is not None]
                
                n_eccentric += sum(1 for r in valid_results if r.get('eccentric_saved'))
                n_lensed += sum(1 for r in valid_results if r.get('lensed_saved'))
                n_unlensed += sum(1 for r in valid_results if r.get('unlensed_saved'))

                save_csv(results_dir / f"{args.path_name}_parameters.csv", valid_results)

                del batch_sample, args_list, results, valid_results
                
                gc.collect()

                count += 1

    print(f"\nData generation complete!")
    print(f"Saved eccentric: {n_eccentric}, lensed: {n_lensed}, unlensed: {n_unlensed}")

if __name__ == '__main__':
    main()