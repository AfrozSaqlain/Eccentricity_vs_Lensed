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
from scipy.optimize import differential_evolution

import pycbc
import bilby
from pycbc.waveform import get_td_waveform, taper_timeseries

from gwtorch.modules.constants import *

from gwtorch.modules.general_utils import save_csv

from gwtorch.modules.gw_utils import (
    save_qtransform_with_checks,
    generate_prior,
    compute_lensed_waveform,
    inject_noise_with_target_SNR
)

bounds = [
    (10, 100), # Chirp Mass
    (0.1, 1), # Mass Ratio
    (-0.9, 0.9), # spin1z
    (-0.9, 0.9), # spin2z
    (0, 2*np.pi), # coa_phase
]

def objective_function(params_array, args):
    chirp_mass, mass_ratio, spin1z, spin2z, coa_phase = params_array
    signal, distance, detector_name = args
    try:
        mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass=chirp_mass, mass_ratio=mass_ratio)

        template_hp, template_hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                                    lambda1=0, lambda2=0,
                                                    spin1z=spin1z, spin2z=spin2z,
                                                    distance=distance, delta_t=DELTA_T,
                                                    ecc=0.0, coa_phase=coa_phase, f_lower=F_START)
        
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
    


def generate_waveform(num ,params, peak_window=(2.0, 2.2), detector=default_detector, compute_FF=True, SEED=None):
    try:
        mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
            params['chirp_mass'], params['mass_ratio'])

        m_lens = 10 ** params["Log_Mlz"]
        y_lens = params["yl"]

        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=REFERENCE_DISTANCE, delta_t=DELTA_T,
                                 ecc=params['eccentricity'], coa_phase=params['coa_phase'], f_lower=F_START)

        sp, sc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=REFERENCE_DISTANCE, delta_t=DELTA_T,
                                 ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_START)

        sp_lensed, sc_lensed, t_delay = compute_lensed_waveform(sp, sc, m_lens, y_lens)

        ecc = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        unlensed = taper_timeseries(detector.project_wave(sp, sc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        lensed = taper_timeseries(detector.project_wave(sp_lensed, sc_lensed, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

        target_snr = params['snr']

        ecc_noisy, snr_e, distance_e, ecc_padded = inject_noise_with_target_SNR(ecc, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='eccentric', peak_window=peak_window, detector=detector, SEED=SEED)
        lens_noisy, snr_l, distance_l, lens_padded = inject_noise_with_target_SNR(lensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='lensed', peak_window=peak_window, detector=detector, SEED=SEED)
        unlens_noisy, snr_u, distance_u, unlens_padded = inject_noise_with_target_SNR(unlensed, params, mass1, mass2, m_lens, y_lens, target_snr, num, gw_signal_type='quasi_circular', peak_window=peak_window, detector=detector, SEED=SEED)
        
        ecc_noisy = ecc_noisy.crop(left=24, right=0)
        lens_noisy = lens_noisy.crop(left=24, right=0)
        unlens_noisy = unlens_noisy.crop(left=24, right=0)

        if compute_FF:
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
        else:
            ecc_fitting_factor, lens_fitting_factor = None, None

        lens_dict = {'lensed_signal': lens_noisy, 'snr_l': snr_l, 'distance_l': distance_l, 'lens_FF': lens_fitting_factor}
        ecc_dict = {'ecc_signal': ecc_noisy, 'snr_e': snr_e, 'distance_e': distance_e, 'ecc_FF': ecc_fitting_factor}
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

def generate_sample(args):
    num, params, jump, output_path = args
    peak_window = (2.0, 2.2)
    try:
        waveform_dict = {}
        for detector in detector_list:
            # SEED = None # np.random.randint(2**32) # 42 # https://pycbc.org/pycbc/latest/html/_modules/pycbc/noise/gaussian.html#noise_from_psd If seed is None, PyCBC creates a numpy.random.randint(2**32) and for each subprocess the random number generated is same.

            SEED = {'ecc_seed': np.random.randint(10), 'lens_seed': np.random.randint(10), 'unlens_seed': np.random.randint(10)}
            # SEED = {'ecc_seed': np.random.randint(2**32), 'lens_seed': np.random.randint(2**32), 'unlens_seed': np.random.randint(2**32)}
            # SEED = {'ecc_seed': None, 'lens_seed': None, 'unlens_seed': None}

            ecc_dict, lens_dict, unlens_dict, mass1, mass2, m_lens, y_lens = generate_waveform(num, params, peak_window, detector=detector, SEED=SEED)
            waveform_dict[detector] = {
                'eccentric': ecc_dict,
                'lensed': lens_dict,
                'unlensed': unlens_dict,
                'mass1': mass1,
                'mass2': mass2,
                'm_lens': m_lens,
                'y_lens': y_lens,
            }

        # snr_e_network = np.sqrt(
        #     sum((waveform_dict[det]['eccentric']['snr_e'])**2 
        #         for det in detector_list)
        # )

        # snr_l_network = np.sqrt(
        #     sum((waveform_dict[det]['lensed']['snr_l'])**2 
        #         for det in detector_list)
        # )

        snr_u_network = np.sqrt(
            sum((waveform_dict[det]['unlensed']['snr_u'])**2 for det in detector_list)
        )

        # --- Eccentric lnB (network FF + network SNR²) ---
        rhosq_e = []
        ff_e = []
        for det in detector_list:
            snr = waveform_dict[det]['eccentric']['snr_e']
            ff  = waveform_dict[det]['eccentric']['ecc_FF']
            rhosq_e.append(snr**2)
            ff_e.append(ff)
        rho_e_net_sq = np.sum(rhosq_e)
        FF_e_net = np.dot(rhosq_e, ff_e) / rho_e_net_sq
        ecc_lnB = 0.5 * rho_e_net_sq * (1.0 - FF_e_net**2)

        # --- Lensed lnB (network FF + network SNR²) ---
        rhosq_l = []
        ff_l = []
        for det in detector_list:
            snr = waveform_dict[det]['lensed']['snr_l']
            ff  = waveform_dict[det]['lensed']['lens_FF']
            rhosq_l.append(snr**2)
            ff_l.append(ff)
        rho_l_net_sq = np.sum(rhosq_l)
        FF_l_net = np.dot(rhosq_l, ff_l) / rho_l_net_sq
        lens_lnB = 0.5 * rho_l_net_sq * (1.0 - FF_l_net**2)

        count = {}

        count['eccentric_saved'] = save_qtransform_with_checks(waveform_dict, output_path / "eccentric" / f"{num}.npy")
        count["lensed_saved"] = save_qtransform_with_checks(waveform_dict, output_path / "lensed" / f"{num}.npy")
        count["unlensed_saved"] = save_qtransform_with_checks(waveform_dict, output_path / "unlensed" / f"{num}.npy")

        result = {
            'sample': num,
            'mass1': float(mass1), 'mass2': float(mass2),
            'spin1z': float(params['spin1z']), 'spin2z': float(params['spin2z']),
            'eccentricity': float(params['eccentricity']),
            'coa_phase': float(params['coa_phase']),
            'ra': float(params['ra']), 'dec': float(params['dec']),
            'polarization': float(params['polarization']),
            'm_lens': float(m_lens), 'y_lens': float(y_lens),
            'eccentric_snr': np.sqrt(rho_e_net_sq), 'lensed_snr': np.sqrt(rho_l_net_sq), 'unlensed_snr': snr_u_network,
            'ecc_lnB': ecc_lnB, 'lens_lnB': lens_lnB,
            **count
        }

        del ecc_dict, lens_dict, unlens_dict, waveform_dict
        del mass1, mass2, m_lens, y_lens
        del snr_u_network
        del ecc_lnB, lens_lnB, count
        gc.collect()

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
    
    args = parser.parse_args()

    num_samples = args.num_samples
    snr_prior = args.snr_prior
    batch_size = args.batch_size

    n_eccentric, n_lensed, n_unlensed = 0, 0, 0

    directory = f"./{args.path_name}"

    init_jump = 0   

    if os.path.exists(directory):
        file_pattern = os.path.join(directory, 'unlensed', '*.npy')
        file_list = glob.glob(file_pattern)
        files = [i.split('/')[3].split('.')[0] for i in file_list]
        final_files = set([i.split('-')[0] if '-' in i else i for i in files])
        init_jump = len(final_files)
        
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

    print(f"Generating {len(samples)} samples in each class...")

    def _init_worker(base):
        np.random.seed((base + os.getpid()) % (2**32 - 1))

    with threadpool_limits(limits=1, user_api="blas"):
        with Pool(args.num_workers) as pool: #, initializer=_init_worker, initargs=(np.random.SeedSequence().entropy,)) as pool:
            count = 0
            for i in range(0, len(samples), batch_size):
                print(f"\nStarting batch {int(i / batch_size)}")
                
                batch_sample = samples[i:i+batch_size]
                
                jump = init_jump + (count) * batch_size

                print(f"Jumping {jump} files")

                args_list = [(j + jump, batch_sample[j], jump, output_path) for j in range(len(batch_sample))]

                results = list(tqdm(pool.imap_unordered(generate_sample, args_list), total=len(batch_sample)))

                valid_results = [res for res in results if res is not None]
                
                n_eccentric += sum(1 for r in valid_results if r.get('eccentric_saved'))
                n_lensed += sum(1 for r in valid_results if r.get('lensed_saved'))
                n_unlensed += sum(1 for r in valid_results if r.get('unlensed_saved')) + sum(1 for r in valid_results if r.get('lens_unlensed_saved')) + sum(1 for r in valid_results if r.get('ecc_unlensed_saved'))

                save_csv(results_dir / f"{args.path_name}_parameters.csv", valid_results)

                del batch_sample, args_list, results, valid_results
                
                for _ in range(3):
                    gc.collect()
                
                try:
                    np._NoValue
                except:
                    pass
                    
                # Force pool to restart
                # pool.terminate()
                # pool.join()
                # pool = Pool(args.num_workers)

                count += 1

    print(f"\nData generation complete!")
    print(f"Saved eccentric: {n_eccentric}, lensed: {n_lensed}, unlensed: {n_unlensed}")

if __name__ == '__main__':
    main()