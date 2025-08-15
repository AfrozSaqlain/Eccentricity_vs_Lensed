import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import os
import glob
import bilby
import pycbc
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from gwpy.timeseries import TimeSeries
from bilby.gw.prior import BBHPriorDict
from matplotlib.colors import Normalize
from pycbc.waveform import get_td_waveform, taper_timeseries

from gwtorch.modules.constants import *
from gwtorch.modules.gw_utils import compute_lensed_waveform, inject_signal_with_peak_in_window, get_psd_type, generate_noise, pad_to_shape, _get_category_parts
from gwtorch.modules.general_utils import save_csv

def generate_prior(num_samples, bin = "10-30"):

    priors = BBHPriorDict()

    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'theta_jn', 'phi_12', 'phi_jl', 'luminosity_distance', 'psi', 'phase']:
        priors.pop(key, None)

    priors['mass_1'].minimum = 10
    priors['mass_2'].minimum = 10
    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=10, maximum=100)
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.1, maximum=1)
    priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=-0.9, maximum=0.9)
    priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=-0.9, maximum=0.9)
    priors['eccentricity'] = bilby.core.prior.Uniform(name='eccentricity', minimum=0.1, maximum=0.6)
    priors['coa_phase'] = bilby.core.prior.Uniform(name='coa_phase', minimum=0.0, maximum=2 * np.pi)
    priors['polarization'] = bilby.core.prior.Uniform(minimum=0., maximum=np.pi, boundary="periodic")
    priors['Log_Mlz'] = bilby.core.prior.Uniform(minimum=1, maximum=5)
    priors['yl'] = bilby.core.prior.PowerLaw(alpha=1, minimum=0.01, maximum=1.0)
    # Remove lnB from priors - we'll compute it from the physics
    
    parameters = priors.sample(num_samples)
    return [
        {key: parameters[key][i] for key in parameters}
        for i in range(num_samples)
    ]

def inject_noise_with_target_SNR(gw_signal, params, mass1, mass2, m_lens, y_lens, snr_desired, gw_match, num, gw_signal_type='eccentric', peak_window=(2.0, 2.2), detector=default_detector, SEED = None, bin_type=None):
   
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
                                 ecc=params['eccentricity'], coa_phase=params['coa_phase'], f_lower=F_LOWER)
        
        gw_signal = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
    
    else: 
        hp, hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                 lambda1=0, lambda2=0,
                                 spin1z=params['spin1z'], spin2z=params['spin2z'],
                                 distance=distance, delta_t=DELTA_T,
                                 ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_LOWER) 
        
        hp, hc, t_delay = compute_lensed_waveform(hp, hc, m_lens, y_lens)

        gw_signal = taper_timeseries(detector.project_wave(hp, hc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
    
    sp, _ = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                lambda1=0, lambda2=0,
                                spin1z=params['spin1z'], spin2z=params['spin2z'],
                                distance=distance, delta_t=DELTA_T,
                                ecc=0.0, coa_phase=params['coa_phase'], f_lower=F_LOWER)
    

    template, delta_t, start_time = inject_signal_with_peak_in_window(
                                            signal_ts=gw_signal,
                                            noise_ts=noise,
                                            peak_window=peak_window)
    
    data = pycbc.types.TimeSeries(np.asarray(template) + np.asarray(noise), delta_t=delta_t, epoch=start_time)

    template = pycbc.types.TimeSeries(template, delta_t=delta_t, epoch=start_time)

    snr = pycbc.filter.matched_filter(template = template,  data = data, psd = psd, low_frequency_cutoff=F_LOWER)

    peak_snr = abs(snr).numpy().max()

    # hp = pycbc.types.TimeSeries(hp.data, delta_t=hp.delta_t, epoch=hp.start_time)

    # tlen = max(len(hp), len(sp))
    # hp.resize(tlen)
    # sp.resize(tlen)

    # duration = hp.duration

    # delta_f = 1 / duration
    # flen = tlen // 2 + 1

    # if PSD_TYPE == 'ET':
    #     psd = pycbc.psd.EinsteinTelescopeP1600143(flen, delta_f, F_LOWER)
    # elif PSD_TYPE == 'CE':
    #     psd = pycbc.psd.CosmicExplorerP1600143(flen, delta_f, F_LOWER)
    # elif PSD_TYPE == 'aLIGO':
    #     psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)

    # sp = pycbc.types.TimeSeries(sp.data, delta_t=hp.delta_t, epoch=hp.start_time)
    # match = pycbc.filter.match(hp, sp, psd=psd, low_frequency_cutoff=F_LOWER)[0]
    
    ln_B = 0.5 * (1 - gw_match**2) * peak_snr**2

    if bin_type == '10-30':
        bin_min, bin_max = 10, 30
    else:
        bin_min, bin_max = 30, 100

    if ln_B <= bin_min or ln_B >= bin_max:
        print(f"[Sample {num}]: lnB out of bounds: {ln_B:.2f} for sample {num} {gw_signal_type}")
        return None, None, None, None, None

    return data, peak_snr, distance, template, ln_B

def generate_waveform(num ,params, peak_window=(2.0, 2.2), detector=default_detector, bin_type=None):

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
        
        lensed = taper_timeseries(detector.project_wave(sp_lensed, sc_lensed, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)
        
        unlensed = taper_timeseries(detector.project_wave(sp, sc, **{k: params[k] for k in ['ra', 'dec', 'polarization']}), tapermethod="TAPER_STARTEND", return_lal=False)

        PSD_TYPE, psd = get_psd_type(detector)

        noise = generate_noise(PSD_TYPE=PSD_TYPE)

        tlen = len(noise)

        del noise
        
        hp.resize(tlen)
        sp.resize(tlen)
        sp_lensed.resize(tlen)

        sp = pycbc.types.TimeSeries(sp.data,
                                        delta_t=hp.delta_t,
                                        epoch=hp.start_time)
        
        sp_lensed = pycbc.types.TimeSeries(sp_lensed.data,
                                        delta_t=hp.delta_t,
                                        epoch=hp.start_time)
        
        ecc_match = pycbc.filter.match(hp, sp, psd=psd, low_frequency_cutoff=F_LOWER)[0]
        lens_match = pycbc.filter.match(sp_lensed, sp, psd=psd, low_frequency_cutoff=F_LOWER)[0]
        
        if bin_type == '10-30':
            bin_min, bin_max = 15, 25
        else:
            bin_min, bin_max = 30, 100
        
        target_lnB = np.random.uniform(bin_min, bin_max)
        
        ecc_snr_desired = np.sqrt(2 * target_lnB / (1 - ecc_match**2))
        lens_snr_desired = np.sqrt(2 * target_lnB / (1 - lens_match**2))
        
        ecc_noisy, snr_e, distance_e, ecc_padded, ecc_ln_B = inject_noise_with_target_SNR(ecc, params, mass1, mass2, m_lens, y_lens, ecc_snr_desired, ecc_match, num, gw_signal_type='eccentric', peak_window=peak_window, detector=detector, bin_type=bin_type)

        lens_noisy, snr_l, distance_l, lens_padded, lens_ln_B = inject_noise_with_target_SNR(lensed, params, mass1, mass2, m_lens, y_lens, lens_snr_desired, lens_match, num, gw_signal_type='lensed', peak_window=peak_window, detector=detector, bin_type=bin_type)
        
        if ecc_noisy is None or lens_noisy is None:
            print(f"[Sample {num}] Error: waveform generation failed for {detector.name}")
            return None
        
        ecc_noisy = ecc_noisy.crop(left=24, right=0)
        lens_noisy = lens_noisy.crop(left=24, right=0)

        ecc_dict = {'ecc_signal': ecc_noisy, 'snr_e': snr_e, 'distance_e': distance_e}
        lens_dict = {'lensed_signal': lens_noisy, 'snr_l': snr_l, 'distance_l': distance_l}

        del hp, hc, sp, sc, sp_lensed, sc_lensed
        del ecc, unlensed, lensed, ecc_noisy, lens_noisy, ecc_padded, lens_padded
        gc.collect()

        return ecc_dict, lens_dict, mass1, mass2, m_lens, y_lens, ecc_snr_desired, lens_snr_desired, target_lnB, ecc_ln_B, lens_ln_B
        
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def save_qtransform_with_checks(waveform_dict, params, path):

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

        # --- Bayes factor / match test where applicable ------------------------
        compute_q = True  # default for unlensed case

        # --- Q‑transform & stack ------------------------------------------------

        ts = TimeSeries.from_pycbc(ts_raw)
        q  = ts.q_transform(logf=True, norm='median',
                            frange=(F_LOWER, 512),
                            whiten=True, qrange=(4, 64))
        stacked_q.append(np.abs(q.data).T)

    max_shape = (
        max(q.shape[0] for q in stacked_q),
        max(q.shape[1] for q in stacked_q)
    )
    
    padded_q = [pad_to_shape(q, max_shape) for q in stacked_q]
    normed = [Normalize()(q) for q in padded_q]

    stacked = np.stack(normed, axis=0)             

    stacked_img = np.transpose(stacked, (1, 2, 0))

    out_path = str(path)
    if out_path.endswith('.png'):
        out_path = out_path.replace('.png', '.npy')
    np.save(out_path, np.asarray(stacked_img))
    
    return True

# ---------------------- Main Worker ---------------------- #

def generate_sample(args):
    num, params, jump, output_path, bin_type = args
    peak_window = (2.0, 2.2)
    try:
        waveform_dict = {}
        for detector in detector_list:
            result = generate_waveform(num, params, peak_window, detector=detector, bin_type=bin_type)
            if result is None:
                return None
            
            ecc_dict, lens_dict, mass1, mass2, m_lens, y_lens, ecc_snr_desired, lens_snr_desired, target_lnB, ecc_ln_B, lens_ln_B = result
            waveform_dict[detector] = {
                'eccentric': ecc_dict,
                'lensed': lens_dict,
                'mass1': mass1,
                'mass2': mass2,
                'm_lens': m_lens,
                'y_lens': y_lens
            }

        count = {
            'eccentric_saved': save_qtransform_with_checks(waveform_dict, params, output_path / f"eccentric_{num + jump}.npy"),
            'lensed_saved': save_qtransform_with_checks(waveform_dict, params, output_path / f"lensed_{num + jump}.npy"),
        }

        # snr_e_network = np.sqrt(
        #     sum((waveform_dict[det]['eccentric']['snr_e'])**2 for det in detector_list)
        # )
        # snr_l_network = np.sqrt(
        #     sum((waveform_dict[det]['lensed']['snr_l'])**2 for det in detector_list)
        # )

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
            'ra': float(params['ra']), 'dec': float(params['dec']),
            'polarization': float(params['polarization']),
            'm_lens': float(m_lens), 'y_lens': float(y_lens),
            'Log_Mlz': float(params['Log_Mlz']),
            'yl': float(y_lens),
            'Target_ln_B': float(target_lnB),
            'ecc_ln_B': float(ecc_ln_B),
            # 'ecc_snr_desired': float(ecc_snr_desired),
            # 'eccentric_snr': snr_e_network,
            'lens_ln_B': float(lens_ln_B),
            # 'lens_snr_desired': float(lens_snr_desired),
            # 'lensed_snr': snr_l_network,
            **count
        }

        del ecc_dict, lens_dict, waveform_dict
        gc.collect()

        return result
    
    except Exception as e:
        print(f"[Sample {num}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    


def main():
    parser = argparse.ArgumentParser(description="Generate Q-transform training dataset.")
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument('--bin-type', type=str, default='10-30',
                        help='Defines the bin type for generating the data. Default is 10-30 ln B bin.')
    
    args = parser.parse_args()

    num_samples = args.num_samples
    bin_type = args.bin_type

    jump = 0   
    
    directory = f"./data"

    if os.path.exists(directory):
    
        lensed_pattern = os.path.join(directory, 'lensed_*')
        eccentric_pattern = os.path.join(directory, 'eccentric_*')

        len_lens = len([f for f in glob.glob(lensed_pattern) if os.path.isfile(f)])
        len_eccentric = len([f for f in glob.glob(eccentric_pattern) if os.path.isfile(f)])

        jump = max(len_lens, len_eccentric)

    output_path = Path(f"./data")
    results_dir = Path('./results')
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)

    samples = generate_prior(num_samples=num_samples, bin=bin_type)

    print(f"Generating {len(samples)} samples in each class...")

    args_list = [(i, samples[i], jump, output_path, args.bin_type) for i in range(num_samples)]

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(generate_sample, args_list), total=num_samples))
        # results = list(pool.imap_unordered(generate_sample, args_list))

    valid_results = [res for res in results if res is not None]

    save_csv(results_dir / f"parameters.csv", valid_results)
    save_csv(results_dir / f"snr_lookup_table.csv",
             [{'sample': r['sample'], #'eccentric_snr': r['eccentric_snr'], 'lensed_snr': r['lensed_snr'],
               'eccentric_distance': r['eccentric_distance'],
               'lensed_distance': r['lensed_distance']} for r in valid_results])

    n_eccentric = sum(1 for r in valid_results if r.get('eccentric_saved'))
    n_lensed = sum(1 for r in valid_results if r.get('lensed_saved'))

    print(f"\nData generation complete!")
    print(f"Total valid samples: {len(valid_results)} / {num_samples}")
    print(f"Saved eccentric: {n_eccentric}, lensed: {n_lensed}")


if __name__ == '__main__':
    main()