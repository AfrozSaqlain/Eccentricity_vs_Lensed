import numpy as np
import bilby.gw.conversion
import pycbc
import pycbc.psd
from scipy.optimize import differential_evolution
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries, FrequencySeries

DELTA_T = 1.0 / 4096
F_LOWER = 5.0

def objective_function(params_array, args):
    chirp_mass, mass_ratio, spin1z, spin2z, coa_phase = params_array
    eccentric_signal, F_LOW, DELTA_T, distance  = args
    try:
        mass1, mass2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass=chirp_mass, mass_ratio=mass_ratio)

        template_hp, template_hc = get_td_waveform(approximant='teobresums', mass1=mass1, mass2=mass2,
                                                    lambda1=0, lambda2=0,
                                                    spin1z=spin1z, spin2z=spin2z,
                                                    distance=distance, delta_t=DELTA_T,
                                                    ecc=0.0, coa_phase=coa_phase, f_lower=F_LOW)
        
        ecc_copy = eccentric_signal.copy()
        temp_copy = template_hp.copy()

        target_len = max(len(ecc_copy), len(temp_copy))
        ecc_copy.resize(target_len)
        temp_copy.resize(target_len)

        duration = temp_copy.duration
        flen     = target_len // 2 + 1
        psd = pycbc.psd.EinsteinTelescopeP1600143(flen, 1/duration, F_LOWER)
        
        match_value, _ = pycbc.filter.match(ecc_copy, temp_copy, psd=psd, low_frequency_cutoff=F_LOW)

        return 1.0 - match_value
    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1.0
    
ecc_signal, _ = hp, hc = get_td_waveform(approximant='teobresums', mass1=50, mass2=50,
                                lambda1=0, lambda2=0,
                                spin1z=0.1, spin2z=-0.3,
                                distance=500, delta_t=DELTA_T,
                                ecc=0.3, coa_phase=3.14, f_lower=F_LOWER)

bounds = [
    (10, 100), # Chirp Mass
    (0.1, 1), # Mass Ratio
    (-0.9, 0.9), # spin1z
    (-0.9, 0.9), # spin2z
    (0, 2*np.pi), # coa_phase
]

args_for_optimizer = (ecc_signal, F_LOWER, DELTA_T, 500)

result = differential_evolution(
    func=objective_function,
    bounds=bounds,
    args=(args_for_optimizer,),
    strategy='best1bin',
    maxiter=200,
    popsize=20,
    disp=True,
    workers=10,
)

best_fit_params = result.x
fitting_factor = 1.0 - result.fun

print(f'Fitting Factor: {fitting_factor}')