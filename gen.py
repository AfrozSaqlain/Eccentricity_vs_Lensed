import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

num_processess = os.cpu_count()
num_samples = int(sys.argv[1])
path_name = str(sys.argv[2])

# sys.stdout = open("log.out", "w")
# sys.stderr = open("error.err", "w")

if path_name == 'test':
    num_samples //= 10

os.makedirs(f'data_2/{path_name}', exist_ok=True)
training_data_path = Path(f"data_2/{path_name}")

f_lower = 10.0       

priors = bilby.core.prior.PriorDict()

priors["mass1"] = bilby.core.prior.Constraint(name="mass1", minimum=5, maximum=100)
priors["mass2"] = bilby.core.prior.Constraint(name="mass2", minimum=5, maximum=100)
priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
priors['spin1z'] = bilby.core.prior.Uniform(name='spin1z', minimum=0.09, maximum=0.9)
priors['spin2z'] = bilby.core.prior.Uniform(name='spin2z', minimum=0.09, maximum=0.9)
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

def generate_noise(signal):
    flow = 10
    delta_f = 1 / 8
    flen = int(4096 / (2 * delta_f)) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    delta_t = 1.0 / 4096
    tsamples = int(8 / delta_t)
    # tsamples = len(signal)
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=None)

    return noise


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
        f_lower=10,
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
        f_lower=10,
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

    ####-----------------------Eccentric Signal + Noise---------------------####

    eccentric_signal = taper_timeseries(eccentric_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    noise_eccentric = generate_noise(eccentric_signal)

    # noise_eccentric.resize(len(eccentric_signal))
    eccentric_signal.resize(len(noise_eccentric))
     
    eccentric_noisy = pycbc.types.TimeSeries(np.array(eccentric_signal) + np.array(noise_eccentric), delta_t = eccentric_signal.delta_t, epoch = eccentric_signal.start_time)

    ####-----------------------Lensed Signal + Noise---------------------####

    lensed_signal = taper_timeseries(lensed_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    noise_lensed = generate_noise(lensed_signal)

    # noise_lensed.resize(len(lensed_signal))
    lensed_signal.resize(len(noise_lensed))

    lensed_noisy = pycbc.types.TimeSeries(np.array(lensed_signal) + np.array(noise_lensed), delta_t = lensed_signal.delta_t, epoch = lensed_signal.start_time)

    ####-----------------------Unlensed Signal + Noise---------------------####

    unlensed_signal = taper_timeseries(unlensed_signal, tapermethod="TAPER_STARTEND", return_lal=False)

    noise_unlensed = generate_noise(unlensed_signal)

    # noise_unlensed.resize(len(unlensed_signal))
    unlensed_signal.resize(len(noise_unlensed))

    unlensed_noisy = pycbc.types.TimeSeries(np.array(unlensed_signal) + np.array(noise_unlensed), delta_t = unlensed_signal.delta_t, epoch = unlensed_signal.start_time)

    ####------------------------------------------------------------------####

    # print(eccentric_noisy.duration)
    # print(lensed_noisy.duration)
    # print(unlensed_noisy.duration)

    # return

    ####------------------------------------------------------------------####

    noisy_gwpy_eccentric = TimeSeries.from_pycbc(eccentric_noisy)
    noisy_gwpy_lensed = TimeSeries.from_pycbc(lensed_noisy)
    noisy_gwpy_unlensed = TimeSeries.from_pycbc(unlensed_noisy)

    ####------------------------------------------------------------------####
 
    plt.figure(figsize=(12,8), facecolor=None)
    plt.pcolormesh(noisy_gwpy_eccentric.q_transform(logf=True, norm='mean', frange=(20,512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.savefig(training_data_path / f'eccentric_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,8), facecolor=None)
    plt.pcolormesh(noisy_gwpy_lensed.q_transform(logf=True, norm='mean', frange=(20,512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.savefig(training_data_path / f'lensed_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12,8), facecolor=None)
    plt.pcolormesh(noisy_gwpy_unlensed.q_transform(logf=True, norm='mean', frange=(20,512), whiten=True, qrange=(4, 64)))
    plt.axis('off')
    plt.savefig(training_data_path / f'unlensed_{num}.png', transparent=True, pad_inches=0, bbox_inches='tight')
    plt.close()

num_range = list(range(int(num_samples)))

with Pool(processes=num_processess) as pool:
        qtransforms = pool.map(generate_training_qtransform, num_range)
