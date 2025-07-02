# A guide to Files
- `gen.py`: This is used to generate qtransform spectrograms of synthetic GW signals assuming aLIGO sensitivity. Sampling frequency is 4096 Hz.
- `tranformer.py` and `cnn.ipynb`: These are the neural network code where I have defined the model's structure and done training and testing.

# How GW waveforms are generated

We use `TEOBResumS` to generate signals assuming `f_lower` = 5. Two sets of data are generated: 
- Eccentric: By giving the `eccentricity` parameter, sampled from prior assuming `Uniform` distribution in [0.1, 0.6].
- Non-Eccentric: By giving the `eccentricity` parameter value to be equal to 0.

The Non-Eccentric waveform is then used to generate unlensed data as well as lensed data. Lensing is done using `GWMAT` package.

The waveforms are then projected onto `H!` detector by using `ra`, `dec`, and `polarization` values sampled from the prior

Each wave is then tapered. Then we generated noise from `aLIGOZeroDetHighPower` PSD.

Next we pad each GW signal and add it to the noise such that the peak of the signal lies within 2.2 seconds to 2 seconds window before the noise ends.

The signal is then cropped so that the data is of 8s duration.

Then finally we generate the `q_transforms` of each signal and save them as `PNG` file with naming convention such that if the signal is `lensed` then the name of the file is `lensed_{num},png`, where `num` just represnts file number.

