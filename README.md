# 🔭 GWTorch Project Overview

A machine learning toolkit to classify gravitational wave (GW) signals — eccentric, lensed, and non-eccentric — using CNN and Transformer models. This project involves generation, preprocessing, training, and evaluation of GW signals using Q-transform spectrograms.

# A guide to File Structure

<pre>
.
├── environment.yml
├── gwtorch
│   ├── inference
│   │   ├── cnn_eval.py
│   │   ├── __pycache__
│   │   └── transformer_eval.py
│   ├── modules
│   │   ├── constants.py
│   │   ├── general_utils.py
│   │   ├── gw_utils.py
│   │   ├── neural_net.py
│   │   └── __pycache__
│   ├── training
│   │   ├── binary_cnn_classifier.py
│   │   ├── binary_transformer_classifier.py
│   │   ├── cnn.py
│   │   ├── Meta_model.py
│   │   ├── __pycache__
│   │   └── transformer.py
│   └── waveform_generation
│       ├── gen_ln_B.py
│       ├── gen.py
│       ├── gen_with_additional_functionalities.py
│       ├── gw_signal_gen_on_ln_B_basis.py
│       ├── Prior_Files
│       └── __pycache__
├── install.sh
├── pyproject.toml
└── README.md

</pre>

## 📌 Key Files Description

- `exp_gen.py`: Generates Q-transform spectrograms of synthetic GW signals using specified PSD (sampling frequency = 4096 Hz).
- `cnn.py`, `transformer.py`: Contain CNN and Transformer model definitions with training/testing routines.
- `transformer_with_ROC_and_File_Classification.py`: Extends training by logging ROC curves, AUC scores, and writing misclassified files.
- `cnn_eval.py`, `transformer_eval.py`: Evaluates trained CNN and Transformer models on test data.


# 🖥️ Command Line Interface (CLI)

- **`exp_gen_gwtorch --num-samples [int] --path-name [str] --batch-size [int]`** : Used to generate qtransform of the waveform samples.
- **`cnn_train_gwtorch --batch_size [default: 128] --epochs [default: 20] --lr [default: 3e-4] --gamma [default: 0.7] --model_path [default: cnn_model0.pth]  --k-folds [default: 5] --use-kfold [stores Truth value]`**: Train the CNN model on the data generated using previous code. The directory `models` is created if it doesn't exist.
- **`transformer_train_gwtorch --model [default: ViT] --batch_size [default: 512] --epochs [default 20] --lr [default: 3e-5] --gamma [default: 0.7] --seed [default: 42] --k-folds [default: 5] --use-kfold [stores Truth value]`**:  Train the Transformer model on the data generated.

# 🌊 How GW waveforms are generated

We use `TEOBResumS` to generate signals assuming `f_lower` = 5. Two sets of data are generated: 
- Eccentric: By giving the `eccentricity` parameter, sampled from prior assuming `Uniform` distribution in [0.1, 0.6].
- Non-Eccentric: By giving the `eccentricity` parameter value to be equal to 0.

The Non-Eccentric waveform is then used to generate unlensed data as well as lensed data. Lensing is done using `GWMAT` package.

The waveforms are then projected onto given set of detectors by using `ra`, `dec`, and `polarization` values sampled from the prior

Each wave is then tapered. Then we generated noise from corresponding PSD.

Next we pad each GW signal and add it to the noise such that the peak of the signal lies within 2.2 seconds to 2 seconds window before the noise ends. This is done to make the training process robust to moderate time
translations in the signal.

The signal is then cropped so that the data is of 8s duration.

Then finally we generate the `q_transforms` of each signal and save them as `.npy` file with naming convention such that if the signal is `lensed` then the name of the file is `lensed_{num}.npy`, where `num` just represnts file number.

## 📋 Prior Distributions Table

| **Parameter**   | **Distribution Type**              | **Range / Description**       |
|------------------|------------------------------------|-------------------------------|
| `mass1`         | `Constraint`                       | [10, 100]                     |
| `mass2`         | `Constraint`                       | [10, 100]                     |
| `mass_ratio`    | `UniformInComponentsMassRatio`     | [0.1, 1]                     |
| `chirp_mass`    | `UniformInComponentsChirpMass`     | [10, 100]                     |
| `spin1z`        | `Uniform`                          | [-0.9, 0.9]                    |
| `spin2z`        | `Uniform`                          | [-0.9, 0.9]                    |
| `eccentricity`  | `Uniform`                          | [0.1, 0.6]                    |
| `coa_phase`     | `Uniform`                          | [0.0, 2π]                     |
| `SNR`           | `PowerLaw` (`α = -4`)              | [10, 500]                   |
| `dec`           | `Cosine`                           | [−π/2, π/2]                   |
| `ra`            | `Uniform` (periodic)               | [0, 2π]                       |
| `polarization`  | `Uniform` (periodic)               | [0, π]                        |
| `Log_Mlz`       | `Uniform`                          | [1, 5]                        |
| `yl`            | `PowerLaw` (`α = 1`)               | [0.01, 1.0]                   |


**Note:** The `exp_gen.py` script also generates a Lookup table for parameter values corresponding to each sample and each category. Also sample `data` generated is given in `data_for_reference` folder alongwith its `Parameter Reference` in csv format.