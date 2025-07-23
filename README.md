# 🔭 GWTorch Project Overview

A machine learning toolkit to classify gravitational wave (GW) signals — eccentric, lensed, and non-eccentric — using CNN and Transformer models. This project involves generation, preprocessing, training, and evaluation of GW signals using Q-transform spectrograms.

# A guide to File Structure

<pre>
.
├── gwtorch
│   ├── inference
│   │   └── cnn_eval.py
│   ├── modules
│   │   ├── constants.py
│   │   ├── general_utils.py
│   │   ├── gw_utils.py
│   │   └── neural_net.py
│   ├── training
│   │   ├── cnn.py
│   │   ├── Meta_model.py
│   │   ├── transformer.py
│   │   ├── transformer_with_ROC_and_File_Classification.py
│   │   └── transformer_with_ROC.py
│   └── waveform_generation
│       ├── gen.py
│       ├── gen_with_additional_functionalities.py
│       ├── gw_signal_gen_on_ln_B_basis.py
│       └── Prior_Files
├── notebooks
│   ├── analysis.ipynb
│   ├── eval_based_on_ln_B_cnn.ipynb
│   ├── eval_based_on_ln_B transformer.ipynb
│   ├── gen.ipynb
│   └── gw_utils.py
├── pyproject.toml
├── README.md
└── results
    ├── cnn_results
    │   ├── misclassified_test.pkl
    │   ├── misclassified_test.txt
    │   └── Plots
    ├── Meta_model
    │   ├── lens_model.pth
    │   ├── morph_model.pth
    │   ├── Plots
    │   └── results
    ├── test_data_parameters.csv
    ├── train_data_parameters.csv
    └── transformer_results
        ├── misclassified_test.txt
        ├── misclassified_validation.txt
        ├── parameters_reference.csv
        ├── Plots
        └── snr_lookup_table.csv

</pre>

## 📌 Key Files Description

- `gen_with_additional_functionalities.py`: Generates Q-transform spectrograms of synthetic GW signals using aLIGO sensitivity (sampling frequency = 4096 Hz).
- `gw_signal_gen_on_ln_B_basis.py`: Generates eccentric GW signals and calculates \(\ln B^{\text{Ecc}}_{\text{Qc}}\), grouping samples into three bins with balanced counts.
- `cnn.py`, `transformer.py`: Contain CNN and Transformer model definitions with training/testing routines.
- `transformer_with_ROC_and_File_Classification.py`: Extends training by logging ROC curves, AUC scores, and writing misclassified files.
- `cnn_eval.py`: Evaluates trained CNN models on test data.
- `eval_based_on_ln_B*.ipynb`: Evaluates misclassifications across different \(\ln B\) bins using previously generated data.


# 🖥️ Command Line Interface (CLI)

- **`generate_waveform_gwtorch --num-samples [int] --path-name ['train', 'test']`** : Used to generate qtransform of the waveform samples.
- **`cnn_train_gwtorch --batch_size [default: 128] --epochs [default: 20] --lr [default: 3e-4] --gamma [default: 0.7] --model_path [default: ./models/cnn_model0.pth]`**: Train the CNN model on the data generated using previous code. The directory `models` is created if it doesn't exist.
- **`transformers_train_gwtorch --model [default: ViT] --batch_size [default: 512] --epochs [default 20] --lr [default: 3e-5] --gamma [default: 0.7] --seed [default: 42]`**:  Train the Transformer model on the data generated.

# 🌊 How GW waveforms are generated

We use `TEOBResumS` to generate signals assuming `f_lower` = 5. Two sets of data are generated: 
- Eccentric: By giving the `eccentricity` parameter, sampled from prior assuming `Uniform` distribution in [0.1, 0.6].
- Non-Eccentric: By giving the `eccentricity` parameter value to be equal to 0.

The Non-Eccentric waveform is then used to generate unlensed data as well as lensed data. Lensing is done using `GWMAT` package.

The waveforms are then projected onto `H1` detector by using `ra`, `dec`, and `polarization` values sampled from the prior

Each wave is then tapered. Then we generated noise from `aLIGOZeroDetHighPower` PSD.

Next we pad each GW signal and add it to the noise such that the peak of the signal lies within 2.2 seconds to 2 seconds window before the noise ends. This is done to make the training process robust to moderate time
translations in the signal.

We also ensure that the signal's SNR is always greater than a minimum threshold, i.e. 10 by multiplying the signal by a sampled `scale_factor`.

The signal is then cropped so that the data is of 8s duration.

Then finally we generate the `q_transforms` of each signal and save them as `PNG` file with naming convention such that if the signal is `lensed` then the name of the file is `lensed_{num},png`, where `num` just represnts file number.

## 📋 Prior Distributions Table

| **Parameter**   | **Distribution Type**              | **Range / Description**       |
|------------------|------------------------------------|-------------------------------|
| `mass1`         | `Constraint`                       | [10, 100]                     |
| `mass2`         | `Constraint`                       | [10, 100]                     |
| `mass_ratio`    | `UniformInComponentsMassRatio`     | [0.1, 1]                     |
| `chirp_mass`    | `UniformInComponentsChirpMass`     | [25, 100]                     |
| `spin1z`        | `Uniform`                          | [0.0, 0.9]                    |
| `spin2z`        | `Uniform`                          | [0.0, 0.9]                    |
| `eccentricity`  | `Uniform`                          | [0.1, 0.6]                    |
| `coa_phase`     | `Uniform`                          | [0.0, 2π]                     |
| `distance`      | `Uniform`                          | [100, 1000]                   |
| `dec`           | `Cosine`                           | [−π/2, π/2]                   |
| `ra`            | `Uniform` (periodic)               | [0, 2π]                       |
| `polarization`  | `Uniform` (periodic)               | [0, π]                        |
| `Log_Mlz`       | `Uniform`                          | [3, 5]                        |
| `yl`            | `PowerLaw` (`α = 1`)               | [0.01, 1.0]                   |


**Note:** The `gen_with_additional_functionalities.py` code also generates a Lookup table for SNR values and parameter values corresponding to each sample and each category. Also sample `data` generated is given in `data_for_reference` folder alongwith its `SNR Lookup Table` as well as `Parameter Reference` in csv format.