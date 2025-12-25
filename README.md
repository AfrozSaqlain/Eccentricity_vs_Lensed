# 🔭 GWTorch Project Overview

A machine learning toolkit to classify gravitational wave (GW) signals — eccentric, lensed, and non-eccentric — using CNN and Transformer models. This project involves generation, preprocessing, training, and evaluation of GW signals using Q-transform spectrograms.

To install the package just run: `bash install.sh`

# A guide to File Structure

<pre>
.
├── combine_csv.sh
├── environment.yml
├── gwtorch
│   ├── dataset
│   │   └── dagman.py
│   ├── inference
│   │   ├── cnn_eval.py
│   │   └── transformer_eval.py
│   ├── modules
│   │   ├── constants.py
│   │   ├── general_utils.py
│   │   ├── gw_utils.py
│   │   ├── ml_utils_new.py
│   │   ├── ml_utils.py
│   │   └── neural_net.py
│   ├── training
│   │   ├── binary_cnn_classifier.py
│   │   ├── binary_transformer_classifier.py
│   │   ├── cnn.py
│   │   ├── hybrid_transformer.py
│   │   ├── Meta_model.py
│   │   └── transformer.py
│   └── waveform_generation
│       ├── fitting_factor_gen.py
│       ├── match_gen_dag.py
│       └── match_gen.py
├── install.sh
├── merge.sh
├── move_30_percent.sh
├── notebooks
├── pyproject.toml
└── README.md

</pre>

## 📌 Key Files Description

- `match_gen.py`: Generates Q-transform spectrograms of synthetic GW signals, and computes ln B based on match.
- `match_gen_dag.py`: This is exactly same as previous function but it's suitable to generate dataset using this function when working on an HTCondor based system.
- `fitting_factor_gen.py`: Generates Q-transform spectrograms of synthetic GW signals, and computes ln B based on fitting factor.
- `cnn.py`, `transformer.py`: Contain CNN and Transformer model definitions with training/testing routines.
- `cnn_eval.py`, `transformer_eval.py`: Specifically used to evaluate trained CNN and Transformer models on test data.
- `dagman.py`: Using this script we can simply run `generate_dagman_gwtorch --jobs [number-of-jobs] --memory [RAM]G --disk [DISK]G --cpu [no-of-cpus] --samples [no-of-samples] to create a dagman workflow which you can submit to HTCondor system.


# 🖥️ Command Line Interface (CLI)

- **`generate_match_dataset_gwtorch --num-samples [int] --num-workers [int] --snr-prior [str] --path-name [str] --batch-size [int] --seed [int]`** : Used to generate qtransform of the waveform samples, by invoking `match_gen.py`. You can set `snr-prior` to be either `powerlaw` or `uniform` upon use case.

- **`generate_match_dataset_dag_gwtorch --num-samples [int] --num-workers [int] --snr-prior [str] --path-name [str] --batch-size [int] --seed [int] --snr-min [int] --snr-max [int]`**: When `snr-prior` is set to `uniform` and snr ranges are  give, the program generate the samples with SNR distributed uniformly within that range. However, if `snr-prior` is set to `powerlaw`, the SNR ranges provided will be ignored.

- **`generate_dagman_gwtorch --jobs [int] --memory [str] --disk [str] --samples [int] --cpu [int] --snr-dist [str] --snr-min [int] --snr-max [int]`**:  
  Generates an **HTCondor DAGMan workflow** for parallel dataset generation using `generate_match_dataset_dag_gwtorch`. It automatically creates:
  - Condor submit file for batched data-generation jobs  
  - DAG file with proper job dependencies  
  - Post-processing scripts to merge generated data directories and combine CSV parameter files  

  If `snr-dist=uniform`, the SNR range [`snr-min`, `snr-max`] is evenly split across jobs.  
  If `snr-dist=powerlaw`, the SNR range arguments are ignored.

  Submit the workflow using:
  ```bash
  condor_submit_dag submit/workflow.dag

- **`cnn_train_gwtorch --batch_size [default: 128] --epochs [default: 20] --lr [default: 3e-4] --gamma [default: 0.7] --model_path [default: cnn_model0.pth]  --k-folds [default: 5] --use-kfold [stores Truth value]`**: Train the CNN model on the data generated using previous code. The directory `models` is created if it doesn't exist.

- **`transformer_train_gwtorch --model [default: ViT] --batch_size [default: 512] --epochs [default 20] --lr [default: 3e-5] --gamma [default: 0.7] --seed [default: 42] --k-folds [default: 5] --use-kfold [stores Truth value]`**:  Train the Transformer model on the data generated.

# 🌊 How GW waveforms are generated

We use `TEOBResumS` to generate signals assuming `f_lower` = 5. Two sets of data are generated: 
- Eccentric: By giving the `eccentricity` parameter, sampled from prior assuming `Uniform` distribution in [0.01, 0.6].
- Non-Eccentric: By giving the `eccentricity` parameter value to be equal to 0.

The Non-Eccentric waveform is then used to generate unlensed data as well as lensed data. Lensing is done using `GWMAT` package.

The waveforms are then projected onto given set of detectors by using `ra`, `dec`, and `polarization` values sampled from the prior.

Each wave is then tapered. Then we generated noise from corresponding PSD.

Next we pad each GW signal and add it to the noise such that the peak of the signal lies within 2.2 seconds to 2 seconds window before the noise ends. This is done to make the training process robust to moderate time translations in the signal.

The signal is then cropped so that the data is of 8s duration.

Then finally we generate the `q_transforms` of each signal and save them as `.npy` file with naming convention such that if the signal is `lensed` then the file is located at `lensed / {num}.npy`, where `num` just represnts file number.

## 📋 Prior Distributions Table

| **Parameter**   | **Distribution Type**              | **Range / Description**       |
|------------------|------------------------------------|-------------------------------|
| `mass1`         | `Constraint`                       | [10, 100]                     |
| `mass2`         | `Constraint`                       | [10, 100]                     |
| `mass_ratio`    | `UniformInComponentsMassRatio`     | [0.1, 1]                     |
| `chirp_mass`    | `UniformInComponentsChirpMass`     | [10, 100]                     |
| `spin1z`        | `Uniform`                          | [-0.9, 0.9]                    |
| `spin2z`        | `Uniform`                          | [-0.9, 0.9]                    |
| `eccentricity`  | `Uniform`                          | [0.01, 0.6]                    |
| `coa_phase`     | `Uniform`                          | [0.0, 2π]                     |
| `SNR`           | `PowerLaw` (`α = -4`)              | [4, 500]                   |
| `dec`           | `Cosine`                           | [−π/2, π/2]                   |
| `ra`            | `Uniform` (periodic)               | [0, 2π]                       |
| `polarization`  | `Uniform` (periodic)               | [0, π]                        |
| `Log_Mlz`       | `Uniform`                          | [1, 5]                        |
| `yl`            | `PowerLaw` (`α = 1`)               | [0.01, 1.0]                   |


**Note:** The `match_gen.py` and `fitting_factor_gen` scripts also generates a Lookup table for parameter values corresponding to each sample and each category.