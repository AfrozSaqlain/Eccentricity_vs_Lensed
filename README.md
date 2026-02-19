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
  ```

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
| `anomlay`       | `Uniform`                          | [0, 2π]                    |
| `coa_phase`     | `Uniform`                          | [0.0, 2π]                     |
| `SNR`           | `PowerLaw` (`α = -4`)              | [4, 500]                   |
| `dec`           | `Cosine`                           | [−π/2, π/2]                   |
| `ra`            | `Uniform` (periodic)               | [0, 2π]                       |
| `polarization`  | `Uniform` (periodic)               | [0, π]                        |
| `Log_Mlz`       | `Uniform`                          | [1, 5]                        |
| `yl`            | `PowerLaw` (`α = 1`)               | [0.01, 1.0]                   |


**Note:** The `match_gen.py` d `fitting_factor_gen` scripts also generates a Lookup table for parameter values corresponding to each sample and each category.

---

# ✅ End-to-End Quick Start (Generate → Train → Test/Inference)

This section is added as a practical guide so a new user can run the full GWtorch workflow without guessing missing steps.

## 1) Installation

You can still use:

```bash
bash install.sh
```

Or install with `pip` (from project root):

```bash
pip install -e .
```

If you need ML + GW dependencies together:

```bash
pip install -e ".[ml,gw]"
```

## 2) Dataset Generation (Local)

Generate dataset with 3 classes (`eccentric`, `lensed`, `unlensed`):

```bash
generate_match_dataset_gwtorch \
  --num-samples 1000 \
  --num-workers 8 \
  --snr-prior powerlaw \
  --path-name data \
  --batch-size 150 \
  --seed 42
```

For uniform SNR binning on HTCondor-friendly generator:

```bash
generate_match_dataset_dag_gwtorch \
  --num-samples 1000 \
  --num-workers 8 \
  --snr-prior uniform \
  --snr-min 8 \
  --snr-max 100 \
  --path-name data \
  --batch-size 150 \
  --seed 42
```

### Generated files

- Spectrograms:
  - `data/eccentric/*.npy`
  - `data/lensed/*.npy`
  - `data/unlensed/*.npy`
- Parameter lookup CSV:
  - `results/data_parameters.csv` (if `--path-name data`)
  - In general: `results/<dataset_name>_parameters.csv`

## 3) Train Multi-Class Models

### CNN

```bash
cnn_train_gwtorch \
  --model CNN \
  --batch-size 512 \
  --epochs 20 \
  --lr 3e-5 \
  --gamma 0.7 \
  --seed 42 \
  --path-name data \
  --model-file-name cnn_model
```

### Transformer (ViT or CvT)

```bash
transformer_train_gwtorch \
  --model ViT \
  --batch-size 512 \
  --epochs 20 \
  --lr 3e-5 \
  --gamma 0.7 \
  --seed 42 \
  --path-name data \
  --model-file-name vit_model
```

### Optional K-fold training

Add:

```bash
--use-kfold --k-folds 5
```

## 4) Train Binary Models (Eccentric vs Lensed)

### Binary CNN

```bash
binary_cnn_train_gwtorch \
  --model CNN \
  --batch-size 512 \
  --epochs 20 \
  --lr 3e-5 \
  --gamma 0.7 \
  --seed 42 \
  --path-name data \
  --model-file-name cnn_model_binary
```

### Binary Transformer

```bash
binary_transformer_train_gwtorch \
  --model ViT \
  --batch-size 512 \
  --epochs 20 \
  --lr 3e-5 \
  --gamma 0.7 \
  --seed 42 \
  --path-name data \
  --model-file-name transformer_model_binary
```

## 5) Inference / Testing

### CNN inference

```bash
cnn_inference_gwtorch \
  --model CNN \
  --batch-size 512 \
  --model-path cnn_model.pth \
  --path-name data \
  --meta-csv results/data_parameters.csv \
  --results-dir inference_results \
  --seed 42
```

### Transformer inference

```bash
transformer_inference_gwtorch \
  --model ViT \
  --batch-size 512 \
  --model-path vit_model.pth \
  --path-name data \
  --meta-csv results/data_parameters.csv \
  --results-dir inference_results \
  --seed 42
```

Optional lnB-filtered confusion matrix for transformer inference:

```bash
--lnB-min 0 --lnB-max 10
```

## 6) Grad-CAM

### CNN Grad-CAM

```bash
cnn_gradcam_gwtorch \
  --model CNN \
  --model-path cnn_model.pth \
  --path-name data \
  --results-dir inference_results
```

### Transformer Grad-CAM

```bash
transformer_gradcam_gwtorch \
  --model ViT \
  --model-path vit_model.pth \
  --path-name data \
  --results-dir inference_results
```

## 7) HTCondor Workflow

Create DAG artifacts:

```bash
generate_data_dagman_gwtorch \
  --jobs 10 \
  --memory 20G \
  --disk 20G \
  --samples 100 \
  --cpu 5 \
  --snr-dist powerlaw
```

Submit:

```bash
condor_submit_dag submit/workflow.dag
```

Generate training submit file:

```bash
generate_training_condor_gwtorch \
  --model transformer \
  --model-subtype ViT \
  --classification-scheme multi-class \
  --epochs 20 \
  --batch-size 512 \
  --data-file-path data
```

## 8) Recommended Dataset Naming Convention

If your dataset directory is named `X`, keep metadata file as:

```text
results/X_parameters.csv
```

This helps training/inference find lnB metadata automatically.

## 9) Useful Output Locations

- Trained models: `models/*.pth`
- Training outputs: `results/<Model>_results/`
- Inference outputs: `<results-dir>/<Model>_results/`
- Misclassification files and reports are generated automatically in result directories.

## 10) Troubleshooting

- If multiprocessing fails in restricted environments, reduce workers:
  - generation: `--num-workers 1`
  - training/inference can still run with fewer worker processes.
- If metadata CSV is missing, scripts can still run but lnB-dependent analysis will be limited.
- For tiny datasets, `train_test_split(..., stratify=...)` may fail. Increase `--num-samples`.

---

## Keep Using Prior Settings Above

Your prior distribution table in this README remains the reference for waveform generation settings. The generation scripts (`match_gen.py`, `match_gen_dag.py`, `fitting_factor_gen.py`) sample from those priors exactly as described above.
