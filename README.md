# Bearing Fault Classification — Machine Learning System

A complete MATLAB-based machine learning system for classifying bearing faults in rotating machinery using an Artificial Neural Network (ANN). The system generates physics-based synthetic sensor data, trains a multi-layer ANN, evaluates performance, and supports real-time fault prediction.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset Description](#dataset-description)
4. [ANN Architecture](#ann-architecture)
5. [Physics-Based Data Generation](#physics-based-data-generation)
6. [Preprocessing](#preprocessing)
7. [Training Parameters](#training-parameters)
8. [Performance Metrics](#performance-metrics)
9. [Usage Instructions](#usage-instructions)
10. [Output Files](#output-files)
11. [References](#references)

---

## Project Overview

Bearing faults account for a significant proportion of rotating-machinery failures. Early detection prevents catastrophic breakdowns and reduces maintenance costs. This project implements a supervised classification pipeline that:

- Generates **5,000 realistic sensor samples** using physics-informed relationships between RPM, temperature, vibration, current, pressure, and operating hours.
- Trains a **6 → 20 → 15 → 5 ANN** using the Adam optimiser with cross-entropy loss.
- Evaluates the model with a confusion matrix and per-class Precision / Recall / F1-Score.
- Exposes a real-time prediction function that returns a fault class, confidence score, and an operational alert status.

---

## Repository Structure

```
FaultClassificationML/
├── matlab/
│   ├── generate_sensor_data.m    # Physics-based data generation
│   ├── preprocess_data.m         # Min-Max normalisation + stratified split
│   ├── train_ann_model.m         # ANN training (Adam, cross-entropy)
│   ├── evaluate_model.m          # Confusion matrix, Precision/Recall/F1
│   ├── predict_fault_status.m    # Real-time single-sample inference
│   └── main.m                    # Full pipeline orchestration script
├── data/
│   ├── sensor_data_5000.csv      # Auto-generated — complete dataset
│   ├── training_data.csv         # Auto-generated — 3 500 training samples
│   └── test_data.csv             # Auto-generated — 1 500 test samples
├── results/
│   ├── trained_ann_model.mat     # Auto-generated — saved network weights
│   ├── training_report.txt       # Auto-generated — metrics & validation
│   ├── confusion_matrix.fig/.png # Auto-generated — visualisation
│   ├── performance_metrics.fig/.png
│   ├── training_history.fig/.png
│   └── prediction_results.csv   # Auto-generated — 10 sample predictions
└── README.md
```

---

## Dataset Description

| Property | Value |
|---|---|
| Total samples | 5,000 |
| Samples per class | 1 000 |
| Number of features | 6 |
| Number of classes | 5 |

### Sensor Features

| # | Feature | Unit | Description |
|---|---|---|---|
| 1 | Vibration | mm/s | RMS vibration amplitude |
| 2 | Temperature | °C | Bearing housing temperature |
| 3 | Pressure | bar | Lubrication system pressure |
| 4 | Current | A | Motor drive current |
| 5 | RPM | rev/min | Shaft rotational speed |
| 6 | Operating Hours | h | Cumulative run time |

### Fault Classes

| Label | Class | Typical Signature |
|---|---|---|
| 1 | **Normal** | Low vibration (0–5 mm/s), stable temperature (40–50 °C) |
| 2 | **Bearing Wear** | Gradual vibration rise (5–15 mm/s), slow temperature increase (45–65 °C) |
| 3 | **Misalignment** | High-frequency vibration spikes (10–25 mm/s), elevated pressure (6–8.5 bar) |
| 4 | **Overheating** | Exponential temperature rise (60–95 °C), high current (19–24 A) |
| 5 | **Combined Faults** | Superposition of wear + misalignment + overheating signatures |

---

## ANN Architecture

```
Input (6)  →  Hidden 1 (20, ReLU)  →  Hidden 2 (15, ReLU)  →  Output (5, Softmax)
```

| Layer | Neurons | Activation | Parameters |
|---|---|---|---|
| Input | 6 | — | — |
| Hidden 1 | 20 | ReLU | 6×20 + 20 = 140 |
| Hidden 2 | 15 | ReLU | 20×15 + 15 = 315 |
| Output | 5 | Softmax | 15×5 + 5 = 80 |
| **Total** | — | — | **535** |

**Weight initialisation:** Xavier / Glorot uniform — limits computed as `sqrt(6 / (fanIn + fanOut))`.

---

## Physics-Based Data Generation

Each fault class follows physically motivated sensor relationships:

### Normal Operation
- Vibration scaled linearly with RPM factor (`rpm / 1500`).
- Temperature includes a small drift proportional to operating hours.
- All parameters remain within healthy operating limits.

### Bearing Wear
- A `degradation = hours / 8000` factor (0 → 1) drives both vibration and temperature upward progressively, simulating material removal and increased friction over time.

### Misalignment
- Vibration modelled as `A · |sin(φ)|`, where φ is a random phase angle, replicating the symmetric double-frequency vibration signature of angular misalignment.
- Pressure fluctuates in phase with the vibration spike.

### Overheating
- Temperature follows an exponential saturation model: `T = 60 + 35 · (1 − exp(−loadFactor · rpmFactor))`, consistent with Fourier heat-transfer equations.
- Current rises proportionally with thermal load.

### Combined Faults
- Vibration and temperature are the superposition of wear, misalignment, and overheating components, producing the most erratic and high-magnitude readings.

---

## Preprocessing

1. **Stratified split** — each class is split independently to preserve class balance in both train (70 %) and test (30 %) sets.
2. **Min-Max normalisation** — fit on training data only to prevent data leakage:

   ```
   x_norm = (x − min_train) / (max_train − min_train)
   ```

3. **Clamping** — test values are clamped to [0, 1] to handle the small fraction that may fall outside the training range.
4. **Normalisation parameters** (`normParams.minVals`, `.maxVals`, `.range`) are saved alongside the model so that future raw samples can be scaled consistently.

---

## Training Parameters

| Parameter | Value |
|---|---|
| Optimiser | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Loss function | Categorical cross-entropy |
| Epochs | 100 |
| Batch size | 32 |
| Initial learning rate | 0.01 |
| LR decay | ×0.95 every 10 epochs |
| Weight initialisation | Xavier uniform |

---

## Performance Metrics

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall (Sensitivity) | TP / (TP + FN) |
| F1-Score | 2 · Precision · Recall / (Precision + Recall) |

Per-class metrics are computed from the confusion matrix. Macro-averaged values (unweighted mean across all 5 classes) are reported in `training_report.txt`.

**Alert thresholds for `predict_fault_status`:**

| Confidence | Status |
|---|---|
| ≥ 95 % | `Normal` |
| 80 % – 94 % | `Alert` |
| < 80 % | `Critical` |

---

## Usage Instructions

### Prerequisites

- MATLAB R2019b or later
- No additional toolboxes required

### Run the Full Pipeline

```matlab
cd matlab
main
```

This executes all six steps and writes every output file automatically.

### Run Individual Steps

```matlab
% 1. Generate data
[sensorData, labels, labelNames] = generate_sensor_data(5000);

% 2. Preprocess
[trainData, testData, trainLabels, testLabels, normParams] = ...
    preprocess_data(sensorData, labels, 0.70);

% 3. Train
[trainedNet, trainingInfo] = train_ann_model(trainData, trainLabels, testData, testLabels);

% 4. Evaluate
[metrics, confusionMat, plots] = evaluate_model(trainedNet, testData, testLabels, labelNames);

% 5. Predict a new sample
rawSample  = [8.5, 62.3, 6.1, 18.4, 1450, 3200];   % 1x6 raw sensor values
normSample = (rawSample - normParams.minVals) ./ normParams.range;
normSample = min(max(normSample, 0), 1);
[cls, conf, alert] = predict_fault_status(trainedNet, normSample, labelNames);
```

### Load a Previously Saved Model

```matlab
load('results/trained_ann_model.mat', 'trainedNet', 'normParams', 'labelNames');

rawSample  = [12.0, 71.5, 7.2, 20.1, 1380, 5500];
normSample = (rawSample - normParams.minVals) ./ normParams.range;
normSample = min(max(normSample, 0), 1);
[cls, conf, alert] = predict_fault_status(trainedNet, normSample, labelNames);
fprintf('Fault: %s | Confidence: %.1f%% | Alert: %s\n', labelNames{cls}, conf, alert);
```

---

## Output Files

| File | Location | Description |
|---|---|---|
| `sensor_data_5000.csv` | `data/` | All 5,000 samples with labels |
| `training_data.csv` | `data/` | 3 500 normalised training samples |
| `test_data.csv` | `data/` | 1 500 normalised test samples |
| `trained_ann_model.mat` | `results/` | Network weights + normParams |
| `training_report.txt` | `results/` | Accuracy, Precision, Recall, F1 |
| `confusion_matrix.fig/.png` | `results/` | Confusion matrix visualisation |
| `performance_metrics.fig/.png` | `results/` | Per-class bar chart |
| `training_history.fig/.png` | `results/` | Loss & accuracy curves |
| `prediction_results.csv` | `results/` | 10 sample predictions with confidence |

---

## References

1. Zhao, M., et al. (2020). **A novel bearing faults diagnosis of rotor-bearing systems using a new multiplicative-threshold wavelet packet transform**. *Mechanical Systems and Signal Processing*, 144, 106933. 

2. Wen, L., et al. (2022). **A physics-informed deep learning approach for bearing fault detection**. *Engineering Applications of Artificial Intelligence*, 103, 104295.

3. Immovilli, F., et al. (2021). **Enhanced approach for gears-bearings defect diagnosis based on complex shifted Morlet wavelets**. *ISA Transactions*, 111, 338–348.

4. Li, X., et al. (2022). **Fault Diagnosis Method of Rolling Bearing Based on MSCNN-LSTM**. *Sensors*, 22(18), 6963.

---

*Generated and maintained by the FaultClassificationML project.*
