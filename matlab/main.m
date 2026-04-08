% MAIN  Orchestration script for the Bearing Fault Classification ML System.
%
%   Runs the complete pipeline:
%     1. Generate 5000 physics-based sensor samples
%     2. Preprocess (normalise + train/test split)
%     3. Train ANN model  (6 → 20 → 15 → 5)
%     4. Evaluate performance (confusion matrix, F1, etc.)
%     5. Save trained model and data
%     6. Run sample real-time predictions
%     7. Generate CSV and text reports
%
%   All output files are written to the  ../data/  and  ../results/  folders
%   relative to this script.
%
%   Requirements:
%     MATLAB R2019b or later (no additional toolboxes required).
%
%   Usage:
%     >> cd matlab
%     >> main

clc; clear; close all;

% ========== 0. Setup paths ============================================= %
scriptDir   = fileparts(mfilename('fullpath'));
dataDir     = fullfile(scriptDir, '..', 'data');
resultsDir  = fullfile(scriptDir, '..', 'results');

if ~exist(dataDir,    'dir'), mkdir(dataDir);    end
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

addpath(scriptDir);   % ensure local functions are findable

fprintf('=============================================================\n');
fprintf('  BEARING FAULT CLASSIFICATION — MACHINE LEARNING PIPELINE  \n');
fprintf('=============================================================\n\n');

% ========== 1. Generate Sensor Data ==================================== %
fprintf('[STEP 1] Generating sensor data ...\n');
numSamples = 5000;
[sensorData, labels, labelNames] = generate_sensor_data(numSamples);

% Save complete dataset to CSV
featureNames = {'Vibration_mm_s', 'Temperature_C', 'Pressure_bar', ...
                'Current_A', 'RPM', 'OperatingHours', 'Label', 'LabelName'};
labelNameCol = labelNames(labels)';   % Nx1 cell
csvTable = array2table([sensorData, double(labels)], ...
    'VariableNames', featureNames(1:7));
csvTable.LabelName = labelNameCol;

fullDataPath = fullfile(dataDir, 'sensor_data_5000.csv');
writetable(csvTable, fullDataPath);
fprintf('  Saved: %s\n\n', fullDataPath);

% ========== 2. Preprocess Data ========================================= %
fprintf('[STEP 2] Preprocessing data ...\n');
trainRatio = 0.70;
[trainData, testData, trainLabels, testLabels, normParams] = ...
    preprocess_data(sensorData, labels, trainRatio);

% Save train/test CSVs
trainTable = array2table([trainData, double(trainLabels)], ...
    'VariableNames', [featureNames(1:6), {'Label'}]);
testTable  = array2table([testData,  double(testLabels)],  ...
    'VariableNames', [featureNames(1:6), {'Label'}]);

trainPath = fullfile(dataDir, 'training_data.csv');
testPath  = fullfile(dataDir, 'test_data.csv');
writetable(trainTable, trainPath);
writetable(testTable,  testPath);
fprintf('  Saved: %s\n', trainPath);
fprintf('  Saved: %s\n\n', testPath);

% ========== 3. Train ANN Model ========================================= %
fprintf('[STEP 3] Training ANN model (6 -> 20 -> 15 -> 5) ...\n');
tic;
[trainedNet, trainingInfo] = train_ann_model(trainData, trainLabels, testData, testLabels);
trainTime = toc;
fprintf('  Training time: %.1f seconds\n\n', trainTime);

% ========== 4. Evaluate Model ========================================== %
fprintf('[STEP 4] Evaluating model ...\n');
[metrics, confusionMat, plots] = evaluate_model(trainedNet, testData, testLabels, labelNames);

% ========== 5. Save Model and Results ================================== %
fprintf('[STEP 5] Saving model and results ...\n');

% Save trained network
modelPath = fullfile(resultsDir, 'trained_ann_model.mat');
save(modelPath, 'trainedNet', 'trainingInfo', 'normParams', 'labelNames');
fprintf('  Saved: %s\n', modelPath);

% Save confusion matrix figure
confFigPath = fullfile(resultsDir, 'confusion_matrix.fig');
savefig(plots.confusionFig, confFigPath);
fprintf('  Saved: %s\n', confFigPath);

% Also save as PNG for portability
confPngPath = fullfile(resultsDir, 'confusion_matrix.png');
print(plots.confusionFig, confPngPath, '-dpng', '-r150');
fprintf('  Saved: %s\n', confPngPath);

% Save metrics bar chart
metFigPath = fullfile(resultsDir, 'performance_metrics.fig');
savefig(plots.metricsBarFig, metFigPath);
fprintf('  Saved: %s\n', metFigPath);

metPngPath = fullfile(resultsDir, 'performance_metrics.png');
print(plots.metricsBarFig, metPngPath, '-dpng', '-r150');
fprintf('  Saved: %s\n', metPngPath);

% Save training history plot
histFig = figure('Name', 'Training History', 'Visible', 'off');
subplot(2, 1, 1);
plot(trainingInfo.trainLoss, 'b-', 'LineWidth', 1.5); hold on;
plot(trainingInfo.testLoss,  'r--', 'LineWidth', 1.5);
legend('Train Loss', 'Test Loss'); grid on;
xlabel('Epoch'); ylabel('Cross-Entropy Loss');
title('Training Loss History');

subplot(2, 1, 2);
plot(trainingInfo.trainAcc * 100, 'b-', 'LineWidth', 1.5); hold on;
plot(trainingInfo.testAcc  * 100, 'r--', 'LineWidth', 1.5);
legend('Train Accuracy', 'Test Accuracy'); grid on;
xlabel('Epoch'); ylabel('Accuracy (%)');
title('Training Accuracy History');
ylim([0, 105]);

histFigPath = fullfile(resultsDir, 'training_history.fig');
histPngPath = fullfile(resultsDir, 'training_history.png');
savefig(histFig, histFigPath);
print(histFig, histPngPath, '-dpng', '-r150');
fprintf('  Saved: %s\n', histFigPath);
fprintf('  Saved: %s\n', histPngPath);

% ========== 6. Write Training Report =================================== %
reportPath = fullfile(resultsDir, 'training_report.txt');
fid = fopen(reportPath, 'w');
fprintf(fid, '=============================================================\n');
fprintf(fid, '  BEARING FAULT CLASSIFICATION — TRAINING REPORT\n');
fprintf(fid, '  Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '=============================================================\n\n');
fprintf(fid, 'Dataset\n');
fprintf(fid, '  Total samples   : %d\n', numSamples);
fprintf(fid, '  Training samples: %d (%.0f%%)\n', size(trainData, 1), trainRatio * 100);
fprintf(fid, '  Test samples    : %d (%.0f%%)\n', size(testData,  1), (1 - trainRatio) * 100);
fprintf(fid, '  Features        : 6 (Vibration, Temperature, Pressure, Current, RPM, Hours)\n');
fprintf(fid, '  Classes         : %d\n\n', numel(labelNames));

fprintf(fid, 'Model Architecture\n');
fprintf(fid, '  Input  layer : 6 neurons\n');
fprintf(fid, '  Hidden layer1: 20 neurons (ReLU)\n');
fprintf(fid, '  Hidden layer2: 15 neurons (ReLU)\n');
fprintf(fid, '  Output layer : 5 neurons (Softmax)\n\n');

fprintf(fid, 'Training Parameters\n');
fprintf(fid, '  Optimizer    : %s\n', trainingInfo.params.optimizer);
fprintf(fid, '  Loss         : %s\n', trainingInfo.params.lossFunction);
fprintf(fid, '  Epochs       : %d\n', trainingInfo.params.epochs);
fprintf(fid, '  Batch size   : %d\n', trainingInfo.params.batchSize);
fprintf(fid, '  Learning rate: %.4f\n', trainingInfo.params.learningRate);
fprintf(fid, '  Train time   : %.1f s\n\n', trainTime);

fprintf(fid, 'Performance Summary\n');
fprintf(fid, '  Overall Accuracy: %.2f%%\n\n', metrics.overallAccuracy * 100);
fprintf(fid, '  %-20s  %10s  %10s  %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf(fid, '  %s\n', repmat('-', 1, 55));
for c = 1:numel(labelNames)
    fprintf(fid, '  %-20s  %10.4f  %10.4f  %10.4f\n', ...
        labelNames{c}, metrics.precision(c), metrics.recall(c), metrics.f1Score(c));
end
fprintf(fid, '  %s\n', repmat('-', 1, 55));
fprintf(fid, '  %-20s  %10.4f  %10.4f  %10.4f\n', 'MACRO AVERAGE', ...
    mean(metrics.precision), mean(metrics.recall), mean(metrics.f1Score));
fprintf(fid, '\nConfusion Matrix\n');
headerFmt = ['  ', repmat('%12s', 1, numel(labelNames)), '\n'];
rowFmt    = ['  %-12s', repmat('%12d',  1, numel(labelNames)), '\n'];
fprintf(fid, headerFmt, labelNames{:});
for r = 1:numel(labelNames)
    fprintf(fid, rowFmt, labelNames{r}, confusionMat(r, :));
end
fclose(fid);
fprintf('  Saved: %s\n\n', reportPath);

% ========== 7. Sample Predictions ====================================== %
fprintf('[STEP 6] Running sample real-time predictions ...\n');
numPredSamples = 10;
rng(999, 'twister');
sampleIdx      = randperm(size(testData, 1), numPredSamples);
predResults    = cell(numPredSamples, 5);

fprintf('  %-4s  %-20s  %-20s  %10s  %10s\n', ...
    'No.', 'True Label', 'Predicted Label', 'Confidence', 'Status');
fprintf('  %s\n', repmat('-', 1, 72));

for i = 1:numPredSamples
    sample  = testData(sampleIdx(i), :);
    trueLabel = testLabels(sampleIdx(i));
    [predClass, conf, alertStatus] = predict_fault_status(trainedNet, sample, labelNames);
    predResults(i, :) = {i, labelNames{trueLabel}, labelNames{predClass}, conf, alertStatus};
    fprintf('  %-4d  %-20s  %-20s  %9.2f%%  %s\n', ...
        i, labelNames{trueLabel}, labelNames{predClass}, conf, alertStatus);
end
fprintf('\n');

% Save prediction results CSV
predTable = cell2table(predResults, ...
    'VariableNames', {'SampleNo', 'TrueLabel', 'PredictedLabel', 'Confidence_pct', 'AlertStatus'});
predPath = fullfile(resultsDir, 'prediction_results.csv');
writetable(predTable, predPath);
fprintf('  Saved: %s\n\n', predPath);

% ========== Done ======================================================= %
fprintf('=============================================================\n');
fprintf('  PIPELINE COMPLETE\n');
fprintf('  Overall Test Accuracy : %.2f%%\n', metrics.overallAccuracy * 100);
fprintf('  Macro F1-Score        : %.4f\n',   mean(metrics.f1Score));
fprintf('\n  Output files written to:\n');
fprintf('    %s\n', dataDir);
fprintf('    %s\n', resultsDir);
fprintf('=============================================================\n');
