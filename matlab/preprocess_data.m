function [trainData, testData, trainLabels, testLabels, normParams] = ...
         preprocess_data(sensorData, labels, trainRatio)
% PREPROCESS_DATA  Normalise sensor data and split into train/test sets.
%
%   [trainData, testData, trainLabels, testLabels, normParams] = ...
%       preprocess_data(sensorData, labels, trainRatio)
%
%   Inputs:
%     sensorData - (N x 6) raw sensor matrix
%     labels     - (N x 1) integer class labels
%     trainRatio - fraction of data used for training (default: 0.70)
%
%   Outputs:
%     trainData   - (Ntrain x 6) normalised training features
%     testData    - (Ntest  x 6) normalised test features
%     trainLabels - (Ntrain x 1) training labels
%     testLabels  - (Ntest  x 1) test labels
%     normParams  - struct with fields:
%                     .minVals  (1 x 6) per-feature minimum
%                     .maxVals  (1 x 6) per-feature maximum
%                     .range    (1 x 6) maxVals - minVals
%                   Use normParams to normalise future samples consistently.
%
%   Normalisation: Min-Max scaling to [0, 1] computed on training set only
%   to avoid data leakage.  Features with zero range are set to 0.

    % ------------------------------------------------------------------ %
    %  Input validation
    % ------------------------------------------------------------------ %
    if nargin < 3 || isempty(trainRatio)
        trainRatio = 0.70;
    end

    [N, numFeatures] = size(sensorData);
    if numel(labels) ~= N
        error('sensorData rows (%d) must match numel(labels) (%d).', N, numel(labels));
    end
    if trainRatio <= 0 || trainRatio >= 1
        error('trainRatio must be in (0, 1). Got: %.3f', trainRatio);
    end

    % ------------------------------------------------------------------ %
    %  Stratified train/test split
    %  Ensures each class is proportionally represented in both sets.
    % ------------------------------------------------------------------ %
    rng(0, 'twister');   % reproducible split
    classIDs  = unique(labels);
    trainIdx  = [];
    testIdx   = [];

    for c = 1:numel(classIDs)
        idx     = find(labels == classIDs(c));
        idx     = idx(randperm(numel(idx)));    % shuffle within class
        nTrain  = round(numel(idx) * trainRatio);
        trainIdx = [trainIdx; idx(1:nTrain)];           %#ok<AGROW>
        testIdx  = [testIdx;  idx(nTrain+1:end)];       %#ok<AGROW>
    end

    % Final shuffle so batches are class-mixed
    trainIdx = trainIdx(randperm(numel(trainIdx)));
    testIdx  = testIdx(randperm(numel(testIdx)));

    % ------------------------------------------------------------------ %
    %  Min-Max normalisation (fit on training data only)
    % ------------------------------------------------------------------ %
    trainRaw = sensorData(trainIdx, :);
    testRaw  = sensorData(testIdx,  :);

    minVals = min(trainRaw, [], 1);
    maxVals = max(trainRaw, [], 1);
    rng_vals = maxVals - minVals;

    % Avoid division by zero for constant features
    zeroCols         = (rng_vals == 0);
    rng_vals(zeroCols) = 1;

    normParams.minVals = minVals;
    normParams.maxVals = maxVals;
    normParams.range   = rng_vals;

    trainData = (trainRaw - minVals) ./ rng_vals;
    testData  = (testRaw  - minVals) ./ rng_vals;

    % Clamp to [0, 1] (test values may slightly exceed training range)
    trainData = min(max(trainData, 0), 1);
    testData  = min(max(testData,  0), 1);

    trainLabels = labels(trainIdx);
    testLabels  = labels(testIdx);

    % ------------------------------------------------------------------ %
    %  Summary
    % ------------------------------------------------------------------ %
    fprintf('Data split: %d training samples (%.0f%%), %d test samples (%.0f%%)\n', ...
            numel(trainIdx), trainRatio * 100, numel(testIdx), (1 - trainRatio) * 100);
    fprintf('Feature ranges (training set):\n');
    featureNames = {'Vibration','Temperature','Pressure','Current','RPM','Hours'};
    for f = 1:numFeatures
        fprintf('  %-12s  min=%.3f  max=%.3f\n', featureNames{f}, minVals(f), maxVals(f));
    end

    % ------------------------------------------------------------------ %
    %  Verify no NaN/Inf in output
    % ------------------------------------------------------------------ %
    if any(isnan(trainData(:))) || any(isinf(trainData(:)))
        error('trainData contains NaN or Inf after normalisation.');
    end
    if any(isnan(testData(:))) || any(isinf(testData(:)))
        error('testData contains NaN or Inf after normalisation.');
    end
end
