function [trainedNet, trainingInfo] = train_ann_model(trainData, trainLabels, testData, testLabels)
% TRAIN_ANN_MODEL  Train an Artificial Neural Network for bearing fault classification.
%
%   [trainedNet, trainingInfo] = train_ann_model(trainData, trainLabels, testData, testLabels)
%
%   Architecture:
%     Input   :  6 neurons  (one per sensor feature)
%     Hidden 1: 20 neurons  (ReLU activation)
%     Hidden 2: 15 neurons  (ReLU activation)
%     Output  :  5 neurons  (Softmax — one per fault class)
%
%   Training parameters:
%     Algorithm  : Mini-batch stochastic gradient descent (Adam optimiser)
%     Epochs     : 100
%     Batch size : 32
%     Learn rate : 0.01 (with decay)
%     Loss       : Cross-entropy
%
%   Inputs:
%     trainData   - (Ntrain x 6) normalised training features
%     trainLabels - (Ntrain x 1) integer labels (1-5)
%     testData    - (Ntest  x 6) normalised test features
%     testLabels  - (Ntest  x 1) integer labels (1-5)
%
%   Outputs:
%     trainedNet   - Trained network weights and architecture (struct)
%     trainingInfo - Struct containing:
%                      .trainLoss   (epochs x 1) training cross-entropy loss
%                      .testLoss    (epochs x 1) test cross-entropy loss
%                      .trainAcc    (epochs x 1) training accuracy (fraction)
%                      .testAcc     (epochs x 1) test accuracy (fraction)
%                      .epochs      scalar number of epochs run
%                      .params      struct of hyperparameters used

    % ------------------------------------------------------------------ %
    %  Hyperparameters
    % ------------------------------------------------------------------ %
    epochs        = 100;
    batchSize     = 32;
    learningRate  = 0.01;
    lrDecay       = 0.95;    % multiply lr by this factor each epoch
    lrDecayEvery  = 10;      % epochs between decay steps
    numClasses    = 5;
    numFeatures   = 6;
    hiddenSizes   = [20, 15];

    % ------------------------------------------------------------------ %
    %  Validate inputs
    % ------------------------------------------------------------------ %
    [Ntrain, nFeat] = size(trainData);
    if nFeat ~= numFeatures
        error('trainData must have %d columns. Got: %d', numFeatures, nFeat);
    end
    if numel(trainLabels) ~= Ntrain
        error('trainData rows (%d) must equal numel(trainLabels) (%d).', Ntrain, numel(trainLabels));
    end

    % ------------------------------------------------------------------ %
    %  Initialise weights (Xavier / Glorot uniform)
    % ------------------------------------------------------------------ %
    rng(123, 'twister');

    layerSizes = [numFeatures, hiddenSizes, numClasses];
    numLayers  = numel(layerSizes) - 1;

    W = cell(1, numLayers);   % Weight matrices
    b = cell(1, numLayers);   % Bias vectors

    for l = 1:numLayers
        fanIn  = layerSizes(l);
        fanOut = layerSizes(l + 1);
        limit  = sqrt(6 / (fanIn + fanOut));
        W{l}   = (2 * limit * rand(fanOut, fanIn)) - limit;
        b{l}   = zeros(fanOut, 1);
    end

    % Adam moment estimates
    mW = cellfun(@(w) zeros(size(w)), W, 'UniformOutput', false);
    vW = cellfun(@(w) zeros(size(w)), W, 'UniformOutput', false);
    mb = cellfun(@(bi) zeros(size(bi)), b, 'UniformOutput', false);
    vb = cellfun(@(bi) zeros(size(bi)), b, 'UniformOutput', false);

    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
    t = 0;   % Adam time step

    % ------------------------------------------------------------------ %
    %  One-hot encode labels
    % ------------------------------------------------------------------ %
    Ytrain = oneHotEncode(trainLabels, numClasses);   % (numClasses x Ntrain)
    Ytest  = oneHotEncode(testLabels,  numClasses);

    Xtrain = trainData';   % (numFeatures x Ntrain)
    Xtest  = testData';

    % ------------------------------------------------------------------ %
    %  Training loop
    % ------------------------------------------------------------------ %
    trainingInfo.trainLoss = zeros(epochs, 1);
    trainingInfo.testLoss  = zeros(epochs, 1);
    trainingInfo.trainAcc  = zeros(epochs, 1);
    trainingInfo.testAcc   = zeros(epochs, 1);

    numBatches = floor(Ntrain / batchSize);
    lr         = learningRate;

    for epoch = 1:epochs
        % Shuffle training set each epoch
        shuffIdx = randperm(Ntrain);
        Xs = Xtrain(:, shuffIdx);
        Ys = Ytrain(:, shuffIdx);

        % Mini-batch SGD
        for batch = 1:numBatches
            batchStart = (batch - 1) * batchSize + 1;
            batchEnd   = min(batch * batchSize, Ntrain);
            Xb = Xs(:, batchStart:batchEnd);
            Yb = Ys(:, batchStart:batchEnd);

            % Forward pass (with cached activations for backprop)
            [~, ~, activations] = forwardPassWithCache(Xb, W, b, numLayers);

            % Backward pass (backpropagation)
            [dW, db] = backwardPass(Xb, Yb, activations, W, b, numLayers);

            % Adam parameter update
            t = t + 1;
            for l = 1:numLayers
                mW{l} = beta1 * mW{l} + (1 - beta1) * dW{l};
                vW{l} = beta2 * vW{l} + (1 - beta2) * (dW{l} .^ 2);
                mb{l} = beta1 * mb{l} + (1 - beta1) * db{l};
                vb{l} = beta2 * vb{l} + (1 - beta2) * (db{l} .^ 2);

                mWhat = mW{l} / (1 - beta1^t);
                vWhat = vW{l} / (1 - beta2^t);
                mbhat = mb{l} / (1 - beta1^t);
                vbhat = vb{l} / (1 - beta2^t);

                W{l} = W{l} - lr * mWhat ./ (sqrt(vWhat) + epsilon);
                b{l} = b{l} - lr * mbhat ./ (sqrt(vbhat) + epsilon);
            end
        end

        % Learning rate decay
        if mod(epoch, lrDecayEvery) == 0
            lr = lr * lrDecay;
        end

        % Evaluate on full training and test sets
        [predTrain, probTrain, ~] = forwardPassWithCache(Xtrain, W, b, numLayers);
        [predTest,  probTest,  ~] = forwardPassWithCache(Xtest,  W, b, numLayers);

        trainingInfo.trainLoss(epoch) = crossEntropyLoss(Ytrain, probTrain);
        trainingInfo.testLoss(epoch)  = crossEntropyLoss(Ytest,  probTest);
        trainingInfo.trainAcc(epoch)  = mean(predTrain == trainLabels');
        trainingInfo.testAcc(epoch)   = mean(predTest  == testLabels');

        if mod(epoch, 10) == 0 || epoch == 1
            fprintf('Epoch %3d/%d  TrainLoss=%.4f  TestLoss=%.4f  TrainAcc=%.2f%%  TestAcc=%.2f%%  LR=%.6f\n', ...
                epoch, epochs, ...
                trainingInfo.trainLoss(epoch), trainingInfo.testLoss(epoch), ...
                trainingInfo.trainAcc(epoch) * 100, trainingInfo.testAcc(epoch) * 100, lr);
        end
    end

    % ------------------------------------------------------------------ %
    %  Package trained network
    % ------------------------------------------------------------------ %
    trainedNet.W          = W;
    trainedNet.b          = b;
    trainedNet.layerSizes = layerSizes;
    trainedNet.numLayers  = numLayers;
    trainedNet.numClasses = numClasses;

    trainingInfo.epochs = epochs;
    trainingInfo.params.learningRate = learningRate;
    trainingInfo.params.batchSize    = batchSize;
    trainingInfo.params.epochs       = epochs;
    trainingInfo.params.hiddenSizes  = hiddenSizes;
    trainingInfo.params.optimizer    = 'Adam';
    trainingInfo.params.lossFunction = 'CrossEntropy';

    fprintf('\nTraining complete.\n');
    fprintf('Final train accuracy : %.2f%%\n', trainingInfo.trainAcc(end) * 100);
    fprintf('Final test  accuracy : %.2f%%\n', trainingInfo.testAcc(end)  * 100);
end

% ======================================================================= %
%  Local helper functions
% ======================================================================= %

function [predicted, probs, activations] = forwardPassWithCache(X, W, b, numLayers)
% FORWARDPASSWITHCACHE  Forward pass storing intermediate activations for backprop.

    activations.A = cell(1, numLayers + 1);
    activations.Z = cell(1, numLayers);
    activations.A{1} = X;

    for l = 1:numLayers - 1
        Z                  = W{l} * activations.A{l} + b{l};
        activations.Z{l}   = Z;
        activations.A{l+1} = reluActivation(Z);
    end
    % Output layer
    Z                           = W{numLayers} * activations.A{numLayers} + b{numLayers};
    activations.Z{numLayers}    = Z;
    activations.A{numLayers+1}  = softmaxActivation(Z);

    probs     = activations.A{numLayers + 1};
    [~, predicted] = max(probs, [], 1);
end

function [dW, db] = backwardPass(X, Y, activations, W, b, numLayers)
% BACKWARDPASS  Compute gradients via backpropagation.
%
%   X           - (numFeatures x batchSize) input
%   Y           - (numClasses  x batchSize) one-hot targets
%   activations - struct with fields A (activations) and Z (pre-activations)

    batchSize = size(X, 2);
    dW = cell(1, numLayers);
    db = cell(1, numLayers);

    % Output layer delta: softmax + cross-entropy combined gradient
    delta = activations.A{numLayers + 1} - Y;   % (numClasses x batchSize)

    for l = numLayers:-1:1
        A_prev = activations.A{l};
        dW{l}  = (delta * A_prev') / batchSize;
        db{l}  = sum(delta, 2)     / batchSize;

        if l > 1
            dZ    = W{l}' * delta;
            delta = dZ .* reluGradient(activations.Z{l-1});
        end
    end
end

function A = reluActivation(Z)
    A = max(0, Z);
end

function G = reluGradient(Z)
    G = double(Z > 0);
end

function S = softmaxActivation(Z)
    % Numerically stable softmax
    Z = Z - max(Z, [], 1);
    E = exp(Z);
    S = E ./ sum(E, 1);
end

function loss = crossEntropyLoss(Y, Yhat)
    % Y, Yhat - (numClasses x N)
    epsilon = 1e-12;
    loss = -sum(sum(Y .* log(Yhat + epsilon))) / size(Y, 2);
end

function Y = oneHotEncode(labels, numClasses)
    % labels (N x 1) → Y (numClasses x N)
    N = numel(labels);
    Y = zeros(numClasses, N);
    for i = 1:N
        Y(labels(i), i) = 1;
    end
end
