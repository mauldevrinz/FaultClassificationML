function [metrics, confusionMat, plots] = evaluate_model(trainedNet, testData, testLabels, labelNames)
% EVALUATE_MODEL  Evaluate a trained ANN on test data and produce metrics/plots.
%
%   [metrics, confusionMat, plots] = evaluate_model(trainedNet, testData, testLabels, labelNames)
%
%   Inputs:
%     trainedNet - struct returned by train_ann_model (fields: W, b, numLayers, numClasses)
%     testData   - (Ntest x 6) normalised test features
%     testLabels - (Ntest x 1) integer labels (1-5)
%     labelNames - cell array of class name strings {'Normal', ...}
%
%   Outputs:
%     metrics      - struct with fields:
%                      .overallAccuracy  - scalar
%                      .precision        - (numClasses x 1)
%                      .recall           - (numClasses x 1)
%                      .f1Score          - (numClasses x 1)
%                      .classAccuracy    - (numClasses x 1)
%     confusionMat - (numClasses x numClasses) confusion matrix
%     plots        - struct with figure handles:
%                      .confusionFig   - confusion matrix figure
%                      .metricsBarFig  - per-class bar chart figure

    % ------------------------------------------------------------------ %
    %  Forward pass on test set
    % ------------------------------------------------------------------ %
    numClasses = trainedNet.numClasses;
    numLayers  = trainedNet.numLayers;
    W          = trainedNet.W;
    b          = trainedNet.b;

    Xtest   = testData';   % (numFeatures x Ntest)
    [predictedLabels, probMatrix] = forwardPassEval(Xtest, W, b, numLayers);
    predictedLabels = predictedLabels';   % (Ntest x 1)

    % Confidence scores for each sample
    confidenceScores = max(probMatrix, [], 1)';   % (Ntest x 1)

    % ------------------------------------------------------------------ %
    %  Confusion matrix
    % ------------------------------------------------------------------ %
    confusionMat = zeros(numClasses, numClasses);
    for i = 1:numel(testLabels)
        trueC = testLabels(i);
        predC = predictedLabels(i);
        confusionMat(trueC, predC) = confusionMat(trueC, predC) + 1;
    end

    % ------------------------------------------------------------------ %
    %  Per-class metrics: Precision, Recall, F1-Score
    % ------------------------------------------------------------------ %
    precision     = zeros(numClasses, 1);
    recall        = zeros(numClasses, 1);
    f1Score       = zeros(numClasses, 1);
    classAccuracy = zeros(numClasses, 1);

    for c = 1:numClasses
        TP = confusionMat(c, c);
        FP = sum(confusionMat(:, c)) - TP;   % predicted as c but not c
        FN = sum(confusionMat(c, :)) - TP;   % actual c but predicted otherwise
        TN = sum(confusionMat(:)) - TP - FP - FN;

        precision(c) = TP / max(TP + FP, 1);
        recall(c)    = TP / max(TP + FN, 1);
        denom        = precision(c) + recall(c);
        if denom > 0
            f1Score(c) = 2 * precision(c) * recall(c) / denom;
        end
        classAccuracy(c) = (TP + TN) / max(TP + TN + FP + FN, 1);
    end

    overallAccuracy = sum(diag(confusionMat)) / sum(confusionMat(:));

    metrics.overallAccuracy = overallAccuracy;
    metrics.precision       = precision;
    metrics.recall          = recall;
    metrics.f1Score         = f1Score;
    metrics.classAccuracy   = classAccuracy;
    metrics.predictedLabels = predictedLabels;
    metrics.confidence      = confidenceScores;

    % ------------------------------------------------------------------ %
    %  Print results
    % ------------------------------------------------------------------ %
    fprintf('\n========== MODEL EVALUATION REPORT ==========\n');
    fprintf('Overall Accuracy: %.2f%%\n\n', overallAccuracy * 100);
    fprintf('%-20s  %10s  %10s  %10s  %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy');
    fprintf('%s\n', repmat('-', 1, 65));
    for c = 1:numClasses
        fprintf('%-20s  %10.4f  %10.4f  %10.4f  %10.4f\n', ...
            labelNames{c}, precision(c), recall(c), f1Score(c), classAccuracy(c));
    end
    fprintf('%s\n', repmat('=', 1, 65));

    % Macro-averaged metrics
    fprintf('Macro Precision: %.4f\n', mean(precision));
    fprintf('Macro Recall   : %.4f\n', mean(recall));
    fprintf('Macro F1-Score : %.4f\n', mean(f1Score));
    fprintf('%s\n\n', repmat('=', 1, 65));

    % ------------------------------------------------------------------ %
    %  Confusion matrix figure
    % ------------------------------------------------------------------ %
    plots.confusionFig = figure('Name', 'Confusion Matrix', 'Visible', 'off');
    imagesc(confusionMat);
    colormap('hot');
    colorbar;
    title('Confusion Matrix — Bearing Fault Classification', 'FontSize', 14);
    xlabel('Predicted Class', 'FontSize', 12);
    ylabel('True Class',      'FontSize', 12);
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', labelNames, ...
             'YTick', 1:numClasses, 'YTickLabel', labelNames);
    xtickangle(20);

    % Annotate cells
    for r = 1:numClasses
        for col = 1:numClasses
            text(col, r, num2str(confusionMat(r, col)), ...
                'HorizontalAlignment', 'center', 'FontSize', 11, ...
                'Color', 'cyan', 'FontWeight', 'bold');
        end
    end

    % ------------------------------------------------------------------ %
    %  Per-class metrics bar chart
    % ------------------------------------------------------------------ %
    plots.metricsBarFig = figure('Name', 'Per-Class Metrics', 'Visible', 'off');
    barData = [precision, recall, f1Score] * 100;
    bar(barData);
    set(gca, 'XTickLabel', labelNames);
    xtickangle(20);
    legend({'Precision (%)', 'Recall (%)', 'F1-Score (%)'}, 'Location', 'southeast');
    title('Per-Class Performance Metrics', 'FontSize', 14);
    ylabel('Metric Value (%)', 'FontSize', 12);
    xlabel('Fault Class',      'FontSize', 12);
    ylim([0, 110]);
    grid on;
end

% ======================================================================= %
%  Local helper
% ======================================================================= %

function [predicted, probs] = forwardPassEval(X, W, b, numLayers)
% FORWARDPASSEVAL  Feed-forward inference (no activation caching).
    A = X;
    for l = 1:numLayers - 1
        Z = W{l} * A + b{l};
        A = max(0, Z);   % ReLU
    end
    Z     = W{numLayers} * A + b{numLayers};
    Z     = Z - max(Z, [], 1);   % numerically stable softmax
    E     = exp(Z);
    probs = E ./ sum(E, 1);
    [~, predicted] = max(probs, [], 1);
end
