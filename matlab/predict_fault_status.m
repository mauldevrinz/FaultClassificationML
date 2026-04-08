function [predictedClass, confidence, statusAlert] = predict_fault_status(trainedNet, sensorInput, labelNames)
% PREDICT_FAULT_STATUS  Real-time bearing fault prediction from sensor readings.
%
%   [predictedClass, confidence, statusAlert] = predict_fault_status(trainedNet, sensorInput, labelNames)
%
%   Inputs:
%     trainedNet   - struct returned by train_ann_model
%     sensorInput  - (1 x 6) normalised sensor readings in the same scale used
%                    during training (apply normParams from preprocess_data first)
%     labelNames   - cell array of class name strings {'Normal', ...}
%
%   Outputs:
%     predictedClass - integer (1-5) indicating the fault class
%     confidence     - percentage [0, 100] for the predicted class
%     statusAlert    - string:
%                        'Normal'   if confidence >= 95%
%                        'Alert'    if 80% <= confidence < 95%
%                        'Critical' if confidence < 80%
%
%   Example:
%     normSample = (rawSample - normParams.minVals) ./ normParams.range;
%     normSample = min(max(normSample, 0), 1);
%     [cls, conf, alert] = predict_fault_status(trainedNet, normSample, labelNames);
%     fprintf('Class: %s | Confidence: %.1f%% | Status: %s\n', labelNames{cls}, conf, alert);

    % ------------------------------------------------------------------ %
    %  Input validation
    % ------------------------------------------------------------------ %
    if ~isstruct(trainedNet) || ~isfield(trainedNet, 'W')
        error('trainedNet must be a struct returned by train_ann_model.');
    end
    if ~isequal(size(sensorInput), [1, 6])
        error('sensorInput must be a 1x6 row vector. Got size [%s].', num2str(size(sensorInput)));
    end
    if any(isnan(sensorInput)) || any(isinf(sensorInput))
        error('sensorInput contains NaN or Inf values.');
    end

    % ------------------------------------------------------------------ %
    %  Forward pass
    % ------------------------------------------------------------------ %
    W         = trainedNet.W;
    b         = trainedNet.b;
    numLayers = trainedNet.numLayers;

    X = sensorInput';   % (6 x 1)

    % Hidden layers with ReLU
    A = X;
    for l = 1:numLayers - 1
        Z = W{l} * A + b{l};
        A = max(0, Z);
    end

    % Output layer: numerically stable softmax
    Z     = W{numLayers} * A + b{numLayers};
    Z     = Z - max(Z);
    E     = exp(Z);
    probs = E / sum(E);   % (numClasses x 1) probability vector

    % ------------------------------------------------------------------ %
    %  Extract prediction
    % ------------------------------------------------------------------ %
    [maxProb, predictedClass] = max(probs);
    confidence = maxProb * 100;   % convert to percentage

    % ------------------------------------------------------------------ %
    %  Alert status
    % ------------------------------------------------------------------ %
    if confidence >= 95
        statusAlert = 'Normal';
    elseif confidence >= 80
        statusAlert = 'Alert';
    else
        statusAlert = 'Critical';
    end

    % ------------------------------------------------------------------ %
    %  Display result
    % ------------------------------------------------------------------ %
    if nargin >= 3 && ~isempty(labelNames) && predictedClass <= numel(labelNames)
        className = labelNames{predictedClass};
    else
        className = sprintf('Class %d', predictedClass);
    end

    fprintf('Predicted: %-20s | Confidence: %6.2f%% | Status: %s\n', ...
            className, confidence, statusAlert);

    % Show full probability distribution
    if nargin >= 3 && ~isempty(labelNames)
        fprintf('  Class probabilities:\n');
        for c = 1:numel(probs)
            name = 'Unknown';
            if c <= numel(labelNames)
                name = labelNames{c};
            end
            marker = '';
            if c == predictedClass
                marker = ' <-- predicted';
            end
            fprintf('    %-22s : %6.2f%%%s\n', name, probs(c) * 100, marker);
        end
    end
end
