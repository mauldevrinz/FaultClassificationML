function [sensorData, labels, labelNames] = generate_sensor_data(numSamples)
% GENERATE_SENSOR_DATA  Generate physics-based sensor data for bearing fault classification.
%
%   [sensorData, labels, labelNames] = generate_sensor_data(numSamples)
%
%   Inputs:
%     numSamples  - Total number of samples to generate (default: 5000)
%
%   Outputs:
%     sensorData  - (numSamples x 6) matrix of sensor readings:
%                     Col 1: Vibration (mm/s)
%                     Col 2: Temperature (°C)
%                     Col 3: Pressure (bar)
%                     Col 4: Current (A)
%                     Col 5: RPM (rev/min)
%                     Col 6: Operating Hours
%     labels      - (numSamples x 1) integer labels (1-5)
%     labelNames  - Cell array of class names {'Normal','Bearing Wear',
%                   'Misalignment','Overheating','Combined Faults'}
%
%   Fault Classes (numSamples/5 samples each):
%     1. Normal        - Healthy bearing operation
%     2. Bearing Wear  - Gradual degradation over time
%     3. Misalignment  - Mechanical misalignment with spikes
%     4. Overheating   - Exponential temperature rise
%     5. Combined      - Simultaneous multiple faults
%
%   Physics-based relationships:
%     - Higher RPM amplifies vibration and increases temperature
%     - Bearing wear causes gradual parameter increase over operating hours
%     - Misalignment produces symmetric high-frequency vibration spikes
%     - Overheating follows exponential temperature rise with load dependency
%     - Current correlates with mechanical load and friction

    if nargin < 1
        numSamples = 5000;
    end

    % Validate input
    if numSamples < 5 || mod(numSamples, 5) ~= 0
        error('numSamples must be a positive multiple of 5. Got: %d', numSamples);
    end

    % Seed for reproducibility
    rng(42, 'twister');

    labelNames = {'Normal', 'Bearing Wear', 'Misalignment', 'Overheating', 'Combined Faults'};
    nPerClass  = numSamples / 5;   % samples per class

    sensorData = zeros(numSamples, 6);
    labels     = zeros(numSamples, 1);

    idx = 1;  % global row index

    % ------------------------------------------------------------------ %
    %  CLASS 1: Normal Operation
    %  Vibration: 0-5 mm/s   Temp: 40-50°C   Pressure: 5-6 bar
    %  Current: 15-17 A      RPM: 1400-1600  Hours: 0-2000
    % ------------------------------------------------------------------ %
    for i = 1:nPerClass
        rpm     = 1400 + 200 * rand();
        hours   = 2000 * rand();
        rpmFact = rpm / 1500;                      % normalised RPM factor

        vibration   = (2.5 + 2.5 * rand()) * rpmFact + 0.2 * randn();
        temperature = 44 + 6 * rand() + 0.01 * hours * rpmFact;
        pressure    = 5.2 + 0.6 * rand() + 0.05 * randn();
        current     = 15.5 + 1.5 * rand() + 0.1 * rpmFact;
        operating_hours = hours;

        vibration   = max(0, vibration);
        temperature = max(35, temperature);
        pressure    = max(4.5, pressure);
        current     = max(13, current);

        sensorData(idx, :) = [vibration, temperature, pressure, current, rpm, operating_hours];
        labels(idx)        = 1;
        idx = idx + 1;
    end

    % ------------------------------------------------------------------ %
    %  CLASS 2: Bearing Wear
    %  Progressive degradation; severity increases with operating hours.
    %  Vibration: 5-15 mm/s   Temp: 45-65°C   Pressure: 5.5-7.5 bar
    %  Current: 15-20 A       RPM: 1200-1700  Hours: 500-8000
    % ------------------------------------------------------------------ %
    for i = 1:nPerClass
        rpm             = 1200 + 500 * rand();
        hours           = 500 + 7500 * rand();
        rpmFact         = rpm / 1500;
        degradation     = hours / 8000;            % 0 (new) → 1 (worn)

        vibration   = 5 + 10 * degradation + 3 * rand() * rpmFact ...
                      + 0.5 * randn();
        temperature = 45 + 20 * degradation + 5 * rand() + 0.3 * randn();
        pressure    = 5.5 + 2 * degradation + 0.5 * rand();
        current     = 15 + 5 * degradation + 1.5 * rand();
        operating_hours = hours;

        vibration   = max(3, vibration);
        temperature = max(43, temperature);
        pressure    = max(5, pressure);
        current     = max(14, current);

        sensorData(idx, :) = [vibration, temperature, pressure, current, rpm, operating_hours];
        labels(idx)        = 2;
        idx = idx + 1;
    end

    % ------------------------------------------------------------------ %
    %  CLASS 3: Misalignment
    %  High-frequency vibration spikes; asymmetric loading.
    %  Vibration: 10-25 mm/s   Temp: 50-70°C   Pressure: 6-8.5 bar
    %  Current: 17-22 A        RPM: 1300-1800  Hours: 0-5000
    % ------------------------------------------------------------------ %
    for i = 1:nPerClass
        rpm             = 1300 + 500 * rand();
        hours           = 5000 * rand();
        rpmFact         = rpm / 1500;

        % Symmetric spikes modelled by abs(sin) pattern
        spikePhase  = 2 * pi * rand();
        spikeAmp    = 8 + 7 * rand();
        vibration   = 10 + spikeAmp * abs(sin(spikePhase)) * rpmFact ...
                      + 1.5 * abs(randn());
        temperature = 50 + 20 * rand() + 0.5 * rpmFact * randn();
        pressure    = 6 + 2.5 * rand() + 0.3 * abs(sin(spikePhase));
        current     = 17 + 5 * rand() + 0.5 * rpmFact;
        operating_hours = hours;

        vibration   = max(8, vibration);
        temperature = max(48, temperature);
        pressure    = max(5.5, pressure);
        current     = max(16, current);

        sensorData(idx, :) = [vibration, temperature, pressure, current, rpm, operating_hours];
        labels(idx)        = 3;
        idx = idx + 1;
    end

    % ------------------------------------------------------------------ %
    %  CLASS 4: Overheating
    %  Exponential temperature rise; elevated current from extra friction.
    %  Temp: 60-95°C   Vibration: 3-20 mm/s   Current: 19-24 A
    %  RPM: 1000-2000  Hours: 100-6000
    % ------------------------------------------------------------------ %
    for i = 1:nPerClass
        rpm         = 1000 + 1000 * rand();
        hours       = 100 + 5900 * rand();
        rpmFact     = rpm / 1500;
        loadFact    = 0.5 + 0.5 * rand();         % random load factor

        % Exponential temperature rise driven by load and RPM
        tempBase    = 60 + 35 * (1 - exp(-loadFact * rpmFact));
        temperature = tempBase + 5 * randn();
        vibration   = 3 + 17 * rand() * rpmFact;
        pressure    = 5 + 2.5 * rand() + 0.1 * loadFact;
        current     = 19 + 5 * loadFact * rpmFact + randn();
        operating_hours = hours;

        temperature = max(58, temperature);
        vibration   = max(2, vibration);
        pressure    = max(4.8, pressure);
        current     = max(18, current);

        sensorData(idx, :) = [vibration, temperature, pressure, current, rpm, operating_hours];
        labels(idx)        = 4;
        idx = idx + 1;
    end

    % ------------------------------------------------------------------ %
    %  CLASS 5: Combined Faults
    %  Superposition of bearing-wear + misalignment + overheating signatures.
    %  All parameters elevated and more erratic.
    % ------------------------------------------------------------------ %
    for i = 1:nPerClass
        rpm         = 1100 + 800 * rand();
        hours       = 1000 + 7000 * rand();
        rpmFact     = rpm / 1500;
        degradation = hours / 8000;
        loadFact    = 0.6 + 0.4 * rand();

        % Wear component
        vibWear  = 5 + 8 * degradation;
        tempWear = 45 + 15 * degradation;

        % Misalignment spike component
        spikePhase = 2 * pi * rand();
        vibSpike   = 6 * abs(sin(spikePhase)) * rpmFact;

        % Overheating component
        tempHeat   = 20 * (1 - exp(-loadFact * rpmFact));

        vibration   = vibWear + vibSpike + 2 * abs(randn());
        temperature = tempWear + tempHeat + 3 * randn();
        pressure    = 6 + 3 * rand() + 0.5 * abs(sin(spikePhase));
        current     = 18 + 6 * loadFact * rpmFact + 1.5 * randn();
        operating_hours = hours;

        vibration   = max(8, vibration);
        temperature = max(60, temperature);
        pressure    = max(5.5, pressure);
        current     = max(17, current);

        sensorData(idx, :) = [vibration, temperature, pressure, current, rpm, operating_hours];
        labels(idx)        = 5;
        idx = idx + 1;
    end

    % Shuffle all samples to mix classes
    shuffleIdx = randperm(numSamples);
    sensorData = sensorData(shuffleIdx, :);
    labels     = labels(shuffleIdx);

    fprintf('Generated %d sensor samples across %d fault classes.\n', numSamples, numel(labelNames));
    for c = 1:numel(labelNames)
        fprintf('  Class %d (%s): %d samples\n', c, labelNames{c}, sum(labels == c));
    end
end
