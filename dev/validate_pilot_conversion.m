%% Validate Pilot Conversion (P01-P03)
% Quick sanity checks on converted .set files
%
% Checks:
%  1. File exists (.set and .mat)
%  2. Channel count = 128
%  3. Sample rate = 500 Hz
%  4. Data shape = (128, samples)
%  5. Amplitude reasonable (NOT in mV, should be µV)
%  6. Events recorded

clear; clc

base = fileparts(pwd);
participants = [1, 2, 3];

fprintf("=" * 80 + "\n");
fprintf("PILOT CONVERSION VALIDATION (P01-P03)\n");
fprintf("=" * 80 + "\n\n");

for p = participants
    p_str = sprintf("P%02d", p);
    set_file = fullfile(base, "output/sets", [p_str ".set"]);
    mat_file = fullfile(base, "output/sets", [p_str ".mat"]);
    
    fprintf("[%s] Validating...\n", p_str);
    
    % Check file exists
    if ~isfile(set_file)
        fprintf("  ERROR: .set file does not exist\n\n");
        continue
    end
    if ~isfile(mat_file)
        fprintf("  ERROR: .mat file does not exist\n\n");
        continue
    end
    fprintf("  ✓ Both .set and .mat files exist\n");
    
    % Load .mat
    try
        m = load(mat_file, "EEG");
        EEG = m.EEG;
    catch ME
        fprintf("  ERROR loading .mat: %s\n\n", ME.message);
        continue
    end
    
    % Validate structure
    nbchan = EEG.nbchan;
    pnts = EEG.pnts;
    srate = EEG.srate;
    
    fprintf("  ✓ Loaded EEG struct\n");
    
    % Check channels
    if nbchan ~= 128
        fprintf("  ✗ Channel count mismatch: expected 128, got %d\n", nbchan);
    else
        fprintf("  ✓ Channel count: %d\n", nbchan);
    end
    
    % Check sample rate
    if srate ~= 500
        fprintf("  ✗ Sample rate mismatch: expected 500 Hz, got %d Hz\n", srate);
    else
        fprintf("  ✓ Sample rate: %d Hz\n", srate);
    end
    
    % Check data shape
    if size(EEG.data, 1) ~= nbchan
        fprintf("  ✗ Data shape mismatch: expected (%d, N), got (%d, %d)\n", nbchan, size(EEG.data, 1), size(EEG.data, 2));
    else
        fprintf("  ✓ Data shape: (%d, %d samples)\n", nbchan, pnts);
    end
    
    % Check amplitude
    mean_amp = mean(abs(EEG.data(:)));
    max_amp = max(abs(EEG.data(:)));
    
    if mean_amp < 1.0
        fprintf("  ⚠  Suspiciously small amplitude (mean %.4f, likely unscaled mV)\n", mean_amp);
    elseif mean_amp > 1000
        fprintf("  ⚠  Suspiciously large amplitude (mean %.1f)\n", mean_amp);
    else
        fprintf("  ✓ Amplitude reasonable: mean %.2f µV, max %.2f µV\n", mean_amp, max_amp);
    end
    
    % Check events
    n_events = length(EEG.event);
    if n_events == 0
        fprintf("  ⚠  No events found\n");
    else
        fprintf("  ✓ Events: %d recorded\n", n_events);
    end
    
    fprintf("\n");
end

fprintf("=" * 80 + "\n");
fprintf("VALIDATION COMPLETE\n");
fprintf("=" * 80 + "\n");
