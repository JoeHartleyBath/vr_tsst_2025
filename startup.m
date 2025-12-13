% MATLAB startup.m - Run automatically on MATLAB launch
% Add EEGLAB and AMICA toolboxes to MATLAB path

% Add EEGLAB
try
    addpath(genpath('c:/MATLAB/toolboxes/eeglab'));
    disp('[startup] EEGLAB loaded');
catch
    warning('[startup] Failed to load EEGLAB');
end

% Add AMICA
try
    addpath(genpath('c:/MATLAB/toolboxes/amica'));
    disp('[startup] AMICA loaded');
catch
    warning('[startup] Failed to load AMICA');
end

% Add yamlmatlab (if available)
try
    addpath(genpath('c:/MATLAB/toolboxes/yamlmatlab'));
    disp('[startup] yamlmatlab loaded');
catch
    % Optional, not critical
end

% Save paths for future sessions
savepath;

disp('[startup] VR-TSST pipeline toolboxes loaded');
