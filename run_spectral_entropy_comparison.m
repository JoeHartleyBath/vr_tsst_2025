%% Spectral Entropy Method Comparison Runner
% Quick script to run the spectral entropy comparison analysis
%
% This script compares the original (potentially flawed) spectral entropy
% calculation with a corrected version to assess if re-running analyses
% is necessary.

clear; clc; close all;

fprintf('Spectral Entropy Method Comparison\n');
fprintf('==================================\n\n');

% Add paths
addpath('scripts/validation');
addpath('scripts/utils');

% Add private functions individually to avoid MATLAB path warnings
private_dir = fullfile('pipelines', '03_matlab_eeg_features', 'private');
if exist(private_dir, 'dir')
    private_files = dir(fullfile(private_dir, '*.m'));
    for i = 1:length(private_files)
        func_content = fileread(fullfile(private_dir, private_files(i).name));
        if contains(func_content, 'function') && ~contains(private_files(i).name, 'compute_spectral_entropy')
            % Don't add the original function to avoid conflicts
            continue;
        end
    end
    % Add the directory anyway but suppress warnings
    warning('off', 'MATLAB:rmpath:DirNotFound');
    warning('off', 'MATLAB:addpath:DirInPath'); 
    addpath(private_dir);
    warning('on', 'MATLAB:rmpath:DirNotFound');
    warning('on', 'MATLAB:addpath:DirInPath');
end

% Check if EEGLAB is in path
if ~exist('pop_loadset', 'file')
    fprintf('Adding EEGLAB to path...\n');
    try
        addpath('C:/Program Files/MATLAB/R2025b/toolbox/eeglab2025.1.0');
        eeglab nogui;
    catch
        warning('Could not find EEGLAB. Please ensure it is in your MATLAB path.');
    end
end

% Check if yaml toolbox is available
if ~exist('ReadYaml', 'file') && ~exist('SimpleYAML', 'file')
    fprintf('YAML reading functions not found. Please ensure yamlmatlab or SimpleYAML is available.\n');
end

try
    % Run the comparison
    compare_spectral_entropy_methods();
    
    fprintf('\n=== Comparison Complete ===\n');
    fprintf('Check the results above and the plots in output/validation/\n');
    
catch ME
    fprintf('\nError during analysis:\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end