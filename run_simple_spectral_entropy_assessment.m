%% Simple Spectral Entropy Impact Assessment
% Quick assessment of spectral entropy issue impact without recomputation

clear; clc; close all;

fprintf('Spectral Entropy Impact Assessment\n');
fprintf('==================================\n\n');

% Add validation scripts to path
addpath('scripts/validation');

try
    % Run the simplified assessment
    compare_spectral_entropy_simple();
    
catch ME
    fprintf('\nError during analysis:\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end