function [config_feat, config_cond, config_gen, output_folder, temp_folder, output_csv] = ...
    setup_environment(params)
% SETUP_ENVIRONMENT Load configs, setup paths, create directories, initialize logging
%
% Inputs:
%   params - struct from parse_inputs()
%
% Outputs:
%   config_feat   - Feature extraction configuration
%   config_cond   - Conditions configuration
%   config_gen    - General configuration
%   output_folder - Path to output folder
%   temp_folder   - Path to temp folder
%   output_csv    - Path to output CSV file

    fprintf('=== EEG Feature Extraction Setup ===\n');
    fprintf('Start time: %s\n\n', datestr(now));
    
    % ===== Load configurations =====
    fprintf('Loading configuration files...\n');
    config_feat = yaml.ReadYaml(params.config_file);
    config_cond = yaml.ReadYaml(params.conditions_file);
    config_gen = yaml.ReadYaml(params.general_file);
    fprintf('  ✓ Loaded 3 config files\n\n');
    
    % Override parallel settings from params if provided
    if ~isempty(params.parallel_enabled)
        config_feat.parallel.enabled = params.parallel_enabled;
    end
    if ~isempty(params.num_workers)
        config_feat.parallel.num_workers = params.num_workers;
    end
    
    % ===== Validate configurations =====
    config_feat = validate_config(config_feat, config_cond, config_gen);
    
    % ===== Setup paths =====
    fprintf('Setting up paths...\n');
    
    % Add toolbox paths
    addpath(genpath(config_feat.toolbox_paths.eeglab));
    addpath(genpath(config_feat.toolbox_paths.entropy_hub));
    addpath(genpath(config_feat.toolbox_paths.utils));
    fprintf('  ✓ Added toolbox paths\n');
    
    % Determine output folder
    if ~isempty(params.output_folder)
        output_folder = params.output_folder;
    else
        % Use relative path if config path doesn't exist (cross-system compatibility)
        config_output = fullfile(config_gen.paths.output, config_feat.output.folder);
        if exist(fileparts(config_output), 'dir')
            output_folder = config_output;
        else
            % Fallback to relative path from project root
            output_folder = fullfile('output', config_feat.output.folder);
            fprintf('  ⚠ Config path not found, using relative: %s\n', output_folder);
        end
    end
    
    % Create output folder (with parents if needed)
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
        fprintf('  ✓ Created output folder: %s\n', output_folder);
    else
        fprintf('  ✓ Output folder exists: %s\n', output_folder);
    end
    
    % Setup temp folder
    temp_folder = fullfile(output_folder, 'temp');
    if ~exist(temp_folder, 'dir')
        mkdir(temp_folder);
        fprintf('  ✓ Created temp folder: %s\n', temp_folder);
    else
        fprintf('  ✓ Temp folder exists: %s\n', temp_folder);
    end
    
    % Output CSV path
    output_csv = fullfile(output_folder, config_feat.output.filename);
    fprintf('  ✓ Output file: %s\n', output_csv);
    
    % ===== Setup logging =====
    log_file = fullfile(output_folder, ...
        sprintf('extraction_log_%s.txt', datestr(now, 'yyyymmdd_HHMMSS')));
    diary(log_file);
    fprintf('\n  ✓ Log file: %s\n\n', log_file);
end
