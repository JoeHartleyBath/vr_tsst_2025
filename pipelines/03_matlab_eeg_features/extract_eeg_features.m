function extract_eeg_features(varargin)
% EXTRACT_EEG_FEATURES Extract EEG features for VR-TSST study
%
% Streamlined feature extraction with configuration-driven design.
% Computes aggregated features per condition:
%   • Band power (per region × frequency band)
%   • Power ratios (frontal asymmetry, alpha/beta, theta/beta, etc.)
%   • Entropy (sample & spectral entropy per region)
%
% IMPORTANT: Run from project root with path added
%   PowerShell command to run with 8 workers:
%   matlab -batch "cd('c:\vr_tsst_2025'); addpath('pipelines/03_matlab_eeg_features'); extract_eeg_features('num_workers', 8)"
%
%   Or from within MATLAB:
%   cd c:\vr_tsst_2025
%   addpath('pipelines/03_matlab_eeg_features')
%   extract_eeg_features('num_workers', 8)
%
% Usage:
%   extract_eeg_features()                                    % Default: all 48 participants
%   extract_eeg_features('participants', [1 5 10])           % Specific participants
%   extract_eeg_features('force_reprocess', true)            % Ignore existing temp files
%   extract_eeg_features('output_folder', 'custom/path')     % Custom output location
%   extract_eeg_features('config_file', 'custom.yaml')       % Custom config file
%   extract_eeg_features('parallel', false)                  % Disable parallel processing
%   extract_eeg_features('num_workers', 8)                   % Override worker count (recommended: 8)
%
% Examples:
%   % Process first 10 participants only
%   extract_eeg_features('participants', 1:10)
%
%   % Reprocess all participants (ignore resume)
%   extract_eeg_features('force_reprocess', true)
%
%   % Run single-threaded for debugging
%   extract_eeg_features('participants', 1, 'parallel', false)
%
% Dependencies:
%   - EEGLAB (EEG processing)
%   - EntropyHub or SampEn function (entropy calculations)
%   - yaml.loadFile (YAML parser)
%   - config/eeg_feature_extraction.yaml (feature config)
%   - config/conditions.yaml (condition definitions)
%   - config/general.yaml (paths)
%
% Output:
%   CSV file with one row per (participant, condition) pair
%   Temp files created during processing for resume capability
%
% See also: parse_inputs, setup_environment, process_all_participants

    %% ====================================================================
    %  SETUP
    %% ====================================================================
    
    % Parse input parameters
    params = parse_inputs(varargin{:});
    
    % Setup environment: load configs, validate, create folders, setup logging
    [config_feat, config_cond, config_gen, output_folder, temp_folder, output_csv] = ...
        setup_environment(params);
    
    %% ====================================================================
    %  BUILD OUTPUT SCHEMA
    %% ====================================================================
    
    header_cols = build_output_schema(config_feat);
    
    % Write header to output CSV
    fid = fopen(output_csv, 'w');
    if fid == -1
        error('Could not create output file: %s', output_csv);
    end
    fprintf(fid, '%s\n', strjoin(header_cols, ','));
    fclose(fid);
    fprintf('Header written to: %s\n\n', output_csv);
    
    %% ====================================================================
    %  DETERMINE WORK (RESUME LOGIC)
    %% ====================================================================
    
    participants_to_process = determine_work(params.participants, temp_folder, ...
                                            params.force_reprocess);
    
    %% ====================================================================
    %  PARALLEL PROCESSING SETUP
    %% ====================================================================
    
    setup_parallel_pool(config_feat);
    
    %% ====================================================================
    %  PROCESS PARTICIPANTS
    %% ====================================================================
    
    process_all_participants(participants_to_process, temp_folder, ...
                            config_feat, config_cond, config_gen, header_cols);
    
    %% ====================================================================
    %  MERGE RESULTS
    %% ====================================================================
    
    merged_count = merge_temp_files(params.participants, temp_folder, output_csv);
    
    %% ====================================================================
    %  CLEANUP
    %% ====================================================================
    
    keep_temp = false; % Set to true for debugging
    cleanup_temp_files(params.participants, temp_folder, keep_temp);
    
    %% ====================================================================
    %  SUMMARY
    %% ====================================================================
    
    fprintf('\n=== EXTRACTION COMPLETE ===\n');
    fprintf('  Processed: %d participants\n', length(participants_to_process));
    fprintf('  Merged: %d participants\n', merged_count);
    fprintf('  Output: %s\n', output_csv);
    fprintf('  Completion time: %s\n', datestr(now));
    
    % Close log file
    diary off;
    
    fprintf('\n✓ Feature extraction complete!\n');
end
