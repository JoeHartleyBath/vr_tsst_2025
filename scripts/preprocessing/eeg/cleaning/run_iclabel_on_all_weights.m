% run_iclabel_on_all_weights.m
% Load AMICA weights for all participants, apply ICLabel, and print summary results

clear;

%% Configuration
projectRoot = 'c:/vr_tsst_2025';
ica_weights_dir = fullfile(projectRoot, 'output', 'ica_weights');
cleaned_eeg_dir = fullfile(projectRoot, 'output', 'cleaned_eeg');

%% Find all AMICA weight files
weight_files = dir(fullfile(ica_weights_dir, '*_amica_weights.mat'));
if isempty(weight_files)
    error('No AMICA weight files found in %s', ica_weights_dir);
end

fprintf('Found %d AMICA weight files\n', length(weight_files));
fprintf('Starting ICLabel analysis...\n\n');
fprintf('========================================\n\n');

%% Process each participant
for f = 1:length(weight_files)
    % Extract participant ID
    [~, fname, ~] = fileparts(weight_files(f).name);
    pid = regexprep(fname, '_amica_weights$', '');
    
    fprintf('=== %s ===\n', pid);
    
    % Define file paths
    amica_weights_path = fullfile(ica_weights_dir, weight_files(f).name);
    cleaned_eeg_path = fullfile(cleaned_eeg_dir, sprintf('%s_cleaned.set', pid));
    
    % Check if cleaned EEG exists
    if ~isfile(cleaned_eeg_path)
        fprintf('  ✗ Cleaned EEG file not found\n\n');
        continue;
    end
    
    try
        % Load cleaned EEG
        EEG = pop_loadset('filename', sprintf('%s_cleaned.set', pid), ...
                          'filepath', cleaned_eeg_dir);
        
        % Load AMICA weights
        loaded = load(amica_weights_path);
        if ~isfield(loaded, 'amica_weights')
            fprintf('  ✗ Invalid weight file format\n\n');
            continue;
        end
        
        weights = loaded.amica_weights.weights;
        sphere = loaded.amica_weights.sphere;
        
        % Apply ICA weights if not already present
        if ~isfield(EEG, 'icaweights') || isempty(EEG.icaweights)
            EEG.icaweights = weights;
            EEG.icasphere = sphere;
            EEG = eeg_checkset(EEG);
        end
        
        % Run ICLabel
        EEG = iclabel(EEG);
        
        % Extract results
        if isfield(EEG, 'etc') && isfield(EEG.etc, 'ic_classification') && ...
           isfield(EEG.etc.ic_classification, 'ICLabel')
            
            classifications = EEG.etc.ic_classification.ICLabel.classifications;
            labels = EEG.etc.ic_classification.ICLabel.classes;
            
            % Get primary classification for each component
            [max_prob, primary_class] = max(classifications, [], 2);
            n_total = size(classifications, 1);
            
            % Count by category
            fprintf('  Total ICs: %d\n', n_total);
            for i = 1:length(labels)
                count = sum(primary_class == i);
                pct = 100 * count / n_total;
                if count > 0
                    fprintf('    %s: %d (%.1f%%)\n', labels{i}, count, pct);
                end
            end
            
            % Show top 5 brain components
            brain_probs = classifications(:, 1);
            [sorted_probs, sorted_idx] = sort(brain_probs, 'descend');
            
            fprintf('  Top 5 Brain components:\n');
            for i = 1:min(5, n_total)
                ic_num = sorted_idx(i);
                fprintf('    IC%02d: %.3f (%s)\n', ...
                        ic_num, sorted_probs(i), labels{primary_class(ic_num)});
            end
            
        else
            fprintf('  ✗ ICLabel results not found\n');
        end
        
    catch ME
        fprintf('  ✗ Error: %s\n', ME.message);
    end
    
    fprintf('\n');
end

fprintf('========================================\n');
fprintf('ICLabel analysis complete!\n');
