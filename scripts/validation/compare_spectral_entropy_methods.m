function compare_spectral_entropy_methods()
% COMPARE_SPECTRAL_ENTROPY_METHODS
% Compare original (flawed) vs corrected spectral entropy calculation
% to assess impact on downstream analyses
%
% Original issue: Spectral entropy computed on 0.5-49 Hz range after 1-49 Hz filtering
% Fixed version: Spectral entropy computed only on 1-49 Hz range

fprintf('=== Spectral Entropy Method Comparison ===\n\n');

% Configuration - Use same loading pattern as in project
config_path = fullfile('config', 'general.yaml');
fprintf('Looking for config at: %s\n', config_path);
fprintf('Config file exists: %s\n', string(exist(config_path, 'file')));
fprintf('Current directory: %s\n', pwd);

config = struct();

if exist('ReadYaml', 'file') == 2
    fprintf('Trying ReadYaml...\n');
    try
        config = ReadYaml(config_path);
        fprintf('Config loaded via ReadYaml (%d fields)\n', length(fieldnames(config)));
    catch ME
        fprintf('ReadYaml failed: %s\n', ME.message);
        % Fall through to SimpleYAML
    end
end
if isempty(fieldnames(config)) && exist('SimpleYAML', 'file') == 2
    fprintf('Trying SimpleYAML...\n');
    try
        config = SimpleYAML.readFile(config_path);
        fprintf('Config loaded via SimpleYAML fallback (%d fields)\n', length(fieldnames(config)));
    catch ME
        fprintf('SimpleYAML failed: %s\n', ME.message);
        % Continue with empty config 
    end
end

if isempty(fieldnames(config))
    error('Failed to load config file from: %s', config_path);
end

% Load existing data with current spectral entropy values
fprintf('Loading existing data with current spectral entropy values...\n');
final_data_path = fullfile(config.paths.output, 'final_data.csv');
if ~exist(final_data_path, 'file')
    error('Final data file not found: %s', final_data_path);
end

data_table = readtable(final_data_path);
participants = unique(data_table.participant_id);

% Get spectral entropy columns from existing data
specent_cols = data_table.Properties.VariableNames(...
    contains(data_table.Properties.VariableNames, '_specent_precond') & ...
    ~contains(data_table.Properties.VariableNames, '_Z'));

fprintf('Found %d spectral entropy features: %s\n', length(specent_cols), strjoin(specent_cols, ', '));

% Initialize comparison results
results = struct();
results.participants = participants;
results.regions = extractRegionsFromColumns(specent_cols);
results.original_values = [];
results.corrected_values = [];
results.differences = [];
results.correlations = [];
results.effect_sizes = [];

fprintf('\nRecomputing spectral entropy for %d participants...\n', length(participants));

% Process each participant
original_all = [];
corrected_all = [];
participant_ids = [];
condition_ids = [];

for p_idx = 1:length(participants)
    pid = participants(p_idx);
    fprintf('Processing P%d (%d/%d)...\n', pid, p_idx, length(participants));
    
    try
        % Get participant's data from table
        p_data = data_table(data_table.participant_id == pid, :);
        
        % Load cleaned EEG data
        matname = sprintf('P%d_cleaned.mat', pid);
        eeg_path = fullfile(config.paths.cleaned_eeg, matname);
        
        if ~exist(eeg_path, 'file')
            fprintf('  Warning: No cleaned EEG data found for P%d, skipping.\n', pid);
            continue;
        end
        
        load(eeg_path, 'cleaned_data');
        
        % Load channel labels from .set file
        filtered_set_path = fullfile(config.paths.eeg_data, 'filtered', sprintf('P%02d_filtered.set', pid));
        if exist(filtered_set_path, 'file')
            EEG_meta = pop_loadset(filtered_set_path);
            chan_labels = {EEG_meta.chanlocs.labels};
        else
            fprintf('  Warning: No .set file found for P%d, using default labels.\n', pid);
            chan_labels = arrayfun(@(x) sprintf('Ch%d', x), 1:size(cleaned_data.data, 1), 'UniformOutput', false);
        end
        
        % Process each condition for this participant
        conditions = fieldnames(cleaned_data.data);
        
        for c_idx = 1:length(conditions)
            cond = conditions{c_idx};
            segment = cleaned_data.data.(cond);
            
            if isempty(segment)
                continue;
            end
            
            % Get sampling rate
            if isfield(cleaned_data, 'srate')
                srate = cleaned_data.srate;
            else
                srate = 125; % Default based on your pipeline
            end
            
            % Compute PSD using same method as pipeline
            [psd, freqs] = calc_psd(segment, srate);
            
            % Extract original spectral entropy values from table
            cond_data = p_data(strcmp(p_data.condition, mapConditionName(cond)), :);
            if isempty(cond_data)
                continue;
            end
            
            % Compare spectral entropy calculations by region
            for s_idx = 1:length(specent_cols)
                col_name = specent_cols{s_idx};
                region_name = extractRegionFromColumn(col_name);
                
                % Get channel mask for this region  
                chan_mask = getChannelMask(region_name, chan_labels);
                
                if ~any(chan_mask)
                    continue;
                end
                
                % Original value from existing data
                original_val = cond_data{1, col_name};
                
                % Compute corrected spectral entropy (exclude < 1 Hz)
                corrected_val = compute_spectral_entropy_corrected(psd, freqs, chan_mask);
                
                % Store results
                original_all(end+1) = original_val;
                corrected_all(end+1) = corrected_val;
                participant_ids(end+1) = pid;
                condition_ids{end+1} = cond;
            end
        end
        
    catch ME
        fprintf('  Error processing P%d: %s\n', pid, ME.message);
    end
end

fprintf('\nAnalyzing differences...\n');

% Remove NaN values for analysis
valid_idx = ~isnan(original_all) & ~isnan(corrected_all);
original_clean = original_all(valid_idx);
corrected_clean = corrected_all(valid_idx);

if length(original_clean) < 10
    error('Insufficient valid data points for comparison (%d)', length(original_clean));
end

% Statistical analysis
results.n_comparisons = length(original_clean);
results.correlation = corr(original_clean', corrected_clean');
results.mean_original = mean(original_clean);
results.mean_corrected = mean(corrected_clean);
results.mean_difference = mean(corrected_clean - original_clean);
results.std_difference = std(corrected_clean - original_clean);
results.max_abs_difference = max(abs(corrected_clean - original_clean));

% Effect size (Cohen's d)
pooled_std = sqrt((var(original_clean) + var(corrected_clean)) / 2);
results.cohens_d = results.mean_difference / pooled_std;

% Paired t-test
[~, results.p_value, ~, stats] = ttest(corrected_clean, original_clean);
results.t_statistic = stats.tstat;

% Classification of effect size
if abs(results.cohens_d) < 0.2
    effect_size_desc = 'negligible';
elseif abs(results.cohens_d) < 0.5
    effect_size_desc = 'small';
elseif abs(results.cohens_d) < 0.8
    effect_size_desc = 'medium';
else
    effect_size_desc = 'large';
end

% Print results
fprintf('\n=== RESULTS ===\n');
fprintf('Valid comparisons: %d\n', results.n_comparisons);
fprintf('Correlation (r): %.4f\n', results.correlation);
fprintf('Mean original: %.4f\n', results.mean_original);
fprintf('Mean corrected: %.4f\n', results.mean_corrected);
fprintf('Mean difference: %.4f ± %.4f\n', results.mean_difference, results.std_difference);
fprintf('Max absolute difference: %.4f\n', results.max_abs_difference);
fprintf('Effect size (Cohen''s d): %.4f (%s)\n', results.cohens_d, effect_size_desc);
fprintf('Paired t-test: t=%.3f, p=%.4f\n', results.t_statistic, results.p_value);

% Interpretation and recommendations
fprintf('\n=== INTERPRETATION ===\n');
if results.correlation > 0.95
    fprintf('✓ Very high correlation (r=%.3f) - methods are highly related\n', results.correlation);
elseif results.correlation > 0.9
    fprintf('✓ High correlation (r=%.3f) - methods are strongly related\n', results.correlation);
elseif results.correlation > 0.7
    fprintf('⚠ Moderate correlation (r=%.3f) - some differences exist\n', results.correlation);
else
    fprintf('⚠ Low correlation (r=%.3f) - substantial differences exist\n', results.correlation);
end

if abs(results.cohens_d) < 0.2
    fprintf('✓ Negligible effect size (d=%.3f) - differences are minimal\n', results.cohens_d);
elseif abs(results.cohens_d) < 0.5
    fprintf('⚠ Small effect size (d=%.3f) - minor practical impact\n', results.cohens_d);
else
    fprintf('⚠ Medium to large effect size (d=%.3f) - substantial differences\n', results.cohens_d);
end

fprintf('\n=== RECOMMENDATIONS ===\n');
if results.correlation > 0.95 && abs(results.cohens_d) < 0.2
    fprintf('✓ RE-RUNNING NOT NEEDED: Differences are minimal and unlikely to affect SVM results\n');
elseif results.correlation > 0.9 && abs(results.cohens_d) < 0.3
    fprintf('⚠ RE-RUNNING OPTIONAL: Small differences present but likely minimal impact on conclusions\n');
else
    fprintf('⚠ RE-RUNNING RECOMMENDED: Substantial differences may affect analysis results\n');
end

% Save results
output_dir = fullfile(config.paths.output, 'validation');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save detailed results
save(fullfile(output_dir, 'spectral_entropy_comparison.mat'), 'results', 'original_all', 'corrected_all');

% Create visualization
create_comparison_plots(original_clean, corrected_clean, results, output_dir);

fprintf('\nResults saved to: %s\n', output_dir);
fprintf('Plots saved as PNG files in the same directory.\n');

end

function corrected_entropy = compute_spectral_entropy_corrected(psd, freqs, chan_mask)
% Compute spectral entropy excluding frequencies < 1 Hz
    try
        % Only use frequencies >= 1 Hz (avoid filter artifacts)
        valid_freq_mask = freqs >= 1.0;
        if ~any(valid_freq_mask) || ~any(chan_mask)
            corrected_entropy = NaN;
            return;
        end
        
        % Select channels and valid frequencies
        psd_region = psd(chan_mask, valid_freq_mask);
        
        % Compute entropy for each channel
        entropies = zeros(size(psd_region, 1), 1);
        for ch = 1:size(psd_region, 1)
            p = psd_region(ch, :);
            p = p / sum(p); % Normalize to probability
            p(p <= 0) = eps; % Avoid log(0)
            entropies(ch) = -sum(p .* log2(p));
        end
        
        % Average valid entropies
        valid = ~isnan(entropies) & ~isinf(entropies);
        if any(valid)
            corrected_entropy = mean(entropies(valid));
        else
            corrected_entropy = NaN;
        end
    catch
        corrected_entropy = NaN;
    end
end

function [psd, freqs] = calc_psd(data, srate)
% Same PSD calculation as in the pipeline
    window_length = min(2 * srate, size(data, 2));
    overlap = round(window_length / 2);
    nfft = 2^nextpow2(window_length);
    
    [psd, freqs] = pwelch(data', hamming(window_length), overlap, nfft, srate);
    psd = psd'; % Transpose to [channels × freqs]
end

function chan_mask = getChannelMask(region_name, chan_labels)
% Get channel mask for a brain region
    % Define region mappings (same as in pipeline)
    regions = struct();
    regions.FrontalLeft = {'L1','L2','L3','L4','LL1','LL2','LL3','LB1','LC1','LD1','LD2','LC2','LD3','LE1'};
    regions.FrontalRight = {'R1','R2','R3','R4','RR1','RR2','RR3','RB1','RC1','RD1','RD2','RC2','RD3','RE1'};
    regions.OverallFrontal = [regions.FrontalLeft, regions.FrontalRight, {'Z1','Z2','Z3'}];
    regions.TemporalLeft = {'LC3','LC4','LC5','LC6','LD4','LD5','LD6','LE2','LE3','LM'};
    regions.TemporalRight = {'RC3','RC4','RC5','RC6','RD4','RD5','RD6','RE2','RE3','RM'};
    regions.Temporal = [regions.TemporalLeft, regions.TemporalRight];
    regions.Central = {'Z4','Z5','Z6','RB2','RB3','RC7','LB2','LB3','LC7'};
    regions.ParietalLeft = {'LL4','LL5','LL6','LL7','LG1','LG2','LH1','LH2','LH3','LI1','LI2'};
    regions.ParietalRight = {'RR4','RR5','RR6','RR7','RG1','RG2','RH1','RH2','RH3','RI1','RI2'};
    regions.Parietal = [regions.ParietalLeft, regions.ParietalRight];
    regions.FrontalMidline = {'Z1','Z2','Z3'};
    
    % Map abbreviated names to full names
    region_map = containers.Map(...
        {'fl','fr','f','tleft','tright','t','c','pleft','pright','p','fm'}, ...
        {'FrontalLeft','FrontalRight','OverallFrontal','TemporalLeft','TemporalRight',...
         'Temporal','Central','ParietalLeft','ParietalRight','Parietal','FrontalMidline'});
    
    if region_map.isKey(region_name)
        full_region_name = region_map(region_name);
    else
        full_region_name = region_name;
    end
    
    if isfield(regions, full_region_name)
        chan_mask = ismember(chan_labels, regions.(full_region_name));
    else
        chan_mask = false(size(chan_labels));
    end
end

function region_name = extractRegionFromColumn(col_name)
% Extract region name from column name like 'eeg_fl_specent_precond'
    parts = split(col_name, '_');
    if length(parts) >= 3
        region_name = parts{2}; % e.g., 'fl' from 'eeg_fl_specent_precond'
    else
        region_name = '';
    end
end

function regions = extractRegionsFromColumns(specent_cols)
% Extract all unique region names
    regions = {};
    for i = 1:length(specent_cols)
        region = extractRegionFromColumn(specent_cols{i});
        if ~ismember(region, regions)
            regions{end+1} = region;
        end
    end
end

function condition_name = mapConditionName(internal_name)
% Map internal condition names to those in the CSV
    % This may need adjustment based on your specific naming
    condition_map = containers.Map(...
        {'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', 'HighStress_LowCog_Task',...
         'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', 'LowStress_LowCog_Task'}, ...
        {'High Stress - High Cog', 'High Stress - High Cog', 'High Stress - Low Cog',...
         'Low Stress - High Cog', 'Low Stress - High Cog', 'Low Stress - Low Cog'});
    
    if condition_map.isKey(internal_name)
        condition_name = condition_map(internal_name);
    else
        condition_name = internal_name;
    end
end

function create_comparison_plots(original, corrected, results, output_dir)
% Create visualization plots
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Scatter plot
    subplot(2,2,1);
    scatter(original, corrected, 20, 'filled', 'Alpha', 0.6);
    hold on;
    % Unity line
    min_val = min([original, corrected]);
    max_val = max([original, corrected]);
    plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
    xlabel('Original Spectral Entropy');
    ylabel('Corrected Spectral Entropy');
    title(sprintf('Scatter Plot (r=%.3f)', results.correlation));
    grid on;
    axis equal;
    
    % Difference histogram
    subplot(2,2,2);
    differences = corrected - original;
    histogram(differences, 30, 'FaceAlpha', 0.7);
    xlabel('Difference (Corrected - Original)');
    ylabel('Count');
    title(sprintf('Difference Distribution (Mean=%.4f)', results.mean_difference));
    grid on;
    
    % Bland-Altman plot
    subplot(2,2,3);
    means = (original + corrected) / 2;
    scatter(means, differences, 20, 'filled', 'Alpha', 0.6);
    hold on;
    yline(results.mean_difference, 'r-', 'LineWidth', 2);
    yline(results.mean_difference + 1.96*results.std_difference, 'r--');
    yline(results.mean_difference - 1.96*results.std_difference, 'r--');
    xlabel('Mean of Methods');
    ylabel('Difference (Corrected - Original)');
    title('Bland-Altman Plot');
    grid on;
    
    % Summary statistics
    subplot(2,2,4);
    axis off;
    stats_text = {
        sprintf('N = %d', results.n_comparisons)
        sprintf('Correlation = %.4f', results.correlation)
        sprintf('Mean difference = %.4f', results.mean_difference)
        sprintf('Cohen''s d = %.4f', results.cohens_d)
        sprintf('p-value = %.4f', results.p_value)
        ''
        'Effect Size Interpretation:'
        sprintf('%.4f = %s effect', results.cohens_d, ...
                getEffectSizeLabel(abs(results.cohens_d)))
    };
    text(0.1, 0.9, stats_text, 'FontSize', 12, 'VerticalAlignment', 'top');
    
    sgtitle('Spectral Entropy Method Comparison', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save plot
    saveas(gcf, fullfile(output_dir, 'spectral_entropy_comparison.png'));
    close(gcf);
    
    fprintf('Comparison plot saved.\n');
end

function label = getEffectSizeLabel(d)
    if d < 0.2
        label = 'negligible';
    elseif d < 0.5
        label = 'small';
    elseif d < 0.8
        label = 'medium';
    else
        label = 'large';
    end
end