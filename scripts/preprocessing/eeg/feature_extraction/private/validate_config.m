function config_feat = validate_config(config_feat, config_cond, config_gen)
% VALIDATE_CONFIG Validate loaded configuration files
%
% Checks that all required fields exist and have valid values.
% Throws descriptive errors if validation fails.
% Returns config_feat with converted cell arrays.

    fprintf('Validating configuration...\n');
    
    % ===== Feature extraction config =====
    required_fields = {'frequency_bands', 'regions', 'features', 'output', ...
                      'parallel', 'toolbox_paths'};
    for i = 1:length(required_fields)
        if ~isfield(config_feat, required_fields{i})
            error('Missing required field in eeg_feature_extraction.yaml: %s', ...
                  required_fields{i});
        end
    end
    
    % Check frequency bands
    bands = config_feat.frequency_bands;
    band_names = fieldnames(bands);
    if isempty(band_names)
        error('No frequency bands defined in config.');
    end
    for i = 1:length(band_names)
        band = bands.(band_names{i});
        
        % Handle cell array from yaml.ReadYaml (converts [4, 7.5] to {[4] [7.5]})
        if iscell(band)
            band = [band{:}];  % Convert {[4] [7.5]} to [4 7.5]
        end
        
        if ~isnumeric(band) || length(band) ~= 2 || band(1) >= band(2)
            error('Invalid frequency band definition: %s = %s', ...
                  band_names{i}, mat2str(band));
        end
        
        % Store back converted value
        bands.(band_names{i}) = band;
    end
    config_feat.frequency_bands = bands;  % Update config with converted values
    fprintf('  ✓ Frequency bands: %d defined\n', length(band_names));
    
    % Check regions
    regions = config_feat.regions;
    region_names = fieldnames(regions);
    if isempty(region_names)
        error('No regions defined in config.');
    end
    for i = 1:length(region_names)
        chans = regions.(region_names{i});
        if ~iscell(chans) || isempty(chans)
            error('Invalid region definition: %s', region_names{i});
        end
    end
    fprintf('  ✓ Regions: %d defined\n', length(region_names));
    
    % Check features enabled
    if ~any([config_feat.features.band_power, ...
             config_feat.features.ratios, ...
             config_feat.features.entropy])
        error('No features enabled! Enable at least one feature type.');
    end
    fprintf('  ✓ Features enabled: ');
    enabled = {};
    if config_feat.features.band_power, enabled{end+1} = 'band_power'; end
    if config_feat.features.ratios, enabled{end+1} = 'ratios'; end
    if config_feat.features.entropy, enabled{end+1} = 'entropy'; end
    fprintf('%s\n', strjoin(enabled, ', '));
    
    % Check toolbox paths
    toolboxes = fieldnames(config_feat.toolbox_paths);
    for i = 1:length(toolboxes)
        path = config_feat.toolbox_paths.(toolboxes{i});
        if ~isfolder(path)
            warning('Toolbox path not found: %s = %s', toolboxes{i}, path);
        end
    end
    
    % ===== Conditions config =====
    if ~isfield(config_cond, 'conditions')
        error('Missing ''conditions'' field in conditions.yaml');
    end
    cond_names = fieldnames(config_cond.conditions);
    if isempty(cond_names)
        error('No conditions defined in conditions.yaml');
    end
    
    % Just report condition count (include_in_analysis field doesn't exist in our YAML)
    fprintf('  ✓ Conditions: %d defined\n', length(cond_names));
    
    % ===== General config =====
    if ~isfield(config_gen, 'paths')
        error('Missing ''paths'' field in general.yaml');
    end
    required_paths = {'eeg_data', 'cleaned_eeg', 'events', 'output'};
    for i = 1:length(required_paths)
        if ~isfield(config_gen.paths, required_paths{i})
            error('Missing required path in general.yaml: %s', required_paths{i});
        end
    end
    fprintf('  ✓ Paths: all required paths defined\n');
    
    fprintf('Configuration validation passed.\n\n');
end
