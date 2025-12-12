function header_cols = build_output_schema(config_feat)
% BUILD_OUTPUT_SCHEMA Build CSV column header based on enabled features
%
% Inputs:
%   config_feat - Feature extraction configuration
%
% Outputs:
%   header_cols - Cell array of column names

    fprintf('Building output schema...\n');
    
    cols = {'Participant', 'Condition'};
    
    % Band power columns: Region_Band_Power
    region_names = fieldnames(config_feat.regions);
    band_names = fieldnames(config_feat.frequency_bands);
    
    if config_feat.features.band_power
        for ri = 1:length(region_names)
            for bi = 1:length(band_names)
                cols{end+1} = sprintf('%s_%s_Power', region_names{ri}, band_names{bi}); %#ok<AGROW>
            end
        end
    end
    
    % Ratio columns
    if config_feat.features.ratios
        for i = 1:length(config_feat.ratios)
            cols{end+1} = config_feat.ratios{i}; %#ok<AGROW>
        end
    end
    
    % Entropy columns: Region_EntropyType
    if config_feat.features.entropy
        for ri = 1:length(region_names)
            for ei = 1:length(config_feat.entropy_metrics)
                cols{end+1} = sprintf('%s_%s', region_names{ri}, ... 
                                     config_feat.entropy_metrics{ei}); %#ok<AGROW>
            end
        end
    end
    
    header_cols = cols;
    
    % Report schema
    fprintf('  Total columns: %d\n', length(cols));
    fprintf('    Metadata: 2 (Participant, Condition)\n');
    if config_feat.features.band_power
        fprintf('    Band power: %d\n', length(region_names) * length(band_names));
    end
    if config_feat.features.ratios
        fprintf('    Ratios: %d\n', length(config_feat.ratios));
    end
    if config_feat.features.entropy
        fprintf('    Entropy: %d\n', length(region_names) * length(config_feat.entropy_metrics));
    end
    fprintf('\n');
end
