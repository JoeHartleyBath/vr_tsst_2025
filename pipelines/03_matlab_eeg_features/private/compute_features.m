function feats = compute_features(data, srate, frequency_bands, regions, chan_labels, config)
% COMPUTE_FEATURES Extract all configured features from EEG data window
%
% Inputs:
%   data            - EEG data matrix [channels × timepoints]
%   srate           - Sampling rate (Hz)
%   frequency_bands - Struct with frequency band definitions
%   regions         - Struct with region channel lists
%   chan_labels     - Cell array of channel labels
%   config          - Feature extraction configuration
%
% Outputs:
%   feats - Struct with fields:
%     .band_power - Cell array [regions × bands] of log power values
%     .ratios     - Struct with ratio values (if enabled)
%     .entropy    - Cell array [regions × entropy_types] (if enabled)

    feats = struct();
    
    % ===== Compute PSD =====
    [psd, freqs] = calc_psd(data, srate);
    
    % ===== Band power per region =====
    region_names = fieldnames(regions);
    band_names = fieldnames(frequency_bands);
    
    feats.band_power = cell(length(region_names), length(band_names));
    
    for ri = 1:length(region_names)
        % Get channel mask for this region
        region_chans = regions.(region_names{ri});
        chan_mask = ismember(chan_labels, region_chans);
        
        for bi = 1:length(band_names)
            band_range = frequency_bands.(band_names{bi});
            
            % Compute band power
            bp = compute_band_power(psd, freqs, chan_mask, band_range);
            feats.band_power{ri, bi} = bp;
        end
    end
    
    % ===== Power ratios =====
    if config.features.ratios
        feats.ratios = compute_ratios(feats.band_power, region_names, band_names);
    end
    
    % ===== Entropy =====
    if config.features.entropy
        feats.entropy = cell(length(region_names), length(config.entropy_metrics));
        
        for ri = 1:length(region_names)
            region_chans = regions.(region_names{ri});
            chan_mask = ismember(chan_labels, region_chans);
            region_data = data(chan_mask, :);
            
            if ~isempty(region_data)
                for ei = 1:length(config.entropy_metrics)
                    metric = config.entropy_metrics{ei};
                    
                    if strcmp(metric, 'SampleEntropy')
                        feats.entropy{ri, ei} = compute_sample_entropy(region_data);
                    elseif strcmp(metric, 'SpectralEntropy')
                        feats.entropy{ri, ei} = compute_spectral_entropy(psd(chan_mask, :), freqs);
                    end
                end
            else
                % No channels in region
                for ei = 1:length(config.entropy_metrics)
                    feats.entropy{ri, ei} = NaN;
                end
            end
        end
    end
end
