function se = compute_sample_entropy(data)
% COMPUTE_SAMPLE_ENTROPY Compute sample entropy for multi-channel data
%
% Inputs:
%   data - Multi-channel data [channels × timepoints]
%
% Outputs:
%   se - Average sample entropy across channels

    try
        m = 2; % Embedding dimension
        r = 0.2 * std(data(:)); % Tolerance
        
        entropies = zeros(size(data, 1), 1);
        for ch = 1:size(data, 1)
            sig = data(ch, :);
            sig = (sig - mean(sig)) / std(sig); % Normalize
            entropies(ch) = SampEn(m, r, sig);
        end
        
        % Average valid entropies
        valid = ~isnan(entropies) & ~isinf(entropies);
        if any(valid)
            se = mean(entropies(valid));
        else
            se = NaN;
        end
    catch
        se = NaN;
    end
end
