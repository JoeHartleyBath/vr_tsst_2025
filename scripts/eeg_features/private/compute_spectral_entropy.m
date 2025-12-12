function spec_ent = compute_spectral_entropy(psd, ~)
% COMPUTE_SPECTRAL_ENTROPY Compute spectral entropy from PSD
%
% Inputs:
%   psd   - Power spectral density [channels × frequencies]
%   freqs - Frequency vector (unused, kept for consistency)
%
% Outputs:
%   spec_ent - Average spectral entropy across channels

    try
        entropies = zeros(size(psd, 1), 1);
        for ch = 1:size(psd, 1)
            p = psd(ch, :);
            p = p / sum(p); % Normalize to probability
            p(p <= 0) = eps; % Avoid log(0)
            entropies(ch) = -sum(p .* log2(p));
        end
        
        % Average valid entropies
        valid = ~isnan(entropies) & ~isinf(entropies);
        if any(valid)
            spec_ent = mean(entropies(valid));
        else
            spec_ent = NaN;
        end
    catch
        spec_ent = NaN;
    end
end
