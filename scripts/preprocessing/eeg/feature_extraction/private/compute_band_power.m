function bp = compute_band_power(psd, freqs, chan_mask, band_range)
% COMPUTE_BAND_POWER Compute band power for a region
%
% Inputs:
%   psd        - Power spectral density [channels × frequencies]
%   freqs      - Frequency vector (Hz)
%   chan_mask  - Logical mask for channels in this region
%   band_range - [min_freq, max_freq] for band
%
% Outputs:
%   bp - Log10 band power (scalar)

    if ~any(chan_mask)
        bp = NaN;
        return;
    end
    
    % Frequency mask
    freq_mask = freqs >= band_range(1) & freqs <= band_range(2);
    
    if ~any(freq_mask)
        bp = NaN;
        return;
    end
    
    % Integrate power (trapezoidal)
    region_psd = psd(chan_mask, :);
    mean_psd = mean(region_psd, 1);
    bp = trapz(freqs(freq_mask), mean_psd(freq_mask));
    
    % Log transform
    bp = log10(max(bp, 1e-10));
end
