function [psd, freqs] = calc_psd(data, srate)
% CALC_PSD Calculate power spectral density using Welch's method
%
% Inputs:
%   data  - EEG data matrix [channels × timepoints]
%   srate - Sampling rate (Hz)
%
% Outputs:
%   psd   - Power spectral density [channels × frequencies]
%   freqs - Frequency vector (Hz)

    window_length = min(2 * srate, size(data, 2));
    overlap = round(window_length / 2);
    nfft = 2^nextpow2(window_length);
    
    [psd, freqs] = pwelch(data', hamming(window_length), overlap, nfft, srate);
    psd = psd'; % Transpose to [channels × freqs]
end
