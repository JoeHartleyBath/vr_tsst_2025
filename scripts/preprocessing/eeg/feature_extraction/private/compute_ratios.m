function ratios = compute_ratios(band_power, region_names, band_names)
% COMPUTE_RATIOS Compute power ratios from pre-computed band powers
%
% Inputs:
%   band_power   - Cell array [regions × bands] of log power values
%   region_names - Cell array of region names
%   band_names   - Cell array of band names
%
% Outputs:
%   ratios - Struct with ratio values

    ratios = struct();
    
    % Helper to get band power by name
    get_bp = @(region, band) band_power{strcmp(region_names, region), strcmp(band_names, band)};
    
    % Frontal_Alpha_Asymmetry: log(RightFrontal_Alpha) - log(LeftFrontal_Alpha)
    left_alpha = get_bp('FrontalLeft', 'Alpha');
    right_alpha = get_bp('FrontalRight', 'Alpha');
    ratios.Frontal_Alpha_Asymmetry = right_alpha - left_alpha;
    
    % Alpha_Beta_Ratio: OverallFrontal_Alpha / OverallFrontal_Beta
    frontal_alpha = get_bp('OverallFrontal', 'Alpha');
    frontal_beta = get_bp('OverallFrontal', 'Beta');
    ratios.Alpha_Beta_Ratio = frontal_alpha - frontal_beta;
    
    % Theta_Beta_Ratio: FrontalMidline_Theta / FrontalMidline_Beta
    fm_theta = get_bp('FrontalMidline', 'Theta');
    fm_beta = get_bp('FrontalMidline', 'Beta');
    ratios.Theta_Beta_Ratio = fm_theta - fm_beta;
    
    % RightFrontal_Alpha
    ratios.RightFrontal_Alpha = right_alpha;
    
    % Theta_Alpha_Ratio: FrontalMidline_Theta / FrontalMidline_Alpha
    fm_alpha = get_bp('FrontalMidline', 'Alpha');
    ratios.Theta_Alpha_Ratio = fm_theta - fm_alpha;
end
