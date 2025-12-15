% summarize_iclabel_all_P26.m
% Summarize the number and proportion of ICs for each ICLabel class for P26

infile = fullfile('c:/vr_tsst_2025/output/ica_weights', 'P26_iclabel_snapshot.mat');
class_names = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other'};
if exist(infile, 'file')
    S = load(infile);
    if isfield(S, 'iclabel_results')
        if isfield(S.iclabel_results, 'classifications')
            labels = S.iclabel_results.classifications;
        else
            error('No classifications field found in iclabel_results.');
        end
    else
        error('iclabel_results not found in snapshot.');
    end
    [maxval, maxidx] = max(labels, [], 2);
    n_total = size(labels, 1);
    fprintf('ICLabel class summary for P26:\n');
    for c = 1:numel(class_names)
        n_class = sum(maxidx == c);
        fprintf('%-14s: %3d (%.1f%%)\n', class_names{c}, n_class, 100*n_class/n_total);
    end
    fprintf('\nTop 5 ICs for each class (by probability):\n');
    for c = 1:numel(class_names)
        [sorted_probs, sorted_idx] = sort(labels(:,c), 'descend');
        fprintf('\n%s:\n', class_names{c});
        for i = 1:min(5, n_total)
            fprintf('  IC %3d: %.3f\n', sorted_idx(i), sorted_probs(i));
        end
    end
else
    error('ICLabel snapshot file not found: %s', infile);
end
