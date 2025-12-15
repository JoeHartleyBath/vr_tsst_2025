% summarize_iclabel_brain_P26.m
% Summarize the number and proportion of ICs labeled as 'Brain' in the ICLabel snapshot for P26

infile = fullfile('c:/vr_tsst_2025/output/ica_weights', 'P26_iclabel_snapshot.mat');
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
    % ICLabel: columns are [Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other]
    [maxval, maxidx] = max(labels, [], 2);
    brain_idx = (maxidx == 1);
    n_brain = sum(brain_idx);
    n_total = size(labels, 1);
    fprintf('ICLabel summary for P26:\n');
    fprintf('Total ICs: %d\n', n_total);
    fprintf('Brain components: %d (%.1f%%)\n', n_brain, 100*n_brain/n_total);
    % Optionally, print the top 10 brain-labeled ICs with their probabilities
    brain_probs = labels(:,1);
    [sorted_probs, sorted_idx] = sort(brain_probs, 'descend');
    fprintf('Top 10 ICs with highest Brain probability:\n');
    for i = 1:min(10, n_total)
        fprintf('IC %d: %.3f\n', sorted_idx(i), sorted_probs(i));
    end
else
    error('ICLabel snapshot file not found: %s', infile);
end
