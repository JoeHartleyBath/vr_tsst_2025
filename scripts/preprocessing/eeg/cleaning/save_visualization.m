function save_visualization(EEG, folder, filename)
% Save EEG visualization to PNG (headless-compatible)
if ~exist(folder, 'dir')
    mkdir(folder);
end
try
    fig = figure('Visible', 'off');
    % Plot mean EEG across all channels for a quick diagnostic
    t = (0:size(EEG.data,2)-1) / EEG.srate;
    plot(t, mean(double(EEG.data),1));
    xlabel('Time (s)'); ylabel('Mean Amplitude (uV)');
    title('Mean EEG across channels');
    grid on;
    set(fig, 'PaperPositionMode', 'auto');
    exportgraphics(fig, fullfile(folder, filename), 'Resolution', 150);
    close(fig);
catch ME
    warning('Could not save visualization %s: %s', filename, ME.message);
end
end
