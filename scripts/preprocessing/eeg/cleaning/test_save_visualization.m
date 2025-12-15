% Test save_visualization for pipeline
EEG = pop_importdata('dataformat','array','data',randn(128,1000));
EEG.srate = 125;
EEG.chanlocs = readlocs('config/chanlocs/NA-271.elc');
folder = 'output/vis/P99';
filename = 'test_save.png';
save_visualization(EEG, folder, filename);