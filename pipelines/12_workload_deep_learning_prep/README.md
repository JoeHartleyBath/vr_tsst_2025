# Pipeline 12: Workload Deep Learning Preparation

## Purpose
This pipeline transforms the cleaned multimodal data (EEG/ECG/GSR) into tensor-ready formats for the Deep Learning model.

## key Steps
1.  **Windowing**: Slice continuous data into X-second windows.
2.  **Feature Selection**: Select specific subset of features agreed upon in `DATA_FORMAT.md`.
3.  **Formatting**: Export as Numpy arrays or Parquet files for PyTorch/TensorFlow dataloaders.

## Input
*   `output/final_data.rds` or `output/cleaned_eeg/`

## Output
*   `output/adaptive_workload/training_data/`
