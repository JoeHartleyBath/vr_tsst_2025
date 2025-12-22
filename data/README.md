# Data Folder Structure

## Experimental Design Files

### `experimental_counterbalance.xlsx`
**Purpose:** Defines the experimental counterbalancing scheme for VR-TSST conditions.

**Used by:** 
- `scripts/utils/data_prep_helpers.R` - Maps participants to their assigned condition order
- Baseline correction and condition assignment in R preprocessing pipeline

**Structure:**
- Columns: Participant, Round1, Round2, Round3, Round4
- Each round specifies which condition variant the participant experienced
- Conditions include: Stress Subtraction, Stress Addition, Control Subtraction, Control Addition

**Important:** This file is critical for proper baseline correction. Do not modify unless updating the experimental design.

---

## Raw Data Folders

### `raw/eeg/`
Raw EEG data files (.xdf format)

### `raw/metadata/`
Physiological sensor data (HR, GSR, Eye tracking) merged with event markers
- Format: `P##.csv` (one file per participant)
- Contains: Timestamps, Study_Phase, sensor readings, behavioral responses

### `raw/events/`
Event markers and timing information

### `raw/subjective/`
Subjective ratings data (if stored separately from metadata)

---

**Note:** Raw data files are large and not tracked in git. Ensure backups are maintained.

**Last Updated:** December 12, 2025
