# Stage 4b: Subjective Ratings Preprocessing

Processes subjective questionnaire data including affect ratings, NASA-TLX, IMI, and MPS scores.

## Usage

From project root:
```bash
python pipelines/04b_subjective_preprocessing/preprocess_subjective.py
```

Or using the stage runner:
```bash
python scripts/stages/run_stage_4b_subjective.py
```

## Input
- `data/RAW/subjective/PQs_*_compiled.csv` - Raw questionnaire responses per participant
- `data/subjective_baseline_measures.xlsx` - Baseline affect ratings
- `data/experimental_counterbalance.xlsx` - Condition order mapping

## Output
- `output/aggregated/subjective.csv` - Long-format data (one row per participant-condition)

## Features Extracted

### Affect Ratings (Baseline-Adjusted)
- Stress, Calm, Happy, Sad, Pleasure, Arousal

### NASA-TLX Workload
- Mental Demand
- Performance
- Effort

### IMI (Intrinsic Motivation Inventory)
- Interest/Enjoyment Score
- Effort/Importance Score
- Pressure/Tension Score
- Perceived Competence Score

### MPS (Multimodal Presence Scale)
- Physical Presence Score
- Social Presence Score
