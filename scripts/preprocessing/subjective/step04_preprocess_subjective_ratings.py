"""
Subjective Ratings Preprocessing Script

Processes subjective ratings from VR-TSST study including:
- Affect ratings (stress, calm, happy, sad, valence, arousal)
- NASA-TLX workload measures
- IMI (Intrinsic Motivation Inventory) scores
- MPS (Multimodal Presence Scale) scores

Outputs long-format CSV with baseline-adjusted ratings and composite scores.

Usage:
    python scripts/preprocessing/subjective/step04_preprocess_subjective_ratings.py

Requirements:
    - Must be run from project root: c:/vr_tsst_2025/
    - Input data: data/raw/subjective/PQs_*_compiled.csv
    - Baseline data: data/processed/VR-TSST Baseline Measures.xlsx
    - Counterbalance: data/processed/VR-TSST Counterbalance sheet.xlsx

Outputs:
    - output/aggregated/subjective.csv (long format: one row per participant-condition)

Author: VR-TSST Project
Date: December 2025
"""

import os
import pandas as pd
import sys

# Define base directory (project root)
base_directory = r"c:\vr_tsst_2025"

# Validate base directory exists
if not os.path.exists(base_directory):
    raise FileNotFoundError(f"Base directory not found: {base_directory}")

# Change to base directory for consistent relative paths
os.chdir(base_directory)

print(f"Working directory: {os.getcwd()}")


# Load CSV file with error handling
def load_csv(file_path):
    """Load CSV file with validation."""
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
        return None

# Subtract baseline data
def subtract_baseline(data, baseline_data, subtract_columns, participant_number):
    """
    Subtract baseline affect ratings from in-task ratings.
    
    Args:
        data: DataFrame with in-task ratings
        baseline_data: DataFrame with pre-task baseline ratings
        subtract_columns: List of columns to baseline-adjust
        participant_number: Participant ID
    
    Returns:
        DataFrame with baseline-adjusted columns added
    """
    try:
        baseline = baseline_data[baseline_data["P#"] == participant_number]
        
        if baseline.empty:
            print(f"WARNING: No baseline data found for participant {participant_number}")
            return data

        for j in range(len(data)):  # Process all rows
            for col in subtract_columns:
                # Handle 'Pleasure' column name mapping
                if col == "Pleasure":
                    if "Valence" in data.columns and "Pleasure" in baseline.columns:
                        data.loc[j, "Pleasure_Baseline"] = data.loc[j, "Valence"] - baseline["Pleasure"].values[0]
                    else:
                        print(f"WARNING: Missing 'Valence' or baseline 'Pleasure' for P{participant_number}")
                else:
                    # Standard baseline adjustment
                    if col in data.columns and col in baseline.columns:
                        data.loc[j, f"{col}_Baseline"] = data.loc[j, col] - baseline[col].values[0]
                    else:
                        if col not in data.columns:
                            print(f"WARNING: '{col}' column not found in P{participant_number} data")
                        if col not in baseline.columns:
                            print(f"WARNING: '{col}' column not found in baseline data")
                            
    except KeyError as e:
        print(f"KeyError: {e} for participant {participant_number}")
    except IndexError as e:
        print(f"IndexError: {e} for participant {participant_number}")
    except Exception as e:
        print(f"Unexpected error: {e} for participant {participant_number}")
    
    return data

# Calculate IMI and MPS composite scores
def calculate_scores(df):
    """
    Calculate composite scores for IMI and MPS questionnaires.
    
    IMI: Intrinsic Motivation Inventory (Interest, Effort, Pressure, Competence)
    MPS: Multimodal Presence Scale (Physical, Social presence)
    """
    # IMI Interest/Enjoyment subscale
    df['IMI_Interest_Score'] = (
        df['IMI_Effort1'] + df['IMI_Effort2'] + 
        (8 - df['IMI_Effort3']) + (8 - df['IMI_Effort4']) + 
        df['IMI_Effort5']
    )
    
    # IMI Effort/Importance subscale
    df['IMI_Effort_Score'] = (
        df['IMI_Effort1'] + (8 - df['IMI_Effort2']) + 
        df['IMI_Effort3'] + df['IMI_Effort4'] + (8 - df['IMI_Effort5'])
    )
    
    # IMI Pressure/Tension subscale
    df['IMI_Pressure_Score'] = (
        (8 - df['IMI_Pressure1']) + df['IMI_Pressure2'] + 
        (8 - df['IMI_Pressure3']) + df['IMI_Pressure4'] + df['IMI_Pressure5']
    )
    
    # IMI Perceived Competence subscale
    df['IMI_Competence_Score'] = (
        df['IMI_Competence1'] + df['IMI_Competence2'] + 
        df['IMI_Competence3'] + df['IMI_Competence4'] + df['IMI_Competence5'] + 
        (8 - df[' IMI_Competence6'])
    )
    
    # MPS Physical Presence subscale (average of 5 items)
    df['MPS_Phys_Presence_Score'] = (
        df['MpqPhys2'] + df['MpqPhys3'] + df['MpqPhys4'] + df['MpqPhys5'] + df['MpqPhys10']
    ) / 5
    
    # MPS Social Presence subscale (average of 5 items)
    df['MPS_Social_Presence_Score'] = (
        df['MpqSocial1'] + df['MpqSocial2'] + df['MpqSocial3'] + df['MpqSocial4'] + df['MpqSocial5']
    ) / 5
    
    return df


# Condition name mapping (matches existing pipeline convention)
CONDITION_MAP = {
    'Calm Addition': 'LowStress_LowCog_Task',
    'Calm Subtraction': 'LowStress_HighCog_Task',
    'Stress Addition': 'HighStress_LowCog_Task',
    'Stress Subtraction': 'HighStress_HighCog_Task'
}


def main():
    """Main processing pipeline."""
    
    print("\n" + "="*60)
    print("VR-TSST Subjective Ratings Preprocessing")
    print("="*60 + "\n")
    
    # Configuration
    participants = list(range(1, 49))
    subtract_columns = ['Stress', 'Calm', 'Happy', 'Sad', 'Pleasure', 'Arousal']
    columns_nasa = ["NASA_Mental", "NASA_Performance", "NASA_Effort"]
    
    # Load baseline and counterbalance data
    print("Loading baseline and counterbalance data...")
    
    baseline_path = "data/processed/VR-TSST Baseline Measures.xlsx"
    counterbalance_path = "data/processed/VR-TSST Counterbalance sheet.xlsx"
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline data not found: {baseline_path}")
    if not os.path.exists(counterbalance_path):
        raise FileNotFoundError(f"Counterbalance data not found: {counterbalance_path}")
    
    baseline_data = pd.read_excel(baseline_path)
    counterbalance_data = pd.read_excel(counterbalance_path)
    
    print(f"  ✓ Loaded baseline data: {len(baseline_data)} participants")
    print(f"  ✓ Loaded counterbalance data: {len(counterbalance_data)} participants\n")
    
    # Create list to collect all rows (long format)
    all_rows = []
    
    # Process each participant
    print("Processing participants...")
    successful_count = 0
    failed_participants = []
    
    for participant_number in participants:
        try:
            # Construct input file path
            file_path = f"data/raw/subjective/PQs_{participant_number}_compiled.csv"
            
            # Load participant data
            df = load_csv(file_path)
            if df is None:
                print(f"  ✗ P{participant_number:02d}: Missing data file")
                failed_participants.append(participant_number)
                continue
            
            # Baseline adjustment
            df = subtract_baseline(df, baseline_data, subtract_columns, participant_number)
            
            # Get condition order for this participant
            condition_info = counterbalance_data[counterbalance_data['Participant'] == participant_number]
            
            if condition_info.empty:
                print(f"  ✗ P{participant_number:02d}: No counterbalance data")
                failed_participants.append(participant_number)
                continue
            
            participant_conditions = [condition_info[f'Round {i+1}'].values[0] for i in range(4)]
            
            # Calculate composite scores
            df = calculate_scores(df)
            
            # Convert to long format: one row per condition
            for idx, condition_short in enumerate(participant_conditions):
                
                if idx >= len(df):
                    print(f"  ⚠ P{participant_number:02d}: Missing data for condition {idx+1}")
                    continue
                
                # Map to full condition name
                condition_full = CONDITION_MAP.get(condition_short, condition_short)
                
                # Build row dictionary
                row = {
                    'Participant_ID': participant_number,
                    'Condition': condition_full,
                }
                
                # Add baseline-adjusted affect ratings
                for metric in subtract_columns:
                    col_name = f'{metric}_Baseline'
                    if col_name in df.columns:
                        row[metric] = df.loc[idx, col_name]
                    else:
                        row[metric] = None
                
                # Add NASA-TLX ratings
                for col in columns_nasa:
                    if col in df.columns:
                        row[col] = df.loc[idx, col]
                    else:
                        row[col] = None
                
                # Add composite scores
                score_columns = [
                    'IMI_Interest_Score', 'IMI_Effort_Score', 'IMI_Pressure_Score',
                    'IMI_Competence_Score', 'MPS_Phys_Presence_Score', 'MPS_Social_Presence_Score'
                ]
                for score_col in score_columns:
                    if score_col in df.columns:
                        row[score_col] = df.loc[idx, score_col]
                    else:
                        row[score_col] = None
                
                all_rows.append(row)
            
            successful_count += 1
            print(f"  ✓ P{participant_number:02d}: Processed {len(participant_conditions)} conditions")
            
        except Exception as e:
            print(f"  ✗ P{participant_number:02d}: Error - {e}")
            failed_participants.append(participant_number)
    
    # Create final DataFrame in long format
    print(f"\nCreating final dataset...")
    final_df = pd.DataFrame(all_rows)
    
    # Ensure output directory exists
    output_path = "output/aggregated/subjective.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV (overwrite mode)
    final_df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Successfully processed: {successful_count} participants")
    print(f"✗ Failed: {len(failed_participants)} participants")
    if failed_participants:
        print(f"  Failed IDs: {failed_participants}")
    print(f"\nOutput saved to: {output_path}")
    print(f"  - Shape: {final_df.shape} (rows × columns)")
    print(f"  - Format: Long (one row per participant-condition)")
    print(f"  - Expected rows: {successful_count * 4} (4 conditions per participant)")
    print(f"  - Actual rows: {len(final_df)}")
    print("="*60 + "\n")
    
    return final_df


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)
