import os
import pandas as pd

# Correctly define the base directory path using raw string
base_directory = r"C:\Users\Joe\Documents\EEG-TSST"

# Check if base directory exists before changing the directory
if not os.path.exists(base_directory):
    raise FileNotFoundError(f"Base directory not found: {base_directory}")

os.chdir(base_directory)

# Load CSV file with error handling
def load_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

# Subtract baseline data
def subtract_baseline(data, baseline_data, substract_columns, participant_number):
    try:
        baseline = baseline_data[baseline_data["P#"] == participant_number]
        
        if baseline.empty:
            print(f"No baseline data found for participant {participant_number}")
            return data

        for j in range(4):
            for col in substract_columns:
                if col == "Pleasure":
                    if "Valence" in data.columns:
                        data.loc[j, "Pleasure_Baseline"] = data.loc[j, "Valence"] - baseline["Pleasure"].values[0]
                    else:
                        print(f"'Valence' column not found in participant {participant_number}'s data")
                else:
                    if col in data.columns:
                        data.loc[j, f"{col}_Baseline"] = data.loc[j, col] - baseline[col].values[0]
                    else:
                        print(f"'{col}' column not found in participant {participant_number}'s data")
    except KeyError as e:
        print(f"KeyError: {e} for participant {participant_number}")
    except IndexError as e:
        print(f"IndexError: {e} for participant {participant_number}")
    except Exception as e:
        print(f"Unexpected error: {e} for participant {participant_number}")
    
    return data

# Calculate IMI and MPS scores
def calculate_scores(df):
    df['IMI_Interest_Score'] = (df['IMI_Effort1'] + df['IMI_Effort2'] + 
                                (8 - df['IMI_Effort3']) + (8 - df['IMI_Effort4']) + 
                                df['IMI_Effort5'])
    df['IMI_Effort_Score'] = (df['IMI_Effort1'] + (8 - df['IMI_Effort2']) + 
                              df['IMI_Effort3'] + df['IMI_Effort4'] + (8 - df['IMI_Effort5']))
    df['IMI_Pressure_Score'] = ((8 - df['IMI_Pressure1']) + df['IMI_Pressure2'] + 
                                (8 - df['IMI_Pressure3']) + df['IMI_Pressure4'] + df['IMI_Pressure5'])
    df['IMI_Competence_Score'] = (df['IMI_Competence1'] + df['IMI_Competence2'] + 
                                  df['IMI_Competence3'] + df['IMI_Competence4'] + df['IMI_Competence5'] + 
                                  (8 - df[' IMI_Competence6']))
    df['MPS_Phys_Presence_Score'] = (df['MpqPhys2'] + df['MpqPhys3'] + df['MpqPhys4'] + df['MpqPhys5'] + df['MpqPhys10']) / 5
    df['MPS_Social_Presence_Score'] = (df['MpqSocial1'] + df['MpqSocial2'] + df['MpqSocial3'] + df['MpqSocial4'] + df['MpqSocial5']) / 5
    return df

# Calculate averages for each condition
def calculate_averages(df, participant_conditions):
    averages = {}
    score_columns = ['IMI_Interest_Score', 'IMI_Effort_Score', 'IMI_Pressure_Score', 
                     'IMI_Competence_Score', 'MPS_Phys_Presence_Score', 'MPS_Social_Presence_Score']
    for score in score_columns:
        for idx, cond in enumerate(participant_conditions):
            avg_column_name = f"{cond} {score}_Avg"
            if score in df.columns:
                averages[avg_column_name] = df.loc[idx, score]
    return averages

# Define lists and columns
participants = list(range(1, 49))
substract_columns = ['Stress', 'Calm', 'Happy', 'Sad', 'Pleasure', 'Arousal']
columns_nasa = ["NASA_Mental", "NASA_Performance", "NASA_Effort"]
condition_order = ['Calm Addition', 'Calm Subtraction', 'Stress Addition', 'Stress Subtraction']

# Load baseline and counterbalance data
baseline_path = os.path.join(base_directory, "Main_Study_Data_Processed", "VR-TSST Baseline Measures.xlsx")
baseline_data = pd.read_excel(baseline_path)
counterbalance_path = os.path.join(base_directory, "Main_Study_Data_Processed", "VR-TSST Counterbalance sheet.xlsx")
counterbalance_data = pd.read_excel(counterbalance_path)

# Create a combined CSV file for analysis
combined_columns = ['PN'] + [
    f'{cond} {metric}' for metric in substract_columns for cond in condition_order
] + [
    f'{cond} {col}' for col in columns_nasa for cond in condition_order
] + [
    f'{cond} {score}_Avg' for score in ['IMI_Interest_Score', 'IMI_Effort_Score', 'IMI_Pressure_Score', 
                                        'IMI_Competence_Score', 'MPS_Phys_Presence_Score', 'MPS_Social_Presence_Score'] 
    for cond in condition_order
]
combined_df = pd.DataFrame(columns=combined_columns)

# Process each participant
for participant_number in participants:
    try:
        file_path = os.path.join(base_directory, "Main_Study_Data_Raw/In_VR_Questions", f"PQs_{participant_number}_compiled.csv")
        df = load_csv(file_path)
        
        if df is None:
            print(f"Skipping participant {participant_number} due to missing file.")
            continue
        
        df = subtract_baseline(df, baseline_data, substract_columns, participant_number)
        
        condition_info = counterbalance_data[counterbalance_data['Participant'] == participant_number]
        participant_conditions = [condition_info[f'Round {i+1}'].values[0] for i in range(4)]
        
        # Add the 'Condition' column
        df["Condition"] = None
        for idx, cond in enumerate(participant_conditions):
            df.at[idx, 'Condition'] = cond
        
        df["Condition"] = pd.Categorical(df["Condition"], categories=condition_order, ordered=True)
        
        # Calculate scores
        df = calculate_scores(df)
        
        # Calculate averages
        averages = calculate_averages(df, participant_conditions)
        
        temp_df = pd.DataFrame(index=[0], columns=combined_columns)
        temp_df['PN'] = participant_number
        
        for metric in substract_columns:
            for idx, cond in enumerate(participant_conditions):
                if f'{metric}_Baseline' in df.columns:
                    temp_df.at[0, f'{cond} {metric}'] = df[f'{metric}_Baseline'][idx] if len(df[f'{metric}_Baseline']) > idx else None
                else:
                    print(f"Column '{metric}_Baseline' not found for participant {participant_number}")
        
        for col in columns_nasa:
            for idx, cond in enumerate(participant_conditions):
                if col in df.columns:
                    temp_df.at[0, f'{cond} {col}'] = df[col][idx] if len(df[col]) > idx else None
                else:
                    print(f"Column '{col}' not found for participant {participant_number}")

        for avg_col, avg_value in averages.items():
            temp_df[avg_col] = avg_value

        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing participant {participant_number}: {e}")

        # Apply renaming to all relevant columns in the combined DataFrame
        combined_df = combined_df.rename(columns=lambda x: x.replace("Stress Addition", "High_Stress_Addition_Task")
                                             .replace("Stress Subtraction", "High_Stress_Subtraction_Task")
                                             .replace("Calm Addition", "Low_Stress_Addition_Task")
                                             .replace("Calm Subtraction", "Low_Stress_Subtraction_Task"))

combined_output_path = os.path.join(base_directory, "Main_Study_Subjective_Aggregated.csv")
# Append new data to the existing file, if it exists
combined_df.to_csv(combined_output_path, index=False, mode='a', header=not os.path.exists(combined_output_path))
print("New data appended and CSV file saved successfully.")
