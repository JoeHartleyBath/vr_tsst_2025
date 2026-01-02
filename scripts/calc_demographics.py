import pandas as pd
import numpy as np
import os

def calculate_demographics():
    # Path to QuestionPro data
    data_path = r"data/QuestionPro-SR-RawData-1602876606-01-02-2026-T065743.833.xlsx"
    
    # Read Excel, finding the header dynamically
    try:
        # Read first few lines to debug
        df_temp = pd.read_excel(data_path, header=None, nrows=20)
        header_row = None
        for i, row in df_temp.iterrows():
            # Check if 'Response ID' or 'Participant Number' is in the row
            row_values = [str(x) for x in row.values]
            if "Response ID" in row_values or "Participant Number" in row_values:
                print(f"Found header at line index {i}")
                header_row = i
                break
        
        if header_row is None:
            print("Could not find header row")
            return

        df = pd.read_excel(data_path, header=header_row)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return
        print(f"Error reading CSV: {e}")
        return

    # Columns of interest
    # Note: Column names might have extra spaces or special characters, so we'll be careful
    # Based on inspection:
    # "Participant Number"
    # "Age"
    # "Gender"
    # "To what extent have you used VR equipment before?"

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Filter for valid participants if necessary (e.g., "Response Status" == "Completed")
    if "Response Status" in df.columns:
        df = df[df["Response Status"] == "Completed"]

    # Extract relevant columns
    cols_to_keep = [
        "Participant Number", 
        "Age", 
        "Gender", 
        "To what extent have you used VR equipment before?"
    ]
    
    # Check if columns exist
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        # Try to find partial matches
        for target in missing_cols:
            matches = [c for c in df.columns if target in c]
            print(f"Possible matches for '{target}': {matches}")
        return

    subset = df[cols_to_keep].copy()

    # Convert Age to numeric
    subset["Age"] = pd.to_numeric(subset["Age"], errors='coerce')

    # Calculate Stats
    n_total = len(subset)
    age_mean = subset["Age"].mean()
    age_sd = subset["Age"].std()
    
    gender_counts = subset["Gender"].value_counts()
    vr_exp_counts = subset["To what extent have you used VR equipment before?"].value_counts()

    # Output results
    output_file = r"results/demographics_summary.txt"
    csv_output_file = r"results/demographics.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save clean CSV for merging
    subset.rename(columns={
        "Participant Number": "participant_id",
        "Age": "age",
        "Gender": "gender",
        "To what extent have you used VR equipment before?": "vr_experience"
    }, inplace=True)
    subset.to_csv(csv_output_file, index=False)
    print(f"Demographics CSV written to {csv_output_file}")

    with open(output_file, "w") as f:
        f.write("--- Demographics Summary ---\n")
        f.write(f"Total Participants (N): {n_total}\n\n")
        
        f.write("--- Age ---\n")
        f.write(f"Mean: {age_mean:.2f}\n")
        f.write(f"SD: {age_sd:.2f}\n\n")
        
        f.write("--- Gender ---\n")
        f.write(gender_counts.to_string())
        f.write("\n\n")
        
        f.write("--- VR Experience ---\n")
        f.write(vr_exp_counts.to_string())
        f.write("\n")

    print(f"Demographics summary written to {output_file}")
    print(open(output_file).read())

if __name__ == "__main__":
    calculate_demographics()
