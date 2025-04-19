import csv
import pandas as pd

def clean_player_logs(filepath, output_path):
    cleaned_rows = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip headers or short rows
            if len(row) < 5 or row[0] == "Gcar":
                continue

            # Skip inactive/DNP/DND
            status = row[7].strip() if len(row) > 7 else ""
            if status in ["Inactive", "Did Not Play", "Did Not Dress"]:
                continue

            cleaned_rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(cleaned_rows)

    # Add headers (adjust if you want to remove extra metadata columns)
    df.columns = [
        "Gcar", "Gtm", "Date", "Team", "At", "Opponent", "Result", "GS", "MP",
        "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
        "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF",
        "PTS", "GmSc", "+/-", "Player"
    ]

    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")

clean_player_logs()