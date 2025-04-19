import pandas as pd
import os

# Set base directory where NBA_data is located
base_dir = "NBA_data"

# Get all team folders (end with _data)
team_folders = [folder for folder in os.listdir(base_dir) if folder.endswith("_data")]

# Hold all DataFrames
all_team_stats = []
all_player_logs = []

# Loop through each team folder
for folder in team_folders:
    team_name = folder.replace("_data", "")
    team_folder_path = os.path.join(base_dir, folder)
    
    # File paths
    stats_file = os.path.join(team_folder_path, f"{team_name}_team_stats.csv")
    logs_file = os.path.join(team_folder_path, f"{team_name}_players_game_logs.csv")
    
    # Read and append if files exist
    if os.path.exists(stats_file):
        df_stats = pd.read_csv(stats_file)
        df_stats["Team"] = team_name  # Add team name column
        all_team_stats.append(df_stats)

    if os.path.exists(logs_file):
        df_logs = pd.read_csv(logs_file)
        df_logs["Team"] = team_name  # Add team name column
        all_player_logs.append(df_logs)

# Concatenate into master DataFrames
team_df = pd.concat(all_team_stats, ignore_index=True)
player_df = pd.concat(all_player_logs, ignore_index=True)

# Save to CSVs
team_df.to_csv("all_teams_stats.csv", index=False)
player_df.to_csv("all_players_game_logs.csv", index=False)

print("✅ Clean datasets saved: all_teams_stats.csv & all_players_game_logs.csv")
