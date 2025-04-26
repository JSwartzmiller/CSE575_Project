import pandas as pd
import os

#set directory to NBA_data folder
base_directory = "NBA_data"

#Get all team folders
team_folders = [folder for folder in os.listdir(base_directory) if folder.endswith("_data")]

#arrays to store all dataframes in same place
all_team_stats = []
all_player_logs = []

#iertate through each team foler and get data
for folder in team_folders:
    team_name = folder.replace("_data", "")
    team_folder_path = os.path.join(base_directory, folder)
    
    #get the stats from each team folder
    stats_file = os.path.join(team_folder_path, f"{team_name}_team_stats.csv")
    logs_file = os.path.join(team_folder_path, f"{team_name}_players_game_logs.csv")
    
    #Add team stats to all team stats
    if os.path.exists(stats_file):
        df_stats = pd.read_csv(stats_file)
        df_stats["Team"] = team_name
        all_team_stats.append(df_stats)

    #add gamelogs to every players stats 
    if os.path.exists(logs_file):
        df_logs = pd.read_csv(logs_file)
        df_logs["Team"] = team_name
        all_player_logs.append(df_logs)

#add all stats to df to save only one csv for all stats
team_df = pd.concat(all_team_stats, ignore_index=True)
player_df = pd.concat(all_player_logs, ignore_index=True)

#save the data frames to csv files
team_df.to_csv("all_teams_stats.csv", index=False)
player_df.to_csv("all_players_game_logs.csv", index=False)

