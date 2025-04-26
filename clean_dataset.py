import pandas as pd
from dotenv import load_dotenv
import os
import re
from datetime import datetime

#load enviroment variables
load_dotenv()

#change file path to slective path in .env file
df = pd.read_csv(os.getenv("ALL_PLAYERS_DATASET_FILEPATH"))

#drop the career game counter column
df.drop(columns=["Gcar"], inplace=True)

#make team first featue and player second feature
features = df.columns.tolist()
#remove team and players from current position and place in front
features.remove('Team')
features.remove('Player')
feature_order = ['Team', 'Player'] + features
df = df[feature_order]

#get rid of repeated headers from fetch phase
duplicate_rows = df[
    (df['Date'] == 'Date')|(df['MP'] == 'MP')|(df['PTS'] == 'PTS')|(df['FG'] == 'FG')
]
df = df.drop(duplicate_rows.index)

#rename game count column
df.rename(columns={'Gtm': 'Game'}, inplace=True)

#change the functionality of the home and away column overall
df.rename(columns={'Unnamed: 4': 'Home'}, inplace=True)
df['Home'] = df['Home'].apply(lambda x: 1 if pd.isna(x) else 0) #make home team value 1

#remove games if player didn't record major stats
key_stats = ['MP', 'PTS', 'FG', 'AST', 'TRB']
df = df.dropna(subset=key_stats, how='all')

#create function that stores amount of overtimes a game went to
#*************function was generated using chat gpt*************
def extract_overtime(result):
    if isinstance(result, str):
        match = re.search(r'\((\d*)OT\)', result)
        if match:
            overtime = match.group(1)
            if overtime == '':
                return 1
            else:
                return int(overtime)
    return 0
df['Overtime'] = df['Result'].apply(extract_overtime)

#create fucntion to remove OT from results column
#*************function was generated using chat gpt*************
def remove_ot(result):
    if isinstance(result, str):
        return re.sub(r'\s*\(\d*OT\)', '', result)
    return result
df['Result'] = df['Result'].apply(remove_ot)

#create function to split the results up to store team points independently
def split_game_result(result):
    win_loss, score = result.split(',', 1)
    win_loss = win_loss.strip()
    team_pts, opp_pts = score.strip().split('-')
    return win_loss, int(team_pts), int(opp_pts)
df['Win_Loss'], df['Team_PTS'], df['Opp_PTS'] = zip(*df['Result'].apply(split_game_result))

#fromat the newly created features and remove results feature all together
cols = df.columns.tolist()
cols.remove('Overtime')
cols.remove('Win_Loss')
cols.remove('Team_PTS')
cols.remove('Opp_PTS')
result_index = cols.index('Result')
cols[result_index+1:result_index+1] = ['Overtime', 'Win_Loss', 'Team_PTS', 'Opp_PTS']
df = df[cols]
df.drop(columns=['Result'], inplace=True)

#Rename the GS column and change to binary classifiers
df.rename(columns={'GS': 'Starter'}, inplace=True)
df['Starter'] = df['Starter'].apply(lambda x: 1 if x == '*' else 0)

#get array with features that data type needs changed
int_columns = [
    'Game', 'Home', 'Overtime', 'Team_PTS', 'Opp_PTS', 'Starter',
    'FG', 'FGA', '3P', '3PA', '2P', '2PA',
    'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
    'BLK', 'TOV', 'PF', 'PTS'
]
float_columns = [
    'FG%', '3P%', '2P%', 'eFG%', 'FT%', 'GmSc', '+/-'
]

#convert features to specific data type
for column in int_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
for column in float_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0).astype(float)


#create function to round minutes played to the nearest minute and convert as int
def minutes_played(mp):
    mins, secs = mp.split(':')
    total_minutes = int(mins) + int(secs) / 60
    return round(total_minutes)
df['MP'] = df['MP'].apply(minutes_played)
df['MP'] = df['MP'].astype(int)

#save df to specified location as .csv file
df.to_csv(os.getenv("NEW_CLEAN_DATASET_FILEPATH"), index=False)