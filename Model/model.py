import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBRegressor
import joblib

#import the scraped data
player_stats  = pd.read_csv("../Web Scrape/combined_data/clean_player_games.csv", parse_dates=["Date"])
defense_stats = pd.read_csv("../Web Scrape/combined_data/defensive_matchup.csv")
season_stats = pd.read_csv("../Web Scrape/combined_data/all_teams_stats.csv")

#function to clean the strings in data set
def clean(df):
    df.columns = df.columns.str.strip()
    return df

#apply the column name cleaning to imported data sets
player_stats  = clean(player_stats)
defense_stats = clean(defense_stats)
season_stats = clean(season_stats)

#Uppercase the team and opponenet columns
for stat_table in (player_stats, defense_stats):
    for column in ("Team", "Opp"):
        if column in stat_table.columns:
            stat_table[column] = stat_table[column].str.upper().str.strip()
season_stats["Team"] = season_stats["Team"].str.upper().str.strip()
season_stats["Player"] = season_stats["Player"].str.strip()


#conevert headings to match the main player dataset
defense_stats = defense_stats.rename(columns={"Team": "Opp", "PTS": "PTS_allowed"})
#clean the opponent values 
defense_stats["Opp"] = defense_stats["Opp"].astype(str).str.upper().str.strip()
#adjust the team abbreviations to match throughout
defense_stats["Opp"] = defense_stats["Opp"].replace({"BKN": "BRK","CHI": "CHO"})

#compute the average points allowed for each team
opponenet_defense = (defense_stats.groupby("Opp")["PTS_allowed"].mean().rename("Opp_PTS_allowed").reset_index())
#merge the calculate defensive data into the dataset
player_stats = player_stats.merge(opponenet_defense, on="Opp", how="left")


#calculate rolling feature to account for rhythm in players game
player_stats = player_stats.sort_values(["Player", "Date"])
#calculate for each of the main stats
for stat in ["PTS", "AST", "TRB"]:
    #for both 5 and 10 games
    for last_games in (5, 10):
        column = f"{stat}_avg_{last_games}"
        player_stats[column] = (player_stats.groupby("Player")[stat].rolling(last_games, min_periods=1).mean().shift(1).reset_index(level=0, drop=True))


#uf rolling average couldn't be calculated fill with the avg
specific_columns = [f"{stat}_avg_{num_games}" 
        for stat in ["PTS","AST","TRB"] 
        for num_games in (5,10)]
player_stats[specific_columns] = (player_stats.groupby("Player")[specific_columns].transform(lambda s: s.fillna(s.mean())).fillna(0))


#can change the target vairable here
target = "PTS"


#create a list of the columns containing numbers
number_columns = [column 
            for column in player_stats.columns 
            if player_stats[column].dtype != "object" and column != target]

#create a list of the columns with strings as categorical variables
category_columns = [c 
            for c in ("Home", "Team", "Opp") 
            if c in player_stats.columns]


#create the input feature X and the corresponding target y
X = player_stats[number_columns + category_columns]
y = player_stats[target]

#split the data into training and testing (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

#chat gpt helped with the processing and the model pipeline to be produced
preprocessor = ColumnTransformer([
    ("number", "passthrough", number_columns),
    ("category", OneHotEncoder(handle_unknown="ignore"), category_columns),
])

#create the XGBoost regessor 
reg = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

#add preprocessing and regression into a single entity
pipe = Pipeline([("prep", preprocessor), ("reg", reg)])

#train the new pipeline on the training data
pipe.fit(X_train, y_train)
#predict on the testing data
y_pred = pipe.predict(X_test)

#calculate rmse and r2 score of the newly created model
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

#calculate the residual standard deviation for comparing the propbability of betting odds
residuals = y_train - pipe.predict(X_train)
error_std = residuals.std()
print(f"Training residual σ ≈ {error_std:.3f}")

#function that can pull a specific games features to calulate predictions
def build_row(player: str, date: str) -> pd.DataFrame:
    #pulls player stats of specific game enterd as input
    row = player_stats[
        (player_stats.Player == player) &
        (player_stats.Date   == pd.to_datetime(date))
    ].copy()
    #returns error if not found
    if row.empty:
        raise ValueError("Player/date combo not found.")
    return row

#function that predicts the probability of the over/under betting line
def predict_over_under(player, date, line_value):
    #build specific row for specific game
    row = build_row(player, date)
    #choose the model's input cplumns
    X_row = row[number_columns + category_columns]
    #calulate the expected points using the trained model
    mu = float(pipe.predict(X_row)[0])
    #compute the probability based on the predicted value
    p_over = norm.sf(line_value, loc=mu, scale=error_std)
    
    #return summary of overall calculations made in function
    return {
        "player"     : player,
        "date"       : str(date),
        "pred_pts"   : round(mu, 2),
        "line_pts"   : line_value,
        "prob_over"  : round(p_over, 4),
        "prob_under" : round(1 - p_over, 4),
    }


print(predict_over_under("Trae Young", "2024-12-29", 27.5))