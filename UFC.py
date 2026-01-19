import pandas as pd
import numpy as np

# --- Step 1: Load the datasets 
df_gaethje = pd.read_excel('Justin_Gaethje.xlsx')
df_pimblet = pd.read_excel('PADDY_PIMBLET.xlsx')

# --- Step 2: Clean Height, Weight, Reach 
def clean_height(height_str):
    if pd.isnull(height_str): return None
    parts = height_str.replace('"', '').split("'")
    feet = int(parts[0])
    inches = int(parts[1]) if len(parts) > 1 else 0
    return (feet * 12 + inches) # Convert to inches

def clean_weight(weight_str):
    if pd.isnull(weight_str): return None
    return float(weight_str.replace(' lbs', ''))

def clean_reach(reach_str):
    if pd.isnull(reach_str): return None
    return float(reach_str.replace('"', ''))

# Apply cleaning functions to df_gaethje
df_gaethje['Height1'] = df_gaethje['Height1'].apply(clean_height)
df_gaethje['Weight1'] = df_gaethje['Weight1'].apply(clean_weight)
df_gaethje['Reach1'] = df_gaethje['Reach1'].apply(clean_reach)
df_gaethje['Height2'] = df_gaethje['Height2'].apply(clean_height)
df_gaethje['Weight2'] = df_gaethje['Weight2'].apply(clean_weight)
df_gaethje['Reach2'] = df_gaethje['Reach2'].apply(clean_reach)

# Apply cleaning functions to df_pimblet
df_pimblet['Height1'] = df_pimblet['Height1'].apply(clean_height)
df_pimblet['Weight1'] = df_pimblet['Weight1'].apply(clean_weight)
df_pimblet['Reach1'] = df_pimblet['Reach1'].apply(clean_reach)
df_pimblet['Height2'] = df_pimblet['Height2'].apply(clean_height)
df_pimblet['Weight2'] = df_pimblet['Weight2'].apply(clean_weight)
df_pimblet['Reach2'] = df_pimblet['Reach2'].apply(clean_reach)

# --- Step 3: One-hot encode 'Stance', map 'RES' 
df_gaethje = pd.get_dummies(df_gaethje, columns=['Stance1', 'Stance2'], prefix=['Stance1', 'Stance2'], dtype=int)
df_pimblet = pd.get_dummies(df_pimblet, columns=['Stance1', 'Stance2'], prefix=['Stance1', 'Stance2'], dtype=int)

df_gaethje['RES'] = df_gaethje['RES'].map({'W': 1, 'L': 0})
df_pimblet['RES'] = df_pimblet['RES'].map({'W': 1, 'L': 0})

# --- Step 4: Clean 'TIME', calculate 'AGE' 
def clean_time(time_str):
    if pd.isnull(time_str): return None
    parts = str(time_str).split(':')
    if len(parts) == 3: # HH:MM:SS
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2: # MM:SS
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    return None

df_gaethje['TIME_SECONDS'] = df_gaethje['TIME'].apply(clean_time)
df_pimblet['TIME_SECONDS'] = df_pimblet['TIME'].apply(clean_time)

current_year = 2026
df_gaethje['AGE1'] = current_year - df_gaethje['DOB1']
df_gaethje['AGE2'] = current_year - df_gaethje['DOB2']
df_pimblet['AGE1'] = current_year - df_pimblet['DOB1']
df_pimblet['AGE2'] = current_year - df_pimblet['DOB2']

df_gaethje = df_gaethje.drop(columns=['TIME', 'DOB1', 'DOB2'])
df_pimblet = df_pimblet.drop(columns=['TIME', 'DOB1', 'DOB2'])

# --- Step 5: One-hot encode 'DECISION' (from cell 4938853e) ---
df_gaethje = pd.get_dummies(df_gaethje, columns=['DECISION'], prefix='DECISION', dtype=int)
df_pimblet = pd.get_dummies(df_pimblet, columns=['DECISION'], prefix='DECISION', dtype=int)

# --- Step 6: Create renaming_dict and rename columns consistently 
# Need to construct a temporary combined history to get all unique columns for renaming_dict
# This was previously done by df_combined_history logic, but now df_fights is the target
temp_combined_for_renaming = pd.concat([df_gaethje, df_pimblet], ignore_index=True, join='outer').fillna(0)

renaming_dict = {}
all_columns_from_temp = temp_combined_for_renaming.columns

for col in all_columns_from_temp:
    new_name = col
    if col.endswith('1') and not col.startswith('Stance'):
        new_name = 'Fighter_A_' + col.replace('1', '').replace('.', '_').replace(' ', '_')
    elif col.endswith('2') and not col.startswith('Stance'):
        new_name = 'Fighter_B_' + col.replace('2', '').replace('.', '_').replace(' ', '_')
    elif col == 'AGE1':
        new_name = 'Fighter_A_AGE'
    elif col == 'AGE2':
        new_name = 'Fighter_B_AGE'
    elif col.startswith('Stance1_'):
        new_name = 'Fighter_A_' + col.replace('Stance1_', 'Stance_')
    elif col.startswith('Stance2_'):
        new_name = 'Fighter_B_' + col.replace('Stance2_', 'Stance_')
    elif col == 'TIME_SECONDS':
        new_name = 'Fight_Duration_Seconds'
    elif col == 'RND':
        new_name = 'Fight_Round_Ended'
    elif col.startswith('DECISION_'):
        new_name = col.replace('DECISION_', 'Decision_')
    elif col == 'RES':
        new_name = 'y'

    if new_name != col:
        renaming_dict[col] = new_name

df_gaethje.rename(columns=renaming_dict, inplace=True)
df_pimblet.rename(columns=renaming_dict, inplace=True)

# --- Step 7: Engineer Core Difference Features & stance_same 
def get_stance_from_row(row, fighter_prefix):
    stances = ['Orthodox', 'Southpaw', 'Switch']
    for stance in stances:
        col_name = f'{fighter_prefix}_Stance_{stance}'
        if col_name in row.index and row[col_name] == 1:
            return stance
    return 'Unknown'

df_gaethje['Fighter_A_Stance_Actual'] = df_gaethje.apply(lambda row: get_stance_from_row(row, 'Fighter_A'), axis=1)
df_gaethje['Fighter_B_Stance_Actual'] = df_gaethje.apply(lambda row: get_stance_from_row(row, 'Fighter_B'), axis=1)
df_gaethje['stance_same'] = (df_gaethje['Fighter_A_Stance_Actual'] == df_gaethje['Fighter_B_Stance_Actual']).astype(int)
df_gaethje = df_gaethje.drop(columns=['Fighter_A_Stance_Actual', 'Fighter_B_Stance_Actual'])

df_pimblet['Fighter_A_Stance_Actual'] = df_pimblet.apply(lambda row: get_stance_from_row(row, 'Fighter_A'), axis=1)
df_pimblet['Fighter_B_Stance_Actual'] = df_pimblet.apply(lambda row: get_stance_from_row(row, 'Fighter_B'), axis=1)
df_pimblet['stance_same'] = (df_pimblet['Fighter_A_Stance_Actual'] == df_pimblet['Fighter_B_Stance_Actual']).astype(int)
df_pimblet = df_pimblet.drop(columns=['Fighter_A_Stance_Actual', 'Fighter_B_Stance_Actual'])

feature_cols_for_diff = [
    'Height', 'Weight', 'Reach', 'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
    'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub__Avg', 'AGE'
]

for col in feature_cols_for_diff:
    if f'Fighter_A_{col}' in df_gaethje.columns and f'Fighter_B_{col}' in df_gaethje.columns:
        df_gaethje[f'{col}_diff'] = df_gaethje[f'Fighter_A_{col}'] - df_gaethje[f'Fighter_B_{col}']
    if f'Fighter_A_{col}' in df_pimblet.columns and f'Fighter_B_{col}' in df_pimblet.columns:
        df_pimblet[f'{col}_diff'] = df_pimblet[f'Fighter_A_{col}'] - df_pimblet[f'Fighter_B_{col}']

# --- Step 8: Engineer Fight-Context Features 
def calculate_win_streak(series):
    win_streak = []
    current_streak = 0
    for res in series:
        if res == 1: # Win
            current_streak += 1
        else: # Loss
            current_streak = 0
        win_streak.append(current_streak)
    return pd.Series(win_streak, index=series.index)

# Process df_gaethje
df_gaethje['win_by_Decision'] = df_gaethje.get('Decision_U-DEC', 0) + df_gaethje.get('Decision_M-DEC', 0)
df_gaethje['ko_indicator'] = df_gaethje.get('Decision_KO/TKO', 0)
df_gaethje['finish_indicator'] = df_gaethje.get('Decision_KO/TKO', 0) + df_gaethje.get('Decision_SUB', 0)

df_gaethje['avg_fight_time_last_5'] = df_gaethje['Fight_Duration_Seconds'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df_gaethje['ko_rate_last_5'] = df_gaethje['ko_indicator'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df_gaethje['finish_rate_last_5'] = df_gaethje['finish_indicator'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df_gaethje['recent_win_streak'] = calculate_win_streak(df_gaethje['y']).shift(1).fillna(0)

# Process df_pimblet
df_pimblet['win_by_Decision'] = df_pimblet.get('Decision_U-DEC', 0) + df_pimblet.get('Decision_M-DEC', 0)
df_pimblet['ko_indicator'] = df_pimblet.get('Decision_KO/TKO', 0)
df_pimblet['finish_indicator'] = df_pimblet.get('Decision_KO/TKO', 0) + df_pimblet.get('Decision_SUB', 0)

df_pimblet['avg_fight_time_last_5'] = df_pimblet['Fight_Duration_Seconds'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df_pimblet['ko_rate_last_5'] = df_pimblet['ko_indicator'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df_pimblet['finish_rate_last_5'] = df_pimblet['finish_indicator'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
df_pimblet['recent_win_streak'] = calculate_win_streak(df_pimblet['y']).shift(1).fillna(0)


all_cols = list(set(df_gaethje.columns) | set(df_pimblet.columns))

# Add missing columns to df_gaethje and fill with 0
missing_in_gaethje = set(all_cols) - set(df_gaethje.columns)
for col in missing_in_gaethje:
    df_gaethje[col] = 0

# Add missing columns to df_pimblet and fill with 0
missing_in_pimblet = set(all_cols) - set(df_pimblet.columns)
for col in missing_in_pimblet:
    df_pimblet[col] = 0

# Sort all_cols to ensure consistent column order across runs
all_cols.sort()

# Ensure both DataFrames have the same column order
df_gaethje = df_gaethje[all_cols]
df_pimblet = df_pimblet[all_cols]

# Concatenate the DataFrames
df_fights = pd.concat([df_gaethje, df_pimblet], ignore_index=True)

print("First 5 rows of df_fights:")
print(df_fights.head())
print("\nShape of df_fights:", df_fights.shape)

columns_to_drop = []

for col in df_fights.columns:
    if (col.startswith('Fighter_A_') and not col.startswith('Fighter_A_Stance_')) or \
       (col.startswith('Fighter_B_') and not col.startswith('Fighter_B_Stance_')) or \
       col.startswith('Fighter_A_Stance_') or \
       col.startswith('Fighter_B_Stance_') or \
       col.startswith('Decision_'):
        columns_to_drop.append(col)

# Remove 'y' from columns_to_drop if it somehow got in there (it shouldn't based on prefixes)
if 'y' in columns_to_drop:
    columns_to_drop.remove('y')

# Keep derived decision features if they are present and not already dropped
# Ensure we don't drop 'win_by_Decision', 'ko_indicator', 'finish_indicator'
columns_to_drop = [col for col in columns_to_drop if col not in ['win_by_Decision', 'ko_indicator', 'finish_indicator']]

df_fights = df_fights.drop(columns=columns_to_drop, errors='ignore')

print("Shape of df_fights after dropping columns:", df_fights.shape)
print("\nFirst 5 rows of df_fights after dropping columns:")
print(df_fights.head())

import numpy as np

# 1. Extract the last row from the processed df_gaethje DataFrame for Justin Gaethje's most recent statistics.
# 2. Extract the last row from the processed df_pimblet DataFrame for Paddy Pimblett's most recent statistics.
gaethje_last_row = df_gaethje.iloc[-1].copy()
pimblett_last_row = df_pimblet.iloc[-1].copy()

# Prepare Gaethje's stats as Fighter_A and Pimblett's stats as Fighter_B
upcoming_fight_data = {}

# Map Gaethje's stats as Fighter_A
for col in gaethje_last_row.index:
    if col.startswith('Fighter_A_') or col in ['Fight_Duration_Seconds', 'Fight_Round_Ended', 'y', 'win_by_Decision', 'ko_indicator', 'finish_indicator', 'avg_fight_time_last_5', 'ko_rate_last_5', 'finish_rate_last_5', 'recent_win_streak']:
        upcoming_fight_data[col] = gaethje_last_row[col]

# Map Pimblett's stats as Fighter_B (Pimblett's own stats are 'Fighter_A_' in his history)
for col in pimblett_last_row.index:
    if col.startswith('Fighter_A_'):
        upcoming_fight_data[col.replace('Fighter_A_', 'Fighter_B_')] = pimblett_last_row[col]

upcoming_fight_prediction_data = pd.DataFrame([upcoming_fight_data])

# 3. Recalculate all difference features
feature_cols_for_diff = [
    'Height', 'Weight', 'Reach', 'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
    'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub__Avg', 'AGE'
]

for col in feature_cols_for_diff:
    col_a = f'Fighter_A_{col}'
    col_b = f'Fighter_B_{col}'
    if col_a in upcoming_fight_prediction_data.columns and col_b in upcoming_fight_prediction_data.columns:
        upcoming_fight_prediction_data[f'{col}_diff'] = upcoming_fight_prediction_data[col_a] - upcoming_fight_prediction_data[col_b]
    else:
        upcoming_fight_prediction_data[f'{col}_diff'] = 0

# 4. Recalculate the `stance_same` feature for the upcoming fight
def get_stance_from_df_row(df_row, fighter_prefix):
    stances = ['Orthodox', 'Southpaw', 'Switch']
    for stance in stances:
        col_name = f'{fighter_prefix}_Stance_{stance}'
        if col_name in df_row.index and df_row[col_name] == 1:
            return stance
    return 'Unknown'

gaethje_stance_actual = get_stance_from_df_row(gaethje_last_row, 'Fighter_A')
pimblett_stance_actual = get_stance_from_df_row(pimblett_last_row, 'Fighter_A') # Paddy's own stance in his history is 'Fighter_A'

upcoming_fight_prediction_data['stance_same'] = int(gaethje_stance_actual == pimblett_stance_actual)

# 5. Explicitly set the one-hot encoded stance columns
# For Fighter_A (Gaethje)
for stance_type in ['Orthodox', 'Southpaw', 'Switch']:
    col_name = f'Fighter_A_Stance_{stance_type}'
    upcoming_fight_prediction_data[col_name] = int(gaethje_stance_actual == stance_type)

# For Fighter_B (Pimblett)
for stance_type in ['Orthodox', 'Southpaw', 'Switch']:
    col_name = f'Fighter_B_Stance_{stance_type}'
    upcoming_fight_prediction_data[col_name] = int(pimblett_stance_actual == stance_type)

# 6. Align the columns of this upcoming_fight_prediction_data DataFrame with df_fights
# Get columns from the training dataset, excluding 'y'
prediction_target_columns = df_fights.drop(columns=['y']).columns

# Add missing columns to upcoming_fight_prediction_data and fill with 0
missing_cols_in_upcoming = set(prediction_target_columns) - set(upcoming_fight_prediction_data.columns)
for col in missing_cols_in_upcoming:
    upcoming_fight_prediction_data[col] = 0

# Remove extra columns from upcoming_fight_prediction_data that are not in the training set
extra_cols_in_upcoming = set(upcoming_fight_prediction_data.columns) - set(prediction_target_columns)
if extra_cols_in_upcoming:
    upcoming_fight_prediction_data = upcoming_fight_prediction_data.drop(columns=list(extra_cols_in_upcoming))

# Reorder columns to match the training set
upcoming_fight_prediction_data = upcoming_fight_prediction_data[prediction_target_columns]

print("Upcoming Fight Prediction Data Head:")
print(upcoming_fight_prediction_data.head())
print("\nShape of Upcoming Fight Prediction Data:", upcoming_fight_prediction_data.shape)
print("\nColumns of Upcoming Fight Prediction Data:\n", upcoming_fight_prediction_data.columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Separate features (X) from the target variable (y)
X = df_fights.drop(columns=['y'])
y = df_fights['y']

# 2. Split the features (X) and target (y) into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize an empty dictionary, models, to store the trained models
models = {}

# 4. Create and train pipelines for each model
# Logistic Regression
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced', random_state=42))
])
log_reg_pipeline.fit(X_train, y_train)
models['Logistic Regression'] = log_reg_pipeline

# Random Forest Classifier
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
rf_pipeline.fit(X_train, y_train)
models['Random Forest'] = rf_pipeline

# Gradient Boosting Classifier
gbc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])
gbc_pipeline.fit(X_train, y_train)
models['Gradient Boosting'] = gbc_pipeline

print("Models trained successfully within pipelines.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

import numpy as np

# 1. Extract static career-level statistics for Justin Gaethje (Fighter_A) and Paddy Pimblett (Fighter_B)
# These rows contain the most recent, fully processed stats including engineered context features.
# gaethje_last_row has Justin Gaethje's stats as 'Fighter_A_...' and his context features (ko_rate_last_5 etc.)
# pimblett_last_row has Paddy Pimblett's stats as 'Fighter_A_...' and his context features

fighter_a_stats = {
    'SLpM': gaethje_last_row['Fighter_A_SLpM'],
    'Str_Acc': gaethje_last_row['Fighter_A_Str_Acc'],
    'SApM': gaethje_last_row['Fighter_A_SApM'],
    'Str_Def': gaethje_last_row['Fighter_A_Str_Def'],
    'TD_Avg': gaethje_last_row['Fighter_A_TD_Avg'],
    'TD_Acc': gaethje_last_row['Fighter_A_TD_Acc'],
    'TD_Def': gaethje_last_row['Fighter_A_TD_Def'],
    'Sub__Avg': gaethje_last_row['Fighter_A_Sub__Avg'],
    'ko_rate': gaethje_last_row['ko_rate_last_5'],
    'finish_rate': gaethje_last_row['finish_rate_last_5']
}

fighter_b_stats = {
    'SLpM': pimblett_last_row['Fighter_A_SLpM'], # Pimblett's own stats from his history are 'Fighter_A_'
    'Str_Acc': pimblett_last_row['Fighter_A_Str_Acc'],
    'SApM': pimblett_last_row['Fighter_A_SApM'],
    'Str_Def': pimblett_last_row['Fighter_A_Str_Def'],
    'TD_Avg': pimblett_last_row['Fighter_A_TD_Avg'],
    'TD_Acc': pimblett_last_row['Fighter_A_TD_Acc'],
    'TD_Def': pimblett_last_row['Fighter_A_TD_Def'],
    'Sub__Avg': pimblett_last_row['Fighter_A_Sub__Avg'],
    'ko_rate': pimblett_last_row['ko_rate_last_5'],
    'finish_rate': pimblett_last_row['finish_rate_last_5']
}

# 2. Define simulation parameters
num_simulations = 1000
max_rounds = 5 # Assuming a championship fight or main event (typical for these fighters)
round_duration_seconds = 300 # 5 minutes per round

# 3. Initialize an empty list called mc_outcomes
mc_outcomes = []

# 4. Loop num_simulations times
for _ in range(num_simulations):
    fighter_a_rounds_won = 0
    fighter_b_rounds_won = 0
    fight_finished = False
    fight_winner = None # 1 for Gaethje, 0 for Pimblett

    # Round loop
    for round_num in range(1, max_rounds + 1):
        # Simulate offensive metrics using Poisson distribution for events per round
        # SLpM, TD_Avg, Sub__Avg are rates. Convert them to per-round rates.

        # Fighter A's offensive actions
        strikes_landed_a = np.random.poisson(fighter_a_stats['SLpM'] * (round_duration_seconds / 60))
        takedowns_landed_a = np.random.poisson(fighter_a_stats['TD_Avg'] * (round_duration_seconds / 900))
        submission_attempts_a = np.random.poisson(fighter_a_stats['Sub__Avg'] * (round_duration_seconds / 900))

        # Fighter B's offensive actions
        strikes_landed_b = np.random.poisson(fighter_b_stats['SLpM'] * (round_duration_seconds / 60))
        takedowns_landed_b = np.random.poisson(fighter_b_stats['TD_Avg'] * (round_duration_seconds / 900))
        submission_attempts_b = np.random.poisson(fighter_b_stats['Sub__Avg'] * (round_duration_seconds / 900))

        # 6. Implement a mechanism to check for fight-ending events (KO/TKO or Submission)
        # Distribute per-fight finish rate across rounds for a per-round finish probability
        # Note: This simplifies the interaction and assumes independent finish probabilities per round
        prob_finish_a_per_round = fighter_a_stats['finish_rate'] / max_rounds
        prob_finish_b_per_round = fighter_b_stats['finish_rate'] / max_rounds

        # Determine if a finish happens
        if np.random.rand() < prob_finish_a_per_round: # If Gaethje finishes Pimblett
            fight_winner = 1 # Justin Gaethje wins
            fight_finished = True
            break # End the fight simulation for this round
        elif np.random.rand() < prob_finish_b_per_round: # If Pimblett finishes Gaethje
            fight_winner = 0 # Paddy Pimblett wins
            fight_finished = True
            break # End the fight simulation for this round

        # 7. If no finish, determine the round winner based on accumulated offensive actions
        # Using a simple weighted scoring system for the round
        # These weights are simplified, but reflect offensive output contributions
        round_score_a = (strikes_landed_a * 0.1) + (takedowns_landed_a * 5) + (submission_attempts_a * 2)
        round_score_b = (strikes_landed_b * 0.1) + (takedowns_landed_b * 5) + (submission_attempts_b * 2)

        if round_score_a > round_score_b:
            fighter_a_rounds_won += 1
        elif round_score_b > round_score_a:
            fighter_b_rounds_won += 1
        else:
            # Handle tie rounds, e.g., by giving a slight edge or coin flip (here, a coin flip)
            if np.random.rand() < 0.5:
                fighter_a_rounds_won += 1
            else:
                fighter_b_rounds_won += 1

    # 8. If the fight goes to decision (all rounds completed without a finish)
    if not fight_finished:
        if fighter_a_rounds_won > fighter_b_rounds_won:
            fight_winner = 1 # Justin Gaethje wins by decision
        elif fighter_b_rounds_won > fighter_a_rounds_won:
            fight_winner = 0 # Paddy Pimblett wins by decision
        else:
            # If still a tie after rounds (very rare for decision), assign randomly
            if np.random.rand() < 0.5:
                fight_winner = 1
            else:
                fight_winner = 0

    # 9. Record the winner of each simulated fight.
    mc_outcomes.append(fight_winner)

# 10. After all simulations are complete, calculate the overall win probability for Justin Gaethje.
monte_carlo_win_prob = np.mean(mc_outcomes)

# 11. Print the calculated Monte Carlo win probability for Justin Gaethje.
print(f"Monte Carlo Win Probability for Justin Gaethje (Fighter_A) from {num_simulations} simulations: {monte_carlo_win_prob:.4f}")

ensemble_predictions = {}

# 2. Iterate through the models dictionary and get predictions
for model_name, model in models.items():
    # predict_proba returns probabilities for both classes [prob_class_0, prob_class_1]
    # We want the probability of Fighter_A (Justin Gaethje) winning, which is class 1.
    prob_fighter_a_win = model.predict_proba(upcoming_fight_prediction_data)[:, 1][0]
    ensemble_predictions[model_name] = prob_fighter_a_win

# 3. Add the monte_carlo_win_prob to the ensemble_predictions dictionary
ensemble_predictions['Monte Carlo Simulation'] = monte_carlo_win_prob

# 4. Calculate the average of all probabilities to get the final ensemble prediction
final_ensemble_prediction = np.mean(list(ensemble_predictions.values()))

# 5. Print all individual probabilities and the final ensemble prediction
print("Individual Win Probabilities for Justin Gaethje:")
for model_name, prob in ensemble_predictions.items():
    print(f"  {model_name}: {prob:.4f}")

print(f"\nFinal Ensemble Predicted Win Probability for Justin Gaethje: {final_ensemble_prediction:.4f}")
