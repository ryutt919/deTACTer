
import pandas as pd
import yaml
import os
import glob
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
import socceraction.vaep.features as features
import socceraction.vaep.labels as labels
from socceraction.vaep import formula as vaep_formula

# Load configuration
with open('c:/Users/Public/Documents/DIK/deTACTer/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SPADL_DIR = config['data']['spadl_output_dir']
NB_PREV_ACTIONS = config['features']['nb_prev_actions']
RESULTS_DIR = "results"
PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.yaml")

def load_spadl_data():
    all_games = []
    files = glob.glob(os.path.join(SPADL_DIR, "game_*.csv"))
    print(f"Loading {len(files)} games for VAEP calculation...")
    for f in files:
        df = pd.read_csv(f)
        all_games.append(df)
    return all_games

def get_feature_functions():
    # Define the features to use (Must match optimization script if feasible, or be standard)
    return [
        features.actiontype_onehot,
        features.result_onehot,
        features.actiontype_result_onehot,
        features.bodypart_onehot,
        features.time,
        features.startlocation,
        features.endlocation,
        features.movement,
        features.space_delta,
        features.team,
    ]

def generate_features_and_labels(games):
    X_list = []
    Y_list = []
    indices_list = [] # To track which game/action corresponds to which feature row
    
    xfns = get_feature_functions()

    print("Generating features and labels...")
    for game in games:
        try:
            gamestates = features.gamestates(game, NB_PREV_ACTIONS)
            X_game = pd.concat([fn(gamestates) for fn in xfns], axis=1)
            
            Y_scores = labels.scores(game, nr_actions=10)
            Y_concedes = labels.concedes(game, nr_actions=10)
            Y_game = pd.concat([Y_scores, Y_concedes], axis=1)
            Y_game.columns = ['scores', 'concedes']
            
            X_list.append(X_game)
            Y_list.append(Y_game)
            
            # Keep track of indices to map back to original data
            # We preserve game_id, period_id, time_seconds, action_id (if exists) 
            # or just use the index of the action in the game df
            meta = game[['game_id', 'period_id', 'time_seconds', 'team_id', 'player_id', 'type_name', 'result_name']].copy()
            indices_list.append(meta)
            
        except Exception as e:
            print(f"Skipping game {game.game_id.iloc[0] if not game.empty else '?'} error: {e}")
            continue

    if not X_list:
        raise ValueError("No data generated.")

    X = pd.concat(X_list).reset_index(drop=True)
    Y = pd.concat(Y_list).reset_index(drop=True)
    meta = pd.concat(indices_list).reset_index(drop=True)
    
    return X, Y, meta

def train_and_predict(X, Y):
    # Load optimal params if available
    params = {}
    if os.path.exists(PARAMS_PATH):
        print(f"Loading best parameters from {PARAMS_PATH}")
        with open(PARAMS_PATH, 'r') as f:
            params = yaml.safe_load(f)
    else:
        print("Using default parameters from config")
        params = {
            'n_estimators': config['model']['n_estimators'],
            'learning_rate': config['model']['learning_rate'],
            'max_depth': config['model']['max_depth'],
        }
    
    # Add common params
    params['verbose'] = 0
    params['random_state'] = 42
    
    # predictions container
    preds_scores = pd.Series(index=X.index, dtype='float64')
    preds_concedes = pd.Series(index=X.index, dtype='float64')
    
    # K-Fold CV to generate out-of-sample predictions for all data
    # (For a large dataset, a simple 5-fold is usually enough to get valid ratings)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Training models and predicting (CV)...")
    
    model_scores = CatBoostClassifier(**params)
    model_concedes = CatBoostClassifier(**params)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train_s, Y_test_s = Y['scores'].iloc[train_idx], Y['scores'].iloc[test_idx]
        Y_train_c, Y_test_c = Y['concedes'].iloc[train_idx], Y['concedes'].iloc[test_idx]
        
        # Train scores
        model_scores.fit(X_train, Y_train_s)
        preds_scores.iloc[test_idx] = model_scores.predict_proba(X_test)[:, 1]
        
        # Train concedes
        model_concedes.fit(X_train, Y_train_c)
        preds_concedes.iloc[test_idx] = model_concedes.predict_proba(X_test)[:, 1]
        
    return preds_scores, preds_concedes

def main():
    games = load_spadl_data()
    X, Y, meta = generate_features_and_labels(games)
    
    # Predict Goal/Concede Probabilities
    P_scores, P_concedes = train_and_predict(X, Y)
    
    # Calculate VAEP
    # VAEP = P_score_diff - P_concede_diff
    # socceraction provides formula.value which does this logic
    # value(actions, Pscores, Pconcedes, Pscores_prev, Pconcedes_prev)
    # But usually we need the probabilities for the current state. 
    # The formula essentially is: V(Si) = P(G|Si) - P(C|Si)
    # VAEP(ai) = V(Si) - V(Si-1)
    
    # We have P(G|Si) and P(C|Si) in P_scores, P_concedes (for the state *after* action i? or *before*?)
    # socceraction's `features.gamestates` typically represents the state *before* action i is performed?
    # NO: gamestates(actions, 3) gives the context *leading up to* action i.
    # The classifiers are usually trained to predict if a goal occurs *after* the current state (which includes action i).
    # So P_scores[i] is the value of the state *after* action i.
    # We need the value of the state *before* action i, which is P_scores[i-1] (roughly, handling game boundaries).
    
    # socceraction.vaep.formula.value does this calculation for us usually.
    # It requires the predictions (Y_hat).
    
    print("Calculating VAEP values...")
    # vaep_values is usually a DataFrame with offensive_value, defensive_value, vaep_value
    vaep_values = vaep_formula.value(meta, P_scores, P_concedes)
    
    # Combine with metadata
    final_df = pd.concat([meta, vaep_values], axis=1)
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, "vaep_values.csv")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    final_df.to_csv(output_path, index=False)
    print(f"Saved VAEP calculations to {output_path}")
    
    # Also save aggregated stats per player/game for quick check
    summary_path = os.path.join(RESULTS_DIR, "vaep_summary.csv")
    summary = final_df.groupby(['player_id', 'game_id'])['vaep_value'].sum().reset_index()
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
