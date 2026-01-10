
import pandas as pd
import yaml
import os
import glob
import optuna
import joblib
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import socceraction.vaep.features as features
import socceraction.vaep.labels as labels
from socceraction.vaep import features as fs
from socceraction.vaep import labels as lab

# Load configuration
with open('c:/Users/Public/Documents/DIK/deTACTer/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SPADL_DIR = config['data']['spadl_output_dir']
STUDY_NAME = config['optimization']['study_name']
N_TRIALS = config['optimization']['n_trials']

def load_spadl_data():
    all_games = []
    files = glob.glob(os.path.join(SPADL_DIR, "game_*.csv"))
    print(f"Loading {len(files)} games for optimization...")
    for f in files:
        df = pd.read_csv(f)
        all_games.append(df)
    return all_games

def generate_features(games, nb_prev_actions):
    """
    Generate features dynamically based on nb_prev_actions.
    Only generates for a subset of games to speed up if needed, 
    but for correctness we use all or a reliable sample.
    """
    X_all = []
    Y_all = []
    
    # Use a subset of games for faster optimization trials if dataset is huge.
    # For ~200 games, it might slow down TPE significantly to re-gen every time.
    # Let's use a subset (e.g., 20 games) for hyperparameter tuning to keep it responsive.
    sample_games = games[:50] if len(games) > 50 else games
    
    for game in sample_games:
        try:
            gamestates = fs.gamestates(game, nb_prev_actions)
            X0 = fs.actiontype_onehot(gamestates)
            X1 = fs.result_onehot(gamestates)
            X2 = fs.actiontype_result_onehot(gamestates)
            X3 = fs.bodypart_onehot(gamestates)
            X4 = fs.time(gamestates)
            X5 = fs.startlocation(gamestates)
            X6 = fs.endlocation(gamestates)
            X7 = fs.movement(gamestates)
            X8 = fs.space_delta(gamestates)
            X9 = fs.team(gamestates)
            
            X = pd.concat([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=1)
            
            Y_scores = lab.scores(game, nr_actions=10)
            Y_concedes = lab.concedes(game, nr_actions=10)
            
            Y = pd.concat([Y_scores, Y_concedes], axis=1)
            Y.columns = ['scores', 'concedes']
            
            X_all.append(X)
            Y_all.append(Y)
            
        except Exception as e:
            continue
            
    if not X_all:
        raise ValueError("No data generated.")
        
    X_train = pd.concat(X_all).reset_index(drop=True)
    Y_train = pd.concat(Y_all).reset_index(drop=True)
    
    return X_train, Y_train

def objective(trial, games):
    # Suggest nb_prev_actions
    nb_prev = trial.suggest_int('nb_prev_actions', 
                                config['optimization']['params']['nb_prev_actions']['low'], 
                                config['optimization']['params']['nb_prev_actions']['high'])
    
    # Generate data for this trial
    X, Y = generate_features(games, nb_prev)
    
    # Hyperparameters for Model
    params = {
        'n_estimators': trial.suggest_int('n_estimators', config['optimization']['params']['n_estimators']['low'], config['optimization']['params']['n_estimators']['high']),
        'learning_rate': trial.suggest_float('learning_rate', config['optimization']['params']['learning_rate']['low'], config['optimization']['params']['learning_rate']['high'], log=config['optimization']['params']['learning_rate'].get('log', False)),
        'max_depth': trial.suggest_int('max_depth', config['optimization']['params']['max_depth']['low'], config['optimization']['params']['max_depth']['high']),
        'verbose': 0,
        'random_state': 42
    }
    
    model_scores = CatBoostClassifier(**params)
    model_concedes = CatBoostClassifier(**params)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model_scores.fit(X_tr, Y_tr['scores'])
    preds_scores = model_scores.predict_proba(X_val)[:, 1]
    
    model_concedes.fit(X_tr, Y_tr['concedes'])
    preds_concedes = model_concedes.predict_proba(X_val)[:, 1]
    
    auc_scores = roc_auc_score(Y_val['scores'], preds_scores)
    auc_concedes = roc_auc_score(Y_val['concedes'], preds_concedes)
    
    return (auc_scores + auc_concedes) / 2

def main():
    games = load_spadl_data()
    
    study = optuna.create_study(direction=config['optimization']['direction'], study_name=STUDY_NAME)
    # Pass games context to lambda
    study.optimize(lambda trial: objective(trial, games), n_trials=N_TRIALS)
    
    print("Best hyperparameters: ", study.best_params)
    
    # Save best parameters
    if not os.path.exists("results"):
        os.makedirs("results")
    
    res_path = os.path.join("results", "best_params.yaml")
    with open(res_path, 'w') as f:
        yaml.dump(study.best_params, f)
    
    print(f"Saved best parameters to {res_path}")

if __name__ == "__main__":
    main()
