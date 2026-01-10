
import pandas as pd
import yaml
import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import socceraction.vaep.features as features
import socceraction.vaep.labels as labels

CONFIG_PATH = 'config.yaml'
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = '../config.yaml'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

SPADL_DIR = config['data']['spadl_output_dir']
if not os.path.exists(SPADL_DIR):
    SPADL_DIR = os.path.join('..', config['data']['spadl_output_dir'])

# Load best params to check if nb_prev_actions is optimized
BEST_PARAMS_PATH = 'results/best_params.yaml'
if os.path.exists(BEST_PARAMS_PATH):
    with open(BEST_PARAMS_PATH, 'r') as f:
        best_params = yaml.safe_load(f)
        NB_PREV_ACTIONS = best_params.get('nb_prev_actions', config['features']['nb_prev_actions'])
else:
    NB_PREV_ACTIONS = config['features']['nb_prev_actions']

print(f"Using nb_prev_actions = {NB_PREV_ACTIONS}")

def load_spadl_data():
    all_games = []
    files = glob.glob(os.path.join(SPADL_DIR, "game_*.csv"))
    if not files:
        files = glob.glob("data/spadl/game_*.csv")
        
    print(f"Loading {len(files)} games for model comparison...")
    for f in files:
        df = pd.read_csv(f)
        all_games.append(df)
    return all_games

def get_feature_functions():
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

def generate_dataset(games):
    X_list = []
    Y_list = []
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
        except Exception as e:
            continue

    if not X_list:
        raise ValueError("No data generated.")

    X = pd.concat(X_list).reset_index(drop=True)
    Y = pd.concat(Y_list).reset_index(drop=True)
    return X, Y

def main():
    games = load_spadl_data()
    X, Y = generate_dataset(games)
    
    # Rename columns to ensure LightGBM compatibility (no special chars or duplicate names ideally)
    # Simple rename by index ensures safety
    X.columns = [f"feat_{i}" for i in range(X.shape[1])]
    
    print(f"Dataset shape: {X.shape}")
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    models = {
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        # Scoring Model
        model.fit(X_train, Y_train['scores'])
        pred_scores = model.predict_proba(X_val)[:, 1]
        auc_scores = roc_auc_score(Y_val['scores'], pred_scores)
        
        # Conceding Model
        model.fit(X_train, Y_train['concedes'])
        pred_concedes = model.predict_proba(X_val)[:, 1]
        auc_concedes = roc_auc_score(Y_val['concedes'], pred_concedes)
        
        results[name] = {
            'AUC (Scores)': auc_scores,
            'AUC (Concedes)': auc_concedes,
            'Average AUC': (auc_scores + auc_concedes) / 2,
            'pred_scores': pred_scores,
            'pred_concedes': pred_concedes
        }
        print(f"{name} Results: {results[name]['Average AUC']:.4f}")

    # Plotting
    plt.figure(figsize=(15, 5))

    # Scoring ROC
    plt.subplot(1, 2, 1)
    for name in models.keys():
        fpr, tpr, _ = roc_curve(Y_val['scores'], results[name]['pred_scores'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['AUC (Scores)']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Scoring')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Conceding ROC
    plt.subplot(1, 2, 2)
    for name in models.keys():
        fpr, tpr, _ = roc_curve(Y_val['concedes'], results[name]['pred_concedes'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['AUC (Concedes)']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Conceding')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.tight_layout()
    output_img = 'results/model_comparison_roc_extended.png'
    if not os.path.exists('results'):
        if os.path.exists('../results'):
             output_img = '../results/model_comparison_roc_extended.png'
        else:
             os.makedirs('results')
             
    plt.savefig(output_img)
    print(f"Saved comparison plot to {output_img}")

if __name__ == "__main__":
    main()
