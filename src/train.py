"""
Model Training Module
Trains ML models for NBA game prediction
"""

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

def load_training_data():
    print("Loading the training data")

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['home_win']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['home_win']

    print(f"\nTraining: {len(X_train)}")
    print(f"\nTest samples: {len(X_test)}")

    return X_train, y_train, X_test, y_test

def train_logreg(X_train, y_train):
    print("\nTraining Logistic Regression")
    # As far as the parameters go, I am not sure what will work best. Might optimize this later but works fine for now
    # That goes for the every model from now on
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)

    model.fit(X_train, y_train)
    print("\nLogistic Regression done")

    return model

def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest")

    model = RandomForestClassifier(
        n_estimators=150,  # not too many trees, keeps training fast
        max_depth=10,      # prevents overfitting
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("\nRandom Forest done")
    return model

def train_xgboost(X_train, y_train):
    print("\nTrainig XGBoost")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
    )

    model.fit(X_train, y_train)
    print("\nXGBoost done")

    return model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]    

    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred_proba)

    results = {
        'model': model_name,
        'accuracy': accuracy,
        'brier_score': brier,
    }

    print("\nResults:")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nBrierScore: {brier:.4f}")
    
    return results

def save_model(model, name):
    filename = f"models/{name.lower().replace(' ','_')}_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model: {filename}")

def main():
    print("\nNBA Game Win Probability Prediction model training")

    X_train, y_train, X_test, y_test = load_training_data()

    models = {}
    results_list = []    

    # Logistic Regression
    lr = train_logreg(X_train, y_train)
    results_list.append(evaluate_model(lr, X_test, y_test, "Logistic Regression"))
    save_model(lr, "Logistic Regression")
    models['Logistic Regression'] = lr

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    results_list.append(evaluate_model(rf, X_test, y_test, "Random Forest"))
    save_model(rf, "Random Forest")
    models['Random Forest'] = rf

    # XGBoost
    xgb = train_xgboost(X_train, y_train)
    results_list.append(evaluate_model(xgb, X_test, y_test, "XGBoost"))
    save_model(xgb, "XGBoost")
    models['XGBoost'] = xgb    

    results_df = pd.DataFrame(results_list).sort_values('accuracy', ascending=False)
    print("\nModel Comparison (sorted by accuracy)")
    print(results_df.to_string(index=False))
    
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("\nSaved model comparison to results/model_comparison.csv")
    
    print("\nTraining done")
    return models, results_df    

if __name__ == "__main__":
    main()