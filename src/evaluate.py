"""
Model Evaluation Module
Evaluates saved models and prints key metrics
"""
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, brier_score_loss

MODELS = {
    'Logistic Regression': 'models/logistic_regression_model.pkl',
    'Random Forest': 'models/random_forest_model.pkl',
    'XGBoost': 'models/xgboost_model.pkl'
}

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def load_test_data():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['home_win']
    return X_test, y_test

def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        'accuracy': accuracy_score(y, pred),
        'brier_score': brier_score_loss(y, prob),
        'predictions': pred,
        'probabilities': prob
    }

def print_results(results, model_name):
    print(f"\n{model_name}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Brier Score: {results['brier_score']:.3f}")

def main():
    print("\nNBA Game Win Probability Prediction model evaluation")
    X_test, y_test = load_test_data()
    
    all_results = {}
    for name, path in MODELS.items():
        model = load_model(path)
        results = evaluate(model, X_test, y_test)
        print_results(results, name)
        all_results[name] = results
    
    # Simple comparison
    comparison = pd.DataFrame({
        name: {'Accuracy': r['accuracy'], 
               'Brier': r['brier_score']} 
        for name, r in all_results.items()
    }).T.sort_values('Accuracy', ascending=False)
    
    print("\n Model Comparison ")
    print(comparison)
    
    return all_results

if __name__ == "__main__":
    results = main()