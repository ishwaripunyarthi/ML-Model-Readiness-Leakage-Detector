import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def final_evaluation_pipeline(df, target_col, time_col, threshold=0.02):

    df = df.copy()

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, target_col])

    df = df.sort_values(by=time_col).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found. Upload a dataset with numeric features.")

    X = X[numeric_cols]

    X = X.fillna(X.median())

    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_random = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42
    )

    model_random.fit(X_train_r, y_train_r)
    y_pred_random = model_random.predict_proba(X_test_r)[:, 1]
    auc_random = roc_auc_score(y_test_r, y_pred_random)

    
    split_index = int(len(df) * 0.7)

    X_train_t = X.iloc[:split_index]
    y_train_t = y.iloc[:split_index]
    X_test_t = X.iloc[split_index:]
    y_test_t = y.iloc[split_index:]

    if len(X_train_t) == 0 or len(X_test_t) == 0:
        raise ValueError("Time-based split failed due to insufficient data.")

    model_time = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42
    )

    model_time.fit(X_train_t, y_train_t)
    y_pred_time = model_time.predict_proba(X_test_t)[:, 1]
    auc_time = roc_auc_score(y_test_t, y_pred_time)

    
    leakage_score = auc_random - auc_time

    if leakage_score > threshold:
        verdict = "⚠️ Potential Temporal Leakage Detected"
        deployment_decision = "⛔ DO NOT DEPLOY this model"
    else:
        verdict = "✅ No Significant Temporal Leakage"
        deployment_decision = "🚀 SAFE TO DEPLOY this model"

    return {
        "Random Split AUC": round(float(auc_random), 4),
        "Time-Based Split AUC": round(float(auc_time), 4),
        "Leakage Score": round(float(leakage_score), 4),
        "Leakage Verdict": verdict,
        "Deployment Decision": deployment_decision
    }



if __name__ == "__main__":
    df = pd.read_csv("creditcard.csv")

    results = final_evaluation_pipeline(
        df=df,
        target_col="Class",
        time_col="Time"
    )

    print(results)
