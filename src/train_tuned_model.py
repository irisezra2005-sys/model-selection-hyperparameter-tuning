import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. LOAD DATASET

URL = "/content/winequality-red.csv"

df = pd.read_csv(URL)


# 2. CREATE TARGET VARIABLE


# Quality >= 7 is considered good quality (1)
# Quality < 7 is considered not good quality (0)

df["target"] = (df["quality"] >= 7).astype(int)

# Remove original quality column
df = df.drop(columns=["quality"])

# 3. SEPARATE FEATURES AND TARGET

X = df.drop(columns=["target"])
y = df["target"]



# 4. TRAIN / VALIDATION / TEST SPLIT


# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split remaining 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Dataset split:")
print(f"Training set   : {X_train.shape}")
print(f"Validation set : {X_val.shape}")
print(f"Test set       : {X_test.shape}")

# 5. DEFINE MODELS

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(random_state=42))
])

gb_pipeline = Pipeline([
    ("model", GradientBoostingClassifier(random_state=42))
])

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=42))
])


# 6. DEFINE HYPERPARAMETER GRIDS

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

gb_params = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 5],
    "model__min_samples_split": [2, 5]
}

svm_params = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"]
}

# 7. STORE MODELS AND PARAMETER GRIDS

models = {
    "Random Forest": (rf_pipeline, rf_params),
    "Gradient Boosting": (gb_pipeline, gb_params),
    "SVM": (svm_pipeline, svm_params)
}


# 8. GRID SEARCH

results = {}
best_models = {}

for name, (model, param_grid) in models.items():

    print("\n" + "=" * 60)
    print(f"Tuning {name}")
    print("=" * 60)

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    best_model = grid_search.best_estimator_

    best_models[name] = best_model

 -
    # Validation predictions
   
    val_predictions = best_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )
    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )
    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    # Test predictions

    test_predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )
    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )
    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    # Store results
  
    results[name] = {
        "Best Parameters": grid_search.best_params_,
        "CV F1 Score": grid_search.best_score_,
        "Validation Accuracy": val_accuracy,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall,
        "Validation F1": val_f1,
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1,
        "Training Time (seconds)": training_time
    }

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(f"\nBest CV F1 Score: {grid_search.best_score_:.4f}")

    print(f"Validation Accuracy : {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall   : {val_recall:.4f}")
    print(f"Validation F1       : {val_f1:.4f}")

    print(f"\nTest Accuracy : {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall   : {test_recall:.4f}")
    print(f"Test F1       : {test_f1:.4f}")

    print(f"\nTraining Time: {training_time:.2f} seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))


# 9. COMPARATIVE RESULTS TABLE

results_df = pd.DataFrame(results).T

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df[
        [
            "CV F1 Score",
            "Validation Accuracy",
            "Validation F1",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1",
            "Training Time (seconds)"
        ]
    ].round(4)
)

# 10. SELECT BEST MODEL

best_model_name = results_df["Validation F1"].idxmax()

final_model = best_models[best_model_name]

print("\n" + "=" * 60)
print("FINAL MODEL SELECTION")
print("=" * 60)

print(f"Selected Model: {best_model_name}")
print(
    f"Validation F1 Score: "
    f"{results_df.loc[best_model_name, 'Validation F1']:.4f}"
)


# 11. SAVE BEST MODEL

joblib.dump(
    final_model,
    "best_tuned_model.pkl"
)

print("\nBest tuned model saved as:")
print("best_tuned_model.pkl")


# 12. SAVE COMPARISON RESULTS

results_df.to_csv(
    "model_comparison_results.csv"
)

print("\nComparison results saved as:")
print("model_comparison_results.csv")
