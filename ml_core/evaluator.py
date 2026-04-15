from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score


class ModelEvaluator:

    def __init__(self, task_type):
        self.task_type = task_type

    # 1️⃣ Evaluation
    def evaluate(self, models, X_test, y_test):
        results = {}

        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)

                if self.task_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    results[name] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }

                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    results[name] = {
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2
                    }

            except Exception as e:
                print(f"❌ Evaluation failed for {name}: {e}")

        return results

    # 2️⃣ Cross-Validation
    def cross_validate(self, models, X, y):
        cv_results = {}

        for name, model in models.items():
            try:
                if self.task_type == "classification":
                    scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
                else:
                    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

                cv_results[name] = {
                    "cv_mean": scores.mean(),
                    "cv_std": scores.std()
                }

            except Exception as e:
                print(f"❌ CV failed for {name}: {e}")

        return cv_results

    # 3️⃣ Leaderboard (FIXED INDENTATION ✅)
    def get_leaderboard(self, results, cv_results=None):
        leaderboard = []

        for model, metrics in results.items():

            if self.task_type == "classification":
                score = metrics["f1_score"]
            else:
                score = metrics["r2"]

            # Combine with CV
            if cv_results and model in cv_results:
                score = (score + cv_results[model]["cv_mean"]) / 2

            leaderboard.append((model, score))

        leaderboard.sort(key=lambda x: x[1], reverse=True)

        return leaderboard

    # 4️⃣ Best Model
    def get_best_model(self, models, leaderboard):
        best_model_name = leaderboard[0][0]
        best_model = models[best_model_name]

        return best_model_name, best_model