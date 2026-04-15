import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

from ml_core.preprocessing import DataPreprocessor
from ml_core.trainer import ModelTrainer
from ml_core.evaluator import ModelEvaluator
from ml_core.utils import detect_target_column

def run_training_pipeline(file_path, job_id, update_fn=None):
    try:
        if update_fn: update_fn(job_id, {"progress": 10})

        df = pd.read_csv(file_path)

        target = detect_target_column(df)

        task_type = "classification" if df[target].nunique() < 20 else "regression"

        if update_fn: update_fn(job_id, {"progress": 25})

        preprocessor = DataPreprocessor(target)
        X, y = preprocessor.fit_transform(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if update_fn: update_fn(job_id, {"progress": 40})

        trainer = ModelTrainer(task_type)
        models = trainer.train(X_train, y_train)

        if update_fn: update_fn(job_id, {"progress": 60})

        evaluator = ModelEvaluator(task_type)
        cv_results = evaluator.cross_validate(models, X, y)
        results = evaluator.evaluate(models, X_test, y_test)

        leaderboard = evaluator.get_leaderboard(results, cv_results)

        best_model_name, best_model = evaluator.get_best_model(models, leaderboard)

        if update_fn: update_fn(job_id, {"progress": 80})

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{job_id}.pkl"
        joblib.dump(best_model, model_path)

        if update_fn: update_fn(job_id, {"progress": 100})

        return {
            "status": "completed",
            "target_column": target,
            "task_type": task_type,
            "best_model": best_model_name,
            "accuracy": leaderboard[0][1],
            "model_path": model_path,

            "leaderboard": leaderboard,          # 🔥 ranking
            "cv_results": cv_results,            # 🔁 CV scores
            "metrics": results                  # 📊 all model metrics
        }   

    except Exception as e:
        return {"status": "failed", "error": str(e)}