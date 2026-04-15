from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor


class ModelTrainer:

    def __init__(self, task_type):
        self.task_type = task_type
        self.models = self._get_models()

    def _get_models(self):
        if self.task_type == "classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "XGBoost": XGBClassifier(eval_metric='mlogloss')
            }

        elif self.task_type == "regression":
            return {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Random Forest": RandomForestRegressor(),
                "SVR": SVR(),
                "XGBoost": XGBRegressor()
            }

    def train(self, X_train, y_train):
        trained_models = {}

        for name, model in self.models.items():
            try:
                print(f"\n🔧 Training: {name}")

                # 🔥 Hyperparameter tuning for selected models
                if name == "Random Forest":
                    param_grid = {
                        "n_estimators": [50, 100],
                        "max_depth": [None, 10, 20]
                    }

                    grid = GridSearchCV(
                        model,
                        param_grid,
                        cv=3,
                        scoring='f1_weighted' if self.task_type == "classification" else 'r2'
                    )

                    grid.fit(X_train, y_train)
                    trained_models[name] = grid.best_estimator_

                    print(f"✅ Best Params: {grid.best_params_}")

                elif name == "XGBoost":
                    param_grid = {
                        "n_estimators": [50, 100],
                        "max_depth": [3, 6],
                        "learning_rate": [0.01, 0.1]
                    }

                    grid = GridSearchCV(
                        model,
                        param_grid,
                        cv=3,
                        scoring='f1_weighted' if self.task_type == "classification" else 'r2'
                    )

                    grid.fit(X_train, y_train)
                    trained_models[name] = grid.best_estimator_

                    print(f"✅ Best Params: {grid.best_params_}")

                else:
                    model.fit(X_train, y_train)
                    trained_models[name] = model

            except Exception as e:
                print(f"❌ Error in {name}: {e}")

        return trained_models