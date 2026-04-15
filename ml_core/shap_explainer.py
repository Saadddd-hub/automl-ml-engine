import shap
import numpy as np


class ShapExplainer:

    def __init__(self, model):
        self.model = model
        self.explainer = None

    def create_explainer(self, X_sample):
        """
        Create SHAP explainer based on model type
        """
        try:
            # Tree-based models (XGBoost, RandomForest)
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fallback (for other models like SVM, Logistic)
            self.explainer = shap.KernelExplainer(self.model.predict, X_sample)

    def explain(self, X_sample):
        """
        Generate SHAP values
        """
        shap_values = self.explainer.shap_values(X_sample)
        return shap_values

    def summary_plot(self, shap_values, X_sample):
        """
        Global feature importance
        """
        shap.summary_plot(shap_values, X_sample)

    def force_plot(self, shap_values, X_sample):
        """
        Explain single prediction
        """
        shap.force_plot(self.explainer.expected_value, shap_values[0], X_sample[0])