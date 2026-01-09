import mlflow
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTrainer:
    def __init__(
        self,
        iterations=800,
        depth=8,
        learning_rate=0.05,
        random_seed=42
    ):
        self.params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "loss_function": "RMSE",
            "random_seed": random_seed,
            "verbose": False,
        }

        self.model = None

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "rmse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def _log_shap(self, X_val):
        """
        Compute SHAP values and log plots/artifacts to MLflow.
        """

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_val)

        # --- Summary plot ---
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_val,
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        plt.close()

        mlflow.log_artifact("shap_summary.png")

        # --- Mean absolute SHAP values ---
        shap_importance = (
            pd.DataFrame({
                "feature": X_val.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0)
            })
            .sort_values("mean_abs_shap", ascending=False)
        )

        shap_importance.to_csv("shap_feature_importance.csv", index=False)
        mlflow.log_artifact("shap_feature_importance.csv")

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        cat_cols,
        experiment_name="final_best_model",
        run_name="catboost_final"
    ):
        """
        Train CatBoost model, log MLflow metrics, model, and SHAP explanations.
        """

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):

            # -------------------------
            # Params & metadata
            # -------------------------
            mlflow.log_params(self.params)
            mlflow.log_param("model_family", "catboost")
            mlflow.log_param("encoding", "catboost_native")
            mlflow.log_param("n_features", X_train.shape[1])

            cat_feature_indices = [
                X_train.columns.get_loc(col) for col in cat_cols
            ]

            # -------------------------
            # Train
            # -------------------------
            self.model = CatBoostRegressor(**self.params)

            self.model.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
                eval_set=(X_val, y_val),
                use_best_model=True
            )

            # -------------------------
            # Evaluate
            # -------------------------
            preds = self.model.predict(X_val)

            metrics = self.evaluate(
                np.expm1(y_val),
                np.expm1(preds)
            )

            mlflow.log_metrics(metrics)

            # -------------------------
            # SHAP explainability
            # -------------------------
            self._log_shap(X_val)

            # -------------------------
            # Log model
            # -------------------------
            mlflow.catboost.log_model(
                self.model,
                artifact_path="model"
            )

        return self.model
