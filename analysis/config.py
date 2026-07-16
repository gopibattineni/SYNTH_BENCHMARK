"""Central configuration for the benchmark analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Results"

GENERATORS = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
    "TabDDPM",
    "ForestDiffusion",
]

CLASSIFIERS = [
    "Logistic Regression",
    "SVM",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "AdaBoost",
    "Gradient Boosting",
    "Naive Bayes",
    "MLP",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVR_RBF",
    "SVR_Linear",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "MLP",
]

CLASSIFICATION_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
REGRESSION_METRICS = ["R2", "RMSE", "MAE", "MSE"]

FIDELITY_METRICS = [
    "Quality_Score",
    "KS_Complement",
    "TV_Complement",
    "JS_Divergence",
    "Wasserstein_Distance",
    "Column_Shapes",
    "Column_Pair_Trends",
    "Correlation_Similarity",
    "CSTest",
]

PRIVACY_METRICS = [
    "Mahalanobis_Distance",
    "Mean_Distance",
    "Median_Distance",
    "NNDR",
    "MIA_AUC",
    "Cosine_Similarity",
    "Distance_to_Closest_Record",
    "Attribute_Disclosure",
    "Identifiability",
]

LEAKAGE_LEVELS = [0, 5, 10, 20, 30, 40, 50]

MODEL_ALIASES = {
    "logreg": "Logistic Regression",
    "logistic regression": "Logistic Regression",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "knn": "KNN",
    "decision tree": "Decision Tree",
    "decisiontree": "Decision Tree",
    "random forest": "Random Forest",
    "randomforest": "Random Forest",
    "extra trees": "Extra Trees",
    "extratrees": "Extra Trees",
    "adaboost": "AdaBoost",
    "gradient boosting": "Gradient Boosting",
    "gradientboost": "Gradient Boosting",
    "gradientboosting": "Gradient Boosting",
    "naive bayes": "Naive Bayes",
    "naivebayes": "Naive Bayes",
    "mlp": "MLP",
    "linear regression": "Linear Regression",
    "linearregression": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "svr_rbf": "SVR_RBF",
    "svr_linear": "SVR_Linear",
    "svr poly": "SVR_Poly",
}

GENERATOR_ALIASES = {
    "wgan-gp": "WGAN_GP",
    "wgan_gp": "WGAN_GP",
    "wgan gp": "WGAN_GP",
    "gaussian copula": "GaussianCopula",
    "gaussiancopula": "GaussianCopula",
    "copulagan": "CopulaGAN",
    "ctab-gan": "CTABGAN",
    "ctabgan+": "CTABGAN",
    "tab-ddpm": "TabDDPM",
    "tabddpm": "TabDDPM",
    "forestdiffusion": "ForestDiffusion",
    "forest diffusion": "ForestDiffusion",
}

# Metrics where higher raw values indicate worse performance (invert for scoring).
LOWER_IS_BETTER = {
    "JS_Divergence",
    "JS Divergence",
    "Wasserstein_Distance",
    "Wasserstein Distance",
    "MIA_AUC",
    "MIA",
    "Mahalanobis_Distance",
    "Mean_Distance",
    "Median_Distance",
    "Distance_to_Closest_Record",
    "Attribute_Disclosure",
    "Identifiability",
    "Accuracy_Drop",
    "F1_Drop",
    "Precision_Drop",
    "Recall_Drop",
    "R2_Drop",
    "MSE_Increase",
    "RMSE_Increase",
    "MAE_Increase",
    "RMSE",
    "MAE",
    "MSE",
    "Utility_Gap",
}

N_SEEDS = 10

FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "pdf", "svg", "eps"]

# IEEE / Nature friendly typography
PLOT_RC = {
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
}


@dataclass
class PipelineConfig:
    """Runtime configuration for a pipeline run."""

    repo_root: Path = REPO_ROOT
    output_dir: Path = RESULTS_DIR
    generators: list[str] = field(default_factory=lambda: GENERATORS.copy())
    n_seeds: int = N_SEEDS
    utility_weight: float = 0.40
    privacy_weight: float = 0.30
    fidelity_weight: float = 0.30
    exclude_dirs: tuple[str, ...] = (
        "Datasets",
        ".git",
        "Results",
        "__pycache__",
        ".venv",
        "venv",
        "excel sheets",
        "docs",
        ".cursor",
        "node_modules",
    )
    figure_formats: list[str] = field(default_factory=lambda: FIGURE_FORMATS.copy())
    figure_dpi: int = FIGURE_DPI

    def subdirs(self) -> dict[str, Path]:
        base = self.output_dir
        mapping = {
            "master": base / "Master_Data",
            "processed": base / "Processed_Data",
            "tables": base / "Tables",
            "figures_utility": base / "Figures" / "Utility",
            "figures_privacy": base / "Figures" / "Privacy",
            "figures_fidelity": base / "Figures" / "Fidelity",
            "figures_tradeoff": base / "Figures" / "Tradeoff",
            "figures_statistical": base / "Figures" / "Statistical",
            "figures_leakage": base / "Figures" / "Leakage",
            "figures_benchmark": base / "Figures" / "Benchmark",
            "figures_similarity": base / "Figures" / "Statistical",
            "supplementary": base / "Supplementary",
            "latex": base / "Latex",
        }
        for path in mapping.values():
            path.mkdir(parents=True, exist_ok=True)
        return mapping
