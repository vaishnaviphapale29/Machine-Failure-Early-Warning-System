from pathlib import Path
import logging

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data" / "ai4i2020.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"
LOG_PATH = BASE_DIR / "project.log"

MODEL_PATH = OUTPUT_DIR / "Machine_Failure_Model.pkl"
METRICS_PATH = OUTPUT_DIR / "model_comparison.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# =========================================================
# DATASET CONFIGURATION
# =========================================================

TARGET_COLUMN = "Machine failure"

NUMERICAL_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

CATEGORICAL_COLUMNS = ["Type"]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# =========================================================
# STEP 1: LOAD DATASET
# =========================================================

def LoadDataset(DataPath):

    logger.info("STEP 1: LOAD DATASET")

    if not DataPath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DataPath}"
        )

    Data = pd.read_csv(DataPath)

    if Data.empty:
        raise ValueError("Dataset is empty.")

    logger.info("Dataset loaded successfully.")
    logger.info("Dataset shape: %s", Data.shape)

    return Data

# =========================================================
# STEP 2: VALIDATE DATASET
# =========================================================

def ValidateDataset(Data):

    logger.info("STEP 2: VALIDATE DATASET")

    RequiredColumns = FEATURE_COLUMNS + [TARGET_COLUMN]

    MissingColumns = [
        Column
        for Column in RequiredColumns
        if Column not in Data.columns
    ]

    if MissingColumns:
        raise ValueError(
            f"Missing required columns: {MissingColumns}"
        )

    Data = Data.copy()

    Data[TARGET_COLUMN] = pd.to_numeric(
        Data[TARGET_COLUMN],
        errors="coerce"
    )

    if Data[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target column contains missing or invalid values."
        )

    if not set(Data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    Data = Data.drop_duplicates().reset_index(drop=True)

    logger.info("Dataset validation completed.")
    logger.info("Dataset shape after duplicate removal: %s", Data.shape)

    return Data

# =========================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# =========================================================

def PerformEDA(Data):

    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")

    print("\nFirst Five Records")
    print(Data.head())

    print("\nDataset Shape")
    print(Data.shape)

    print("\nDataset Information")
    Data.info()

    print("\nMissing Values")
    print(Data.isnull().sum())

    print("\nStatistical Summary")
    print(Data[FEATURE_COLUMNS].describe(include="all"))

    print("\nMachine Failure Distribution")
    print(Data[TARGET_COLUMN].value_counts())

    print("\nFailure Percentage")
    print(
        Data[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # -----------------------------------------------------
    # GRAPH 1: MACHINE FAILURE DISTRIBUTION
    # -----------------------------------------------------

    logger.info("Displaying failure distribution graph.")

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=Data,
        x=TARGET_COLUMN
    )

    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of Records")
    plt.xticks([0, 1], ["No Failure", "Failure"])
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "failure_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # -----------------------------------------------------
    # GRAPH 2: FEATURE CORRELATION
    # -----------------------------------------------------

    logger.info("Displaying feature correlation heatmap.")

    NumericalData = Data[
        NUMERICAL_COLUMNS + [TARGET_COLUMN]
    ]

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        NumericalData.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Numerical Feature Correlation")
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "feature_correlation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    logger.info("EDA completed successfully.")

# =========================================================
# STEP 4: PREPARE FEATURES AND TARGET
# =========================================================

def PrepareData(Data):

    logger.info("STEP 4: PREPARE FEATURES AND TARGET")

    X = Data[FEATURE_COLUMNS].copy()
    y = Data[TARGET_COLUMN].astype(int).copy()

    logger.info("Input features selected: %s", FEATURE_COLUMNS)
    logger.info("Target column: %s", TARGET_COLUMN)

    logger.info(
        "UDI, Product ID and failure-cause columns "
        "are excluded from model inputs."
    )

    return X, y

# =========================================================
# STEP 5: SPLIT DATASET
# =========================================================

def SplitDataset(X, y):

    logger.info("STEP 5: SPLIT DATASET")

    if y.value_counts().min() < 2:
        raise ValueError(
            "Not enough examples of each target class "
            "for stratified train-test splitting."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    logger.info("Training records: %s", len(X_train))
    logger.info("Testing records: %s", len(X_test))

    return X_train, X_test, y_train, y_test

# =========================================================
# STEP 6: CREATE PREPROCESSING PIPELINE
# =========================================================

def CreatePreprocessor():

    logger.info("STEP 6: CREATE PREPROCESSING PIPELINE")

    NumericalPipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    CategoricalPipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    Preprocessor = ColumnTransformer([
        (
            "numerical",
            NumericalPipeline,
            NUMERICAL_COLUMNS
        ),
        (
            "categorical",
            CategoricalPipeline,
            CATEGORICAL_COLUMNS
        )
    ])

    return Preprocessor

# =========================================================
# STEP 7: CREATE MACHINE LEARNING MODELS
# =========================================================

def CreateModels():

    logger.info("STEP 7: CREATE MACHINE LEARNING MODELS")

    LogisticRegressionModel = Pipeline([
        ("preprocessor", CreatePreprocessor()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    RandomForestModel = Pipeline([
        ("preprocessor", CreatePreprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return {
        "Logistic Regression": LogisticRegressionModel,
        "Random Forest": RandomForestModel
    }

# =========================================================
# STEP 8: TRAIN AND COMPARE MODELS
# =========================================================

def TrainAndCompareModels(Models, X_train, y_train):

    logger.info("STEP 8: TRAIN AND COMPARE MODELS")

    MinimumClassCount = y_train.value_counts().min()

    NumberOfFolds = min(5, int(MinimumClassCount))

    if NumberOfFolds < 2:
        raise ValueError(
            "Insufficient class samples for cross-validation."
        )

    CrossValidator = StratifiedKFold(
        n_splits=NumberOfFolds,
        shuffle=True,
        random_state=42
    )

    Results = []
    TrainedModels = {}

    for ModelName, Model in Models.items():

        logger.info("Training model: %s", ModelName)

        Scores = cross_val_score(
            Model,
            X_train,
            y_train,
            cv=CrossValidator,
            scoring="f1",
            n_jobs=1
        )

        Model.fit(X_train, y_train)

        Results.append({
            "Model": ModelName,
            "CV_F1_Mean": Scores.mean(),
            "CV_F1_Std": Scores.std()
        })

        TrainedModels[ModelName] = Model

        logger.info(
            "%s CV F1: %.4f (+/- %.4f)",
            ModelName,
            Scores.mean(),
            Scores.std()
        )

    ResultsData = pd.DataFrame(Results)

    ResultsData = ResultsData.sort_values(
        by="CV_F1_Mean",
        ascending=False
    ).reset_index(drop=True)

    ResultsData.to_csv(
        METRICS_PATH,
        index=False
    )

    BestModelName = ResultsData.loc[0, "Model"]
    BestModel = TrainedModels[BestModelName]

    print("\nCross-Validation Model Comparison")
    print(ResultsData.to_string(index=False))

    logger.info("Selected model: %s", BestModelName)

    return BestModelName, BestModel

# =========================================================
# STEP 9: EVALUATE MODEL
# =========================================================

def EvaluateModel(Model, X_test, y_test):

    logger.info("STEP 9: EVALUATE MODEL")

    Predictions = Model.predict(X_test)
    Probabilities = Model.predict_proba(X_test)[:, 1]

    Metrics = {
        "Accuracy": accuracy_score(y_test, Predictions),
        "Precision": precision_score(
            y_test, Predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, Predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, Predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test, Probabilities
        )
    }

    print("\nModel Evaluation Results")
    print("=" * 55)

    for MetricName, Value in Metrics.items():
        print(f"{MetricName}: {Value:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            Predictions,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # GRAPH 3: CONFUSION MATRIX
    # -----------------------------------------------------

    logger.info("Displaying confusion matrix.")

    Matrix = confusion_matrix(
        y_test,
        Predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        Matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"]
    )

    plt.title("Machine Failure Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # -----------------------------------------------------
    # SAVE TEST PREDICTIONS
    # -----------------------------------------------------

    PredictionResults = X_test.copy()

    PredictionResults["Actual_Failure"] = y_test.to_numpy()
    PredictionResults["Predicted_Failure"] = Predictions
    PredictionResults["Failure_Probability"] = Probabilities

    PredictionResults.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    logger.info("Test predictions saved.")

    return Metrics

# =========================================================
# STEP 10: SAVE TRAINED MODEL
# =========================================================

def SaveModel(Model, ModelName, Metrics):

    logger.info("STEP 10: SAVE TRAINED MODEL")

    ModelBundle = {
        "model": Model,
        "model_name": ModelName,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": Metrics
    }

    joblib.dump(
        ModelBundle,
        MODEL_PATH
    )

    logger.info("Model saved at: %s", MODEL_PATH)

# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    try:

        print("=" * 65)
        print(" MACHINE FAILURE EARLY-WARNING SYSTEM ")
        print("=" * 65)

        Data = LoadDataset(DATA_PATH)

        Data = ValidateDataset(Data)

        PerformEDA(Data)

        X, y = PrepareData(Data)

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = SplitDataset(X, y)

        Models = CreateModels()

        BestModelName, BestModel = TrainAndCompareModels(
            Models,
            X_train,
            y_train
        )

        Metrics = EvaluateModel(
            BestModel,
            X_test,
            y_test
        )

        SaveModel(
            BestModel,
            BestModelName,
            Metrics
        )

        print("\nProject completed successfully.")
        print(f"Best Model: {BestModelName}")
        print(f"Saved Model: {MODEL_PATH}")
        print(f"Charts saved in: {IMAGE_DIR}")

        logger.info("PROJECT COMPLETED SUCCESSFULLY")

    except Exception:

        logger.exception("Project execution failed.")
        raise


if __name__ == "__main__":
    main()