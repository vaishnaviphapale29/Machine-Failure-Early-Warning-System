import joblib
import pandas as pd
from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "Machine_Failure_Model.pkl"

# =========================================================
# STEP 1: LOAD TRAINED MODEL
# =========================================================

def LoadModel():

    print("\n" + "=" * 55)
    print("STEP 1: LOAD TRAINED MODEL")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run Machine_Failure_Early_Warning_System.py first."
        )

    ModelBundle = joblib.load(MODEL_PATH)

    # Extract the trained model from the saved dictionary
    Model = ModelBundle["model"]

    print("Model loaded successfully.")
    print("Selected Model:", ModelBundle["model_name"])

    return Model

# =========================================================
# STEP 2: GET NEW MACHINE DATA
# =========================================================

def GetMachineData():

    print("\n" + "=" * 55)
    print("STEP 2: ENTER MACHINE DETAILS")
    print("=" * 55)

    AirTemperature = float(
        input("Air Temperature [K]: ")
    )

    ProcessTemperature = float(
        input("Process Temperature [K]: ")
    )

    RotationalSpeed = float(
        input("Rotational Speed [rpm]: ")
    )

    Torque = float(
        input("Torque [Nm]: ")
    )

    ToolWear = float(
        input("Tool Wear [min]: ")
    )

    MachineType = input(
        "Machine Type (L / M / H): "
    ).strip().upper()

    if MachineType not in ["L", "M", "H"]:
        raise ValueError(
            "Invalid machine type. Enter L, M, or H."
        )

    if AirTemperature <= 0:
        raise ValueError("Air temperature must be positive.")

    if ProcessTemperature <= 0:
        raise ValueError("Process temperature must be positive.")

    if RotationalSpeed <= 0:
        raise ValueError("Rotational speed must be positive.")

    if Torque < 0:
        raise ValueError("Torque cannot be negative.")

    if ToolWear < 0:
        raise ValueError("Tool wear cannot be negative.")

    MachineData = pd.DataFrame([{
        "Air temperature [K]": AirTemperature,
        "Process temperature [K]": ProcessTemperature,
        "Rotational speed [rpm]": RotationalSpeed,
        "Torque [Nm]": Torque,
        "Tool wear [min]": ToolWear,
        "Type": MachineType
    }])

    return MachineData

# =========================================================
# STEP 3: PREDICT MACHINE FAILURE
# =========================================================

def PredictFailure(Model, MachineData):

    print("\n" + "=" * 55)
    print("STEP 3: MACHINE FAILURE PREDICTION")
    print("=" * 55)

    Prediction = Model.predict(MachineData)[0]

    if Prediction == 1:
        print("Prediction: MACHINE FAILURE")
        print("Warning: The model predicts a possible failure.")
    else:
        print("Prediction: NO MACHINE FAILURE")
        print("The model predicts no failure for this input.")

    if hasattr(Model, "predict_proba"):

        Probabilities = Model.predict_proba(
            MachineData
        )[0]

        Classes = list(Model.classes_)

        if 1 in Classes:

            FailureProbability = (
                Probabilities[Classes.index(1)] * 100
            )

            print(
                f"Estimated failure probability: "
                f"{FailureProbability:.2f}%"
            )

    print("=" * 55)

# =========================================================
# STEP 4: MAIN FUNCTION
# =========================================================

def main():

    print("\n")
    print("=" * 55)
    print(" MACHINE FAILURE EARLY-WARNING SYSTEM")
    print(" NEW DATA TESTING")
    print("=" * 55)

    try:

        Model = LoadModel()

        MachineData = GetMachineData()

        PredictFailure(
            Model,
            MachineData
        )

    except ValueError as Error:
        print("\nInvalid input:", Error)

    except FileNotFoundError as Error:
        print("\nError:", Error)

    except Exception as Error:
        print("\nUnexpected error:", Error)

# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":
    main()
