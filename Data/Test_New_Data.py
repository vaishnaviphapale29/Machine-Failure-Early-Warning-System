import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# STEP 1: DEFINE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "Machine_Failure_Model.pkl"

# --------------------------------------------------
# STEP 2: LOAD TRAINED MODEL
# --------------------------------------------------

def LoadModel():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run Machine_Failure_Early_Warning_System.py first."
        )

    ModelBundle = joblib.load(MODEL_PATH)

    Model = ModelBundle["model"]

    print("\nModel loaded successfully.")
    print("Selected Model:", ModelBundle["model_name"])

    return Model


# --------------------------------------------------
# STEP 3: GET NEW MACHINE DATA
# --------------------------------------------------

def GetMachineData():

    print("\nEnter machine details:")

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
            "Machine Type must be L, M, or H."
        )

    MachineData = pd.DataFrame([{
        "Type": MachineType,
        "Air temperature [K]": AirTemperature,
        "Process temperature [K]": ProcessTemperature,
        "Rotational speed [rpm]": RotationalSpeed,
        "Torque [Nm]": Torque,
        "Tool wear [min]": ToolWear
    }])

    return MachineData

# --------------------------------------------------
# STEP 4: PREDICT MACHINE FAILURE
# --------------------------------------------------

def PredictFailure(Model, MachineData):

    Prediction = Model.predict(MachineData)[0]

    print("\n" + "=" * 45)
    print("MACHINE FAILURE PREDICTION")
    print("=" * 45)

    if Prediction == 1:
        print("Warning: Possible machine failure detected!")
    else:
        print("Prediction: No failure detected by the model.")

    if hasattr(Model, "predict_proba"):

        Probabilities = Model.predict_proba(
            MachineData
        )[0]

        Classes = list(Model.classes_)

        if 1 in Classes:
            FailureProbability = Probabilities[
                Classes.index(1)
            ] * 100

            print(
                f"Estimated failure probability: "
                f"{FailureProbability:.2f}%"
            )

    print("=" * 45)

# --------------------------------------------------
# STEP 5: MAIN FUNCTION
# --------------------------------------------------

def main():

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

if __name__ == "__main__":
    main()
