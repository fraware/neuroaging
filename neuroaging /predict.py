import os
import numpy as np
import pandas as pd
from train import load_psd_data
import joblib
 
def predict_test_data(model_path: str,
                      test_psd_dir: str,
                      test_subjects_csv: str,
                      submission_csv: str):
    """
    Load a trained model, predict on test PSD data, and create a submission CSV.
    """
    # Load the model
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Load test PSD data
    X_test, ids = load_test_data(test_psd_dir, test_subjects_csv)

    # Predict
    y_pred = model.predict(X_test)

    # Save to submission
    df = pd.DataFrame({"id": ids, "age": y_pred})
    df.to_csv(submission_csv, index=False)
    print(f"Submission saved to {submission_csv}")


def load_test_data(psd_dir: str, subjects_csv: str):
    """
    Similar to load_psd_data, but we might not have 'age' for test set.
    Return X_test and an array of subject IDs for building submission file.
    """
    df = pd.read_csv(subjects_csv)  # e.g. {id}
    X_list, id_list = [], []

    for sub in df["id"].values:
        sub_str = str(sub)
        sub_path = os.path.join(psd_dir, sub_str)
        ec_dir = os.path.join(sub_path, "EC")
        eo_dir = os.path.join(sub_path, "EO")
        if not (os.path.isdir(ec_dir) and os.path.isdir(eo_dir)):
            # skip if missing
            continue

        ec_file = os.path.join(ec_dir, "1.npy")  # if there's only one file
        eo_file = os.path.join(eo_dir, "1.npy")

        if not os.path.exists(ec_file) or not os.path.exists(eo_file):
            continue

        ec_data = np.load(ec_file)
        eo_data = np.load(eo_file)

        # scale each trial
        from sklearn.preprocessing import StandardScaler
        ec_data = StandardScaler().fit_transform(ec_data).flatten()
        eo_data = StandardScaler().fit_transform(eo_data).flatten()
        data_concat = np.hstack([ec_data, eo_data])
        X_list.append(data_concat)
        id_list.append(sub)

    return np.array(X_list), np.array(id_list)


if __name__ == "__main__":
    MODEL_PATH = "svr_brain_age_model.joblib"
    TEST_PSD_DIR = "data/NeuroTex/test_psds"
    TEST_SUBJECTS_CSV = "data/NeuroTex/test_subjects.csv"
    SUBMISSION_CSV = "submission.csv"

    predict_test_data(MODEL_PATH, TEST_PSD_DIR, TEST_SUBJECTS_CSV, SUBMISSION_CSV)
