import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib  # for saving/loading the model

def load_psd_data(psd_dir: str, participants_csv: str):
    """
    Loads PSD data and ages from a given directory with 'EC'/'EO' subfolders,
    and a CSV file containing {id, age} columns.

    Returns X, y arrays for training.
    """
    df = pd.read_csv(participants_csv)
    X_list, y_list = [], []

    # for each subject folder
    for sub in os.listdir(psd_dir):
        sub_path = os.path.join(psd_dir, sub)
        ec_dir = os.path.join(sub_path, "EC")
        eo_dir = os.path.join(sub_path, "EO")
        if not (os.path.isdir(ec_dir) and os.path.isdir(eo_dir)):
            continue

        ec_files = set(os.listdir(ec_dir))
        eo_files = set(os.listdir(eo_dir))
        valid_files = ec_files.intersection(eo_files)

        for fname in valid_files:
            ec_data = np.load(os.path.join(ec_dir, fname))  # shape: (channels, freqs)
            eo_data = np.load(os.path.join(eo_dir, fname))  # shape: (channels, freqs)

            # scale each separately, or combine then scale
            scaler_ec = StandardScaler()
            ec_data_scaled = scaler_ec.fit_transform(ec_data).flatten()

            scaler_eo = StandardScaler()
            eo_data_scaled = scaler_eo.fit_transform(eo_data).flatten()

            data_concat = np.hstack([ec_data_scaled, eo_data_scaled])
            X_list.append(data_concat)

            # Find age from CSV (subject ID might be 'sub' as str or int)
            # Adjust as needed if sub is numeric vs. string
            row = df.loc[df['id'] == sub]
            if not row.empty:
                age = row['age'].values[0]
                y_list.append(age)

    return np.array(X_list), np.array(y_list)


def train_svr_model(X, y, c=0.0009765625, kernel="linear", gamma=1.0, nested_cv=False):
    """
    Train an SVR model on X, y.
    If nested_cv=True, runs nested cross-validation to find best hyperparameters.
    Otherwise, uses the provided hyperparams.
    """
    if nested_cv:
        print("Running nested cross-validation. This can be slow...")
        outer_results = []
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        for train_ix, val_ix in cv_outer.split(X):
            x_train, x_val = X[train_ix], X[val_ix]
            y_train_, y_val_ = y[train_ix], y[val_ix]

            # Inner CV for hyperparameter search
            cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
            model = SVR()
            param_grid = {
                "C": [0.0009765625, 0.001953125, 0.00390625],  # example subset
                "kernel": ["linear", "rbf"],
                "gamma": [1e-9, 1e-7, 1e-5]  # example subset
            }
            search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error',
                                  cv=cv_inner, refit=True, error_score='raise')
            result = search.fit(x_train, y_train_)
            best_model = result.best_estimator_
            y_hat = best_model.predict(x_val)
            mae_svm = mean_absolute_error(y_val_, y_hat)
            outer_results.append(mae_svm)
            print('>MAE=%.3f, cfg=%s' % (mae_svm, result.best_params_))

        print('Nested CV Mean MAE=%.3f' % np.mean(outer_results))
        # Return the last best model from GridSearchCV
        return best_model  # you might want to re-fit best params on entire X, y
    else:
        # Train directly with given hyperparams
        print(f"Training SVR with kernel={kernel}, C={c}, gamma={gamma}")
        model = SVR(kernel=kernel, C=c, gamma=gamma)
        model.fit(X, y)
        return model


if __name__ == "__main__":
    # Example usage: train with combined data (HBN + Brain Age).
    
    # 1) Load HBN data
    hbn_psd_dir = "data/HBN_psd"
    hbn_participants_csv = "data/HBN_participants.csv"
    X_hbn, y_hbn = load_psd_data(hbn_psd_dir, hbn_participants_csv)

    # 2) Load Brain Age data (assuming you preprocessed similarly)
    ba_psd_dir = "data/NeuroTex/psds_tutorial"
    ba_participants_csv = "data/NeuroTex/train_subjects.csv"
    X_ba, y_ba = load_psd_data(ba_psd_dir, ba_participants_csv)

    # 3) Combine
    X_train = np.concatenate([X_hbn, X_ba], axis=0)
    y_train = np.concatenate([y_hbn, y_ba], axis=0)

    # 4) Train model (using chosen hyperparams or nested CV)
    model = train_svr_model(X_train, y_train,
                            c=0.0009765625,
                            kernel="linear",
                            gamma=1e-9,
                            nested_cv=False)

    # 5) Save the model
    joblib.dump(model, "svr_brain_age_model.joblib")
    print("Model saved to svr_brain_age_model.joblib")
