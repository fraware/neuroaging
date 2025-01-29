import os
import numpy as np
import pandas as pd
import mne
from pyprep.find_noisy_channels import NoisyChannels
from tqdm import tqdm
from mne.time_frequency import psd_array_welch
from mne.io import RawArray
 
def preprocess_hbn_data(raw_dir: str, preprocessed_dir: str, montage_file: str,
                        l_freq=0.1, h_freq=45, sfreq=100):
    """
    Preprocess the HBN .npy EEG data:
      - Load data
      - Split into EC/EO segments
      - Detect & interpolate bad channels
      - Filter & downsample
      - Save to disk
    """
    montage = mne.channels.read_custom_montage(montage_file)
    info = mne.create_info(montage.ch_names, sfreq=500, ch_types='eeg')  # original SF is 500

    # For each subject, create directories for EC/EO
    subjects = [f.replace('.npy', '') for f in os.listdir(raw_dir) if f.endswith('.npy')]
    for sub in tqdm(subjects):
        sub_save_path = os.path.join(preprocessed_dir, sub)
        if not os.path.exists(sub_save_path):
            os.makedirs(os.path.join(sub_save_path, "EC"), exist_ok=True)
            os.makedirs(os.path.join(sub_save_path, "EO"), exist_ok=True)
        else:
            # If it already exists, skip
            continue

        npy_file = os.path.join(raw_dir, f"{sub}.npy")
        file_data = np.reshape(np.load(npy_file, allow_pickle=True), (1,))
        try:
            data = file_data[0]["EEG"][0]["data"][0]
            if data.shape[0] < 129:
                continue
        except KeyError:
            continue

        # Event info
        events = pd.DataFrame(file_data[0]["EEG"][0]["event"][0][0])
        EO_times = events[events.type == "20  "]["sample"].to_numpy(dtype=int)
        EC_times = events[events.type == "30  "]["sample"].to_numpy(dtype=int)

        n = 1
        for i, j in zip(EO_times, EC_times):
            try:
                EO = data[:, i:j]
                # get the next EO time if it exists
                next_index = np.where(EO_times == j)[0]
                if len(next_index) == 0:
                    # no next index found
                    continue
                n_idx = next_index[0] + 1
                if n_idx < len(EO_times):
                    EC = data[:, j:EO_times[n_idx]]
                else:
                    continue
            except IndexError:
                continue

            # Now do MNE preprocessing for each of (EO, EC)
            for k, trial_data in enumerate([EO, EC]):
                raw = RawArray(trial_data, info.copy())
                try:
                    # find bad channels
                    nc = NoisyChannels(raw)
                    nc.find_all_bads()
                    if len(nc.get_bads()) > 30:
                        continue
                except:
                    continue

                raw.info['bads'].extend(nc.get_bads())
                raw.interpolate_bads(reset_bads=True)
                raw.filter(l_freq, h_freq)
                raw.resample(sfreq)
                out_data, _ = raw[:]

                # Save the resulting data
                if k == 0:
                    # EO
                    eo_path = os.path.join(sub_save_path, "EO", f"{n}.npy")
                    np.save(eo_path, out_data)
                else:
                    # EC
                    ec_path = os.path.join(sub_save_path, "EC", f"{n}.npy")
                    np.save(ec_path, out_data)
            n += 1


def compute_psd_hbn(preprocessed_dir: str, psd_dir: str, montage_file: str,
                    fmin=1, fmax=45, sfreq=100):
    """
    For each subject's EC/EO files, compute PSD and save to disk.
    """
    montage = mne.channels.read_custom_montage(montage_file)
    info = mne.create_info(montage.ch_names, sfreq, 'eeg')

    subjects = os.listdir(preprocessed_dir)
    for sub in tqdm(subjects):
        sub_src_path = os.path.join(preprocessed_dir, sub)
        sub_save_path = os.path.join(psd_dir, sub)
        if not os.path.exists(sub_save_path):
            os.makedirs(os.path.join(sub_save_path, "EC"), exist_ok=True)
            os.makedirs(os.path.join(sub_save_path, "EO"), exist_ok=True)
        else:
            continue

        # List of files in EC and EO
        ec_files = set(os.listdir(os.path.join(sub_src_path, "EC")))
        eo_files = set(os.listdir(os.path.join(sub_src_path, "EO")))
        common = ec_files.intersection(eo_files)
        
        for fname in common:
            for condition in ["EC", "EO"]:
                fpath = os.path.join(sub_src_path, condition, fname)
                data = np.load(fpath)
                raw = RawArray(data, info.copy())
                arr, _ = raw[:]
                psds, freqs = psd_array_welch(arr, sfreq, fmin=fmin, fmax=fmax, n_fft=sfreq)
                
                out_path = os.path.join(sub_save_path, condition, fname)
                np.save(out_path, psds)


if __name__ == "__main__":
    # Example usage for HBN data
    raw_dir = "data/HBN_raw"   # Where .npy HBN data are stored
    preproc_dir = "data/HBN_preprocessed"
    psd_dir = "data/HBN_psd"

    # Change to your own channel location file
    montage_file = "data/GSN_HydroCel_129.sfp"

    # 1) Preprocess
    preprocess_hbn_data(raw_dir, preproc_dir, montage_file)

    # 2) Compute PSD
    compute_psd_hbn(preproc_dir, psd_dir, montage_file)
