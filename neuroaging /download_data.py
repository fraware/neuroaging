import os
import pandas as pd
import wget
import scipy.io as sio
import numpy as np
from tqdm import tqdm


def download_hbn_data(participants_tsv_url: str,
                      hbn_output_dir: str,
                      participants_output_csv: str,
                      s3_url_template: str):
    """
    Download HBN participants file and each subject's .mat EEG data.
    
    Parameters
    ----------
    participants_tsv_url : str
        URL to the participants.tsv file.
    hbn_output_dir : str
        Path to save the .mat or .npy EEG data.
    participants_output_csv : str
        Where to save the final participants.csv file.
    s3_url_template : str
        Format string for S3 URL, e.g.:
        'https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG/{}/EEG/raw/mat_format/RestingState.mat'
    """
    # 1) Download participants TSV
    if not os.path.exists('participants.tsv'):
        print("Downloading participants file...")
        wget.download(participants_tsv_url, 'participants.tsv')
        
    participants = pd.read_csv('participants.tsv', sep='\t')[["participant_id", "Age"]]
    participants.rename(columns={"participant_id": "id", "Age": "age"}, inplace=True)
    participants.drop_duplicates(subset="id", inplace=True)
    participants.dropna(subset=["age"], inplace=True)
    
    if not os.path.exists(hbn_output_dir):
        os.makedirs(hbn_output_dir)
    
    # 2) Save participants to CSV
    participants.to_csv(participants_output_csv, index=False)
    print(f"Saved participants CSV to {participants_output_csv}")
    print(f"Number of participants: {len(participants)}")
    
    # 3) Download EEG data
    for subject in tqdm(participants["id"]):
        save_file = os.path.join(hbn_output_dir, f"{subject}.npy")
        if not os.path.exists(save_file):
            # Construct S3 URL
            url = s3_url_template.format(subject)
            try:
                # Download the .mat file
                mat_file = wget.download(url)
                sub_dict = sio.loadmat(mat_file)
                np.save(save_file, sub_dict)
                os.remove(mat_file)  # remove the .mat after converting to .npy
            except Exception as e:
                # If download fails or file isn't found
                print(f"\nWarning: Could not download or process subject {subject}. Error: {e}")


if __name__ == "__main__":
    # Example usage:
    PARTICIPANTS_TSV_URL = "https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG/participants.tsv"
    HBN_OUTPUT_DIR = "data/HBN_raw"
    PARTICIPANTS_OUTPUT_CSV = "data/HBN_participants.csv"
    S3_TEMPLATE = "https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG/{}/EEG/raw/mat_format/RestingState.mat"

    download_hbn_data(
        participants_tsv_url=PARTICIPANTS_TSV_URL,
        hbn_output_dir=HBN_OUTPUT_DIR,
        participants_output_csv=PARTICIPANTS_OUTPUT_CSV,
        s3_url_template=S3_TEMPLATE
    )
