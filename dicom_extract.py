import os
import zipfile
import pandas as pd
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

# =====================================================
# PATH CONFIG
# =====================================================
BASE_DIR = "/home/djariwal/Desktop/RSNA_training"
DATA_DIR = f"{BASE_DIR}/Data"
ZIP_DIR = f"{DATA_DIR}/Training_Data_Zip"
EXTRACT_DIR = f"{ZIP_DIR}/extracted_new"

METADATA_CSV = f"{DATA_DIR}/train_balanced_50k.csv"
DICOM_INDEX_CSV = f"{BASE_DIR}/dicom_index.csv"
OUTPUT_DIR = f"{BASE_DIR}/png_dataset_25k"

TARGET_SIZE = 256
LABEL_COL = "pe_present_on_image"

TOTAL_DATASET = 25000
RATIO_0 = 0.75
RATIO_1 = 0.25

TARGET_0 = int(TOTAL_DATASET * RATIO_0)
TARGET_1 = int(TOTAL_DATASET * RATIO_1)


# =====================================================
# STAGE 1 — UNZIP ALL VALID ZIP FILES
# =====================================================
def unzip_all():
    """Extracts all valid ZIP files into extracted/ directory."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print("\n=== Extracting ZIP batches (skipping bad zips) ===")
    for f in os.listdir(ZIP_DIR):
        if not f.endswith(".zip"):
            continue

        zip_path = os.path.join(ZIP_DIR, f)

        print(f"\nChecking {f} ...")
        if not zipfile.is_zipfile(zip_path):
            print(f"❌ Skipping INVALID ZIP: {f}")
            continue

        print(f"Extracting {f} ...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(EXTRACT_DIR)
            print(f"✔ Extracted {f}")
        except Exception as e:
            print(f"❌ Failed to extract {f}: {e}")
            continue

    print("\n✔ ZIP extraction stage complete.")


# =====================================================
# STAGE 2 — DICOM INDEXING
# =====================================================
def index_dicoms():
    """Walk folder and index SOPInstanceUID → full path."""
    dicoms = []

    print("\n=== Indexing DICOM files ===")
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".dcm"):
                sop = f.replace(".dcm", "")
                full = os.path.join(root, f)
                dicoms.append((sop, full))

    df = pd.DataFrame(dicoms, columns=["SOPInstanceUID", "filepath"])
    df.to_csv(DICOM_INDEX_CSV, index=False)

    print("✔ DICOM index saved:", DICOM_INDEX_CSV)
    print("Total DICOMs indexed:", len(df))

    return df


# =====================================================
# WINDOWING FUNCTIONS
# =====================================================
def apply_window(img, wl, ww):
    low = wl - ww / 2
    high = wl + ww / 2
    img = np.clip(img, low, high)
    return ((img - low) / (high - low) * 255).astype("uint8")


def pseudo_window_fusion(img):
    p1 = apply_window(img,   40,  400)   # soft tissue
    p2 = apply_window(img, -600, 1400)   # lung
    p3 = apply_window(img,  100,  600)   # bone/blood
    return np.stack([p1, p2, p3], axis=-1)


# =====================================================
# STAGE 3 — EXPORT PNGS (GRAY + PSEUDO)
# =====================================================
def export_images(df, split_name):
    print(f"\n=== Exporting {split_name} images ===")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["filepath"]
        uid = row["SOPInstanceUID"]
        label = row[LABEL_COL]

        # create output directories
        gray_dir = os.path.join(OUTPUT_DIR, split_name, f"label_{label}", "gray")
        pseudo_dir = os.path.join(OUTPUT_DIR, split_name, f"label_{label}", "pseudo")

        os.makedirs(gray_dir, exist_ok=True)
        os.makedirs(pseudo_dir, exist_ok=True)

        f_gray = os.path.join(gray_dir, f"{uid}_gray.png")
        f_pseudo = os.path.join(pseudo_dir, f"{uid}_pseudo.png")

        # read dicom
        try:
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32)
        except Exception as e:
            print("[Decode Error]", path, e)
            continue

        # ---- GREYSCALE ----
        gray = apply_window(img, wl=40, ww=400)
        gray = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE))
        cv2.imwrite(f_gray, gray)

        # ---- PSEUDO ----
        pseudo = pseudo_window_fusion(img)
        pseudo = cv2.resize(pseudo, (TARGET_SIZE, TARGET_SIZE))
        cv2.imwrite(f_pseudo, pseudo)


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    print("\n======= RSNA PIPELINE START =======")

    unzip_all()

    dicom_df = index_dicoms()

    print("\n=== Loading Metadata ===")
    meta = pd.read_csv(METADATA_CSV)

    print("\n=== Merging metadata + filepaths ===")
    df = meta.merge(dicom_df, on="SOPInstanceUID", how="inner")
    print("Total matched images:", len(df))

    df0 = df[df[LABEL_COL] == 0].iloc[:TARGET_0]
    df1 = df[df[LABEL_COL] == 1].iloc[:TARGET_1]

    final_df = pd.concat([df0, df1]).sample(frac=1, random_state=42)

    print("\nFinal dataset distribution:")
    print(final_df[LABEL_COL].value_counts())

    # split 70/15/15
    train = final_df.sample(frac=0.70, random_state=42)
    leftover = final_df.drop(train.index)
    val = leftover.sample(frac=0.50, random_state=42)
    test = leftover.drop(val.index)

    print("\nSplit sizes:")
    print("Train:", len(train))
    print("Val:  ", len(val))
    print("Test: ", len(test))

    export_images(train, "train")
    export_images(val, "val")
    export_images(test, "test")

    print("\n======= PIPELINE COMPLETE =======")


if __name__ == "__main__":
    main()
