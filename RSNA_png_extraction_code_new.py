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
OUTPUT_DIR = f"{BASE_DIR}/new_pseudo_dataset_25k"

TARGET_SIZE = 256
LABEL_COL = "pe_present_on_image"

TOTAL_DATASET = 25000
RATIO_0 = 0.75
RATIO_1 = 0.25

TARGET_0 = int(TOTAL_DATASET * RATIO_0)
TARGET_1 = int(TOTAL_DATASET * RATIO_1)


# =====================================================
# STAGE 1 — UNZIP BATCH FILES
# =====================================================
def unzip_all():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print("\n=== Extracting ZIP batches (skipping invalid zips) ===")
    for f in os.listdir(ZIP_DIR):
        if not f.endswith(".zip"):
            continue

        zip_path = os.path.join(ZIP_DIR, f)
        print(f"\nChecking {f} ...")

        if not zipfile.is_zipfile(zip_path):
            print(f"❌ Skipping invalid ZIP: {f}")
            continue

        print(f"Extracting {f} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(EXTRACT_DIR)

        print(f"✔ Extracted {f}")

    print("\n✔ ZIP extraction complete.")


# =====================================================
# STAGE 2 — INDEX DICOMS
# =====================================================
def index_dicoms():
    dicoms = []
    print("\n=== Indexing DICOM files ===")

    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".dcm"):
                # Format: StudyID_SeriesID_SOPInstanceUID.dcm
                sop = f.replace(".dcm", "").split("_")[-1]
                full = os.path.join(root, f)
                dicoms.append((sop, full))

    df = pd.DataFrame(dicoms, columns=["SOPInstanceUID", "filepath"])
    df.to_csv(DICOM_INDEX_CSV, index=False)

    print("✔ DICOM index saved:", DICOM_INDEX_CSV)
    print("Total DICOMs indexed:", len(df))
    return df


# =====================================================
# WINDOWING
# =====================================================
def apply_window(img, wl, ww):
    low = wl - ww / 2
    high = wl + ww / 2
    img = np.clip(img, low, high)
    return ((img - low) / (high - low) * 255).astype("uint8")


# def pseudo_window_fusion(img):
#     p1 = apply_window(img,   40, 400)   # soft
#     p2 = apply_window(img, -600, 1400)  # lung
#     p3 = apply_window(img,  100, 600)   # blood/bone
#     return np.stack([p1, p2, p3], axis=-1)

# def pseudo_window_fusion(img):
#     """
#     Improved pseudo-RGB windowing for pulmonary embolism detection.
#     Produces balanced, contrast-enhanced 3-channel CT images.
#     """

#     windows = [
#         (40, 400),      # Soft tissue – PE, heart, clot visibility
#         (100, 700),     # Vascular window – contrast filling defects
#         (-500, 1500)    # Lung parenchyma – but compressed
#     ]

#     channels = []
#     for i, (wl, ww) in enumerate(windows):

#         low = wl - ww / 2
#         high = wl + ww / 2

#         # Clip HU → window range
#         win = np.clip(img, low, high)

#         # Normalize to 0–1
#         win = (win - low) / (high - low + 1e-8)

#         # --- Optional tuning per-channel ---
#         if i == 2:  
#             # Lung channel compression prevents saturation
#             win = np.power(win, 0.6)     # gamma boost
#         if i == 1:
#             # Vascular clarity improvement
#             win = np.power(win, 1.2)

#         channels.append(win)

#     rgb = np.stack(channels, axis=-1)

#     # convert to 8-bit
#     rgb = (rgb * 255).astype(np.uint8)

#     return rgb

import cv2
import numpy as np

def pseudo_window_fusion(img):
    """
    Improved pseudo-RGB fusion for CT:
    Lung, Soft-tissue, and Vessel windows with gamma & normalization.
    """

    windows = [
        (-600, 1500),  # Lung
        (50,   350),   # Soft tissue
        (300,  700)    # Vessel / contrast
    ]

    channels = []
    for wl, ww in windows:
        low = wl - ww/2
        high = wl + ww/2

        w = np.clip(img, low, high)
        w = (w - low) / (high - low + 1e-8)

        # Gamma correction to reduce brightness
        w = np.power(w, 0.7)

        channels.append(w)

    rgb = np.stack(channels, axis=-1)

    # Normalize each channel independently
    rgb = rgb / (rgb.max(axis=(0, 1), keepdims=True) + 1e-8)

    rgb = (rgb * 255).astype(np.uint8)

    return rgb


# =====================================================
# EXPORT IMAGES
# =====================================================
def export_images(df, split_name):
    print(f"\n=== Exporting {split_name} ===")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["filepath"]
        uid  = row["SOPInstanceUID"]
        label = row[LABEL_COL]

        # Output directories
        out_gray = os.path.join(OUTPUT_DIR, "greyscale", split_name, f"label_{label}")
        out_ps   = os.path.join(OUTPUT_DIR, "pseudo",    split_name, f"label_{label}")
        os.makedirs(out_gray, exist_ok=True)
        os.makedirs(out_ps, exist_ok=True)

        f_gray = os.path.join(out_gray, f"{uid}.png")
        f_ps   = os.path.join(out_ps,   f"{uid}.png")

        # Load DICOM
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
        cv2.imwrite(f_ps, pseudo)


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    print("\n======= RSNA PIPELINE START =======")

    unzip_all()
    dicom_df = index_dicoms()

    print("\n=== Loading metadata ===")
    meta = pd.read_csv(METADATA_CSV)

    print("\n=== Merging metadata with DICOM paths ===")
    df = meta.merge(dicom_df, on="SOPInstanceUID", how="inner")
    print("Matched images:", len(df))

    # Class split
    df0 = df[df[LABEL_COL] == 0].iloc[:TARGET_0]
    df1 = df[df[LABEL_COL] == 1].iloc[:TARGET_1]

    final_df = pd.concat([df0, df1]).sample(frac=1, random_state=42)
    print("\nFinal dataset distribution:")
    print(final_df[LABEL_COL].value_counts())

    # Train / Val / Test
    train = final_df.sample(frac=0.70, random_state=42)
    leftover = final_df.drop(train.index)
    val = leftover.sample(frac=0.50, random_state=42)
    test = leftover.drop(val.index)

    print("\nSplit sizes:")
    print("Train:", len(train))
    print("Val:  ", len(val))
    print("Test: ", len(test))

    # Export images
    export_images(train, "train")
    export_images(val, "val")
    export_images(test, "test")

    print("\n======= PIPELINE COMPLETE =======")


if __name__ == "__main__":
    main()
