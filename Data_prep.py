import os
import cv2
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = "/home/djariwal/Desktop/RSNA_training"
DATA_DIR = f"{BASE_DIR}/Data"
METADATA = f"{DATA_DIR}/train_balanced_50k.csv"
IMAGES_ROOT = f"{DATA_DIR}/Training_Data_Zip/extracted"
OUTPUT_DIR  = f"{BASE_DIR}/png_dataset_25k"

LABEL_COL = "pe_present_on_image"
TARGET_SIZE = 256

TOTAL_DATASET = 25000
RATIO_LABEL0 = 0.75
RATIO_LABEL1 = 0.25

# final target counts
TARGET_0 = int(TOTAL_DATASET * RATIO_LABEL0)
TARGET_1 = int(TOTAL_DATASET * RATIO_LABEL1)


# =====================================================
# WINDOWING FUNCTIONS
# =====================================================
def apply_window(img, wl, ww):
    low = wl - ww / 2
    high = wl + ww / 2
    img = np.clip(img, low, high)
    return ((img - low) / (high - low) * 255).astype("uint8")

def pseudo_window_fusion(img):
    p1 = apply_window(img,   40,  400)
    p2 = apply_window(img, -600, 1500)
    p3 = apply_window(img,  300,  700)
    return np.stack([p1, p2, p3], axis=-1)


# =====================================================
# EXPORT FUNCTION
# =====================================================
def export(df, split_name):
    print(f"\n=== Exporting {split_name} ===")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dcm_path = row["filepath"]
        label    = row[LABEL_COL]
        uid      = row["SOPInstanceUID"]

        outdir = os.path.join(OUTPUT_DIR, split_name, f"label_{label}")
        os.makedirs(outdir, exist_ok=True)

        outfile = os.path.join(outdir, f"{uid}.png")

        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            img = ds.pixel_array.astype(np.float32)
        except Exception as e:
            print("[Decode Error]", dcm_path, e)
            continue

        fused = pseudo_window_fusion(img)
        fused = cv2.resize(fused, (TARGET_SIZE, TARGET_SIZE))
        cv2.imwrite(outfile, fused)


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    print("Loading metadata...")
    df = pd.read_csv(METADATA)

    print("Indexing available DICOM files...")
    dcm_map = {}
    for root, _, files in os.walk(IMAGES_ROOT):
        for f in files:
            if f.endswith(".dcm"):
                sop = f.replace(".dcm", "").split("_")[-1]
                dcm_map[sop] = os.path.join(root, f)

    print("Matching metadata to existing images...")
    df["filepath"] = df["SOPInstanceUID"].map(dcm_map)
    df = df.dropna(subset=["filepath"])

    print("Available matched images:", len(df))

    # Get the two classes
    df0 = df[df[LABEL_COL] == 0]
    df1 = df[df[LABEL_COL] == 1]

    print("Found label 0:", len(df0))
    print("Found label 1:", len(df1))

    # SAMPLE EXACT TARGET COUNTS
    df0 = df0.iloc[:TARGET_0]
    df1 = df1.iloc[:TARGET_1]

    df_final = pd.concat([df0, df1]).reset_index(drop=True)

    print("\nFinal dataset size:", len(df_final))
    print(df_final[LABEL_COL].value_counts())

    # Train/Val/Test SPLITS
    print("\nPerforming 70/15/15 split...")

    train = df_final.sample(frac=0.70, random_state=42)
    leftover = df_final.drop(train.index)

    val = leftover.sample(frac=0.50, random_state=42)
    test = leftover.drop(val.index)

    print("\nSplit sizes:")
    print("Train:", len(train))
    print("Val:  ", len(val))
    print("Test: ", len(test))

    export(train, "train")
    export(val, "val")
    export(test, "test")

    print("\n=== PNG export DONE (25k dataset) ===")


if __name__ == "__main__":
    main()
