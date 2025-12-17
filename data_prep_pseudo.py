import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
from PIL import Image

# ===================================================================
# PATH CONFIGURATION (AS REQUESTED)
# ===================================================================
BASE_DIR = "/home/djariwal/Desktop/RSNA_training"
DATA_DIR = f"{BASE_DIR}/Data"
METADATA = f"{DATA_DIR}/train_balanced_50k.csv"
IMAGES_ROOT = f"{DATA_DIR}/Training_Data_Zip/extracted"
OUTPUT_DIR  = f"{BASE_DIR}/rgb_dataset_25k"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# DATASET CONFIG
# ===================================================================
TOTAL_TARGET = 25000
RATIO_LABEL0 = 0.75
RATIO_LABEL1 = 0.25

RATIO_VAL  = 0.15
RATIO_TEST = 0.15

# Total splits: 70/15/15 (train/val/test)
RATIO_TRAIN = 1 - (RATIO_VAL + RATIO_TEST)

# Pseudo-window set D (Lung + Soft + Vessel)
WINDOWS = [
    (-600, 1400),  # Lung
    (40, 400),     # Soft tissue
    (100, 600)     # Vessel/Bone
]

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
def to_hu(ds):
    """Convert DICOM to Hounsfield Units."""
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return img * slope + intercept

def apply_window(img, wl, ww):
    low = wl - ww / 2
    high = wl + ww / 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    return img

def to_pseudo_rgb(img_hu):
    ch = [apply_window(img_hu, wl, ww) for wl, ww in WINDOWS]
    rgb = np.stack(ch, axis=-1)
    return (rgb * 255).astype(np.uint8)

# ===================================================================
# STEP 1 — Load metadata & map to DICOM paths
# ===================================================================
print("📌 Loading metadata...")
df = pd.read_csv(METADATA)

print("📌 Indexing available DICOM files...")
dcm_map = {}
for root, _, files in os.walk(IMAGES_ROOT):
    for f in files:
        if f.lower().endswith(".dcm"):
            sop = f.replace(".dcm","")
            full = os.path.join(root, f)
            dcm_map[sop] = full

df["filepath"] = df["SOPInstanceUID"].map(dcm_map)
df = df.dropna(subset=["filepath"])
print("✔ Matched images:", len(df))

# ===================================================================
# STEP 2 — SAMPLE EXACT COUNTS
# ===================================================================
df0 = df[df["pe_present_on_image"] == 0].sample(int(TOTAL_TARGET * RATIO_LABEL0), random_state=42)
df1 = df[df["pe_present_on_image"] == 1].sample(int(TOTAL_TARGET * RATIO_LABEL1), random_state=42)

df_final = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_final["pe_present_on_image"].value_counts())

# ===================================================================
# STEP 3 — SPLIT INTO TRAIN/VAL/TEST
# ===================================================================
train_df = df_final.sample(frac=RATIO_TRAIN, random_state=42)
leftover  = df_final.drop(train_df.index)

val_df  = leftover.sample(frac=0.5, random_state=42)
test_df = leftover.drop(val_df.index)

print("\nSPLIT SIZES:")
print("Train:", len(train_df))
print("Val:  ", len(val_df))
print("Test: ", len(test_df))

splits = {
    "train": train_df,
    "val":   val_df,
    "test":  test_df
}

# ===================================================================
# STEP 4 — Convert & Save Pseudo RGB PNGs
# ===================================================================
def convert_and_save(df_split, phase):
    print(f"\n🚀 Exporting {phase}...")
    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        dcm_path = row["filepath"]
        label    = row["pe_present_on_image"]
        uid      = row["SOPInstanceUID"]

        out_dir = os.path.join(OUTPUT_DIR, phase, f"label_{label}")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{uid}.png")

        try:
            dcm = pydicom.dcmread(dcm_path, force=True)
            img_hu = to_hu(dcm)
            arr = to_pseudo_rgb(img_hu)

            Image.fromarray(arr).save(out_path)

        except Exception as e:
            print(f"⚠️ Error {uid}: {e}")
            continue

# Run conversion
convert_and_save(train_df, "train")
convert_and_save(val_df, "val")
convert_and_save(test_df, "test")

print("\n🎉 DONE! Pseudo-window RGB dataset created successfully!")
print(f"📁 Saved at: {OUTPUT_DIR}")
