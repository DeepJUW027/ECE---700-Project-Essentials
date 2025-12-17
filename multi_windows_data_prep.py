import os
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from PIL import Image

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/home/djariwal/Desktop/RSNA_training"
DATA_DIR = f"{BASE_DIR}/Data"

# IMPORTANT FIX
IMAGES_ROOT = f"{DATA_DIR}/Training_Data_Zip/extracted_new"

METADATA = f"{DATA_DIR}/train_balanced_50k.csv"
OUTPUT_DIR = f"{BASE_DIR}/rgb_dataset_25k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATASET CONFIG
# ============================================================
TOTAL_TARGET = 25000
RATIO_L0 = 0.75
RATIO_L1 = 0.25

RATIO_VAL  = 0.15
RATIO_TEST = 0.15
RATIO_TRAIN = 1 - (RATIO_VAL + RATIO_TEST)

# ============================================================
# MULTI-PSEUDO WINDOWS (8 VARIANTS)
# ============================================================
WINDOW_PRESETS = {
    "w1_lung_soft_vessel": [
        (-600, 1400), (40, 400), (100, 600)
    ],
    "w2_highbone_soft": [
        (300, 1500), (40, 400), (60, 300)
    ],
    "w3_soft_detail": [
        (40, 200), (60, 300), (80, 400)
    ],
    "w4_lowdose": [
        (-800, 1600), (40, 400), (200, 700)
    ],
    "w5_vessel_enhance": [
        (300, 500), (60, 200), (100, 300)
    ],
    "w6_combined_CT": [
        (-600, 1500), (0, 300), (50, 350)
    ],
    "w7_bone_lung": [
        (-600, 1400), (300, 1800), (50, 350)
    ],
    "w8_soft_fine": [
        (40, 300), (40, 500), (100, 300)
    ]
}

# ============================================================
# FUNCTIONS
# ============================================================
def to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return arr * slope + intercept

def apply_window(img, wl, ww):
    low = wl - ww/2
    high = wl + ww/2
    img = np.clip(img, low, high)
    return (img - low) / (high - low + 1e-8)

def make_pseudo(img_hu, preset):
    chans = [apply_window(img_hu, wl, ww) for wl, ww in preset]
    rgb = np.stack(chans, axis=-1)
    return (rgb * 255).astype(np.uint8)

def make_greyscale(img_hu):
    win = apply_window(img_hu, 40, 400)
    return (win * 255).astype(np.uint8)

# ============================================================
# STEP 1 — Load metadata & map DICOMs
# ============================================================
print("📌 Loading metadata...")
df = pd.read_csv(METADATA)

print("📌 Mapping DICOMs in IMAGES_ROOT...")
dcm_map = {}

for root, _, files in os.walk(IMAGES_ROOT):
    for f in files:
        if f.lower().endswith(".dcm"):
            sop = f.replace(".dcm","").split("_")[-1]   # FIXED
            full = os.path.join(root, f)
            dcm_map[sop] = full

df["filepath"] = df["SOPInstanceUID"].map(dcm_map)
df = df.dropna(subset=["filepath"])
print("✔ Matched images:", len(df))

if len(df) == 0:
    raise SystemExit("❌ No matched images. SOP extraction failed. Check IMAGES_ROOT.")

# ============================================================
# STEP 2 — Sampling
# ============================================================
df0 = df[df.pe_present_on_image == 0].sample(int(TOTAL_TARGET * RATIO_L0), random_state=42)
df1 = df[df.pe_present_on_image == 1].sample(int(TOTAL_TARGET * RATIO_L1), random_state=42)

df_final = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_final.pe_present_on_image.value_counts())

# ============================================================
# STEP 3 — Split train/val/test
# ============================================================
train_df = df_final.sample(frac=RATIO_TRAIN, random_state=42)
remaining = df_final.drop(train_df.index)

val_df  = remaining.sample(frac=0.5, random_state=42)
test_df = remaining.drop(val_df.index)

print("\nTrain:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

splits = {"train": train_df, "val": val_df, "test": test_df}

# ============================================================
# STEP 4 — Export all window presets
# ============================================================
def export_images(df_split, phase, label, out_root, preset):
    os.makedirs(out_root, exist_ok=True)

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"{phase}-{label}"):
        path = row["filepath"]
        uid  = row["SOPInstanceUID"]

        try:
            ds = pydicom.dcmread(path, force=True)
            img_hu = to_hu(ds)
        except:
            continue

        out_dir = os.path.join(out_root, phase, f"label_{label}")
        os.makedirs(out_dir, exist_ok=True)

        img = make_pseudo(img_hu, preset)
        Image.fromarray(img).save(os.path.join(out_dir, uid + ".png"))

# GREYSCALE EXPORT
def export_greyscale(df_split, phase, label):
    out_root = os.path.join(OUTPUT_DIR, "grey")
    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"grey-{phase}"):
        try:
            ds = pydicom.dcmread(row["filepath"], force=True)
            img_hu = to_hu(ds)
            arr = make_greyscale(img_hu)
            out_dir = os.path.join(out_root, phase, f"label_{label}")
            os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(arr).save(os.path.join(out_dir, row["SOPInstanceUID"] + ".png"))
        except:
            continue

# ============================================================
# RUN EXPORTS
# ============================================================
for preset_name, preset_windows in WINDOW_PRESETS.items():
    print(f"\n🚀 EXPORTING PRESET: {preset_name}")
    out_root = os.path.join(OUTPUT_DIR, preset_name)

    for phase, df_split in splits.items():
        for label in [0, 1]:
            export_images(df_split[df_split.pe_present_on_image == label], 
                          phase, label, out_root, preset_windows)

print("\n🚀 EXPORTING GREYSCALE")
for phase, df_split in splits.items():
    for label in [0, 1]:
        export_greyscale(df_split[df_split.pe_present_on_image == label], phase, label)

print("\n🎉 DONE — Multi-window pseudo dataset + grey channel created!")
print(f"📁 Output directory: {OUTPUT_DIR}")
