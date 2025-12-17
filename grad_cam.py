import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# 1. CONFIG
# ================================================================

# Models
GREY_MODEL_PATH = "/home/djariwal/Desktop/RSNA_training/Results_Greyscale/model_greyscale.h5"
PSEUDO_MODEL_PATH = "/home/djariwal/Desktop/RSNA_training/Models/model_training_pseudo_25k_40eps.h5"

# Image roots (test splits)
GREY_TEST_ROOT = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/greyscale/test"
PSEUDO_TEST_ROOT = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/pseudo/test"

# Error-analysis CSVs (already generated)
ERROR_DIR = "/home/djariwal/Desktop/RSNA_training/Results_Error_Analysis"
GREY_EDGE_CSV  = os.path.join(ERROR_DIR, "Greyscale_Edge_Cases_grey.csv")
GREY_CONF_CSV  = os.path.join(ERROR_DIR, "Greyscale_Confidently_Wrong_grey.csv")
PSEUDO_EDGE_CSV = os.path.join(ERROR_DIR, "Pseudo_Edge_Cases_pseudo.csv")
PSEUDO_CONF_CSV = os.path.join(ERROR_DIR, "Pseudo_Confidently_Wrong_pseudo.csv")

# Output root
INTERPRET_ROOT = "/home/djariwal/Desktop/RSNA_training/Interpretation_Results"
os.makedirs(INTERPRET_ROOT, exist_ok=True)

IMG_SIZE = (228, 228)
VALID_EXT = (".png", ".jpg", ".jpeg")

# Last conv layer name for Grad-CAM (DenseNet121)
# If model_greyscale has a different backbone, adjust GREY_LAST_CONV accordingly.
GREY_LAST_CONV   = "conv5_block16_concat"
PSEUDO_LAST_CONV = "conv5_block16_concat"

# ================================================================
# 2. UTILITIES
# ================================================================
def load_rgb_image(path):
    """Read image as RGB and resize."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def find_image(root, filename):
    """
    Given root like .../greyscale/test, try label_0 and label_1 subdirs.
    Returns full path or None.
    """
    for lbl in ["label_0", "label_1"]:
        p = os.path.join(root, lbl, filename)
        if os.path.exists(p):
            return p
    return None

def grad_cam_single(img_rgb, model, last_conv_layer):
    """
    img_rgb: (H, W, 3) uint8 or float32
    model: keras model
    last_conv_layer: name of last conv layer
    """
    img = tf.cast(img_rgb, tf.float32)
    img_batch = tf.expand_dims(img, axis=0)

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_batch)
        # Assuming binary classifier with single sigmoid output
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # (Hc, Wc, C)

    # Weighted combination
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-12)

    # Resize to input size and colorize
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay on image (convert img_rgb to BGR for cv2)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap, overlay_rgb

def saliency_map(img_rgb, model):
    """
    Vanilla gradient-based saliency map.
    """
    img = tf.cast(img_rgb, tf.float32)
    img_batch = tf.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        preds = model(img_batch, training=False)
        # focus on positive class score
        score = preds[:, 0]

    grads = tape.gradient(score, img_batch)[0]  # (H, W, 3)
    # Take max magnitude over channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()

    # Normalize to [0, 1]
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-12)

    return saliency

def save_img(path, array, cmap=None, title=None):
    plt.figure(figsize=(4, 4))
    if cmap:
        plt.imshow(array, cmap=cmap)
    else:
        plt.imshow(array)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ================================================================
# 3. BUILD LIST OF FILENAMES TO INTERPRET
#    -> Intersection of filenames from grey & pseudo error CSVs
# ================================================================
def collect_filenames_from_csv(csv_path, col="filename"):
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found, skipping: {csv_path}")
        return set()
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        print(f"⚠️ Column '{col}' not in {csv_path}, columns are {df.columns.tolist()}")
        return set()
    return set(df[col].astype(str).tolist())

print("=== Collecting filenames from error CSVs ===")
grey_set = collect_filenames_from_csv(GREY_EDGE_CSV,  "filename") | \
           collect_filenames_from_csv(GREY_CONF_CSV,  "filename")

pseudo_set = collect_filenames_from_csv(PSEUDO_EDGE_CSV, "filename") | \
             collect_filenames_from_csv(PSEUDO_CONF_CSV, "filename")

common_filenames = sorted(grey_set & pseudo_set)
print(f"Total filenames (grey):   {len(grey_set)}")
print(f"Total filenames (pseudo): {len(pseudo_set)}")
print(f"Common filenames (both):  {len(common_filenames)}")

if len(common_filenames) == 0:
    print("⚠️ No common filenames found between greyscale and pseudo error CSVs.")
    print("   You can change the logic above to use union instead of intersection if desired.")
    # fall back to union to avoid doing nothing
    common_filenames = sorted(grey_set | pseudo_set)
    print(f"Using union instead, total filenames: {len(common_filenames)}")

# ================================================================
# 4. LOAD MODELS
# ================================================================
print("\n=== Loading models ===")
grey_model = tf.keras.models.load_model(GREY_MODEL_PATH)
print("✅ Greyscale model loaded:", GREY_MODEL_PATH)

pseudo_model = tf.keras.models.load_model(PSEUDO_MODEL_PATH)
print("✅ Pseudo model loaded   :", PSEUDO_MODEL_PATH)

# ================================================================
# 5. MAIN LOOP: PROCESS EACH FILENAME
# ================================================================
for filename in common_filenames:
    if not filename.lower().endswith(VALID_EXT):
        # Ensure it has an extension – if CSV has bare IDs, append .png
        filename_png = filename + ".png"
    else:
        filename_png = filename

    base = os.path.splitext(filename_png)[0]
    print(f"\n🔍 Processing {filename_png} ...")

    out_dir = os.path.join(INTERPRET_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- GREYSCALE ----------------
    grey_path = find_image(GREY_TEST_ROOT, filename_png)
    if grey_path is None:
        print(f"  ⚠️ Greyscale image not found for {filename_png}, skipping grey.")
    else:
        try:
            grey_img = load_rgb_image(grey_path)

            # Save input
            save_img(os.path.join(out_dir, "grey_input.png"), grey_img, title="Greyscale Input")

            # Grad-CAM
            grey_heat, grey_overlay = grad_cam_single(
                grey_img, grey_model, last_conv_layer=GREY_LAST_CONV
            )
            save_img(os.path.join(out_dir, "grey_gradcam_heatmap.png"),
                     grey_heat, cmap="jet", title="Grey Grad-CAM Heatmap")
            save_img(os.path.join(out_dir, "grey_gradcam_overlay.png"),
                     grey_overlay, title="Grey Grad-CAM Overlay")

            # Saliency
            grey_sal = saliency_map(grey_img, grey_model)
            save_img(os.path.join(out_dir, "grey_saliency.png"),
                     grey_sal, cmap="hot", title="Grey Saliency")

        except Exception as e:
            print(f"  ❌ Error processing greyscale for {filename_png}: {e}")

    # ---------------- PSEUDO ----------------
    pseudo_path = find_image(PSEUDO_TEST_ROOT, filename_png)
    if pseudo_path is None:
        print(f"  ⚠️ Pseudo image not found for {filename_png}, skipping pseudo.")
    else:
        try:
            pseudo_img = load_rgb_image(pseudo_path)

            # Save input
            save_img(os.path.join(out_dir, "pseudo_input.png"), pseudo_img, title="Pseudo Input")

            # Grad-CAM
            pseudo_heat, pseudo_overlay = grad_cam_single(
                pseudo_img, pseudo_model, last_conv_layer=PSEUDO_LAST_CONV
            )
            save_img(os.path.join(out_dir, "pseudo_gradcam_heatmap.png"),
                     pseudo_heat, cmap="jet", title="Pseudo Grad-CAM Heatmap")
            save_img(os.path.join(out_dir, "pseudo_gradcam_overlay.png"),
                     pseudo_overlay, title="Pseudo Grad-CAM Overlay")

            # Saliency
            pseudo_sal = saliency_map(pseudo_img, pseudo_model)
            save_img(os.path.join(out_dir, "pseudo_saliency.png"),
                     pseudo_sal, cmap="hot", title="Pseudo Saliency")

        except Exception as e:
            print(f"  ❌ Error processing pseudo for {filename_png}: {e}")

print("\n🎉 DONE! All interpretation outputs saved under:")
print(INTERPRET_ROOT)
