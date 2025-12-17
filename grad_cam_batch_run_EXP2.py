import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "/home/djariwal/Desktop/RSNA_training"

IMG_DIR = f"{BASE_DIR}/grad_cam_results"
MODEL_ROOT = f"{BASE_DIR}/batch_run_results_EXP2"

OUT_ROOT = f"{IMG_DIR}/interpretation"
os.makedirs(OUT_ROOT, exist_ok=True)

IMG_SIZE = (228, 228)
LAST_CONV = "conv5_block16_concat"

WINDOW_MAP = {
    "w1": "w1_lung_soft_vessel",
    "w2": "w2_highbone_soft",
    "w3": "w3_soft_detail",
    "w4": "w4_lowdose",
    "w5": "w5_vessel_enhance",
    "w6": "w6_combined_CT",
    "w7": "w7_bone_lung",
    "w8": "w8_soft_fine",
    "grey": "grey"
}

# ============================================================
# UTILITIES
# ============================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_img(path, arr, cmap=None, title=None):
    plt.figure(figsize=(4,4))
    if cmap:
        plt.imshow(arr, cmap=cmap)
    else:
        plt.imshow(arr)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ============================================================
# GRAD-CAM
# ============================================================
def grad_cam(img_rgb, model):
    img = tf.cast(img_rgb, tf.float32)
    img_batch = tf.expand_dims(img, axis=0)

    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_out, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        0.6,
        heatmap_color,
        0.4,
        0
    )
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return heatmap, overlay

# ============================================================
# SALIENCY
# ============================================================
def saliency_map(img_rgb, model):
    img = tf.cast(img_rgb, tf.float32)
    img_batch = tf.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        preds = model(img_batch, training=False)
        score = preds[:, 0]

    grads = tape.gradient(score, img_batch)[0]
    sal = tf.reduce_max(tf.abs(grads), axis=-1).numpy()

    sal -= sal.min()
    sal /= sal.max() + 1e-8
    return sal

# ============================================================
# MAIN LOOP
# ============================================================
for key, model_folder in WINDOW_MAP.items():

    print(f"\n🔍 Processing {key} ...")

    img_path = os.path.join(IMG_DIR, f"{key}.png")
    model_path = os.path.join(MODEL_ROOT, model_folder, "model.h5")

    if not os.path.exists(img_path):
        print(f"❌ Image missing: {img_path}")
        continue

    if not os.path.exists(model_path):
        print(f"❌ Model missing: {model_path}")
        continue

    out_dir = os.path.join(OUT_ROOT, key)
    os.makedirs(out_dir, exist_ok=True)

    # Load
    model = tf.keras.models.load_model(model_path)
    img = load_image(img_path)

    # Save input
    save_img(os.path.join(out_dir, "input.png"), img, title="Input")

    # Grad-CAM
    heat, overlay = grad_cam(img, model)
    save_img(os.path.join(out_dir, "gradcam_heatmap.png"),
             heat, cmap="jet", title="Grad-CAM Heatmap")
    save_img(os.path.join(out_dir, "gradcam_overlay.png"),
             overlay, title="Grad-CAM Overlay")

    # Saliency
    sal = saliency_map(img, model)
    save_img(os.path.join(out_dir, "saliency.png"),
             sal, cmap="hot", title="Saliency Map")

    print(f"✔ Saved interpretation for {key}")

print("\n🎉 DONE — All Grad-CAM + saliency maps generated")
print(f"📁 Output: {OUT_ROOT}")
