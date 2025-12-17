import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ============================================================
# GPU CHECK
# ============================================================
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR      = "/home/djariwal/Desktop/RSNA_training"
DATASET_DIR   = f"{BASE_DIR}/rgb_dataset_25k"
OUTPUT_ROOT   = f"{BASE_DIR}/batch_run_results"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

Img_size  = (228, 228)
batch_size = 12

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_one_window(window_name):
    print(f"\n==============================")
    print(f"▶ TRAINING WINDOW: {window_name}")
    print(f"==============================")

    window_dir = os.path.join(DATASET_DIR, window_name)

    Train_path = os.path.join(window_dir, "train")
    Val_path   = os.path.join(window_dir, "val")
    Test_path  = os.path.join(window_dir, "test")

    # Output directory
    OUT = os.path.join(OUTPUT_ROOT, window_name)
    os.makedirs(OUT, exist_ok=True)

    # ======================
    # LOAD DATA
    # ======================
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Train_path, seed=123, image_size=Img_size, batch_size=batch_size)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        Val_path, seed=123, image_size=Img_size, batch_size=batch_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        Test_path, seed=123, image_size=Img_size, batch_size=batch_size)

    # ======================
    # MODEL SETUP
    # ======================
    base_model = tf.keras.applications.DenseNet121(
        input_shape=(228, 228, 3), include_top=False, weights="imagenet")

    for layer in base_model.layers:
        layer.trainable = True

    x = tf.keras.layers.Flatten()(base_model.output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=pred)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    anne = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.01)

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=10)

    # ======================
    # TRAIN
    # ======================
    history = model.fit(
        train_ds, epochs=40, validation_data=valid_ds,
        callbacks=[anne, earlystop]
    )

    # Save model + metrics
    model.save(f"{OUT}/model.h5")
    pd.DataFrame(history.history).to_csv(f"{OUT}/metrics.csv", index=False)

    # ======================
    # PLOT CURVES
    # ======================
    plt.figure(figsize=(8,6))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"Accuracy – {window_name}")
    plt.savefig(f"{OUT}/accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"Loss – {window_name}")
    plt.savefig(f"{OUT}/loss_curve.png")
    plt.close()

    # ======================
    # EVALUATION FUNCTION
    # ======================
    def evaluate_split(ds, split):

        print(f"  → Evaluating {split}...")

        y_true, y_prob, y_prob0 = [], [], []
        file_names = [os.path.basename(p) for p in ds.file_paths]

        # run model
        for images, labels in ds:
            preds = model.predict(images)
            y_true.extend(labels.numpy())
            y_prob.extend(preds)
            y_prob0.extend(1 - preds)

        y_true = np.array(y_true).flatten()
        y_prob = np.array(y_prob).flatten()
        y_prob0 = np.array(y_prob0).flatten()

        y_pred = (y_prob > 0.5).astype(int)

        # CONFUSION MATRIX
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["0","1"])
        plt.figure(figsize=(6,6))
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"{split} Confusion Matrix – {window_name}")
        plt.savefig(f"{OUT}/confusion_matrix_{split}.png")
        plt.close()

        # SAVE RESULTS CSV
        df = pd.DataFrame({
            "filename": file_names,
            "prob_y0": y_prob0,
            "prob_y1": y_prob,
            "true_label": y_true,
            "pred_label": y_pred
        })
        df.to_csv(f"{OUT}/{split}_results.csv", index=False)

        # SAVE REPORT
        report = classification_report(y_true, y_pred)
        with open(f"{OUT}/{split}_classification_report.txt", "w") as f:
            f.write(report)

        return df

    # RUN EVALS
    evaluate_split(train_ds, "train")
    evaluate_split(valid_ds, "val")
    evaluate_split(test_ds,  "test")

    print(f"✔ DONE: {window_name}")


# ============================================================
# LOOP OVER ALL WINDOW FOLDERS
# ============================================================
window_folders = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print("\n🔁 WINDOWS FOUND:", window_folders)

for w in window_folders:
    train_one_window(w)

print("\n🎉 ALL WINDOWS TRAINED SUCCESSFULLY!")
print("📁 Results saved inside:", OUTPUT_ROOT)
