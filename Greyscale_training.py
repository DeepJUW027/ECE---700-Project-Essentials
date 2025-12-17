import pandas as pd
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ============================================================
# GPU CHECK
# ============================================================
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# ============================================================
# PATHS
# ============================================================
Train_path = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/greyscale/train"
Val_path   = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/greyscale/val"
Test_path  = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/greyscale/test"

RESULTS_DIR = "/home/djariwal/Desktop/RSNA_training/Results_Greyscale"
os.makedirs(RESULTS_DIR, exist_ok=True)

Img_size = (228, 228)
batch_size = 12

# ============================================================
# LOAD DATASET
# ============================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    Train_path, seed=123, image_size=Img_size, batch_size=batch_size)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    Val_path, seed=123, image_size=Img_size, batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    Test_path, seed=123, image_size=Img_size, batch_size=batch_size)

class_names = train_ds.class_names
print("Classes:", class_names)

# ============================================================
# MODEL
# ============================================================
anne = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=2, min_lr=0.01)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=10)

base_model = tf.keras.applications.DenseNet121(
    input_shape=(228, 228, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = tf.keras.layers.Flatten()(x)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model1 = tf.keras.models.Model(inputs=base_model.input, outputs=pred)

model1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history = model1.fit(train_ds, epochs=40, validation_data=valid_ds,
                     callbacks=[anne, earlystop])

# ============================================================
# SAVE MODEL + HISTORY
# ============================================================
model1.save(f"{RESULTS_DIR}/model_greyscale.h5")
pd.DataFrame(history.history).to_csv(
    f"{RESULTS_DIR}/training_history_greyscale.csv", index=False)

# ============================================================
# PLOT ACCURACY & LOSS (Saved)
# ============================================================
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy (Greyscale)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.grid(True)
plt.savefig(f"{RESULTS_DIR}/accuracy_curve_greyscale.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss (Greyscale)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'])
plt.grid(True)
plt.savefig(f"{RESULTS_DIR}/loss_curve_greyscale.png")
plt.close()

# ============================================================
# FUNCTION FOR EVALUATION + SAVING CONFUSION MATRIX + RESULTS
# ============================================================
def evaluate_and_save(ds, split_name):
    y_true, y_pred, y_pred0, y_pred1 = [], [], [], []
    file_paths = ds.file_paths
    file_names = [os.path.basename(p) for p in file_paths]

    # Run inference
    for images, labels in ds:
        preds = model1.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_pred0.extend(1 - preds)
        y_pred1.extend(preds)

    # Convert to arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_pred0 = np.array(y_pred0).flatten()
    y_pred1 = np.array(y_pred1).flatten()

    y_pred_binary = (y_pred > 0.5).astype(int)

    # ======================================================
    # SAVE CONFUSION MATRIX
    # ======================================================
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Class 0', 'Class 1'])

    plt.figure(figsize=(6,6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{split_name} Confusion Matrix (Greyscale)")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{split_name.lower()}_greyscale.png")
    plt.close()

    # ======================================================
    # SAVE RESULTS CSV
    # ======================================================
    df = pd.DataFrame({
        "filename": file_names,
        "prob_y0": y_pred0,
        "prob_y1": y_pred1,
        "true_label": y_true,
        "pred_label": y_pred_binary
    })

    df.to_csv(f"{RESULTS_DIR}/{split_name.lower()}_results_greyscale.csv", index=False)

    # ======================================================
    # SAVE CLASSIFICATION REPORT
    # ======================================================
    report = classification_report(y_true, y_pred_binary)
    with open(f"{RESULTS_DIR}/{split_name.lower()}_classification_report_greyscale.txt", "w") as f:
        f.write(report)

    print(f"✔ Saved results for {split_name}")
    return df


# ============================================================
# RUN EVALS
# ============================================================
train_results = evaluate_and_save(train_ds, "Train")
val_results   = evaluate_and_save(valid_ds, "Val")
test_results  = evaluate_and_save(test_ds, "Test")

print("🎉 All evaluation files saved with _greyscale suffix!")  