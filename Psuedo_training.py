import pandas as pd
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# =========================================================
# PATHS
# =========================================================
Train_path = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/pseudo/train"
Val_path   = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/pseudo/val"
test_path  = "/home/djariwal/Desktop/RSNA_training/png_dataset_25k/pseudo/test"

MODEL_DIR   = "/home/djariwal/Desktop/RSNA_training/Models"
RESULTS_DIR = "/home/djariwal/Desktop/RSNA_training/Results_Pseudo"

# Create results folder
os.makedirs(RESULTS_DIR, exist_ok=True)

Img_size = (228, 228)
batch_size = 12

# =========================================================
# DATASETS
# =========================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    Train_path, seed=123, image_size=Img_size, batch_size=batch_size
)
valid_ds = tf.keras.utils.image_dataset_from_directory(
    Val_path, seed=123, image_size=Img_size, batch_size=batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path, seed=123, image_size=Img_size, batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# =========================================================
# MODEL
# =========================================================
base_model = tf.keras.applications.DenseNet121(
    input_shape=(228, 228, 3), include_top=False, weights='imagenet'
)
for layer in base_model.layers:
    layer.trainable = True

x = tf.keras.layers.Flatten()(base_model.output)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model1 = tf.keras.models.Model(inputs=base_model.input, outputs=pred)

model1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

anne = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.01)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
callbacks1 = [anne, earlystop]

# =========================================================
# TRAIN
# =========================================================
history1 = model1.fit(train_ds, epochs=40, validation_data=valid_ds, callbacks=callbacks1)

# Save model + logs
model1.save(f"{MODEL_DIR}/model_training_pseudo_25k_40eps.h5")

pd.DataFrame(history1.history).to_csv(f"{MODEL_DIR}/model_training_pseudo_25k_40eps.csv", index=False)
pd.DataFrame(history1.history).to_json(f"{MODEL_DIR}/model_training_pseudo_25k_40eps.json")

# =========================================================
# GPU RESET
# =========================================================
from tensorflow.keras import backend as K
def reset_gpu_memory():
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    print("✔ GPU Reset Done")

reset_gpu_memory()

# =========================================================
# SAVE RESULTS FUNCTION
# =========================================================
def evaluate_and_save(ds, ds_name, suffix="_pseudo"):
    print(f"\n===== Evaluating {ds_name.upper()} =====")

    y_true, y_pred, y_prob_0, y_prob_1 = [], [], [], []

    for images, labels in ds:
        preds = model1.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob_0.extend(1 - preds)
        y_prob_1.extend(preds)

    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

    # Confusion Matrix Image Save
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Purples, values_format='d', ax=ax)
    plt.title(f"Confusion Matrix ({ds_name.upper()})")
    plt.grid(False)

    cm_filename = f"{RESULTS_DIR}/{ds_name}_confusion_matrix{suffix}.png"
    plt.savefig(cm_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved confusion matrix: {cm_filename}")

    # Prepare CSV results
    filenames = [os.path.basename(p) for p in ds.file_paths]

    results = pd.DataFrame({
        "filename" + suffix: filenames,
        "Prob_0" + suffix: np.array(y_prob_0).flatten(),
        "Prob_1" + suffix: np.array(y_prob_1).flatten(),
        "True_Label": np.array(y_true).flatten(),
        "Prediction": np.array(y_pred_binary).flatten()
    })

    csv_path = f"{RESULTS_DIR}/{ds_name}_results{suffix}.csv"
    results.to_csv(csv_path, index=False)
    print(f"✔ Saved results CSV: {csv_path}")

    return results


# =========================================================
# RUN FOR TRAIN, VAL, TEST
# =========================================================
Train_Results = evaluate_and_save(train_ds, "train", suffix="_pseudo")
Val_Results   = evaluate_and_save(valid_ds, "val", suffix="_pseudo")
Test_Results  = evaluate_and_save(test_ds, "test", suffix="_pseudo")

print("\n🎉 All pseudo-window results saved successfully!")
print(f"📁 Folder: {RESULTS_DIR}")
