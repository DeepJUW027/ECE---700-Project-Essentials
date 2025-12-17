import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS (FIXED SPELLING: "Pseudo" not "Psuedo")
# ============================================================
BASE = "/home/djariwal/Desktop/RSNA_training"

GREY_TEST_CSV   = f"{BASE}/Results_Greyscale/test_results_greyscale.csv"
PSEUDO_TEST_CSV = f"{BASE}/Results_Pseudo/test_results_pseudo.csv"   # FIXED PATH

OUTPUT_DIR = f"{BASE}/Results_Error_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Loading Test Results ===")

# ============================================================
# LOAD CSVs
# ============================================================
df_grey = pd.read_csv(GREY_TEST_CSV)
df_pseudo = pd.read_csv(PSEUDO_TEST_CSV)

# ============================================================
# PROCESSOR FUNCTION
# ============================================================
def process_model(df, name, suffix):
    print(f"\n=== Processing {name} ===")
    print("CSV Columns:", df.columns.tolist())

    # ------------------------------------------------------
    # STANDARDIZE COLUMN NAMES
    # ------------------------------------------------------
    col_map = {
        "filename" + suffix: "filename",
        "Prob_0" + suffix: "prob_y0",
        "Prob_1" + suffix: "prob_y1",
        "True_Label": "true_label",
        "Prediction": "pred_label",
        "prob_y0": "prob_y0",
        "prob_y1": "prob_y1",
        "true_label": "true_label",
        "pred_label": "pred_label",
    }

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["filename", "prob_y0", "prob_y1", "true_label", "pred_label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column: {col}")

    # ------------------------------------------------------
    # DEFINE ERROR TYPE (TP, TN, FP, FN)
    # ------------------------------------------------------
    df["ErrorType"] = df.apply(
        lambda x:
            "TP" if x["true_label"] == 1 and x["pred_label"] == 1 else
            "TN" if x["true_label"] == 0 and x["pred_label"] == 0 else
            "FP" if x["true_label"] == 0 and x["pred_label"] == 1 else
            "FN",
        axis=1
    )

    # Probability of class 1
    df["PredProb"] = df["prob_y1"]

    # ------------------------------------------------------
    # CONFIDENTLY WRONG:
    # FP with prob > 0.9
    # FN with prob < 0.1
    # ------------------------------------------------------
    confidently_wrong = df[
        ((df["ErrorType"] == "FP") & (df["PredProb"] >= 0.9)) |
        ((df["ErrorType"] == "FN") & (df["PredProb"] <= 0.1))
    ].copy()

    # ------------------------------------------------------
    # EDGE CASE MISCLASSIFICATIONS (near-threshold 0.45–0.55)
    # ------------------------------------------------------
    edge_cases = df[
        (df["ErrorType"].isin(["FP", "FN"])) &
        (df["PredProb"] > 0.45) &
        (df["PredProb"] < 0.55)
    ].copy()

    # ------------------------------------------------------
    # SAVE CSVs
    # ------------------------------------------------------
    cw_path = f"{OUTPUT_DIR}/{name}_Confidently_Wrong{suffix}.csv"
    ec_path = f"{OUTPUT_DIR}/{name}_Edge_Cases{suffix}.csv"

    confidently_wrong.to_csv(cw_path, index=False)
    edge_cases.to_csv(ec_path, index=False)

    print(f"✔ Saved Confident Wrong → {cw_path}")
    print(f"✔ Saved Edge Cases     → {ec_path}")

    # Logging details
    print(f"Total test samples      : {len(df)}")
    print(f"Misclassified (FP+FN)   : {len(df[df['ErrorType'].isin(['FP','FN'])])} "
          f"({len(df[df['ErrorType'].isin(['FP','FN'])])/len(df):.1%})")
    print(f"Edge cases (wrong, 0.45–0.55): {len(edge_cases)}")
    print(f"Confidently wrong       : {len(confidently_wrong)}")

    return confidently_wrong, edge_cases


# ============================================================
# RUN FOR BOTH MODELS
# ============================================================
grey_conf, grey_edge = process_model(df_grey, "Greyscale", "_grey")
pseudo_conf, pseudo_edge = process_model(df_pseudo, "Pseudo", "_pseudo")

print("\n🎉 DONE — All error CSVs created successfully!")
print(f"Location: {OUTPUT_DIR}")
