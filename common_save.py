import pandas as pd
import os

BASE = "/home/djariwal/Desktop/RSNA_training/Results_Error_Analysis"
OUT_DIR = f"{BASE}/Comparisons"
os.makedirs(OUT_DIR, exist_ok=True)

# Input CSVs
GREY_CONF = f"{BASE}/Greyscale_Confidently_Wrong_grey.csv"
GREY_EDGE = f"{BASE}/Greyscale_Edge_Cases_grey.csv"

PSEUDO_CONF = f"{BASE}/Pseudo_Confidently_Wrong_pseudo.csv"
PSEUDO_EDGE = f"{BASE}/Pseudo_Edge_Cases_pseudo.csv"

print("\n=== Loading Files ===")
df_grey_conf = pd.read_csv(GREY_CONF)
df_grey_edge = pd.read_csv(GREY_EDGE)

df_pseudo_conf = pd.read_csv(PSEUDO_CONF)
df_pseudo_edge = pd.read_csv(PSEUDO_EDGE)

print("Rows Loaded:")
print("Grey Confident Wrong :", len(df_grey_conf))
print("Grey Edge Cases      :", len(df_grey_edge))
print("Pseudo Confident Wrong:", len(df_pseudo_conf))
print("Pseudo Edge Cases     :", len(df_pseudo_edge))

# ----------------------------------------------------
# Standardize filename column
# ----------------------------------------------------
df_grey_conf["filename"] = df_grey_conf["filename"].str.strip()
df_pseudo_conf["filename"] = df_pseudo_conf["filename"].str.strip()

df_grey_edge["filename"] = df_grey_edge["filename"].str.strip()
df_pseudo_edge["filename"] = df_pseudo_edge["filename"].str.strip()

# ----------------------------------------------------
# Merge for confident wrong
# ----------------------------------------------------
common_conf = df_grey_conf.merge(df_pseudo_conf, on="filename", suffixes=("_grey", "_pseudo"))
common_conf.to_csv(f"{OUT_DIR}/Common_Confidently_Wrong.csv", index=False)

print("\n✔ Saved Common Confidently Wrong →", f"{OUT_DIR}/Common_Confidently_Wrong.csv")
print("Count:", len(common_conf))

# ----------------------------------------------------
# Merge for edge cases
# ----------------------------------------------------
common_edge = df_grey_edge.merge(df_pseudo_edge, on="filename", suffixes=("_grey", "_pseudo"))
common_edge.to_csv(f"{OUT_DIR}/Common_Edge_Cases.csv", index=False)

print("\n✔ Saved Common Edge Cases →", f"{OUT_DIR}/Common_Edge_Cases.csv")
print("Count:", len(common_edge))

print("\n🎉 DONE — comparison CSVs generated!")
print("Location:", OUT_DIR)
