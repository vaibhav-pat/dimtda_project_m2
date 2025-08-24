# File: create_splits.py (Version 4 - Final, Keys Match Traceback)
import json
import random
from glob import glob
import os
from tqdm import tqdm

# --- Configuration ---
SEED = 42
LANG_SRC = "en"
LANG_TGT = "zh"
DATA_DIR = "./data/DoTA_dataset"
# ---------------------

print(f"🔄 Generating new data splits from files in {DATA_DIR}...")

img_dir = os.path.join(DATA_DIR, "imgs")
all_image_files = glob(os.path.join(img_dir, "*.png"))
all_basenames = [os.path.splitext(os.path.basename(f))[0] for f in all_image_files]

valid_samples = []
print(f"Found {len(all_basenames)} image files. Verifying matching text files...")
for basename in tqdm(all_basenames):
    src_path = os.path.join(DATA_DIR, f"{LANG_SRC}_mmd", f"{basename}.mmd")
    tgt_path = os.path.join(DATA_DIR, f"{LANG_TGT}_mmd", f"{basename}.mmd")
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        valid_samples.append(basename)

print(f"✅ Found {len(valid_samples)} complete and valid samples.")

random.seed(SEED)
random.shuffle(valid_samples)
train_end = int(0.8 * len(valid_samples))
val_end = int(0.9 * len(valid_samples))
train_split, val_split, test_split = valid_samples[:train_end], valid_samples[train_end:val_end], valid_samples[val_end:]

# --- CRITICAL CHANGE: Corrected the key to match the error message EXACTLY ---
final_split_data = {
    "train_name_list": train_split,
    "valid_name_list": val_split, # Corrected to 'valid_name_list' as requested by the script
    "test_name_list": test_split
}
# -------------------------------------------------------------------------

DATA_SPLIT_FILE = os.path.join(DATA_DIR, "generated_split_dataset.json")
with open(DATA_SPLIT_FILE, 'w') as f:
    json.dump(final_split_data, f, indent=2)

print(f"\n✅ Successfully created new data split file at: {DATA_SPLIT_FILE} (with all correct keys)")
print(f"   - Training samples:   {len(train_split)}")
print(f"   - Validation samples: {len(val_split)}")
print(f"   - Test samples:       {len(test_split)}")