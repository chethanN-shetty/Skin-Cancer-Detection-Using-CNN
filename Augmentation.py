import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
from tqdm import tqdm

# ============================
# 1️⃣  CONFIGURATION
# ============================
# Set your paths
DATA_DIR = r"C:\Users\Sudarshan\Downloads\archive (1)\ISIC_2019_Training_Input\ISIC_2019_Training_Input"  # path where original images are stored
CSV_PATH = r"C:\Users\Sudarshan\Downloads\archive (1)\ISIC_2019_Training_GroundTruth.csv"  # path to your CSV file
OUTPUT_DIR = r"C:\Users\Sudarshan\Downloads\archive (1)\augmented_images"  # path where augmented images will be saved

# How many augmented images per original image
AUG_PER_IMAGE = 5

# Image size (resize for augmentation)
IMG_SIZE = (224, 224)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 2️⃣  LOAD AND PREPARE LABELS
# ============================
df = pd.read_csv(CSV_PATH)
print("✅ CSV Loaded:", CSV_PATH)
print("Columns:", df.columns.tolist())

# Find label columns (all except 'image')
label_cols = [c for c in df.columns if c.lower() != "image"]

# Convert one-hot columns → single label column
df["label"] = df[label_cols].idxmax(axis=1)
df = df[["image", "label"]]  # keep only necessary columns

print("\n✅ One-hot columns converted to 'label'")
print(df.head())

# ============================
# 3️⃣  IMAGE DATA GENERATOR
# ============================
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# ============================
# 4️⃣  AUGMENT IMAGES
# ============================
for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting images"):
    img_name = row["image"]
    label = row["label"]

    # Image file path (assuming .jpg or .png — change extension if needed)
    img_path_jpg = os.path.join(DATA_DIR, img_name + ".jpg")
    img_path_png = os.path.join(DATA_DIR, img_name + ".png")

    if os.path.exists(img_path_jpg):
        img_path = img_path_jpg
    elif os.path.exists(img_path_png):
        img_path = img_path_png
    else:
        continue  # skip missing files

    # Load and preprocess
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Create folder for this label
    label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Generate augmented images
    prefix = f"{img_name}_{label}"
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=label_dir,
                              save_prefix=prefix,
                              save_format='jpg'):
        i += 1
        if i >= AUG_PER_IMAGE:
            break

print("\n🎉 Augmentation complete!")
print(f"➡️ Augmented images saved in: {OUTPUT_DIR}")
