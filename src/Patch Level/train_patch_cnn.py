#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# ---------------------------------------------------------------------
# 1) PARAMETERS & FILE PATHS
# ---------------------------------------------------------------------
PATCH_PARQUET = "/home/ubuntu/sai/Capstone_Group_3/src/Data/patch_level_data_compressed.parquet"

TARGET_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 20

# ---------------------------------------------------------------------
# 2) PATCH → IMAGE UTILITIES
# ---------------------------------------------------------------------
def patch_pixels_to_image(df_patch, channel_cols):
    """
    Convert pixel-level rows for ONE patch into [H, W, n_channels].
    We use 'row' and 'col' to place each pixel. Missing spots => 0.
    """
    min_r = df_patch["row"].min()
    max_r = df_patch["row"].max()
    min_c = df_patch["col"].min()
    max_c = df_patch["col"].max()

    patch_height = (max_r - min_r) + 1
    patch_width = (max_c - min_c) + 1
    n_channels = len(channel_cols)

    img = np.zeros((patch_height, patch_width, n_channels), dtype=np.float32)

    for _, px in df_patch.iterrows():
        rr = int(px["row"] - min_r)
        cc = int(px["col"] - min_c)
        vals = [px[ch] for ch in channel_cols]
        img[rr, cc, :] = vals

    return img

def resize_image(image, target_size=(128,128)):
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)  # [1,H,W,C]
    resized = tf.image.resize(tensor, target_size)
    return tf.squeeze(resized, axis=0).numpy()  # [H,W,C]

# ---------------------------------------------------------------------
# 3) DATA GENERATOR
# ---------------------------------------------------------------------
def patch_data_generator(
    df,
    patch_ids,
    channel_cols,
    label_encoder,
    batch_size=8,
    infinite=True,
    target_size=(128,128)
):
    """
    Groups pixel-level rows by patch_id, builds an image, yields X_batch, y_batch.
    We assume each patch has exactly one crop_name (the same for all pixels).
    """
    df_sub = df[df["patch_id"].isin(patch_ids)]
    unique_patches = df_sub["patch_id"].unique()

    while True:
        # Shuffle patch_ids each epoch
        rng = np.random.default_rng()
        shuffled_ids = rng.permutation(unique_patches)

        X_batch, y_batch = [], []
        for p_id in shuffled_ids:
            df_patch = df_sub[df_sub["patch_id"] == p_id]
            # Single crop_name per patch
            crops = df_patch["crop_name"].unique()
            if len(crops) == 0:
                continue
            crop_str = crops[0]
            # Skip unknown label in test
            if crop_str not in label_encoder.classes_:
                continue

            y_val = label_encoder.transform([crop_str])[0]

            # Reconstruct [H,W,channels], then resize
            patch_img = patch_pixels_to_image(df_patch, channel_cols)
            patch_img_resized = resize_image(patch_img, target_size=target_size)

            X_batch.append(patch_img_resized)
            y_batch.append(y_val)

            if len(X_batch) == batch_size:
                yield np.array(X_batch), np.array(y_batch)
                X_batch, y_batch = [], []

        # leftover partial batch
        if len(X_batch) > 0:
            yield np.array(X_batch), np.array(y_batch)
            X_batch, y_batch = [], []

        if not infinite:
            break

# ---------------------------------------------------------------------
# 4) MAIN TRAINING SCRIPT
# ---------------------------------------------------------------------
def main():
    # A) LOAD PARQUET
    df = pd.read_parquet(PATCH_PARQUET)

    # Specify the columns to ignore (keep regardless of NaN)
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}

    df_ignore = df[list(ignore_cols)]
    channel_cols = [c for c in df.columns if c not in ignore_cols]

    df_channel = df[channel_cols].dropna(axis=1)

    df = pd.concat([df_ignore, df_channel], axis=1)
    print("Loaded patch-level parquet.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(5))
    print(df.isna().sum())
    print(df['crop_name'].value_counts().sum())

    #FILTER INVALID CROPS
    df = df.dropna(subset=["crop_name"])
    df = df[df["crop_name"].str.lower() != "none"]

    #FIELD-BASED SPLIT
    unique_fields = df["field_id"].dropna().unique()
    if len(unique_fields) == 0:
        print("No valid field_id found in DataFrame. Exiting.")
        return

    fields_train, fields_test = train_test_split(unique_fields, test_size=0.2, random_state=42)

    train_patch_ids = df.loc[df["field_id"].isin(fields_train), "patch_id"].unique()
    test_patch_ids = df.loc[df["field_id"].isin(fields_test), "patch_id"].unique()
    print(f"#fields train: {len(fields_train)}, #fields test: {len(fields_test)}")
    print(f"#patches train: {len(train_patch_ids)}, #patches test: {len(test_patch_ids)}")

    #LABEL ENCODE
    df_train = df[df["patch_id"].isin(train_patch_ids)].copy()
    df_test = df[df["patch_id"].isin(test_patch_ids)].copy()

    le = LabelEncoder()
    train_crops = df_train["crop_name"].unique()
    le.fit(train_crops)

    df_test = df_test[df_test["crop_name"].isin(le.classes_)]
    if len(df_test) == 0:
        print("No valid test patches after filtering unknown classes. Exiting.")
        return

    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    # Build a list of columns for bands/time
    channel_cols = [c for c in df.columns if c not in ignore_cols]
    # Optional: sort or reorder them
    channel_cols = sorted(channel_cols)
    print("\nChannel/Band columns used:", channel_cols)

    # SETUP DATA GENERATORS
    train_gen = patch_data_generator(
        df_train,
        patch_ids=train_patch_ids,
        channel_cols=channel_cols,
        label_encoder=le,
        batch_size=BATCH_SIZE,
        infinite=True,
        target_size=TARGET_SIZE
    )

    val_gen = patch_data_generator(
        df_test,
        patch_ids=test_patch_ids,
        channel_cols=channel_cols,
        label_encoder=le,
        batch_size=BATCH_SIZE,
        infinite=True,
        target_size=TARGET_SIZE
    )

    # -- PRINT BATCH SHAPE SNIPPET --
    X_temp, y_temp = next(train_gen)
    print("Example training batch shape:", X_temp.shape)  # e.g. (8, 128, 128, n_channels)
    print("Example training labels shape:", y_temp.shape)

    train_gen = patch_data_generator(
        df_train,
        patch_ids=train_patch_ids,
        channel_cols=channel_cols,
        label_encoder=le,
        batch_size=BATCH_SIZE,
        infinite=True,
        target_size=TARGET_SIZE
    )

    train_count = len(train_patch_ids)
    test_count = len(test_patch_ids)
    steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
    validation_steps = math.ceil(test_count / BATCH_SIZE)

    print(f"\nsteps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

    n_channels = len(channel_cols)
    n_classes = len(le.classes_)

    model = models.Sequential([
        layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], n_channels)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if train_count == 0:
        print("No training patches found. Exiting.")
        return

    #TRAIN
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen if validation_steps > 0 else None,
        validation_steps=validation_steps if validation_steps > 0 else None
    )

    #EVALUATE
    test_once_gen = patch_data_generator(
        df_test,
        patch_ids=test_patch_ids,
        channel_cols=channel_cols,
        label_encoder=le,
        batch_size=BATCH_SIZE,
        infinite=False,
        target_size=TARGET_SIZE
    )

    y_true, y_pred = [], []
    for X_batch, y_batch in test_once_gen:
        preds = model.predict(X_batch)
        preds_label = np.argmax(preds, axis=1)
        y_true.extend(y_batch.tolist())
        y_pred.extend(preds_label.tolist())

    if len(y_true) == 0:
        print("No valid test patches for evaluation. Exiting.")
        return

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- TEST EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)

    # I) SAVE MODEL
    model.save("patch_level_cnn.h5")
    print("CNN model saved to patch_level_cnn.h5")

if __name__ == "__main__":
    main()
