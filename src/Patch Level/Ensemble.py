#!/usr/bin/env python3
"""
Ensemble of 3D CNN Models for Multi-Temporal Crop Classification

This script:
  1. Loads the patch-level parquet file (which has 60 band columns for 6 spectral bands over ~T months),
     groups them by band prefix, and reconstructs each patch into a tensor of shape (T, 128, 128, 6).
  2. Splits the data into train and test sets based on field_id.
  3. For each crop class, trains a binary 3D CNN (one-vs-all) model to distinguish that class.
  4. Saves each base model to disk.
  5. Generates out-of-fold predictions for the training set and predictions for the test set
     from the base models.
  6. Trains a logistic regression meta-classifier (stacking ensemble) on the base models’ outputs.
  7. Saves the ensemble meta-model as well as the train and test DataFrames.
"""

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from joblib import dump
import pickle

# ---------------------------
# PARAMETERS & FILE PATHS
# ---------------------------
PATCH_PARQUET = "/home/ubuntu/sai/Capstone_Group_3/src/Data/updated_patch_level_data.parquet"
TARGET_SIZE = (128, 128)  # spatial resolution for each patch
BATCH_SIZE = 8
EPOCHS = 20  # epochs for each base model training
BASE_MODEL_SAVE_DIR = "/home/ubuntu/sai/Capstone_Group_3/src/Models/BaseModels"
ENSEMBLE_MODEL_PATH = "/home/ubuntu/sai/Capstone_Group_3/src/Models/meta_model.joblib"
TRAIN_DF_PATH = "/home/ubuntu/sai/Capstone_Group_3/src/Data/train_df.pkl"
TEST_DF_PATH = "/home/ubuntu/sai/Capstone_Group_3/src/Data/test_df.pkl"

# The six band prefixes
BAND_PREFIXES = ['SA_B11', 'SA_B12', 'SA_B2', 'SA_B6', 'SA_EVI', 'SA_hue']


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def group_band_columns(channel_cols, band_prefixes):
    """Returns a dictionary mapping each band prefix to a sorted list of columns."""
    band_mapping = {}
    for prefix in band_prefixes:
        matching = [col for col in channel_cols if col.startswith(prefix)]
        if not matching:
            raise ValueError(f"No columns found for band prefix: {prefix}")
        matching_sorted = sorted(matching, key=lambda x: int(x.split('_')[-1]))
        band_mapping[prefix] = matching_sorted
    return band_mapping


def patch_pixels_to_image(df_patch, cols):
    """Reconstruct an image (of shape [H, W, len(cols)]) from pixel-level rows."""
    min_r = df_patch["row"].min()
    max_r = df_patch["row"].max()
    min_c = df_patch["col"].min()
    max_c = df_patch["col"].max()
    H = (max_r - min_r) + 1
    W = (max_c - min_c) + 1
    C = len(cols)
    img = np.zeros((H, W, C), dtype=np.float32)
    for _, px in df_patch.iterrows():
        rr = int(px["row"] - min_r)
        cc = int(px["col"] - min_c)
        vals = [px[c] for c in cols]
        img[rr, cc, :] = vals
    return img


def resize_image(image, target_size=(128, 128)):
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)  # [1, H, W, C]
    resized = tf.image.resize(tensor, target_size)
    return tf.squeeze(resized, axis=0).numpy()


# ---------------------------
# DATA GENERATOR FOR TIME-BASED INPUT
# ---------------------------
def patch_data_generator_time(df, patch_ids, band_mapping, label_encoder,
                              batch_size=8, infinite=True, target_size=(128, 128)):
    """
    Yields (X_batch, y_batch) where X_batch is of shape:
         (batch, T, target_size[0], target_size[1], num_bands)
    Each patch is reconstructed by:
      - For each band prefix, extracting the first T columns.
      - Reconstructing an image (shape: [H, W, T]) for that band.
      - Stacking these to form (H, W, T, num_bands) then transposing to (T, H, W, num_bands)
      - Resizing each time slice.
    """
    T = min(len(v) for v in band_mapping.values())
    num_bands = len(band_mapping)
    band_prefixes = sorted(band_mapping.keys())
    df_sub = df[df["patch_id"].isin(patch_ids)]
    unique_patches = df_sub["patch_id"].unique()
    rng = np.random.default_rng()
    while True:
        shuffled = rng.permutation(unique_patches)
        X_batch, y_batch = [], []
        for pid in shuffled:
            df_patch = df_sub[df_sub["patch_id"] == pid]
            crops = df_patch["crop_name"].unique()
            if len(crops) == 0:
                continue
            crop_str = crops[0]
            if crop_str not in label_encoder.classes_:
                continue
            y_val = label_encoder.transform([crop_str])[0]
            band_images = []
            for prefix in band_prefixes:
                cols = band_mapping[prefix][:T]
                band_img = patch_pixels_to_image(df_patch, cols)
                if band_img.shape[-1] > T:
                    band_img = band_img[..., :T]
                elif band_img.shape[-1] < T:
                    pad_width = T - band_img.shape[-1]
                    band_img = np.pad(band_img, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                band_images.append(band_img)
            stacked = np.stack(band_images, axis=-1)  # (H, W, T, num_bands)
            patch_img = np.transpose(stacked, (2, 0, 1, 3))  # (T, H, W, num_bands)
            T_current = patch_img.shape[0]
            resized_slices = []
            for t in range(T_current):
                slice_img = patch_img[t]
                resized_slice = resize_image(slice_img, target_size=target_size)
                resized_slices.append(resized_slice)
            final_patch = np.stack(resized_slices, axis=0)  # (T, target_size[0], target_size[1], num_bands)
            X_batch.append(final_patch)
            y_batch.append(y_val)
            if len(X_batch) == batch_size:
                yield (np.array(X_batch), np.array(y_batch))
                X_batch, y_batch = [], []
        if len(X_batch) > 0:
            yield (np.array(X_batch), np.array(y_batch))
            X_batch, y_batch = [], []
        if not infinite:
            break


# ---------------------------
# LOAD DATA & SPLIT (Train/Test based on field_id)
# ---------------------------
def load_and_split_data():
    df = pd.read_parquet(PATCH_PARQUET)
    print("Loaded DF shape:", df.shape)
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    df_ignore = df[list(ignore_cols)]
    channel_candidates = [c for c in df.columns if c not in ignore_cols]
    df_channel = df[channel_candidates].dropna(axis=1)
    df = pd.concat([df_ignore, df_channel], axis=1)
    print("After dropping NaN band columns, shape:", df.shape)
    df = df.dropna(subset=["crop_name"])
    df = df[df["crop_name"].str.lower() != "none"]
    fields = df["field_id"].dropna().unique()
    if len(fields) == 0:
        raise ValueError("No valid field_id found.")
    f_train, f_test = train_test_split(fields, test_size=0.2, random_state=42)
    train_ids = df.loc[df["field_id"].isin(f_train), "patch_id"].unique()
    test_ids = df.loc[df["field_id"].isin(f_test), "patch_id"].unique()
    print(f"Train patches: {len(train_ids)}, Test patches: {len(test_ids)}")
    return df, train_ids, test_ids


# ---------------------------
# BINARY MODEL CREATION FUNCTION (One-Vs-All)
# ---------------------------
def create_binary_model(input_shape):
    """
    Build a 3D CNN model for binary classification.
    The final layer has one unit with sigmoid activation.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling3D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ---------------------------
# WRAPPER FOR BINARY LABELS
# ---------------------------
def binary_generator(generator, target_label):
    """
    Wrap a multi-class generator so that the labels become binary:
    1 if the original label equals target_label, 0 otherwise.
    """
    for X, y in generator:
        y_bin = (y == target_label).astype(np.int32)
        yield X, y_bin


# ---------------------------
# FUNCTION TO GET PREDICTIONS FROM A MODEL
# ---------------------------
def get_model_predictions(model, df, patch_ids, band_mapping, label_encoder, batch_size, target_size):
    """
    Returns a vector of predictions (probabilities) for each patch in patch_ids.
    """
    gen = patch_data_generator_time(df, patch_ids, band_mapping, label_encoder,
                                    batch_size=batch_size, infinite=False, target_size=target_size)
    preds = []
    for X_batch, _ in gen:
        batch_preds = model.predict(X_batch)
        # For binary model, predict probability of class 1.
        preds.extend(batch_preds.flatten().tolist())
    return np.array(preds)


# ---------------------------
# MAIN SCRIPT: TRAINING BASE MODELS & ENSEMBLE
# ---------------------------
def main():
    # Create directory for saving base models
    os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)

    # Load and split data
    df, train_ids, test_ids = load_and_split_data()

    # Save the train and test DataFrames for future use
    with open(TRAIN_DF_PATH, "wb") as f:
        pickle.dump(df[df["patch_id"].isin(train_ids)], f)
    with open(TEST_DF_PATH, "wb") as f:
        pickle.dump(df[df["patch_id"].isin(test_ids)], f)
    print("Train and test sets saved.")

    # Determine channel columns and group them.
    ignore_cols = {"patch_id", "field_id", "crop_name", "row", "col"}
    channel_cols = sorted([c for c in df.columns if c not in ignore_cols])
    if len(channel_cols) != 60:
        raise ValueError(f"Expected 60 band columns, but got {len(channel_cols)}.")
    band_mapping = group_band_columns(channel_cols, BAND_PREFIXES)
    T = min(len(v) for v in band_mapping.values())
    print(f"Using {T} time steps for each band.")

    # Create a multi-class label encoder
    le = LabelEncoder()
    le.fit(df[df["patch_id"].isin(train_ids)]["crop_name"].unique())
    classes = le.classes_
    num_classes = len(classes)
    print("Crop classes:", classes)

    # For each class, train a binary model (one-vs-all)
    base_models = {}  # dictionary to store base models keyed by class
    for crop in classes:
        print(f"\nTraining base model for class '{crop}' (one-vs-all)...")
        # Get the target numeric label for this class
        target_label = le.transform([crop])[0]
        # Build a generator for binary classification for this class.
        train_gen = patch_data_generator_time(df, train_ids, band_mapping, le,
                                              batch_size=BATCH_SIZE, infinite=True, target_size=TARGET_SIZE)
        train_gen_bin = binary_generator(train_gen, target_label)
        val_gen = patch_data_generator_time(df, test_ids, band_mapping, le,
                                            batch_size=BATCH_SIZE, infinite=True, target_size=TARGET_SIZE)
        val_gen_bin = binary_generator(val_gen, target_label)
        input_shape = (T, TARGET_SIZE[0], TARGET_SIZE[1], len(BAND_PREFIXES))
        model = create_binary_model(input_shape)
        steps_per_epoch = math.ceil(len(train_ids) / BATCH_SIZE)
        validation_steps = math.ceil(len(test_ids) / BATCH_SIZE)
        history = model.fit(
            train_gen_bin,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen_bin,
            validation_steps=validation_steps
        )
        model_path = os.path.join(BASE_MODEL_SAVE_DIR, f"conv3d_model_class_{crop}.h5")
        model.save(model_path)
        print(f"Saved base model for class '{crop}' to {model_path}")
        base_models[crop] = model

    # ---------------------------
    # GET BASE MODEL PREDICTIONS & TRAIN META-CLASSIFIER
    # ---------------------------
    # For stacking, we need base model predictions on the train and test sets.
    # For each patch, each base model outputs a probability.
    # We'll build a feature vector of length = num_classes for each patch.
    def get_ensemble_features(df_subset, patch_ids):
        features = []
        for pid in patch_ids:
            # For each patch, get a list of probabilities from each base model.
            patch_features = []
            for crop in classes:
                model = base_models[crop]
                # Use our generator to get a single patch's prediction.
                gen = patch_data_generator_time(df_subset, [pid], band_mapping, le,
                                                batch_size=1, infinite=False, target_size=TARGET_SIZE)
                X, _ = next(gen)
                prob = model.predict(X)[0][0]  # probability of class (binary model outputs a single probability)
                patch_features.append(prob)
            features.append(patch_features)
        return np.array(features)

    # Get ensemble features for training and test sets.
    df_train = df[df["patch_id"].isin(train_ids)].copy()
    df_test = df[df["patch_id"].isin(test_ids)].copy()
    print("Generating ensemble features for training set...")
    X_train_ensemble = get_ensemble_features(df_train, train_ids)
    y_train = le.transform([df_train[df_train["patch_id"] == pid]["crop_name"].iloc[0] for pid in train_ids])
    print("Generating ensemble features for test set...")
    X_test_ensemble = get_ensemble_features(df_test, test_ids)
    y_test = le.transform([df_test[df_test["patch_id"] == pid]["crop_name"].iloc[0] for pid in test_ids])

    # Train a simple logistic regression meta-classifier.
    from sklearn.linear_model import LogisticRegression
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train_ensemble, y_train)
    print("Meta-classifier trained.")

    # Evaluate ensemble on test set.
    y_pred = meta_model.predict(X_test_ensemble)
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- ENSEMBLE TEST EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save the meta-model.
    from joblib import dump
    dump(meta_model, ENSEMBLE_MODEL_PATH)
    print(f"Ensemble meta-model saved to {ENSEMBLE_MODEL_PATH}")


if __name__ == "__main__":
    main()
