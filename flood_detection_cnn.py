"""
Flood Detection from Satellite Images Using CNN
=================================================
Student     : Madhavapeddy Achyuth Reddy
Programme   : MSc Data Science
University  : University of Europe for Applied Sciences
Supervisor  : Shan Faiz
Course      : Machine Learning
Phase       : Phase 2 - Proposal and Code Implementation

This script implements a CNN-based binary image classification pipeline
to detect flooded vs non-flooded regions from satellite imagery.

Two models are trained, evaluated, and compared:
  1. Custom CNN  - 4-block convolutional network designed from scratch
  2. MobileNetV2 - ImageNet pre-trained transfer learning model

Run with:  python flood_detection_cnn.py
"""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # no display needed - works on any machine/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
SEED        = 42
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS_CNN  = 25
EPOCHS_TL   = 15
EPOCHS_FT   = 10
NUM_CLASSES = 2
CLASS_NAMES = ['Non-Flooded', 'Flooded']
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA'})


# ============================================================
# SECTION 1: DATASET LOADING
# ============================================================
def download_or_generate_dataset():
    """
    Load dataset from a Kaggle input path if available, otherwise
    generate a realistic synthetic satellite-image dataset
    (150 flooded + 150 non-flooded) for full pipeline demonstration.

    Returns
    -------
    data_dir : Path - root containing 'flooded' and 'non_flooded' folders
    """
    kaggle_paths = [
        '/kaggle/input/flood-area-segmentation/Image',
        '/kaggle/input/flood-area-segmentation',
    ]
    for kp in kaggle_paths:
        if os.path.exists(kp) and \
           os.path.exists(os.path.join(kp, 'flooded')) and \
           os.path.exists(os.path.join(kp, 'non_flooded')):
            print(f'Found dataset at: {kp}')
            return Path(kp)

    print('No attached dataset found - generating synthetic dataset...')
    data_dir      = Path('flood_dataset')
    flood_dir     = data_dir / 'flooded'
    non_flood_dir = data_dir / 'non_flooded'
    flood_dir.mkdir(parents=True, exist_ok=True)
    non_flood_dir.mkdir(parents=True, exist_ok=True)

    n_per_class = 150
    for i in range(n_per_class):
        # ---- Flooded image: dark blue/grey water tones ----
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(20, 60)
        img[:, :, 1] = np.random.randint(40, 90)
        img[:, :, 2] = np.random.randint(80, 140)
        noise = np.random.normal(0, 15, (256, 256, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for _ in range(random.randint(2, 6)):
            x1, y1 = random.randint(0, 200), random.randint(0, 200)
            x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
            img[y1:y2, x1:x2] = [random.randint(60, 120),
                                  random.randint(80, 150),
                                  random.randint(30, 80)]
        img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(str(flood_dir / f'flood_{i:03d}.jpg'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # ---- Non-flooded image: varied terrain tones ----
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        terrain = i % 4
        if terrain == 0:
            img2[:, :, 0] = np.random.randint(30, 80)
            img2[:, :, 1] = np.random.randint(90, 160)
            img2[:, :, 2] = np.random.randint(20, 60)
        elif terrain == 1:
            img2[:, :, 0] = np.random.randint(120, 200)
            img2[:, :, 1] = np.random.randint(100, 160)
            img2[:, :, 2] = np.random.randint(60, 110)
        elif terrain == 2:
            v = np.random.randint(100, 180)
            img2[:, :] = [v, v, v]
        else:
            img2[:, :, 0] = np.random.randint(80, 150)
            img2[:, :, 1] = np.random.randint(110, 180)
            img2[:, :, 2] = np.random.randint(50, 100)
        noise2 = np.random.normal(0, 20, (256, 256, 3)).astype(np.int16)
        img2 = np.clip(img2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, 200), random.randint(0, 200)
            x2, y2 = x1 + random.randint(15, 60), y1 + random.randint(15, 60)
            img2[y1:y2, x1:x2] = [random.randint(60, 200),
                                   random.randint(60, 200),
                                   random.randint(30, 150)]
        img2 = cv2.GaussianBlur(img2, (3, 3), 0)
        cv2.imwrite(str(non_flood_dir / f'non_flood_{i:03d}.jpg'),
                    cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    print(f'Generated {n_per_class} flooded + {n_per_class} non-flooded images')
    return data_dir


def load_dataset(data_root, label_map, img_size):
    """
    Load all images from class sub-directories, resize and convert to RGB.

    Returns
    -------
    images : np.ndarray (N, H, W, 3) uint8
    labels : np.ndarray (N,) int32
    """
    images, labels = [], []
    valid_ext = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    data_root = Path(data_root)

    for folder_name, label in label_map.items():
        folder_path = data_root / folder_name
        if not folder_path.exists():
            print(f'  [SKIP] {folder_path} not found')
            continue
        files = [f for f in sorted(folder_path.iterdir())
                 if f.suffix.lower() in valid_ext]
        print(f'  {folder_name:<15s} | label={label} | files={len(files)}')

        for fpath in files:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size[1], img_size[0]),
                              interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(label)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


# ============================================================
# SECTION 2: CUSTOM CNN ARCHITECTURE
# ============================================================
def build_custom_cnn(input_shape=(128, 128, 3), num_classes=2):
    """
    Custom 4-block CNN for flood detection.

    Block 1: Conv(32)x2  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Block 2: Conv(64)x2  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Block 3: Conv(128)x2 -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Block 4: Conv(256)   -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Head: GlobalAvgPool -> Dense(256) -> BN -> ReLU -> Dropout(0.5) -> Softmax
    """
    model = models.Sequential(name='Custom_CNN')
    model.add(layers.Input(shape=input_shape))

    for filters in [32, 64, 128]:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# ============================================================
# SECTION 3: MOBILENETV2 TRANSFER LEARNING MODEL
# ============================================================
def build_mobilenetv2(input_shape=(128, 128, 3), num_classes=2):
    """
    MobileNetV2 transfer learning model.
    Phase 1: freeze base, train head only.
    Phase 2: unfreeze top 30 layers, fine-tune with lr=1e-5.
    """
    base = MobileNetV2(input_shape=input_shape,
                        include_top=False, weights='imagenet')
    base.trainable = False

    inputs  = tf.keras.Input(shape=input_shape)
    x       = mobilenet_preprocess(inputs * 255.0)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256)(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Activation('relu')(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name='MobileNetV2_TL'), base


# ============================================================
# SECTION 4: TRAINING CURVE PLOTTING
# ============================================================
def plot_training_curves(history_dict, model_name, save_path):
    """Plot and save training vs validation accuracy and loss."""
    n_ep   = len(history_dict['accuracy'])
    epochs = range(1, n_ep + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Curves - {model_name}', fontsize=14, fontweight='bold')

    ax1.plot(epochs, history_dict['accuracy'],     'b-o', ms=4, lw=2, label='Train Acc')
    ax1.plot(epochs, history_dict['val_accuracy'], 'r-o', ms=4, lw=2, label='Val Acc')
    ax1.axhline(max(history_dict['val_accuracy']), color='red', ls='--', alpha=0.4,
                 label=f'Best: {max(history_dict["val_accuracy"])*100:.1f}%')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.5)

    ax2.plot(epochs, history_dict['loss'],     'b-o', ms=4, lw=2, label='Train Loss')
    ax2.plot(epochs, history_dict['val_loss'], 'r-o', ms=4, lw=2, label='Val Loss')
    ax2.axhline(min(history_dict['val_loss']), color='red', ls='--', alpha=0.4,
                 label=f'Min: {min(history_dict["val_loss"]):.4f}')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {save_path}')


# ============================================================
# SECTION 5: FULL MODEL EVALUATION
# ============================================================
def full_evaluation(model, X_test, y_test, y_test_oh, class_names,
                     model_name, figures_dir, batch_size=32):
    """
    Complete evaluation: test accuracy/loss, confusion matrix,
    classification report, and ROC curve. Uses pure matplotlib
    (no seaborn) for maximum portability.
    """
    safe = model_name.lower().replace(' ', '_')

    y_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f'\n{model_name} - Test Results')
    print('-' * 45)
    print(f'  Accuracy : {test_acc*100:.2f}%')
    print(f'  Loss     : {test_loss:.4f}')

    print(f'\nClassification Report:')
    print('-' * 45)
    report_str  = classification_report(y_test, y_pred, target_names=class_names)
    report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                          output_dict=True)
    print(report_str)

    # ---- Confusion matrix (pure matplotlib) ----
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = figures_dir / f'fig_cm_{safe}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {cm_path}')

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5,
            label=f'ROC AUC = {roc_auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2)
    ax.fill_between(fpr, tpr, alpha=0.08, color='darkorange')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    roc_path = figures_dir / f'fig_roc_{safe}.png'
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {roc_path}')

    metrics = {
        'Model':         model_name,
        'Accuracy (%)':  round(test_acc * 100, 2),
        'Precision (%)': round(report_dict['weighted avg']['precision'] * 100, 2),
        'Recall (%)':    round(report_dict['weighted avg']['recall'] * 100, 2),
        'F1-Score (%)':  round(report_dict['weighted avg']['f1-score'] * 100, 2),
        'ROC AUC (%)':   round(roc_auc_val * 100, 2),
    }
    return metrics, y_pred, y_prob


# ============================================================
# SECTION 6: GRAD-CAM VISUALISATION
# ============================================================
def get_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError('No Conv2D layer found')


def make_gradcam(img_array, model, last_conv_name):
    """Compute a Grad-CAM heatmap for the given image array (1,H,W,3)."""
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output])
    with tf.GradientTape() as tape:
        img_t = tf.cast(img_array, tf.float32)
        conv_out, preds = grad_model(img_t)
        pred_idx = int(tf.argmax(preds[0]))
        class_ch = preds[:, pred_idx]
    grads   = tape.gradient(class_ch, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_idx, float(preds[0][pred_idx])


def overlay_cam(img, heatmap, alpha=0.45):
    """Overlay a Grad-CAM heatmap on the original RGB image."""
    h, w = img.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    jet  = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    jet  = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(jet * alpha + np.clip(img, 0, 1) * (1 - alpha), 0, 1)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print(f'TensorFlow  : {tf.__version__}')
    print(f'NumPy       : {np.__version__}')
    print(f'GPU devices : {tf.config.list_physical_devices("GPU")}')

    # ---- 1. Dataset loading ----
    print('\n' + '=' * 60)
    print('SECTION 1: DATASET LOADING')
    print('=' * 60)
    data_dir  = download_or_generate_dataset()
    label_map = {'flooded': 1, 'non_flooded': 0}

    print('\nLoading images...')
    images_raw, labels_raw = load_dataset(data_dir, label_map, IMG_SIZE)
    print(f'Total images: {len(images_raw)} | shape: {images_raw[0].shape}')
    u, c = np.unique(labels_raw, return_counts=True)
    for ui, ci in zip(u, c):
        print(f'  Class {ui} ({CLASS_NAMES[ui]:<14s}): {ci} images '
              f'({ci/len(labels_raw)*100:.1f}%)')

    # ---- EDA figures ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Class Distribution Analysis', fontsize=15, fontweight='bold')
    colors = ['#1976D2', '#E53935']
    bars = axes[0].bar([CLASS_NAMES[ui] for ui in u], c, color=colors,
                        edgecolor='white', linewidth=1.5, width=0.5)
    axes[0].set_title('Image Count per Class')
    axes[0].set_ylabel('Number of Images')
    for bar, count in zip(bars, c):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                      f'{count}\n({count/len(labels_raw)*100:.1f}%)',
                      ha='center', fontsize=11, fontweight='bold')
    axes[1].pie(c, labels=[CLASS_NAMES[ui] for ui in u], autopct='%1.1f%%',
                 colors=colors, startangle=90,
                 wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Class Proportion')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig01_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved -> figures/fig01_class_distribution.png')

    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    fig.suptitle('Sample Satellite Images\nRow 1: Non-Flooded | Row 2: Flooded',
                  fontsize=13, fontweight='bold')
    for row_idx, cls_label in enumerate([0, 1]):
        cls_images = images_raw[labels_raw == cls_label]
        chosen = np.random.choice(len(cls_images), 6, replace=False)
        for col_idx, img_idx in enumerate(chosen):
            axes[row_idx, col_idx].imshow(cls_images[img_idx])
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].set_title(
                CLASS_NAMES[cls_label], fontsize=9, fontweight='bold',
                color='#1976D2' if cls_label == 0 else '#E53935')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig02_sample_images.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved -> figures/fig02_sample_images.png')

    # ---- 2. Preprocessing ----
    print('\n' + '=' * 60)
    print('SECTION 2: PREPROCESSING - NORMALISATION & SPLIT')
    print('=' * 60)
    images_norm = images_raw.astype(np.float32) / 255.0

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images_norm, labels_raw, test_size=0.15,
        stratify=labels_raw, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176,
        stratify=y_train_val, random_state=SEED)

    total = len(images_norm)
    print(f'Train: {len(X_train)} ({len(X_train)/total*100:.1f}%) | '
          f'Val: {len(X_val)} ({len(X_val)/total*100:.1f}%) | '
          f'Test: {len(X_test)} ({len(X_test)/total*100:.1f}%)')

    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = to_categorical(y_val,   NUM_CLASSES)
    y_test_oh  = to_categorical(y_test,  NUM_CLASSES)

    # ---- 3. Data augmentation ----
    print('\n' + '=' * 60)
    print('SECTION 3: DATA AUGMENTATION')
    print('=' * 60)
    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
        zoom_range=0.15, horizontal_flip=True, vertical_flip=True,
        brightness_range=[0.8, 1.2], shear_range=0.1, fill_mode='nearest')
    val_test_datagen = ImageDataGenerator()

    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    val_steps       = max(1, len(X_val) // BATCH_SIZE)
    print(f'Steps per epoch: {steps_per_epoch} | Val steps: {val_steps}')

    # Augmentation visualisation
    sample_img = X_train[0:1]
    aug_demo = train_datagen.flow(sample_img, y_train_oh[0:1], batch_size=1, seed=0)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle('Data Augmentation - 1 Original Image, 9 Augmented Versions',
                  fontsize=13, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i == 0:
            ax.imshow(sample_img[0])
            ax.set_title('Original', fontsize=10, fontweight='bold', color='#1976D2')
        else:
            aug_img, _ = next(aug_demo)
            ax.imshow(np.clip(aug_img[0], 0, 1))
            ax.set_title(f'Augmented #{i}', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig03_augmentation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved -> figures/fig03_augmentation.png')

    # ---- 4. Custom CNN ----
    print('\n' + '=' * 60)
    print('SECTION 4: CUSTOM CNN - BUILD & TRAIN')
    print('=' * 60)
    cnn_model = build_custom_cnn((*IMG_SIZE, 3), NUM_CLASSES)
    cnn_model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                       loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()
    print(f'Total parameters: {cnn_model.count_params():,}')

    train_gen = train_datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE, seed=SEED)
    val_gen   = val_test_datagen.flow(X_val, y_val_oh, batch_size=BATCH_SIZE, seed=SEED)

    cnn_callbacks = [
        callbacks.ModelCheckpoint('best_cnn.keras', monitor='val_accuracy',
                                    save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=5, min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                  restore_best_weights=True, verbose=1),
    ]

    cnn_history = cnn_model.fit(
        train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS_CNN,
        validation_data=val_gen, validation_steps=val_steps,
        callbacks=cnn_callbacks, verbose=1)

    plot_training_curves(cnn_history.history, 'Custom CNN',
                          FIGURES_DIR / 'fig04_cnn_curves.png')

    cnn_metrics, y_pred_cnn, y_prob_cnn = full_evaluation(
        cnn_model, X_test, y_test, y_test_oh, CLASS_NAMES, 'Custom CNN',
        FIGURES_DIR, BATCH_SIZE)

    # ---- 5. MobileNetV2 transfer learning ----
    print('\n' + '=' * 60)
    print('SECTION 5: MOBILENETV2 TRANSFER LEARNING')
    print('=' * 60)
    tl_model, tl_base = build_mobilenetv2((*IMG_SIZE, 3), NUM_CLASSES)
    tl_model.compile(optimizer=optimizers.Adam(1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    tl_callbacks = [
        callbacks.ModelCheckpoint('best_mobilenet.keras', monitor='val_accuracy',
                                    save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=4, min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                  restore_best_weights=True, verbose=1),
    ]

    train_gen_tl = train_datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE, seed=SEED)
    val_gen_tl   = val_test_datagen.flow(X_val, y_val_oh, batch_size=BATCH_SIZE, seed=SEED)

    print(f'PHASE 1: Feature Extraction (max {EPOCHS_TL} epochs, frozen base)...')
    tl_hist_p1 = tl_model.fit(
        train_gen_tl, steps_per_epoch=steps_per_epoch, epochs=EPOCHS_TL,
        validation_data=val_gen_tl, validation_steps=val_steps,
        callbacks=tl_callbacks, verbose=1)
    n_p1 = len(tl_hist_p1.history['accuracy'])

    # Phase 2: fine-tune top 30 layers
    tl_base.trainable = True
    for layer in tl_base.layers[:len(tl_base.layers) - 30]:
        layer.trainable = False
    tl_model.compile(optimizer=optimizers.Adam(1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    train_gen_ft = train_datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE, seed=SEED)
    val_gen_ft   = val_test_datagen.flow(X_val, y_val_oh, batch_size=BATCH_SIZE, seed=SEED)

    print(f'\nPHASE 2: Fine-Tuning (max {EPOCHS_FT} epochs, lr=1e-5)...')
    tl_hist_p2 = tl_model.fit(
        train_gen_ft, steps_per_epoch=steps_per_epoch, epochs=EPOCHS_FT,
        initial_epoch=n_p1, validation_data=val_gen_ft, validation_steps=val_steps,
        callbacks=tl_callbacks, verbose=1)

    merged_hist = {k: tl_hist_p1.history[k] + tl_hist_p2.history.get(k, [])
                   for k in tl_hist_p1.history}
    plot_training_curves(merged_hist, 'MobileNetV2 Transfer Learning',
                          FIGURES_DIR / 'fig05_mobilenet_curves.png')

    tl_metrics, y_pred_tl, y_prob_tl = full_evaluation(
        tl_model, X_test, y_test, y_test_oh, CLASS_NAMES, 'MobileNetV2',
        FIGURES_DIR, BATCH_SIZE)

    # ---- 6. Model comparison ----
    print('\n' + '=' * 60)
    print('SECTION 6: MODEL COMPARISON')
    print('=' * 60)
    comparison_df = pd.DataFrame([cnn_metrics, tl_metrics]).set_index('Model')
    print(comparison_df.to_string())

    x, width = np.arange(len(comparison_df.columns)), 0.35
    colors = ['#1976D2', '#E53935']
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (idx, color) in enumerate(zip(comparison_df.index, colors)):
        values = comparison_df.loc[idx].tolist()
        bars = ax.bar(x + i*width - width/2, values, width,
                       label=idx, color=color, alpha=0.87, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.columns, fontsize=11)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Custom CNN vs MobileNetV2 - Performance Comparison',
                  fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig06_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved -> figures/fig06_model_comparison.png')

    # Overlay ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for y_prob, name, color in [(y_prob_cnn, 'Custom CNN', '#1976D2'),
                                  (y_prob_tl, 'MobileNetV2', '#E53935')]:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC={auc(fpr,tpr):.4f})')
        ax.fill_between(fpr, tpr, alpha=0.05, color=color)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig07_roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved -> figures/fig07_roc_comparison.png')

    # ---- 7. Grad-CAM ----
    print('\n' + '=' * 60)
    print('SECTION 7: GRAD-CAM VISUALISATION')
    print('=' * 60)
    last_conv = get_last_conv(cnn_model)
    print(f'Last Conv2D layer: {last_conv}')

    np.random.seed(SEED)
    idx_s = np.random.choice(len(X_test), min(8, len(X_test)), replace=False)
    n_show = len(idx_s)

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle('Grad-CAM Interpretability - Custom CNN\n'
                  'Row 1: Original | Row 2: Heatmap | Row 3: Overlay',
                  fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, n_show, figure=fig, hspace=0.05, wspace=0.05)

    for col, idx in enumerate(idx_s):
        img    = X_test[idx]
        img_in = np.expand_dims(img, 0).astype(np.float32)
        true_l = CLASS_NAMES[y_test[idx]]
        hm, pred_cls, conf = make_gradcam(img_in, cnn_model, last_conv)
        pred_l  = CLASS_NAMES[pred_cls]
        overlay = overlay_cam(img, hm)
        ok = true_l == pred_l

        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(img); ax0.axis('off')
        ax0.set_title(f'T:{true_l[:3]} P:{pred_l[:3]}\n{conf*100:.0f}%',
                      fontsize=8, color='#2E7D32' if ok else '#C62828', fontweight='bold')

        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(hm, cmap='jet'); ax1.axis('off')

        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(overlay); ax2.axis('off')

    plt.savefig(FIGURES_DIR / 'fig08_gradcam.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved -> figures/fig08_gradcam.png')

    # ---- 8. Error analysis ----
    print('\n' + '=' * 60)
    print('SECTION 8: ERROR ANALYSIS')
    print('=' * 60)
    misclassified = np.where(y_pred_cnn != y_test)[0]
    fp = np.sum((y_pred_cnn == 1) & (y_test == 0))
    fn = np.sum((y_pred_cnn == 0) & (y_test == 1))
    print(f'Total test : {len(y_test)}')
    print(f'Correct    : {len(y_test)-len(misclassified)} '
          f'({(1-len(misclassified)/len(y_test))*100:.1f}%)')
    print(f'Errors     : {len(misclassified)} '
          f'({len(misclassified)/len(y_test)*100:.1f}%)')
    print(f'False Positives (Non-Flood -> Flood): {fp}')
    print(f'False Negatives (Flood -> Non-Flood): {fn}')

    if len(misclassified) > 0:
        n_show = min(10, len(misclassified))
        fig, axes = plt.subplots(2, 5, figsize=(20, 9))
        fig.suptitle(f'Error Analysis - Misclassified Images (Custom CNN)\n'
                     f'Total errors: {len(misclassified)} | FP={fp} FN={fn}',
                     fontsize=13, fontweight='bold')
        for ax, idx in zip(axes.flat, misclassified[:n_show]):
            ax.imshow(X_test[idx])
            ax.set_title(
                f'TRUE: {CLASS_NAMES[y_test[idx]]}\n'
                f'PRED: {CLASS_NAMES[y_pred_cnn[idx]]} '
                f'({y_prob_cnn[idx][y_pred_cnn[idx]]*100:.0f}%)',
                fontsize=8.5, color='#C62828', fontweight='bold')
            ax.axis('off')
        for ax in axes.flat[n_show:]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fig09_error_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved -> figures/fig09_error_analysis.png')

    # ---- 9. Save models & summary ----
    print('\n' + '=' * 60)
    print('SECTION 9: SAVE MODELS & FINAL SUMMARY')
    print('=' * 60)
    cnn_model.save('flood_custom_cnn.keras')
    tl_model.save('flood_mobilenetv2.keras')
    print('Models saved: flood_custom_cnn.keras, flood_mobilenetv2.keras')

    print('\nSaved figures:')
    for f in sorted(FIGURES_DIR.iterdir()):
        print(f'  {f.name}  ({f.stat().st_size/1024:.1f} KB)')

    print('\n' + '=' * 65)
    print('              FINAL RESULTS SUMMARY')
    print('=' * 65)
    print(comparison_df.to_string())
    print('=' * 65)
    print(f'\nBest model: {comparison_df["Accuracy (%)"].idxmax()}')
    print('\nDone. All figures saved to ./figures/')


if __name__ == '__main__':
    main()
