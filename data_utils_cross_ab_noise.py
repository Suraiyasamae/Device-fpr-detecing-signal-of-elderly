import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple


def add_gaussian_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:  # แก้ไขค่า default เป็น 0.3
    """เพิ่ม Gaussian noise ให้กับข้อมูล

    Args:
        data: ข้อมูลที่ต้องการเพิ่ม noise
        noise_level: ระดับของ noise (0.0 ถึง 1.0) โดย:
            - 0.1 = noise 10%
            - 0.2 = noise 20%
            - 0.5 = noise 50%
            เป็นต้น

    Returns:
        ข้อมูลที่เพิ่ม noise แล้ว
    """
    # คำนวณ standard deviation ตามระดับ noise ที่ต้องการ
    data_std = np.std(data)
    noise_std = data_std * noise_level

    # สร้าง noise
    noise = np.random.normal(0, noise_std, data.shape)

    print(f"\nApplying {noise_level * 100}% Gaussian noise to data...")
    print(f"Original data std: {data_std:.4f}")
    print(f"Noise std: {noise_std:.4f}")

    return data + noise


def get_class_name(feature_value: int) -> str:
    """แปลงค่า feature เป็นชื่อคลาส"""
    class_names = {
        0: "Non-request",
        1: "Both hands",
        2: "Left hand",
        3: "Right hand"
    }
    return class_names.get(feature_value, "Unknown")


def load_and_separate_data():
    """โหลดและแยกข้อมูลปกติและไม่ปกติ"""
    BASE_DATA_PATH = r"C:\Model101\.venv\Scripts\separated_data"

    normal_data = []
    normal_labels = []
    abnormal_data = []
    abnormal_labels = []

    print("Loading and separating normal and abnormal data...")

    filename_to_class = {
        'Non_request': 0,
        'Both_hand': 1,
        'Left_hand': 2,
        'Right_hand': 3
    }

    for filename in os.listdir(BASE_DATA_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(BASE_DATA_PATH, filename)
            is_abnormal = filename.startswith('abnormal_')

            for class_name, class_idx in filename_to_class.items():
                if class_name in filename:
                    data = pd.read_csv(file_path)
                    features = data.iloc[:, 1:-2].values
                    labels = np.full(len(features), class_idx)

                    if is_abnormal:
                        abnormal_data.append(features)
                        abnormal_labels.append(labels)
                    else:
                        normal_data.append(features)
                        normal_labels.append(labels)
                    break

    normal_data = np.vstack(normal_data)
    normal_labels = np.concatenate(normal_labels)
    abnormal_data = np.vstack(abnormal_data)
    abnormal_labels = np.concatenate(abnormal_labels)

    print_data_distribution("Initial", normal_labels, abnormal_labels)

    return normal_data, normal_labels, abnormal_data, abnormal_labels


def balance_data(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Under-sampling เพื่อทำให้แต่ละคลาสมีจำนวนเท่ากัน"""
    unique_labels = np.unique(labels)
    min_size = min(np.sum(labels == label) for label in unique_labels)

    balanced_data = []
    balanced_labels = []

    for label in unique_labels:
        idx = labels == label
        label_data = data[idx]
        if len(label_data) > min_size:
            indices = np.random.choice(len(label_data), size=min_size, replace=False)
            balanced_data.append(label_data[indices])
            balanced_labels.extend([label] * min_size)
        else:
            balanced_data.append(label_data)
            balanced_labels.extend([label] * len(label_data))

    return np.vstack(balanced_data), np.array(balanced_labels)


def prepare_windows(data: np.ndarray, labels: np.ndarray, window_size: int = 20, stride: int = 1) -> Tuple[
    np.ndarray, np.ndarray]:
    """สร้าง sliding windows จากข้อมูล"""
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        label = labels[i + window_size - 1]
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)


def create_train_test_split(X_normal, y_normal, X_abnormal, y_abnormal, test_size=0.2):
    """แบ่ง train/test โดยรักษาสัดส่วน 90:10 ระหว่างข้อมูลปกติและผิดปกติ"""

    n_normal_train = int(len(X_normal) * (1 - test_size))
    n_abnormal_train = int(len(X_abnormal) * (1 - test_size))

    n_normal_test = len(X_normal) - n_normal_train
    n_abnormal_test = len(X_abnormal) - n_abnormal_train

    print("\nSplit sizes:")
    print(f"Normal data - Train: {n_normal_train}, Test: {n_normal_test}")
    print(f"Abnormal data - Train: {n_abnormal_train}, Test: {n_abnormal_test}")

    normal_train_indices = np.random.choice(len(X_normal), size=n_normal_train, replace=False)
    abnormal_train_indices = np.random.choice(len(X_abnormal), size=n_abnormal_train, replace=False)

    normal_test_indices = np.setdiff1d(np.arange(len(X_normal)), normal_train_indices)
    abnormal_test_indices = np.setdiff1d(np.arange(len(X_abnormal)), abnormal_train_indices)

    X_train = np.concatenate([X_normal[normal_train_indices], X_abnormal[abnormal_train_indices]])
    y_train = np.concatenate([y_normal[normal_train_indices], y_abnormal[abnormal_train_indices]])

    # เพิ่ม noise เฉพาะใน test set
    X_normal_test = add_gaussian_noise(X_normal[normal_test_indices])  # ไม่ต้องระบุ noise_level
    X_abnormal_test = add_gaussian_noise(X_abnormal[abnormal_test_indices])  # ไม่ต้องระบุ noise_level

    X_test = np.concatenate([X_normal_test, X_abnormal_test])
    y_test = np.concatenate([y_normal[normal_test_indices], y_abnormal[abnormal_test_indices]])

    train_normal_ratio = len(normal_train_indices) / (len(normal_train_indices) + len(abnormal_train_indices))
    test_normal_ratio = len(normal_test_indices) / (len(normal_test_indices) + len(abnormal_test_indices))

    print(f"\nRatios:")
    print(f"Train set - Normal:Abnormal = {train_normal_ratio:.2f}:{1 - train_normal_ratio:.2f}")
    print(f"Test set - Normal:Abnormal = {test_normal_ratio:.2f}:{1 - test_normal_ratio:.2f}")

    return X_train, X_test, y_train, y_test


def print_data_distribution(stage: str, normal_labels: np.ndarray, abnormal_labels: np.ndarray):
    """แสดงการกระจายของข้อมูล"""
    print(f"\n{stage} data distribution:")
    print("\nNormal data:")
    unique_labels, counts = np.unique(normal_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples")
    print(f"Total normal samples: {len(normal_labels)}")

    print("\nAbnormal data:")
    unique_labels, counts = np.unique(abnormal_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples")
    print(f"Total abnormal samples: {len(abnormal_labels)}")


def prepare_cross_validation_data(X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5):
    """เตรียมข้อมูลสำหรับ cross validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        folds.append((X_tr, y_tr, X_val, y_val))

        print(f"\nFold {fold + 1} distribution:")
        print("Training set:")
        unique_labels, counts = np.unique(y_tr, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_tr) * 100:.2f}%)")

        print("\nValidation set:")
        unique_labels, counts = np.unique(y_val, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_val) * 100:.2f}%)")

    return folds


def prepare_train_test_data(num_augmentations: int = 1):
    """เตรียมข้อมูลสำหรับ train และ test"""
    # 1. โหลดและแยกข้อมูล
    normal_data, normal_labels, abnormal_data, abnormal_labels = load_and_separate_data()

    # 2. Balance ข้อมูลแต่ละประเภท
    print("\nBalancing data...")
    normal_data_balanced, normal_labels_balanced = balance_data(normal_data, normal_labels)
    abnormal_data_balanced, abnormal_labels_balanced = balance_data(abnormal_data, abnormal_labels)
    print_data_distribution("After balancing", normal_labels_balanced, abnormal_labels_balanced)

    # 3. สร้าง windows
    X_normal, y_normal = prepare_windows(normal_data_balanced, normal_labels_balanced)
    X_abnormal, y_abnormal = prepare_windows(abnormal_data_balanced, abnormal_labels_balanced)

    # 4. แบ่ง train/test (noise จะถูกเพิ่มเฉพาะใน test set)
    X_train, X_test, y_train, y_test = create_train_test_split(X_normal, y_normal, X_abnormal, y_abnormal)

    # แปลงเป็น tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # แสดงการกระจายข้อมูลสุดท้าย
    print("\nFinal data distribution:")
    print("\nTraining set:")
    unique_labels, counts = np.unique(y_train.numpy(), return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_train) * 100:.2f}%)")

    print("\nTest set (with 10% Gaussian noise):")
    unique_labels, counts = np.unique(y_test.numpy(), return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_test) * 100:.2f}%)")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_train_test_data()

    # สร้าง folds สำหรับ cross validation จาก training set
    cv_folds = prepare_cross_validation_data(X_train, y_train)

    print("\nPreprocessing completed successfully!")
    print("\n--------------------------------------------------------")