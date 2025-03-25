import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from typing import Dict, List, Tuple

# กำหนดพาทของโฟลเดอร์ที่เก็บข้อมูล
BASE_DATA_PATH = r"C:\model101\.venv\Scripts\separated_data"


def get_class_name(feature_value: int) -> str:
    """
    แปลงค่า feature เป็นชื่อคลาส
    """
    class_names = {
        0: "Non-request",
        1: "Both hands",
        2: "Left hand",
        3: "Right hand"
    }
    return class_names.get(feature_value, "Unknown")


def load_data_from_files() -> Tuple[np.ndarray, np.ndarray]:
    """
    โหลดข้อมูลจากทุกไฟล์และรวมเข้าด้วยกัน โดยใช้คอลัมน์ feature เป็น label

    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, labels)
    """
    all_features = []
    all_labels = []

    print("Loading data from files...")
    for filename in os.listdir(BASE_DATA_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(BASE_DATA_PATH, filename)
            print(f"Processing {filename}...")

            # อ่านข้อมูลจากไฟล์
            data = pd.read_csv(file_path)

            # ดึงข้อมูล features (ทุกคอลัมน์ยกเว้น 2 คอลัมน์สุดท้าย)
            feature_cols = data.iloc[:, 1:-2].values

            # ดึง feature column (คอลัมน์สุดท้ายที่สอง) เป็น label
            labels = data.iloc[:, -2].values

            all_features.append(feature_cols)
            all_labels.append(labels)

    # รวมข้อมูลทั้งหมด
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def balance_classes(X: torch.Tensor, y: torch.Tensor, target_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ทำ random under-sampling บน majority class
    """
    X_np = X.numpy()
    y_np = y.numpy()

    # นับจำนวนข้อมูลในแต่ละคลาส
    unique_classes, class_counts = np.unique(y_np, return_counts=True)
    print("\nOriginal class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls} ({get_class_name(cls)}): {count} samples")

    # ถ้าไม่ระบุ target_size ให้ใช้ขนาดของคลาสที่ใหญ่เป็นอันดับ 2
    if target_size is None:
        sorted_counts = np.sort(class_counts)
        target_size = int(sorted_counts[-2])  # ใช้ขนาดของคลาสที่ใหญ่เป็นอันดับ 2

    # ทำ under-sampling สำหรับแต่ละคลาส
    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_np == cls)[0]
        if len(cls_indices) > target_size:
            # สุ่มเลือกข้อมูล
            selected_indices = np.random.choice(cls_indices, size=target_size, replace=False)
            balanced_indices.extend(selected_indices)
        else:
            # ใช้ข้อมูลทั้งหมดถ้ามีน้อยกว่า target_size
            balanced_indices.extend(cls_indices)

    # สุ่มสลับลำดับข้อมูล
    np.random.shuffle(balanced_indices)

    # เลือกข้อมูลตาม indices
    X_balanced = X_np[balanced_indices]
    y_balanced = y_np[balanced_indices]

    # แปลงกลับเป็น tensor
    X_balanced = torch.tensor(X_balanced, dtype=torch.float32)
    y_balanced = torch.tensor(y_balanced, dtype=torch.long)

    return X_balanced, y_balanced


def prepare_data(features: np.ndarray, labels: np.ndarray, window_size: int = 20, stride: int = 1) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    เตรียมข้อมูลโดยการสร้าง sliding windows
    """
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, stride):
        window_features = features[i:i + window_size]
        window_label = labels[i + window_size - 1]
        X.append(window_features)
        y.append(window_label)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def prepare_train_test_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    เตรียมข้อมูลสำหรับการเทรนและทดสอบ

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        (X_train, X_test, y_train, y_test, class_weights)
    """
    # โหลดข้อมูลจากไฟล์
    features, labels = load_data_from_files()

    # เตรียมข้อมูลด้วย sliding windows
    X, y = prepare_data(features, labels)

    print("\nInitial data distribution:")
    unique_labels, counts = np.unique(y.numpy(), return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y) * 100:.2f}%)")

    # ทำ balancing
    print("\nBalancing dataset...")
    X_balanced, y_balanced = balance_classes(X, y)

    print("\nBalanced data distribution:")
    unique_labels, counts = np.unique(y_balanced.numpy(), return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_balanced) * 100:.2f}%)")

    # แบ่งข้อมูลตามคลาส
    X_classes = [X_balanced[y_balanced == i] for i in range(len(unique_labels))]
    y_classes = [y_balanced[y_balanced == i] for i in range(len(unique_labels))]

    # ทำ stratified split
    X_train, X_test, y_train, y_test = [], [], [], []
    for X_class, y_class in zip(X_classes, y_classes):
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )
        X_train.append(X_train_class)
        X_test.append(X_test_class)
        y_train.append(y_train_class)
        y_test.append(y_test_class)

    # รวมข้อมูล
    X_train = torch.cat(X_train)
    X_test = torch.cat(X_test)
    y_train = torch.cat(y_train)
    y_test = torch.cat(y_test)

    # คำนวณ class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_balanced.numpy()),
        y=y_balanced.numpy()
    )

    print("\n--------------------------------------------------------")
    print("\nFinal class distribution:")
    print("\nTraining set:")
    unique_labels_train, counts_train = np.unique(y_train.numpy(), return_counts=True)
    for label, count in zip(unique_labels_train, counts_train):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_train) * 100:.2f}%)")

    print("\nTest set:")
    unique_labels_test, counts_test = np.unique(y_test.numpy(), return_counts=True)
    for label, count in zip(unique_labels_test, counts_test):
        print(f"Class {label} ({get_class_name(label)}): {count} samples ({count / len(y_test) * 100:.2f}%)")

    return X_train, X_test, y_train, y_test, class_weights


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_weights = prepare_train_test_data()
    print("\nPreprocessing completed successfully!")
    print("\n--------------------------------------------------------")