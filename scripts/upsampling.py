# upsampling.py
import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote_upsampling(X, y, preprocessor, label_encoder=None):
    """
    Applies preprocessing + SMOTE oversampling to balance the dataset.
    Returns X_resampled, y_resampled, and prints class distribution.

    X: features before preprocessing
    y: encoded labels (0,1,2)
    preprocessor: ColumnTransformer (scaler + one-hot)
    label_encoder: (optional) to print original class names
    """

    print("\n🔄 Step 1: Applying preprocessing...")
    X_preprocessed = preprocessor.fit_transform(X)

    print("\n🔄 Step 2: Applying SMOTE upsampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

    print("\n📌 Class distribution AFTER SMOTE:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, cnt in zip(unique, counts):
        if label_encoder:
            print(f"{label_encoder.inverse_transform([cls])[0]} : {cnt}")
        else:
            print(f"{cls} : {cnt}")

    return X_resampled, y_resampled
