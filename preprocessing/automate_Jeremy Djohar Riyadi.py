import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    # Hapus kolom ID
    df = df.drop(
        columns=["transaction_id"]
    )

    # Encoding
    le = LabelEncoder()

    df["merchant_category"] = le.fit_transform(
        df["merchant_category"]
    )

    # Scaling
    feature_cols = [
        col for col in df.columns
        if col != "is_fraud"
    ]

    scaler = StandardScaler()

    df[feature_cols] = scaler.fit_transform(
        df[feature_cols]
    )

    # Simpan hasil
    df.to_csv(
        output_file,
        index=False
    )

if __name__ == "__main__":
    preprocess_data(
        "../credit_card_fraud_raw.csv",
        "credit_card_fraud_preprocessed.csv"
    )