import pandas as pd
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def csv_to_dataframe(file_path):
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path, sep=";")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

file_path = "Scripts/student_mat.csv"
# data = csv_to_dataframe(file_path)
# print(data.head())




def preprocess_data(csv_data, target_column):
    """Split features and target, encode categorical features,
    and scale numeric features.
    """

    if target_column not in csv_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in CSV.\n"
            f"Available columns: {list(csv_data.columns)}"
        )

    # Split features and target
    X = csv_data.drop(columns=[target_column])
    y = csv_data[target_column]

    # Find numeric and categorical columns
    numeric_features = X.select_dtypes(
        include="number"
    ).columns

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns

    # Preprocessing:
    # - StandardScaler for numeric features
    # - OneHotEncoder for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )

    # Transform the data
    X_processed = preprocessor.fit_transform(X)
    print(X_processed,y)
    return X_processed, y
    

#function run_knn do dopracowania

def run_knn(X, y, k=5):
    """Train and evaluate KNN classifier."""

    if k <= 0:
        raise ValueError("Number of neighbors (k) must be positive.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("\nKNN Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


#testing csv_to_dataframe and preprocess_data_functions
data = csv_to_dataframe(file_path)
print(data)
X_processed, y = preprocess_data(
    data,
    target_column=data.columns[-1]
)

run_knn(X_processed, y)