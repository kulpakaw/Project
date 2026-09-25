import pandas as pd
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


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