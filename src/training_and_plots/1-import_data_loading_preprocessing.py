import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)
 
print("✅ Tüm kütüphaneler başarıyla yüklendi!\n")
 
# ── Veri Yükleme ──────────────────────────────────────────────
TRAIN_PATH = "/content/makine öğrenmesi projem/veri setlerim/processed/train_cleaned.csv"
TEST_PATH  = "/content/makine öğrenmesi projem/veri setlerim/processed/test_cleaned.csv"
 
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
 
print(f"📦 Train boyutu : {train.shape}")
print(f"📦 Test  boyutu : {test.shape}")
print(f"\n🎯 Hedef dağılımı:\n{train['Transported'].value_counts()}\n")
 
# ── Ön İşleme ─────────────────────────────────────────────────
def preprocess(df, is_train=True):
    df = df.copy()
    df.drop(columns=["PassengerId", "Name"], errors="ignore", inplace=True)
 
    cabin_split      = df["Cabin"].str.split("/", expand=True)
    df["Cabin_Deck"] = cabin_split[0]
    df["Cabin_Side"] = cabin_split[2]
    df.drop(columns=["Cabin"], inplace=True)
 
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
 
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
 
    if is_train:
        X = df.drop(columns=["Transported"])
        y = df["Transported"]
        return X, y
    else:
        return df, None
 
X_train, y_train = preprocess(train, is_train=True)
X_test,  _       = preprocess(test,  is_train=False)
 
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
 
print(f"✅ Ön işleme tamamlandı!")
print(f"   Özellik sayısı : {X_train.shape[1]}")
print(f"   Train örnek    : {X_train.shape[0]}")
print(f"   Test  örnek    : {X_test.shape[0]}")