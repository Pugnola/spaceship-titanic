# Spaceship Titanic — Machine Learning Project

Kaggle "Spaceship Titanic" yarışması veri seti üzerinde 7 farklı klasik makine öğrenmesi modelinin karşılaştırmalı analizi. Çalışma kapsamında veri temizleme, ön işleme, çapraz doğrulama (cross-validation), grid search ile hiperparametre optimizasyonu ve sonuçların görselleştirilmesi yer alır.

---

## İçerik

- **Veri seti:** Spaceship Titanic ([Kaggle](https://www.kaggle.com/competitions/spaceship-titanic))
- **Problem:** İkili sınıflandırma — yolcunun başka bir boyuta ışınlanıp ışınlanmadığının tahmini (`Transported`)
- **Karşılaştırılan modeller:**
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - AdaBoost

---

## Proje Yapısı

```
spaceship-titanic/
├── data/
│   ├── raw/                          # Orijinal Kaggle CSV'leri (gitignore)
│   └── cleaned/                      # Temizlenmiş veri
├── src/
│   ├── data_analysis_and_preprocessing/
│   │   ├── clean_data.py             # NaN temizleme
│   │   ├── analyze_transported.py    # Hedef değişken analizi
│   │   └── dataset_analysis.py       # Veri seti özet istatistikleri
│   └── training_and_plots/
│       ├── 1-import_data_loading_preprocessing.py
│       ├── 2-model_training_and_cross_validation.py
│       ├── 3-grid_search_and_hyperparameter_optimization.py
│       ├── 4-results_and_comparison_plots.py
│       ├── 5-confusion_matrix_comparisons.py
│       ├── 6-roc_curve_and_auc_comparison.py
│       ├── 7-grid_search_before_after_comparison.py
│       └── 8-feature_importance.py
├── report_project_template_and_user_guide/
│   └── usage_commands.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Kurulum

```bash
# Sanal ortam oluştur
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Bağımlılıkları kur
pip install -r requirements.txt
```

---

## Veri

Kaggle'dan veri setini indirin: <https://www.kaggle.com/competitions/spaceship-titanic/data>

`train.csv` ve `test.csv` dosyalarını `data/raw/` klasörüne yerleştirin.

```bash
# Veriyi temizle
cd src/data_analysis_and_preprocessing
python clean_data.py
```

Bu işlem `data/cleaned/train_cleaned.csv` ve `data/cleaned/test_cleaned.csv` dosyalarını üretir.

---

## Çalıştırma

> **Not:** Bu projedeki eğitim scriptleri Google Colab hücrelerinde sırayla çalıştırılmak üzere yazılmıştır. Her dosya bir önceki hücrenin değişkenlerine bağımlıdır.

Önerilen çalıştırma sırası:

1. `1-import_data_loading_preprocessing.py` — Kütüphane importları, veri yükleme, ön işleme
2. `2-model_training_and_cross_validation.py` — 7 modelin 5-fold CV ile değerlendirilmesi
3. `3-grid_search_and_hyperparameter_optimization.py` — Grid search
4. `4-results_and_comparison_plots.py` — Karşılaştırma grafikleri
5. `5-confusion_matrix_comparisons.py` — Confusion matrix'ler
6. `6-roc_curve_and_auc_comparison.py` — ROC eğrileri ve AUC skorları
7. `7-grid_search_before_after_comparison.py` — GS öncesi/sonrası karşılaştırma
8. `8-feature_importance.py` — Random Forest özellik önemleri

---

## Yöntem

- **Veri temizleme:** `dropna()` ile NaN içeren satırlar çıkarıldı
- **Kategorik dönüşüm:** Label encoding
- **Ölçeklendirme:** `StandardScaler` (mesafe bazlı ve doğrusal modeller için)
- **Cross-validation:** 5-fold CV, `scoring="f1"`
- **Hiperparametre optimizasyonu:** `GridSearchCV` ile her model için
- **Değerlendirme:** Accuracy, Precision, Recall, F1, ROC-AUC

---
