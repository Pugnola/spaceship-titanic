models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42),        True),
    "Decision Tree"      : (DecisionTreeClassifier(random_state=42),                    False),
    "KNN"                : (KNeighborsClassifier(n_neighbors=7),                        True),
    "Naive Bayes"        : (GaussianNB(),                                               True),
    "SVM"                : (SVC(kernel="rbf", probability=True, random_state=42),       True),
    "Random Forest"      : (RandomForestClassifier(n_estimators=200, random_state=42),  False),
    "AdaBoost"           : (AdaBoostClassifier(n_estimators=100, random_state=42),      False),
}
 
results = {}
 
print("=" * 60)
print("  MODEL EĞİTİMİ & DEĞERLENDİRME (5-Fold Cross Validation)")
print("=" * 60)
 
for name, (model, use_scaled) in models.items():
    X = X_train_sc if use_scaled else X_train.values
 
    # Normal eğitim skorları (sadece referans)
    model.fit(X, y_train)
    y_pred = model.predict(X)
 
    acc  = accuracy_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred)
    rec  = recall_score(y_train, y_pred)
    f1   = f1_score(y_train, y_pred)
 
    # CV skorları — 4 metriğin hepsi (gerçek değerler)
    cv_acc  = cross_val_score(model, X, y_train, cv=5, scoring="accuracy")
    cv_prec = cross_val_score(model, X, y_train, cv=5, scoring="precision")
    cv_rec  = cross_val_score(model, X, y_train, cv=5, scoring="recall")
    cv_f1   = cross_val_score(model, X, y_train, cv=5, scoring="f1")
 
    results[name] = {
        # Normal skorlar (sadece referans, baz alınmaz)
        "Accuracy"    : acc,
        "Precision"   : prec,
        "Recall"      : rec,
        "F1-Score"    : f1,
        # CV skorlar (gerçek değerler — baz alınan)
        "CV Accuracy" : cv_acc.mean(),
        "CV Precision": cv_prec.mean(),
        "CV Recall"   : cv_rec.mean(),
        "CV F1"       : cv_f1.mean(),
        "CV Acc (std)": cv_acc.std(),
        "CV Pre (std)": cv_prec.std(),
        "CV Rec (std)": cv_rec.std(),
        "CV F1 (std)" : cv_f1.std(),
    }
 
    print(f"\n🔹 {name}")
    print(f"   Accuracy  : {acc:.4f}  |  Precision: {prec:.4f}")
    print(f"   Recall    : {rec:.4f}  |  F1-Score : {f1:.4f}")
    print(f"   ── Cross Validation (5-Fold) ──────────────────")
    print(f"   CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"   CV Precision : {cv_prec.mean():.4f} ± {cv_prec.std():.4f}")
    print(f"   CV Recall    : {cv_rec.mean():.4f} ± {cv_rec.std():.4f}")
    print(f"   CV F1        : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
 
print("\n" + "=" * 60)