print("=" * 60)
print("  GRID SEARCH — HİPERPARAMETRE OPTİMİZASYONU")
print("=" * 60)
print("⏳ Bu işlem 5-10 dakika sürebilir...\n")
 
param_grids = {
    "Logistic Regression": {
        "model" : LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
        "scaled": True
    },
    "Decision Tree": {
        "model" : DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]},
        "scaled": False
    },
    "KNN": {
        "model" : KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
        "scaled": True
    },
    "Naive Bayes": {
        "model" : GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        "scaled": True
    },
    "SVM": {
        "model" : SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        "scaled": True
    },
    "Random Forest": {
        "model" : RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
        "scaled": False
    },
    "AdaBoost": {
        "model" : AdaBoostClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
        "scaled": False
    },
}
 
gs_results     = {}
gs_best_models = {}
 
for name, config in param_grids.items():
    print(f"🔍 {name} aranıyor...")
    X = X_train_sc if config["scaled"] else X_train.values
 
    gs = GridSearchCV(config["model"], config["params"], cv=5, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X, y_train)
 
    best = gs.best_estimator_
    gs_best_models[name] = (best, config["scaled"])
 
    # GS sonrası tüm metrikler CV ile hesaplanıyor
    cv_acc  = cross_val_score(best, X, y_train, cv=5, scoring="accuracy")
    cv_prec = cross_val_score(best, X, y_train, cv=5, scoring="precision")
    cv_rec  = cross_val_score(best, X, y_train, cv=5, scoring="recall")
    cv_f1   = cross_val_score(best, X, y_train, cv=5, scoring="f1")
 
    gs_results[name] = {
        "En İyi Parametreler": gs.best_params_,
        "CV F1 (GS)"         : round(cv_f1.mean(),  4),
        "CV Accuracy"        : round(cv_acc.mean(),  4),
        "CV Precision"       : round(cv_prec.mean(), 4),
        "CV Recall"          : round(cv_rec.mean(),  4),
        "CV Acc (std)"       : round(cv_acc.std(),   4),
        "CV F1 (std)"        : round(cv_f1.std(),    4),
        "CV Pre (std)"       : round(cv_prec.std(),  4),
        "CV Rec (std)"       : round(cv_rec.std(),   4),
    }
 
    print(f"   ── CV Sonuçları ────────────────────────────────")
    print(f"   CV F1        : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"   CV Accuracy  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"   CV Precision : {cv_prec.mean():.4f} ± {cv_prec.std():.4f}")
    print(f"   CV Recall    : {cv_rec.mean():.4f} ± {cv_rec.std():.4f}")
    print(f"   📌 Parametreler: {gs.best_params_}\n")