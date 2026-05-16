from sklearn.model_selection import cross_val_predict

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Confusion Matrix — 7 Model (Grid Search Sonrası, CV Bazlı)",
             fontsize=15, fontweight="bold")

axes_flat = axes.flatten()

for i, name in enumerate(gs_df.index):
    model, use_scaled = gs_best_models[name]
    X = X_train_sc if use_scaled else X_train.values

    # CV bazlı tahmin — model hiç görmediği veriyle test ediliyor
    y_pred_cv = cross_val_predict(model, X, y_train, cv=5)

    cm   = confusion_matrix(y_train, y_pred_cv)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Trans.", "Transported"])
    disp.plot(ax=axes_flat[i], colorbar=False, cmap="Blues")

    cv_f1  = gs_results[name]["CV F1 (GS)"]
    cv_acc = gs_results[name]["CV Accuracy"]
    axes_flat[i].set_title(f"{name}\nCV F1: {cv_f1:.4f} | CV Acc: {cv_acc:.4f}",
                           fontsize=10, fontweight="bold")

axes_flat[-1].set_visible(False)
plt.tight_layout()
plt.show()

# Classification Report — En iyi model (CV bazlı)
best_gs_name              = gs_df.index[0]
best_gs_model, best_gs_sc = gs_best_models[best_gs_name]
X_best                    = X_train_sc if best_gs_sc else X_train.values

# CV bazlı tahmin
y_pred_cv_best = cross_val_predict(best_gs_model, X_best, y_train, cv=5)

print(f"\n📋 CLASSIFICATION REPORT — {best_gs_name} (CV Bazlı):\n")
print(classification_report(y_train, y_pred_cv_best,
                             target_names=["Not Transported", "Transported"]))