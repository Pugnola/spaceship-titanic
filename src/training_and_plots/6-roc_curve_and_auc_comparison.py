from sklearn.model_selection import cross_val_predict

print("=" * 60)
print("  ROC CURVE & AUC SKORLARI (Grid Search Sonrası, CV Bazlı)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))
roc_colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4","#FF5722"]
auc_scores = {}

for (name, (model, use_scaled)), color in zip(gs_best_models.items(), roc_colors):
    X = X_train_sc if use_scaled else X_train.values

    # Hücre 5 ile aynı yöntem — cross_val_predict ile CV bazlı tahmin
    y_prob_cv = cross_val_predict(model, X, y_train, cv=5, method="predict_proba")[:, 1]

    fpr, tpr, _ = roc_curve(y_train, y_prob_cv)
    auc         = roc_auc_score(y_train, y_prob_cv)
    auc_scores[name] = auc

    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
    print(f"🔹 {name:25s} → CV AUC: {auc:.4f}")

ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Rastgele Tahmin (AUC = 0.50)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — 7 Model (GS Sonrası, CV Bazlı)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.show()

print("\n📊 CV AUC SKORU SIRALAMA (Yüksekten Düşüğe):")
print("-" * 40)
for name, auc in sorted(auc_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"   {name:25s}: {auc:.4f}")