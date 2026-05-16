print("=" * 60)
print("  FEATURE IMPORTANCE — Random Forest (GS Sonrası)")
print("=" * 60)
 
rf_best_model = gs_best_models["Random Forest"][0]
feature_names = X_train.columns.tolist()
importances   = pd.Series(rf_best_model.feature_importances_, index=feature_names)
importances   = importances.sort_values(ascending=True)
 
print("\n📊 Özellik Önem Sıralaması (Yüksekten Düşüğe):")
print("-" * 40)
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"   {feat:20s}: {imp:.4f}")
 
fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
importances.plot(kind="barh", color=colors_fi, ax=ax)
ax.set_title("🌳 Random Forest (GS Sonrası) — Feature Importance",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Önem Skoru")
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)
 
plt.tight_layout()
plt.show()
 
print("\n" + "=" * 60)
print("✅ Tüm analizler tamamlandı!")
print("=" * 60)
print(f"\n🏆 En İyi Model : {gs_df.index[0]}")
print(f"   CV F1        : {gs_results[gs_df.index[0]]['CV F1 (GS)']:.4f}")
print(f"   CV Accuracy  : {gs_results[gs_df.index[0]]['CV Accuracy']:.4f}")
print(f"   CV Precision : {gs_results[gs_df.index[0]]['CV Precision']:.4f}")
print(f"   CV Recall    : {gs_results[gs_df.index[0]]['CV Recall']:.4f}")