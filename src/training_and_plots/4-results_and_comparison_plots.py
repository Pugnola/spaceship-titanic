gs_df = pd.DataFrame(gs_results).T
gs_df = gs_df.sort_values("CV F1 (GS)", ascending=False)

print("=" * 60)
print("📊 SONUÇ TABLOSU — GS Sonrası CV Bazlı (CV F1'e göre sıralı)")
print("=" * 60)
print(gs_df[["CV F1 (GS)", "CV Accuracy", "CV Precision", "CV Recall"]].to_string())

print("\n📌 EN İYİ PARAMETRELER:")
print("-" * 60)
for name in gs_df.index:
    print(f"\n🔹 {name}")
    for k, v in gs_results[name]["En İyi Parametreler"].items():
        print(f"   {k:25s}: {v}")

# ── Grafikler ─────────────────────────────────────────────────
model_names_gs = list(gs_df.index)
colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4","#FF5722"]

cv_f1_vals  = [gs_results[m]["CV F1 (GS)"]  for m in model_names_gs]
cv_acc_vals = [gs_results[m]["CV Accuracy"]  for m in model_names_gs]
cv_pre_vals = [gs_results[m]["CV Precision"] for m in model_names_gs]
cv_rec_vals = [gs_results[m]["CV Recall"]    for m in model_names_gs]
cv_acc_stds = [gs_results[m]["CV Acc (std)"] for m in model_names_gs]

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle("🚀 Spaceship Titanic — Grid Search Sonrası CV Bazlı Karşılaştırma",
             fontsize=16, fontweight="bold", y=1.01)

# Sol üst — CV Accuracy
ax = axes[0, 0]
bars = ax.barh(model_names_gs, cv_acc_vals, color=colors, edgecolor="white", height=0.6)
ax.set_xlim(0.5, 1.0)
ax.set_title("CV Accuracy (GS Sonrası)", fontsize=13, fontweight="bold")
ax.set_xlabel("Skor")
ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="0.80 referans")
for bar, val in zip(bars, cv_acc_vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.legend(fontsize=9)

# Sağ üst — CV F1
ax = axes[0, 1]
bars = ax.barh(model_names_gs, cv_f1_vals, color=colors, edgecolor="white", height=0.6)
ax.set_xlim(0.5, 1.0)
ax.set_title("CV F1 Skoru (GS Sonrası)", fontsize=13, fontweight="bold")
ax.set_xlabel("Skor")
ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="0.80 referans")
for bar, val in zip(bars, cv_f1_vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.legend(fontsize=9)

# Sol alt — CV 4 Metrik grouped bar
ax = axes[1, 0]
x     = np.arange(len(model_names_gs))
width = 0.2
metric_colors = ["#2196F3","#4CAF50","#FF9800","#E91E63"]

ax.bar(x - width*1.5, cv_acc_vals, width, label="CV Accuracy",  color=metric_colors[0], edgecolor="white", alpha=0.85)
ax.bar(x - width*0.5, cv_pre_vals, width, label="CV Precision", color=metric_colors[1], edgecolor="white", alpha=0.85)
ax.bar(x + width*0.5, cv_rec_vals, width, label="CV Recall",    color=metric_colors[2], edgecolor="white", alpha=0.85)
ax.bar(x + width*1.5, cv_f1_vals,  width, label="CV F1",        color=metric_colors[3], edgecolor="white", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(model_names_gs, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0.5, 1.05)
ax.set_title("CV Tüm Metrikler Karşılaştırması", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.4)

# Sağ alt — CV Accuracy ± std
ax = axes[1, 1]
bars = ax.barh(model_names_gs, cv_acc_vals, xerr=cv_acc_stds,
               color=colors, edgecolor="white", height=0.6,
               error_kw=dict(ecolor="black", capsize=4, linewidth=1.5))
ax.set_xlim(0.5, 1.0)
ax.set_title("CV Accuracy — 5-Fold (mean ± std)", fontsize=13, fontweight="bold")
ax.set_xlabel("CV Accuracy")
ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5)
for bar, mean, std in zip(bars, cv_acc_vals, cv_acc_stds):
    ax.text(mean + std + 0.005, bar.get_y() + bar.get_height()/2,
            f"{mean:.4f}±{std:.4f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()