print("=" * 60)
print("  GRID SEARCH ÖNCESİ vs SONRASI — CV Bazlı Karşılaştırma")
print("=" * 60)
 
model_names_gs = list(gs_df.index)
x     = np.arange(len(model_names_gs))
width = 0.35
 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Grid Search Öncesi vs Sonrası — CV Bazlı Karşılaştırma",
             fontsize=14, fontweight="bold")
 
# CV F1 Karşılaştırması
ax = axes[0]
onceki_f1  = [results[m]["CV F1"]         for m in model_names_gs]
sonraki_f1 = [gs_results[m]["CV F1 (GS)"] for m in model_names_gs]
 
bars1 = ax.bar(x - width/2, onceki_f1,  width, label="GS Öncesi CV F1",  color="#90CAF9", edgecolor="white")
bars2 = ax.bar(x + width/2, sonraki_f1, width, label="GS Sonrası CV F1", color="#1565C0", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(model_names_gs, rotation=25, ha="right", fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_title("CV F1 Karşılaştırması", fontsize=12, fontweight="bold")
ax.set_ylabel("CV F1 Skoru")
ax.legend()
ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.4)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
 
# CV Accuracy Karşılaştırması
ax = axes[1]
onceki_acc  = [results[m]["CV Accuracy"]    for m in model_names_gs]
sonraki_acc = [gs_results[m]["CV Accuracy"] for m in model_names_gs]
 
bars3 = ax.bar(x - width/2, onceki_acc,  width, label="GS Öncesi CV Acc",  color="#A5D6A7", edgecolor="white")
bars4 = ax.bar(x + width/2, sonraki_acc, width, label="GS Sonrası CV Acc", color="#2E7D32", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(model_names_gs, rotation=25, ha="right", fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_title("CV Accuracy Karşılaştırması", fontsize=12, fontweight="bold")
ax.set_ylabel("CV Accuracy")
ax.legend()
ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.4)
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
 
plt.tight_layout()
plt.show()
