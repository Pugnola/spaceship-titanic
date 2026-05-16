import pandas as pd

df = pd.read_csv('../data/cleaned/train_cleaned.csv')

# 1. True→1, False→0 DÖNÜŞÜMÜ (önce yapılır)
transported_true = df['Transported'].sum()      # True sayısı (dönüşüm öncesi)
transported_false = (~df['Transported']).sum()  # False sayısı
total = len(df)

df['Transported'] = df['Transported'].astype(int)
df.to_csv('../data/cleaned/train_cleaned.csv', index=False)
print("Dönüşüm tamamlandı: True→1, False→0. Dosya kaydedildi.")

# 2. ANALIZ (dönüşüm sonrası)
print("\n=== IŞINLANMA ANALİZİ ===")
print(f"Toplam yolcu sayısı      : {total}")
print(f"Işınlananlar (1)         : {transported_true}")
print(f"Işınlanamayanlar (0)     : {transported_false}")
print(f"\nValue counts:\n{df['Transported'].value_counts()}")

# 3. VERİ SETİ TANIMI
print("\n=== VERİ SETİ ÖZELLİKLERİ ===")
print(f"Boyut: {df.shape}")
print(f"\nKolon tipleri:\n{df.dtypes}")
print(f"\nİstatistiksel özet:\n{df.describe()}")
print(f"\nİlk 5 satır:\n{df.head()}")
