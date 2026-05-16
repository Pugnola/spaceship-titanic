import pandas as pd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

def analiz_et(dosya_adi, baslik):
    dosya_yolu = os.path.join(script_dir, '../../data/cleaned', dosya_adi)
    df = pd.read_csv(dosya_yolu)

    print("=" * 45)
    print(f"   VERİ SETİ ANALİZİ - {baslik}")
    print("=" * 45)

    print(f"\nVeri seti boyutu       : {df.shape[0]} satır, {df.shape[1]} kolon")
    print(f"Toplam yolcu sayısı    : {len(df)}")

    if 'Transported' in df.columns:
        isinlananlar = (df['Transported'] == 1).sum()
        isinlanamayanlar = (df['Transported'] == 0).sum()
        print(f"\nIşınlananlar (1)       : {isinlananlar}")
        print(f"Işınlanamayanlar (0)   : {isinlanamayanlar}")
        print(f"\nIşınlanma oranı        : %{isinlananlar / len(df) * 100:.2f}")
        print(f"Işınlanamama oranı     : %{isinlanamayanlar / len(df) * 100:.2f}")
    else:
        print("\n(Transported kolonu bu veri setinde yok)")

    print("\n" + "-" * 45)
    print("Kolon listesi:")
    for col in df.columns:
        print(f"  - {col} ({df[col].dtype})")

    dosya_boyutu = os.path.getsize(dosya_yolu) / 1024
    print(f"\nDosya boyutu           : {dosya_boyutu:.2f} KB")
    print("=" * 45)
    print()

analiz_et('train_cleaned.csv', 'TRAIN CLEANED')
analiz_et('test_cleaned.csv', 'TEST CLEANED')