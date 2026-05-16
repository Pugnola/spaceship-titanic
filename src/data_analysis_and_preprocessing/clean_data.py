import pandas as pd

def clean_data():
    # Veri setlerinin dosya yolları
    train_path = '../data/raw/train.csv'
    test_path = '../data/raw/test.csv'

    # Verileri okuma
    print("Veriler okunuyor...")
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı - {e}")
        return

    # Temizleme öncesi boyutlar
    print(f"\nTemizleme öncesi boyutlar:")
    print(f"Train veri seti: {train_df.shape}")
    print(f"Test veri seti: {test_df.shape}")

    # Boş (NaN) değer içeren satırları temizleme (dropna)
    train_df_cleaned = train_df.dropna()
    test_df_cleaned = test_df.dropna()

    # Temizleme sonrası boyutlar
    print(f"\nTemizleme sonrası boyutlar:")
    print(f"Train veri seti: {train_df_cleaned.shape}")
    print(f"Test veri seti: {test_df_cleaned.shape}")

    # Temizlenmiş verileri yeni bir dosyaya (veya aynı dosyaya) kaydetme
    # Orijinal dosyaların üzerine yazmamak için yeni isimlerle kaydediyoruz.
    print("\nTemizlenmiş veriler 'train_cleaned.csv' ve 'test_cleaned.csv' olarak kaydediliyor...")
    train_df_cleaned.to_csv('../data/cleaned/train_cleaned.csv', index=False)
    test_df_cleaned.to_csv('../data/cleaned/test_cleaned.csv', index=False)
    
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    clean_data()
