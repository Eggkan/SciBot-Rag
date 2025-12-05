import pandas as pd
import json
import os  # Dosya varlığını kontrol etmek için


def csv_to_json(csv_file_path, json_file_path):
    print(f"'{csv_file_path}' dosyasını okumaya çalışılıyor...")
    if not os.path.exists(csv_file_path):
        print(f"HATA: '{csv_file_path}' dosyası bulunamadı. Lütfen CSV dosyasının aynı dizinde olduğundan emin olun.")
        return False  # Başarısız olduğunu belirtmek için False döndür

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"HATA: CSV dosyasını okurken bir sorun oluştu: {e}")
        print("Lütfen CSV dosyasının doğru formatta olduğundan ve başlıklarının doğru olduğundan emin olun.")
        return False

    # CSV'deki sütun adlarını kendi dosyanıza göre ayarlayın
    # Örneğin: df = pd.read_csv(csv_file_path, names=['Soru', 'Girdi', 'Cevap'], header=None)
    # Eğer başlık satırı varsa ve başlıklar 'Soru', 'Girdi', 'Cevap' ise bu kısmı değiştirmeyin.

    # Kendi CSV dosyanızdaki sütun başlıklarını buraya yazın:
    QUESTION_COL = 'instruction'  # CSV'deki soru sütununun başlığı
    INPUT_COL = 'input'  # CSV'deki girdi sütununun başlığı
    ANSWER_COL = 'output'  # CSV'deki cevap sütununun başlığı

    # Sütunların CSV'de var olup olmadığını kontrol et
    missing_cols = [col for col in [QUESTION_COL, INPUT_COL, ANSWER_COL] if col not in df.columns]
    if missing_cols:
        print(f"HATA: CSV dosyasında eksik sütunlar var: {', '.join(missing_cols)}")
        print(f"Mevcut sütunlar: {', '.join(df.columns)}")
        return False

    training_data = []
    for index, row in df.iterrows():
        instruction = str(row[QUESTION_COL]).strip()
        input_text = str(row[INPUT_COL]).strip()
        output_text = str(row[ANSWER_COL]).strip()

        if instruction and input_text and output_text:  # Boş veya eksik alanları atla
            training_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })

    if not training_data:
        print(
            "UYARI: Dönüştürülecek geçerli veri bulunamadı. CSV dosyasının boş olmadığından veya sütunlarda veri olduğundan emin olun.")
        return False

    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"Veri '{json_file_path}' dosyasına başarıyla kaydedildi. Toplam {len(training_data)} örnek.")
        return True  # Başarılı olduğunu belirtmek için True döndür
    except Exception as e:
        print(f"HATA: JSON dosyasına yazarken bir sorun oluştu: {e}")
        return False


def verify_json_output(json_file_path, num_examples=3):
    """
    Oluşturulan JSON dosyasını okur ve ilk 'num_examples' kadarını ekrana yazdırır.
    """
    if not os.path.exists(json_file_path):
        print(f"Doğrulama HATA: '{json_file_path}' dosyası bulunamadı. JSON dosyası oluşturulamamış olabilir.")
        return

    print(f"\n--- '{json_file_path}' dosyasının ilk {num_examples} örneği ---")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            print("JSON dosyası boş.")
            return

        for i, example in enumerate(data[:num_examples]):
            print(f"Örnek {i + 1}:")
            print(f"  instruction: {example.get('instruction', 'N/A')}")
            print(f"  input:       {example.get('input', 'N/A')}")
            print(f"  output:      {example.get('output', 'N/A')}")
            print("-" * 20)

        if len(data) > num_examples:
            print(f"... Toplam {len(data)} örnekten {num_examples} tanesi gösterildi.")

        print("\nJSON doğrulama başarılı!")

    except json.JSONDecodeError as e:
        print(f"Doğrulama HATA: JSON dosyasını okurken bir sorun oluştu (geçersiz format): {e}")
    except Exception as e:
        print(f"Doğrulama sırasında beklenmedik bir hata oluştu: {e}")


if __name__ == "__main__":
    csv_path = 'fen_bilimleri_soru_cevap.csv'
    json_path = 'fen_bilimleri_training_data.json'

    # CSV'den JSON'a dönüştürme işlemini çalıştır
    success = csv_to_json(csv_path, json_path)

    # Eğer dönüştürme başarılıysa JSON çıktısını doğrula
    if success:
        verify_json_output(json_path, num_examples=5)  # İlk 5 örneği göster
    else:
        print("\nJSON dönüştürme işlemi başarısız olduğu için doğrulama yapılmadı.")