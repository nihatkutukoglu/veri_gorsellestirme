import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Netflix TV Shows and Movies.csv")

print(df.head())  # İlk birkaç satırı görüntüle
print(df.info())  # Veri seti hakkında genel bilgi al
print(df.describe())  # İstatistiksel özet
print(df.isnull().sum())  # Eksik değerleri kontrol et
print(df.dropna(subset=["imdb_score", "imdb_votes"], inplace=True))
print(df.dtypes)  # Veri tiplerini kontrol et
print("--- Temizlenmiş Veri Özeti ---")
print(df.head())
print(f"\nAnaliz Edilecek Toplam Kayıt Sayısı: {len(df)}")
print(f"Analiz Edilecek Toplam Sütun Sayısı: {len(df.columns)}")
print(f"Toplam Eksik Değer Sayısı: {df.isnull().sum().sum()}")
print(f"Toplam Tekrarlanan Kayıt Sayısı: {df.duplicated().sum()}")
print(f"Toplam Benzersiz Kayıt Sayısı: {df.nunique().sum()}")


# 2. İstatistiksel Analiz (NumPy & Pandas)
# En Yüksek Puanlı Yapımlar
min_votes = 5000
top_rated_movies = (
    df[df["imdb_votes"] >= min_votes]
    .sort_values(by="imdb_score", ascending=False)
    .head(10)
)

top_rated_shows = (
    df[df["imdb_votes"] >= min_votes]
    .sort_values(by="imdb_score", ascending=False)
    .head(10)
)
print("\n--- En İyi 10 Yüksek Oylu Yapım ---")
print(top_rated_movies[["title", "imdb_score", "imdb_votes", "type"]])


# İçerik Türü Dağılımı
type_counts = df["type"].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=type_counts.index, y=type_counts.values)
plt.title("İçerik Türü Dağılımı")
plt.xlabel("İçerik Türü")
plt.ylabel("Sayı")
plt.show()

# -- IMDb Puanı Dağılımı ---
plt.figure(figsize=(10, 6))
sns.histplot(df["imdb_score"], bins=20, kde=True, color="skyblue", edgecolor="black")
plt.title("IMDb Puanı Dağılımı")
plt.xlabel("IMDb Puanı")
plt.ylabel("Adet")
plt.show()


# --- Yıllara Göre İçerik Yayınlama Eğilimi ---
plt.figure(figsize=(12, 7))
yearly_counts = df["release_year"].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker="o")
plt.title("Yıllara Göre İçerik Yayınlama Eğilimi")
plt.xlabel("Yıl")
plt.ylabel("Yayınlanan İçerik Sayısı")
# plt.xticks(ticks=yearly_counts.index, rotation=45) # Tüm yılları göstermek için yorum satırını kaldırabilirsiniz
plt.show()


# --- Filmlerin Süre Dağılımı ---
plt.figure(figsize=(12, 7))
sns.histplot(df["runtime"], bins=30, kde=True, color="darkgreen", edgecolor="black")
plt.title("Filmlerin Süre Dağılımı")
plt.xlabel("Süre (dakika)")
plt.ylabel("Adet")
median_runtime = df["runtime"].median()
plt.axvline(  # Ortanca süreyi gösteren dikey çizgi
    median_runtime,
    color="red",
    linestyle="--",
    linewidth=1,
    label=f"Ortanca Süre: {median_runtime:.0f} dk",
)
plt.legend()
plt.show()

# --- Film Süresi ve IMDb Puanı İlişkisi (Scatter Plot) ---
plt.figure(figsize=(12, 7))

sns.scatterplot(
    x="runtime",
    y="imdb_score",
    data=df,
    color="darkorange",
)

plt.title("Film Süresi ve IMDb Puanı İlişkisi")
plt.xlabel("Süre (dakika)")
plt.ylabel("IMDb Puanı")
plt.xlim(0, 300)  # Süreyi 300 dakika ile sınırla
plt.ylim(0, 10)  # IMDb puanını 10 ile sınırla
plt.show()

print("\nAnaliz tamamlandı.")
