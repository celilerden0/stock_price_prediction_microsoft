"""AMZN günlük OHLCV verisini yfinance ile indirip diske önbellekler.

Kısıt: Modele sadece `Close` girecek olsa da CSV'ye tüm OHLCV kaydedilir.
Dosya zaten varsa tekrar indirilmez, diskten okunur.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Proje kök dizini (src/ -> parent) ve data klasörü.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TICKER = "AMZN"
START_DATE = "2010-01-01"


def load_stock_data(ticker: str = TICKER,
                    start: str = START_DATE,
                    end: str | None = None,
                    csv_path: Path | None = None) -> pd.DataFrame:
    """AMZN OHLCV verisini döndürür.

    `csv_path` varsa diskten okur; yoksa yfinance ile indirip kaydeder.
    yfinance boş dönerse anlamlı bir RuntimeError fırlatır.
    Döndürülen DataFrame'in indeksi tarih (DatetimeIndex), sütunlar OHLCV.
    """
    if csv_path is None:
        csv_path = DATA_DIR / f"{ticker}.csv"
    csv_path = Path(csv_path)

    if csv_path.exists():
        # Diskte varsa yeniden indirme; ilk sütun (Date) indeks olarak okunur.
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"[data_loader] '{csv_path.name}' diskten okundu (indirme atlandi).")
    else:
        # yfinance import'u sadece indirme gerektiğinde yapılır (opsiyonel bağımlılık).
        import yfinance as yf

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"[data_loader] yfinance ile {ticker} indiriliyor: {start} -> {end}")
        # auto_adjust=True: split/temettü düzeltmeli fiyatlar. Tek ticker -> düz sütunlar.
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)

        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance '{ticker}' icin bos veri dondurdu. "
                "Internet baglantisini, ticker sembolunu ve tarih araligini kontrol edin."
            )

        # yfinance bazen MultiIndex sütun döndürür (('Close','AMZN')); düzleştir.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df.to_csv(csv_path)
        print(f"[data_loader] '{csv_path.name}' kaydedildi.")

    if df.empty:
        raise RuntimeError(f"Veri bos: {csv_path}")

    # Satır sayısı ve tarih aralığını raporla.
    print(f"[data_loader] Satir sayisi: {len(df)}")
    print(f"[data_loader] Tarih araligi: {df.index.min().date()} -> {df.index.max().date()}")
    return df


if __name__ == "__main__":
    load_stock_data()
