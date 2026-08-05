"""Keşifsel veri analizi: NaN kontrolü, istatistikler, tarih sürekliliği, fiyat grafiği."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Ekransız (backend) kayıt için; notebook'ta %matplotlib inline geçersiz kılar.
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def explore(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Veriyi inceler ve gerekirse forward-fill ile NaN'leri doldurur.

    - NaN sayısını yazdırır, varsa forward fill uygular ve düzeltilen satır sayısını raporlar.
    - Temel istatistikleri ve tarih sürekliliği (eksik iş günü) bilgisini yazdırır.
    - `Close` fiyatının tüm zaman grafiğini results/price_history.png olarak kaydeder.
    Temizlenmiş DataFrame'i döndürür.
    """
    print("[explore] --- Temel istatistikler ---")
    print(df.describe())

    # NaN kontrolü ve forward fill.
    nan_before = int(df.isna().sum().sum())
    print(f"[explore] Toplam NaN sayisi (fill oncesi): {nan_before}")
    if nan_before > 0:
        # ffill: her sütunda bir önceki geçerli değeri ileri taşır (zaman serisinde uygundur).
        df = df.ffill()
        # ffill'den sonra hâlâ NaN kalırsa (baştaki satırlar) bfill ile tamamla.
        df = df.bfill()
        nan_after = int(df.isna().sum().sum())
        fixed = nan_before - nan_after
        print(f"[explore] Forward fill uygulandi. Duzeltilen NaN sayisi: {fixed}")
    else:
        print("[explore] NaN yok, fill uygulanmadi.")

    # Tarih sürekliliği: iş günü (B) takvimine göre eksik günleri say.
    full_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
    missing_bdays = len(full_range) - len(df.index.intersection(full_range))
    print(f"[explore] Is gunu takviminde eksik gun sayisi (tatiller dahil): {missing_bdays}")
    print(f"[explore] (Bu gunler borsa tatilleri olabilir; seri kronolojik ve monoton mu: "
          f"{df.index.is_monotonic_increasing})")

    # Fiyat grafiği.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df[price_col], color="tab:blue", linewidth=1)
    ax.set_title("AMZN Close Fiyati (Tum Zaman)")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = RESULTS_DIR / "price_history.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[explore] Fiyat grafigi kaydedildi: {out_path.name}")

    return df


if __name__ == "__main__":
    from data_loader import load_stock_data

    explore(load_stock_data())
