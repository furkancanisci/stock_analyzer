# app.py
import os
import time
from datetime import datetime

import pandas as pd
import ta
import yfinance as yf
from flask import Flask, jsonify
from flask_cors import CORS

# --- Flask Uygulaması Tanımlamaları ---
app = Flask(__name__)
CORS(
    app
)  # Tüm kökenlerden gelen isteklere izin ver (Geliştirme için iyi, üretim için kısıtlanmalı)

# --- Dosya Yolu Tanımlamaları (Loglar ve Pozisyonlar için) ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "trading_log.txt")
POSITIONS_FILE = os.path.join(LOG_DIR, "open_positions.csv")

# --- Global Değişkenler (Bellekte tutulacak pozisyonlar) ---
# DİKKAT: Bu global değişkenler, sunucu yeniden başlatıldığında sıfırlanır.
# Üretim ortamı için bir veritabanı (SQLite, PostgreSQL vb.) kullanılmalıdır.
open_positions = {}


def load_open_positions():
    """Kayıtlı açık pozisyonları yükler."""
    global open_positions
    if os.path.exists(POSITIONS_FILE):
        try:
            positions_df = pd.read_csv(POSITIONS_FILE, index_col="ticker")
            open_positions = positions_df.to_dict(orient="index")
            for ticker, data in open_positions.items():
                data["entry_time"] = datetime.fromisoformat(data["entry_time"])
            log_message(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Açık pozisyonlar yüklendi."
            )
        except Exception as e:
            log_message(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pozisyonlar yüklenirken hata oluştu: {e}"
            )
            open_positions = {}
    else:
        open_positions = {}


def save_open_positions():
    """Açık pozisyonları kaydeder."""
    global open_positions
    if open_positions:
        temp_positions = open_positions.copy()
        for ticker, data in temp_positions.items():
            data["entry_time"] = data["entry_time"].isoformat()

        positions_df = pd.DataFrame.from_dict(temp_positions, orient="index")
        positions_df.index.name = "ticker"
        positions_df.to_csv(POSITIONS_FILE)
    elif os.path.exists(POSITIONS_FILE):
        os.remove(POSITIONS_FILE)


# --- Loglama Fonksiyonu ---
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")


# --- Veri Çekme Fonksiyonu ---
def get_stock_data(ticker_symbol, period, interval):
    """
    Belirtilen hisse senedi sembolü için geçmiş verileri çeker.
    """
    log_message(f"{ticker_symbol} için {interval} veri çekiliyor...", level="DEBUG")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker_symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if not data.empty:
                return data
            else:
                log_message(
                    f"UYARI: {ticker_symbol} için {interval} veri boş geldi. {attempt + 1}. deneme...",
                    level="WARN",
                )
        except yf.YFRateLimitError as e:
            log_message(
                f"HIZ LİMİTİ! {ticker_symbol} için {interval} veri çekilemedi. ({e}) 60 saniye bekleniyor... ({attempt + 1}/{max_retries})",
                level="ERROR",
            )
            time.sleep(60)
        except Exception as e:
            log_message(
                f"HATA: {ticker_symbol} için {interval} veri çekilirken beklenmeyen hata: {e}. {attempt + 1}. deneme...",
                level="ERROR",
            )
            time.sleep(10)
    log_message(
        f"HATA: {ticker_symbol} için {interval} veri çekilemedi, tüm denemeler başarısız oldu.",
        level="CRITICAL",
    )
    return pd.DataFrame()


# --- Teknik Gösterge Ekleme Fonksiyonu ---
def add_technical_indicators(df):
    """
    DataFrame'e çeşitli teknik analiz göstergeleri ekler.
    .squeeze() kullanarak boyut uyumsuzluğu hatalarını önler.
    Ekstra sağlamlık kontrolleri eklendi.
    """
    if df.empty:
        log_message("add_technical_indicators: Gelen DataFrame boş.", level="WARN")
        return df

    try:
        # Gerekli temel sütunların varlığını kontrol et
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            missing_cols = set(required_cols) - set(df.columns)
            log_message(
                f"UYARI: Teknik göstergeler için eksik temel sütunlar: {missing_cols}",
                level="ERROR",
            )
            return pd.DataFrame()  # Eksik sütun varsa boş DataFrame dön

        # Sütunları Series olarak çek ve boyutlarını kontrol et
        # .copy() kullanarak orijinal DataFrame'in slice'ını değil, gerçek bir kopyasını al
        close_series = df["Close"].copy().squeeze()
        high_series = df["High"].copy().squeeze()
        low_series = df["Low"].copy().squeeze()
        volume_series = df["Volume"].copy().squeeze()

        # Series'lerin 1D olduğundan ve boş olmadığından emin ol
        # Minimum bar sayısını burada da kontrol edelim ki ta kütüphanesi hata vermesin
        # PSAR için 200 bar, diğerleri için 50 bar genellikle yeterlidir, en büyük olanı baz alalım.
        min_bars_for_ta = 200  # SMA_200 için en az 200 bar

        if (
            close_series.ndim != 1
            or close_series.empty
            or len(close_series) < min_bars_for_ta
        ):
            log_message(
                f"HATA: 'Close' serisi 1D değil ({close_series.ndim}), boş veya yetersiz bar ({len(close_series)}).",
                level="ERROR",
            )
            return pd.DataFrame()
        if (
            high_series.ndim != 1
            or high_series.empty
            or len(high_series) < min_bars_for_ta
        ):
            log_message(
                f"HATA: 'High' serisi 1D değil ({high_series.ndim}), boş veya yetersiz bar ({len(high_series)}).",
                level="ERROR",
            )
            return pd.DataFrame()
        if (
            low_series.ndim != 1
            or low_series.empty
            or len(low_series) < min_bars_for_ta
        ):
            log_message(
                f"HATA: 'Low' serisi 1D değil ({low_series.ndim}), boş veya yetersiz bar ({len(low_series)}).",
                level="ERROR",
            )
            return pd.DataFrame()
        if (
            volume_series.ndim != 1
            or volume_series.empty
            or len(volume_series) < min_bars_for_ta
        ):
            log_message(
                f"HATA: 'Volume' serisi 1D değil ({volume_series.ndim}), boş veya yetersiz bar ({len(volume_series)}).",
                level="ERROR",
            )
            return pd.DataFrame()

        # Teknik göstergeleri hesapla
        df["RSI"] = ta.momentum.RSIIndicator(close_series, window=14).rsi()

        macd = ta.trend.MACD(close_series)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"] = macd.macd_diff()

        df["SMA_20"] = ta.trend.SMAIndicator(close_series, window=20).sma_indicator()
        df["SMA_50"] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()
        df["SMA_200"] = ta.trend.SMAIndicator(close_series, window=200).sma_indicator()

        bollinger = ta.volatility.BollingerBands(close_series)
        df["BBL"] = bollinger.bollinger_lband()
        df["BBM"] = bollinger.bollinger_mavg()
        df["BBH"] = bollinger.bollinger_hband()

        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            close_series, volume_series
        ).on_balance_volume()
        df["PSAR"] = ta.trend.PSARIndicator(
            high_series, low_series, close_series
        ).psar()

        # NaN değerleri temizle, ancak önemli bir miktar veri kalmasını bekle
        initial_len = len(df)
        df.dropna(inplace=True)
        # Temizleme sonrası hala yetersiz bar varsa, boş DF dön
        if len(df) < 50:  # Sinyal kontrolü için en az 50 temiz bar olsun
            log_message(
                f"UYARI: add_technical_indicators sonrası yetersiz bar sayısı ({len(df)}).",
                level="ERROR",
            )
            return pd.DataFrame()

        return df
    except Exception as e:
        log_message(
            f"CRITICAL HATA: Teknik göstergeler eklenirken beklenmedik bir hata oluştu: {e}",
            level="CRITICAL",
        )
        return pd.DataFrame()  # Hata olursa boş DataFrame dön


# --- Mum Grafiği Formasyonları Yardımcı Fonksiyonlar ---
def is_bullish_engulfing(df_slice):
    if len(df_slice) < 2:
        return False
    current_candle = df_slice.iloc[-1]
    prev_candle = df_slice.iloc[-2]

    # NaN kontrolü ve .item() kullanımı
    if any(pd.isna(current_candle[col]) for col in ["Close", "Open"]) or any(
        pd.isna(prev_candle[col]) for col in ["Close", "Open"]
    ):
        return False

    prev_is_bearish = (
        prev_candle["Close"].item() < prev_candle["Open"].item()
    )  # Önceki mum düşüş mumu mu?
    current_is_bullish = (
        current_candle["Close"].item() > current_candle["Open"].item()
    )  # Şimdiki mum yükseliş mumu mu?

    engulfs_open = (
        current_candle["Open"].item() < prev_candle["Close"].item()
    )  # Şimdiki mumun açılışı önceki mumun kapanışından düşük mü?
    engulfs_close = (
        current_candle["Close"].item() > prev_candle["Open"].item()
    )  # Şimdiki mumun kapanışı önceki mumun açılışından yüksek mi?

    prev_body_size = abs(prev_candle["Open"].item() - prev_candle["Close"].item())
    current_body_size = abs(
        current_candle["Open"].item() - current_candle["Close"].item()
    )

    if prev_body_size == 0:
        return False  # Sıfır bölme hatasını önle
    body_size_condition = current_body_size > (
        prev_body_size * 1.2
    )  # Şimdiki mumun gövdesi önceki mumdan %20 daha büyük mü?

    return (
        prev_is_bearish
        and current_is_bullish
        and engulfs_open
        and engulfs_close
        and body_size_condition
    )


def is_bearish_engulfing(df_slice):
    if len(df_slice) < 2:
        return False
    current_candle = df_slice.iloc[-1]
    prev_candle = df_slice.iloc[-2]

    # NaN kontrolü ve .item() kullanımı
    if any(pd.isna(current_candle[col]) for col in ["Close", "Open"]) or any(
        pd.isna(prev_candle[col]) for col in ["Close", "Open"]
    ):
        return False

    prev_is_bullish = (
        prev_candle["Close"].item() > prev_candle["Open"].item()
    )  # Önceki mum yükseliş mumu mu?
    current_is_bearish = (
        current_candle["Close"].item() < current_candle["Open"].item()
    )  # Şimdiki mum düşüş mumu mu?

    engulfs_open = (
        current_candle["Open"].item() > prev_candle["Close"].item()
    )  # Şimdiki mumun açılışı önceki mumun kapanışından yüksek mi?
    engulfs_close = (
        current_candle["Close"].item() < prev_candle["Open"].item()
    )  # Şimdiki mumun kapanışı önceki mumun açılışından düşük mü?

    prev_body_size = abs(prev_candle["Open"].item() - prev_candle["Close"].item())
    current_body_size = abs(
        current_candle["Open"].item() - current_candle["Close"].item()
    )

    if prev_body_size == 0:
        return False  # Sıfır bölme hatasını önle
    body_size_condition = current_body_size > (
        prev_body_size * 1.2
    )  # Şimdiki mumun gövdesi önceki mumdan %20 daha büyük mü?

    return (
        prev_is_bullish
        and current_is_bearish
        and engulfs_open
        and engulfs_close
        and body_size_condition
    )


# --- Sinyal ve Pozisyon Yönetimi ---


def enter_position(ticker, entry_price, signal_type, signal_details):
    global open_positions
    if ticker not in open_positions:
        open_positions[ticker] = {
            "entry_time": datetime.now(),
            "entry_price": entry_price,
            "signal_time": datetime.now(),
            "signal_type": signal_type,
            "signal_details": signal_details,
        }
        log_message(
            f"POZISYON AÇILDI: {ticker} @ {entry_price:.2f}$ | Sinyal: {signal_type} ({signal_details})",
            level="INFO",
        )
        save_open_positions()


def exit_position(ticker, exit_price, signal_type, signal_details):
    global open_positions
    if ticker in open_positions:
        pos_data = open_positions.pop(ticker)
        entry_price = pos_data["entry_price"]

        profit_loss_usd = exit_price - entry_price
        profit_loss_pct = (
            (profit_loss_usd / entry_price) * 100 if entry_price != 0 else 0
        )

        log_message(
            f"POZISYON KAPATILDI: {ticker} @ {exit_price:.2f}$ | Kar/Zarar: {profit_loss_usd:.2f}$ ({profit_loss_pct:.2f}%) | Çıkış Sinyali: {signal_type} ({signal_details})",
            level="INFO",
        )
        save_open_positions()


# --- Sinyal Kontrol Fonksiyonu (API tarafından çağrılacak) ---
def analyze_stock_for_api(ticker):
    """
    Belirli bir hisse senedi için veri çeker, göstergeleri hesaplar ve sinyal verir.
    API tarafından çağrılmak üzere tasarlandı.
    """
    LIVE_DATA_PERIOD_1H = "60d"
    LIVE_DATA_PERIOD_1D = "5y"

    data_1h = get_stock_data(ticker, period=LIVE_DATA_PERIOD_1H, interval="1h")
    data_1d = get_stock_data(ticker, period=LIVE_DATA_PERIOD_1D, interval="1d")

    # API yanıtı için varsayılan boş bir hisse durumu
    stock_status = {
        "symbol": ticker,
        "currentPrice": 0.0,
        "openPrice": 0.0,
        "highPrice": 0.0,
        "lowPrice": 0.0,
        "volume": 0,
        "RSI": 0.0,
        "MACD": 0.0,
        "MACD_Signal": 0.0,
        "SMA_20": 0.0,
        "SMA_50": 0.0,
        "PSAR": 0.0,
        "buy_signal": False,
        "buy_details": "",
        "sell_signal": False,
        "sell_details": "",
        "is_open_position": ticker in open_positions,
        "change": 0.0,
        "status_text": "Veri Yetersiz",
        "status_class": "text-gray-500",
    }

    # Yeterli veri kontrolü
    required_min_bars_1h = 100
    required_min_bars_1d = 200

    if (
        data_1h.empty
        or data_1d.empty
        or len(data_1h) < required_min_bars_1h
        or len(data_1d) < required_min_bars_1d
    ):
        log_message(
            f"UYARI: {ticker} için sinyal kontrolü için yeterli veri yok. 1h bar: {len(data_1h)}, 1d bar: {len(data_1d)}",
            level="WARN",
        )
        stock_status["status_text"] = "Yetersiz Veri"
        stock_status["status_class"] = "text-red-500"
        return {
            "status": "no_data",
            "message": "Yeterli veri alınamadı veya hesaplama için yetersiz bar.",
            "data": stock_status,
        }

    data_1h_processed = add_technical_indicators(data_1h.copy())
    data_1d_processed = add_technical_indicators(data_1d.copy())

    if data_1h_processed.empty or data_1d_processed.empty:
        log_message(
            f"HATA: {ticker} için teknik göstergeler hesaplanamadı.", level="ERROR"
        )
        stock_status["status_text"] = "Gösterge Hatası"
        stock_status["status_class"] = "text-red-500"
        return {
            "status": "error_processing",
            "message": "Teknik göstergeler hesaplanamadı.",
            "data": stock_status,
        }

    try:
        latest_1h_data = data_1h_processed.iloc[-1]
        prev_1h_data = data_1h_processed.iloc[-2]
        latest_1d_data = data_1d_processed.iloc[-1]
        prev_1d_data = data_1d_processed.iloc[-2]

        # Temel fiyat ve hacim bilgileri
        stock_status["currentPrice"] = round(latest_1h_data["Close"].item(), 2)
        stock_status["openPrice"] = round(latest_1h_data["Open"].item(), 2)
        stock_status["highPrice"] = round(latest_1h_data["High"].item(), 2)
        stock_status["lowPrice"] = round(latest_1h_data["Low"].item(), 2)
        stock_status["volume"] = int(latest_1h_data["Volume"].item())

        # Teknik gösterge değerleri (NaN kontrolü dahil)
        indicators_to_assign = [
            "RSI",
            "MACD",
            "MACD_Signal",
            "SMA_20",
            "SMA_50",
            "PSAR",
        ]
        for ind in indicators_to_assign:
            if ind in latest_1h_data and not pd.isna(latest_1h_data[ind].item()):
                stock_status[ind] = round(latest_1h_data[ind].item(), 2)
            else:
                stock_status[ind] = 0.0  # Varsayılan değer

        # Günlük değişimi hesapla
        if (
            not pd.isna(prev_1h_data["Close"].item())
            and prev_1h_data["Close"].item() != 0
        ):
            stock_status["change"] = round(
                (
                    (latest_1h_data["Close"].item() - prev_1h_data["Close"].item())
                    / prev_1h_data["Close"].item()
                )
                * 100,
                2,
            )
        else:
            stock_status["change"] = 0.0

        # --- Sinyal Kontrol Mantığı (Gevşetilmiş versiyon) ---
        buy_signal = False
        buy_details = ""
        sell_signal = False
        sell_details = ""

        # YÜKSELİŞ (ALIŞ) SİNYALİ KONTROLÜ
        if ticker not in open_positions:
            # Tüm koşullar .item() ile güvenli bir şekilde alınmalı ve NaN kontrol edilmeli
            rsi_1h_condition = (
                not pd.isna(latest_1h_data["RSI"].item())
                and latest_1h_data["RSI"].item() > 55
            )

            sma_crossover_1h_condition = (
                not pd.isna(prev_1h_data["SMA_20"].item())
                and not pd.isna(prev_1h_data["SMA_50"].item())
                and not pd.isna(latest_1h_data["SMA_20"].item())
                and not pd.isna(latest_1h_data["SMA_50"].item())
                and (
                    prev_1h_data["SMA_20"].item() < prev_1h_data["SMA_50"].item()
                    and latest_1h_data["SMA_20"].item()
                    > latest_1h_data["SMA_50"].item()
                )
            )

            macd_1h_condition = (
                not pd.isna(latest_1h_data["MACD"].item())
                and not pd.isna(latest_1h_data["MACD_Signal"].item())
                and latest_1h_data["MACD"].item() > 0
                and latest_1h_data["MACD"].item() > latest_1h_data["MACD_Signal"].item()
            )

            daily_trend_condition = (
                not pd.isna(latest_1d_data["SMA_50"].item())
                and not pd.isna(latest_1d_data["SMA_200"].item())
                and not pd.isna(latest_1d_data["MACD"].item())
                and latest_1d_data["SMA_50"].item() > latest_1d_data["SMA_200"].item()
                and latest_1d_data["MACD"].item() > 0
            )

            avg_volume_1h_series = data_1h_processed["Volume"].iloc[-20:]
            max_volume_1h_50_period_series = data_1h_processed["Volume"].iloc[-50:]

            # BURADA CRİTİCAL DÜZELTME: .mean().item() ve .max().item() ile skaler değer garantisi
            avg_volume_1h = (
                avg_volume_1h_series.mean().item()
                if not avg_volume_1h_series.empty
                else 0
            )
            max_volume_1h_50_period = (
                max_volume_1h_50_period_series.max().item()
                if not max_volume_1h_50_period_series.empty
                else 0
            )

            # Hacim ortalamaları/maksimumları NaN ise koşulu devre dışı bırak
            if pd.isna(avg_volume_1h) or pd.isna(max_volume_1h_50_period):
                volume_condition = False
            else:
                volume_condition = (
                    not pd.isna(
                        latest_1h_data["Volume"].item()
                    )  # Hacim de NaN olmamalı
                    and latest_1h_data["Volume"].item() > (avg_volume_1h * 1.2)
                    and latest_1h_data["Volume"].item()
                    > (max_volume_1h_50_period * 0.5)
                )

            if (
                rsi_1h_condition
                and sma_crossover_1h_condition
                and macd_1h_condition
                and daily_trend_condition
                and volume_condition
            ):
                buy_signal = True
                buy_details = (
                    "BUY: RSI>55, SMA_CO, MACD_POS, DailyTrend, HighVol (Gevşetilmiş)"
                )
                enter_position(
                    ticker, latest_1h_data["Close"].item(), "BUY", buy_details
                )

        # DÜŞÜŞ (SATIŞ/KAR ALMA) SİNYALİ KONTROLÜ
        if ticker in open_positions:
            psar_condition = (
                not pd.isna(latest_1h_data["PSAR"].item())
                and latest_1h_data["Close"].item() < latest_1h_data["PSAR"].item()
            )
            close_below_sma20 = (
                not pd.isna(latest_1h_data["SMA_20"].item())
                and latest_1h_data["Close"].item() < latest_1h_data["SMA_20"].item()
            )

            macd_negative_or_crossover = (
                not pd.isna(latest_1h_data["MACD"].item())
                and not pd.isna(latest_1h_data["MACD_Signal"].item())
                and not pd.isna(prev_1h_data["MACD"].item())
                and not pd.isna(prev_1h_data["MACD_Signal"].item())
                and (
                    latest_1h_data["MACD"].item() < 0
                    or (
                        prev_1h_data["MACD"].item() > prev_1h_data["MACD_Signal"].item()
                        and latest_1h_data["MACD"].item()
                        < latest_1h_data["MACD_Signal"].item()
                    )
                )
            )

            if psar_condition or close_below_sma20 or macd_negative_or_crossover:
                sell_signal = True
                sell_details = "SELL: PSAR/SMA20_Down OR MACD_Bearish (Gevşetilmiş)"
                exit_position(
                    ticker, latest_1h_data["Close"].item(), "SELL", sell_details
                )

        stock_status["buy_signal"] = buy_signal
        stock_status["buy_details"] = buy_details
        stock_status["sell_signal"] = sell_signal
        stock_status["sell_details"] = sell_details

        # Genel durum belirleme (web arayüzü için)
        if stock_status["buy_signal"]:
            stock_status["status_text"] = "Yükselecek Sinyali!"
            stock_status["status_class"] = "status-buy-signal"
        elif stock_status["sell_signal"]:
            stock_status["status_text"] = "Düşüş/Satış Sinyali!"
            stock_status["status_class"] = "status-sell-signal"
        elif stock_status["change"] > 0.5:
            stock_status["status_text"] = "Yükselişte"
            stock_status["status_class"] = "status-up"
        elif stock_status["change"] < -0.5:
            stock_status["status_text"] = "Düşüşte"
            stock_status["status_class"] = "status-down"
        else:
            stock_status["status_text"] = "Yatay"
            stock_status["status_class"] = "status-neutral"

        return {"status": "ok", "data": stock_status}

    except Exception as e:
        log_message(
            f"HATA: {ticker} için sinyal işlenirken veya API yanıtı hazırlanırken hata oluştu: {e}",
            level="ERROR",
        )
        stock_status["status_text"] = "İşlem Hatası!"
        stock_status["status_class"] = "text-red-500"
        # Hata anında bile mevcut verileri (currentPrice, change, vs.) döndürmeye çalış
        # Eğer bu değerler de hata verdiyse 0.0 olarak kalacaktır.
        return {
            "status": "error_processing",
            "message": f"İşlem hatası: {e}",
            "data": stock_status,
        }


# --- API Endpoint'leri ---


@app.route("/api/analyze_stocks", methods=["GET"])
def analyze_stocks_api():
    stocks_to_monitor = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "NFLX",
        "JPM",
        "XOM",
        "JNJ",
        "WMT",
    ]
    results = []
    buy_signal_stock = None

    load_open_positions()  # Her API çağrısında pozisyonları yükle

    for ticker in stocks_to_monitor:
        analysis_result = analyze_stock_for_api(ticker)

        # Hata veya veri yoksa bile hissenin varsayılan/hata mesajlı verisini döndür
        results.append(analysis_result["data"])

        if (
            analysis_result["status"] == "ok"
            and analysis_result["data"]["buy_signal"]
            and buy_signal_stock is None
        ):
            buy_signal_stock = analysis_result["data"]

    current_open_positions = []
    for ticker, data in open_positions.items():
        current_open_positions.append(
            {
                "ticker": ticker,
                "entry_time": data["entry_time"].isoformat(),
                "entry_price": data["entry_price"],
            }
        )

    return jsonify(
        {
            "all_stocks_data": results,
            "buy_signal_stock": buy_signal_stock,
            "open_positions": current_open_positions,
        }
    )


# İlk kez çalıştırıldığında pozisyonları yükle
load_open_positions()

if __name__ == "__main__":
    log_message("Flask API sunucusu başlatılıyor...")
    app.run(debug=True, host="0.0.0.0", port=8080)

# PR test
