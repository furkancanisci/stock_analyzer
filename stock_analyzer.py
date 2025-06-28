import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import time
import os

# --- Dosya Yolu Tanımlamaları ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "trading_log.txt")
POSITIONS_FILE = os.path.join(LOG_DIR, "open_positions.csv")

# --- Global Değişkenler ---
# Canlı sistemde pozisyonları kalıcı olarak kaydetmek için
open_positions = {}

def load_open_positions():
    """Kayıtlı açık pozisyonları yükler."""
    global open_positions
    if os.path.exists(POSITIONS_FILE):
        try:
            positions_df = pd.read_csv(POSITIONS_FILE, index_col='ticker')
            open_positions = positions_df.to_dict(orient='index')
            # Timestamp'leri datetime objelerine dönüştür
            for ticker, data in open_positions.items():
                data['entry_time'] = datetime.fromisoformat(data['entry_time'])
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Açık pozisyonlar yüklendi.")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pozisyonlar yüklenirken hata oluştu: {e}")
            open_positions = {} # Hata durumunda boş başlat
    else:
        open_positions = {}

def save_open_positions():
    """Açık pozisyonları kaydeder."""
    global open_positions
    if open_positions:
        # datetime objelerini ISO formatında string'e dönüştür
        temp_positions = open_positions.copy()
        for ticker, data in temp_positions.items():
            data['entry_time'] = data['entry_time'].isoformat()
        
        positions_df = pd.DataFrame.from_dict(temp_positions, orient='index')
        positions_df.index.name = 'ticker'
        positions_df.to_csv(POSITIONS_FILE)
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Açık pozisyonlar kaydedildi.")
    elif os.path.exists(POSITIONS_FILE):
        os.remove(POSITIONS_FILE) # Hiç pozisyon yoksa dosyayı sil

# --- Loglama Fonksiyonu ---
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
            # auto_adjust=True ile FutureWarning'ı önle
            data = yf.download(ticker_symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if not data.empty:
                return data
            else:
                log_message(f"UYARI: {ticker_symbol} için {interval} veri boş geldi. {attempt + 1}. deneme...", level="WARN")
        except yf.YFRateLimitError as e:
            log_message(f"HIZ LİMİTİ! {ticker_symbol} için {interval} veri çekilemedi. ({e}) 60 saniye bekleniyor... ({attempt + 1}/{max_retries})", level="ERROR")
            time.sleep(60)
        except Exception as e:
            log_message(f"HATA: {ticker_symbol} için {interval} veri çekilirken beklenmeyen hata: {e}. {attempt + 1}. deneme...", level="ERROR")
            time.sleep(10) # Diğer hatalarda kısa bekleme
    log_message(f"HATA: {ticker_symbol} için {interval} veri çekilemedi, tüm denemeler başarısız oldu.", level="CRITICAL")
    return pd.DataFrame()

# --- Teknik Gösterge Ekleme Fonksiyonu ---
def add_technical_indicators(df):
    """
    DataFrame'e çeşitli teknik analiz göstergeleri ekler.
    .squeeze() kullanarak boyut uyumsuzluğu hatalarını önler.
    """
    if df.empty:
        return df

    try:
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
        # MACD
        macd = ta.trend.MACD(df['Close'].squeeze())
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        # SMA
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'].squeeze(), window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'].squeeze(), window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(df['Close'].squeeze(), window=200).sma_indicator()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'].squeeze())
        df['BBL'] = bollinger.bollinger_lband()
        df['BBM'] = bollinger.bollinger_mavg()
        df['BBH'] = bollinger.bollinger_hband()
        
        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()
        # Parabolic SAR
        df['PSAR'] = ta.trend.PSARIndicator(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()).psar()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        log_message(f"UYARI: Teknik göstergeler eklenirken hata oluştu: {e}", level="ERROR")
        return pd.DataFrame() # Hata olursa boş DataFrame dön

# --- Mum Grafiği Formasyonları Yardımcı Fonksiyonlar ---
def is_bullish_engulfing(df_slice):
    if len(df_slice) < 2:
        return False
    current_candle = df_slice.iloc[-1]
    prev_candle = df_slice.iloc[-2]
    prev_is_bearish = prev_candle['Close'].item() < prev_candle['Open'].item()
    current_is_bullish = current_candle['Close'].item() > current_candle['Open'].item()
    engulfs_open = current_candle['Open'].item() < prev_candle['Close'].item()
    engulfs_close = current_candle['Close'].item() > prev_candle['Open'].item()
    prev_body_size = abs(prev_candle['Open'].item() - prev_candle['Close'].item())
    current_body_size = abs(current_candle['Open'].item() - current_candle['Close'].item())
    body_size_condition = current_body_size > (prev_body_size * 1.2)
    return (prev_is_bearish and current_is_bullish and 
            engulfs_open and engulfs_close and body_size_condition)

def is_bearish_engulfing(df_slice):
    if len(df_slice) < 2:
        return False
    current_candle = df_slice.iloc[-1]
    prev_candle = df_slice.iloc[-2]
    prev_is_bullish = prev_candle['Close'].item() > prev_candle['Open'].item()
    current_is_bearish = current_candle['Close'].item() < current_candle['Open'].item()
    engulfs_open = current_candle['Open'].item() > prev_candle['Close'].item()
    engulfs_close = current_candle['Close'].item() < prev_candle['Open'].item()
    prev_body_size = abs(prev_candle['Open'].item() - prev_candle['Close'].item())
    current_body_size = abs(current_candle['Open'].item() - current_candle['Close'].item())
    body_size_condition = current_body_size > (prev_body_size * 1.2)
    return (prev_is_bullish and current_is_bearish and 
            engulfs_open and engulfs_close and body_size_condition)

# --- Sinyal ve Pozisyon Yönetimi ---

def enter_position(ticker, entry_price, signal_type, signal_details):
    global open_positions
    if ticker not in open_positions:
        open_positions[ticker] = {
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'signal_time': datetime.now(), 
            'signal_type': signal_type,
            'signal_details': signal_details
        }
        log_message(f"POZISYON AÇILDI: {ticker} @ {entry_price:.2f}$ | Sinyal: {signal_type} ({signal_details})", level="INFO")
        save_open_positions()

def exit_position(ticker, exit_price, signal_type, signal_details):
    global open_positions
    if ticker in open_positions:
        pos_data = open_positions.pop(ticker)
        entry_price = pos_data['entry_price']
        
        profit_loss_usd = (exit_price - entry_price)
        profit_loss_pct = (profit_loss_usd / entry_price) * 100 if entry_price != 0 else 0

        log_message(f"POZISYON KAPATILDI: {ticker} @ {exit_price:.2f}$ | Kar/Zarar: {profit_loss_usd:.2f}$ ({profit_loss_pct:.2f}%) | Çıkış Sinyali: {signal_type} ({signal_details})", level="INFO")
        save_open_positions()

# --- Sinyal Kontrol Fonksiyonu ---
def check_for_signals(ticker, data_1h, data_1d):
    """
    Belirli bir hisse senedi için çoklu zaman dilimi ve hacim onayı ile
    hem alış hem de satış sinyallerini kontrol eder.
    """
    # Gerekli minimum veri kontrolü
    required_min_bars_1h = 100 
    required_min_bars_1d = 200 

    if data_1h.empty or data_1d.empty or \
       len(data_1h) < required_min_bars_1h or len(data_1d) < required_min_bars_1d:
        log_message(f"UYARI: {ticker} için sinyal kontrolü için yeterli veri yok.", level="WARN")
        return

    # En güncel ve önceki veri noktalarını alın
    latest_1h_data = data_1h.iloc[-1]
    prev_1h_data = data_1h.iloc[-2]
    
    latest_1d_data = data_1d.iloc[-1]
    prev_1d_data = data_1d.iloc[-2]

    # Teknik göstergelerin NaN olmadığını kontrol et
    indicators_1h_to_check = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'SMA_200', 'BBL', 'BBM', 'BBH', 'OBV', 'PSAR']
    indicators_1d_to_check = ['MACD', 'MACD_Signal', 'SMA_50', 'SMA_200']

    for col in indicators_1h_to_check:
        if col not in latest_1h_data or pd.isna(latest_1h_data[col].item()):
            log_message(f"UYARI: {ticker} 1h verisinde {col} göstergesi eksik veya NaN.", level="WARN")
            return 
        if col not in prev_1h_data or pd.isna(prev_1h_data[col].item()):
            log_message(f"UYARI: {ticker} 1h önceki verisinde {col} göstergesi eksik veya NaN.", level="WARN")
            return

    for col in indicators_1d_to_check:
        if col not in latest_1d_data or pd.isna(latest_1d_data[col].item()):
            log_message(f"UYARI: {ticker} 1d verisinde {col} göstergesi eksik veya NaN.", level="WARN")
            return
        if col not in prev_1d_data or pd.isna(prev_1d_data[col].item()):
            log_message(f"UYARI: {ticker} 1d önceki verisinde {col} göstergesi eksik veya NaN.", level="WARN")
            return

    # --- YÜKSELİŞ (ALIŞ) SİNYALİ KONTROLÜ ---
    if ticker not in open_positions: # Sadece açık pozisyon yoksa alış sinyali ara
        # Kural 1: Saatlik RSI güçlü momentumu gösteriyor
        rsi_1h_condition = latest_1h_data['RSI'].item() > 55 

        # Kural 2: Saatlik SMA20, SMA50'yi yukarı kesiyor
        sma_crossover_1h_condition = (prev_1h_data['SMA_20'].item() < prev_1h_data['SMA_50'].item() and
                                      latest_1h_data['SMA_20'].item() > latest_1h_data['SMA_50'].item())

        # Kural 3: Saatlik MACD pozitif ve sinyal hattının üzerinde
        macd_1h_condition = (latest_1h_data['MACD'].item() > 0 and 
                             latest_1h_data['MACD'].item() > latest_1h_data['MACD_Signal'].item())
        
        # Kural 4: Günlük grafik de güçlü yükseliş trendinde
        daily_trend_condition = (latest_1d_data['SMA_50'].item() > latest_1d_data['SMA_200'].item() and
                                 latest_1d_data['MACD'].item() > 0)
        
        # Kural 5: Hacim Onayı (gevşetildi)
        avg_volume_1h_series = data_1h['Volume'].iloc[-20:]
        max_volume_1h_50_period_series = data_1h['Volume'].iloc[-50:]

        avg_volume_1h = avg_volume_1h_series.mean().item() if not avg_volume_1h_series.empty else 0
        max_volume_1h_50_period = max_volume_1h_50_period_series.max().item() if not max_volume_1h_50_period_series.empty else 0
        
        volume_condition = (latest_1h_data['Volume'].item() > (avg_volume_1h * 1.2) and 
                            latest_1h_data['Volume'].item() > (max_volume_1h_50_period * 0.5))

        # Kural 6: Mum Grafiği Formasyonu Onayı (Boğa Yutan Mum) - Geçici olarak KAPALI
        candlestick_pattern_condition = False # is_bullish_engulfing(data_1h.iloc[-2:]) # Çok sıkı koşul olduğu için kapatıldı

        # Kural 7: Bollinger Bandı Kırılması - Geçici olarak KAPALI
        bollinger_breakout_condition = False # latest_1h_data['Close'].item() > latest_1h_data['BBH'].item() # Çok sıkı koşul olduğu için kapatıldı

        # Tüm koşullar sağlanıyorsa alış sinyali ver
        if (rsi_1h_condition and 
            sma_crossover_1h_condition and 
            macd_1h_condition and 
            daily_trend_condition and 
            volume_condition and
            candlestick_pattern_condition == False and # Veya doğrudan bu koşulu if'ten çıkarabiliriz
            bollinger_breakout_condition == False): # Veya doğrudan bu koşulu if'ten çıkarabiliriz

            details = ("BUY: RSI>55, SMA_CO, MACD_POS, DailyTrend, HighVol (Gevşetilmiş)")
            enter_position(ticker, latest_1h_data['Close'].item(), "BUY", details)
            return

    # --- DÜŞÜŞ (SATIŞ/KAR ALMA) SİNYALİ KONTROLÜ ---
    if ticker in open_positions: # Sadece açık pozisyon varsa satış sinyali ara
        # Kural 1: Fiyat Parabolik SAR'ın altına düştü
        psar_condition = latest_1h_data['Close'].item() < latest_1h_data['PSAR'].item()

        # Kural 2: Hisse kapanış fiyatı 20 periyotluk Basit Hareketli Ortalamanın (SMA_20) altına düştü
        close_below_sma20 = latest_1h_data['Close'].item() < latest_1h_data['SMA_20'].item()

        # Kural 3: MACD negatif veya MACD hattı sinyal hattının altına düştü
        macd_negative_or_crossover = (latest_1h_data['MACD'].item() < 0 or 
                                      (prev_1h_data['MACD'].item() > prev_1h_data['MACD_Signal'].item() and
                                       latest_1h_data['MACD'].item() < latest_1h_data['MACD_Signal'].item()))
        
        # Kural 4: Ayı Yutan Mum Formasyonu - Geçici olarak KAPALI
        candlestick_pattern_sell_condition = False # is_bearish_engulfing(data_1h.iloc[-2:]) # Çok sıkı koşul olduğu için kapatıldı

        if psar_condition or close_below_sma20 or macd_negative_or_crossover or candlestick_pattern_sell_condition == False:
            details = "SELL: PSAR/SMA20_Down OR MACD_Bearish (Gevşetilmiş)"
            exit_position(ticker, latest_1h_data['Close'].item(), "SELL", details)


# --- Ana İzleme Döngüsü ---
def main_loop():
    stocks_to_monitor = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"] 
    # Daha kısa periyotlar canlı veri için daha anlamlı olabilir. yfinance 1m için sadece 7 günlük data verir.
    # 5m ve 15m interval'ları daha geniş periyotlar için kullanılabilir.
    # Örnek: period="60d", interval="1h" veya period="7d", interval="1m"
    LIVE_DATA_PERIOD_1H = "60d" 
    LIVE_DATA_PERIOD_1D = "5y" 
    CHECK_INTERVAL_MINUTES = 15 # Her 15 dakikada bir kontrol et

    log_message("Hisse senedi takip programı başlatıldı.")
    log_message(f"Takip edilen hisseler: {', '.join(stocks_to_monitor)}")
    log_message(f"Her {CHECK_INTERVAL_MINUTES}.0 dakikada bir kontrol edilecek.")

    # Açık pozisyonları program başlangıcında yükle
    load_open_positions()

    while True:
        current_time = datetime.now()
        log_message(f"\n--- {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Yeni Kontrol Döngüsü Başlatılıyor ---", level="INFO")
        
        for ticker in stocks_to_monitor:
            try:
                log_message(f"{ticker} için veri çekiliyor ve sinyaller kontrol ediliyor...", level="INFO")
                # Canlı veri çekimi
                data_1h = get_stock_data(ticker, period=LIVE_DATA_PERIOD_1H, interval="1h")
                data_1d = get_stock_data(ticker, period=LIVE_DATA_PERIOD_1D, interval="1d")

                if data_1h.empty or data_1d.empty:
                    log_message(f"HATA: {ticker} için yeterli veri alınamadı, bu hisse atlanıyor.", level="ERROR")
                    continue

                # Teknik göstergeleri ekle
                data_1h_processed = add_technical_indicators(data_1h.copy())
                data_1d_processed = add_technical_indicators(data_1d.copy())

                if data_1h_processed.empty or data_1d_processed.empty:
                    log_message(f"HATA: {ticker} için teknik göstergeler hesaplanamadı, bu hisse atlanıyor.", level="ERROR")
                    continue

                # Sinyalleri kontrol et
                check_for_signals(ticker, data_1h_processed, data_1d_processed)

            except Exception as e:
                log_message(f"HATA: {ticker} hissesi işlenirken bir sorun oluştu: {e}", level="ERROR")
            
            # Her hisse arasında kısa bir bekleme
            time.sleep(2) # Yüksek frekansta API limitlerine takılmamak için

        log_message(f"--- Tüm hisseler kontrol edildi. {CHECK_INTERVAL_MINUTES} dakika bekleniyor... ---", level="INFO")
        time.sleep(CHECK_INTERVAL_MINUTES * 60) # Belirlenen süre kadar bekle

if __name__ == "__main__":
    main_loop()