import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import os
import time

# --- Dosya Yolu Tanımlamaları ---
LOG_DIR = "backtest_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

BACKTEST_RESULTS_FILE = os.path.join(LOG_DIR, "backtest_results.csv")
BACKTEST_POSITIONS_FILE = os.path.join(LOG_DIR, "backtest_positions.csv")

# --- Veri Çekme Fonksiyonu ---

def get_stock_data(ticker_symbol, period, interval):
    """
    Belirtilen hisse senedi sembolü için geçmiş verileri çeker.
    Hız limiti hataları için yeniden deneme mekanizması eklendi.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {ticker_symbol} için {interval} veri çekiliyor...")
    max_retries = 7 # Maksimum deneme sayısı
    retry_delay_seconds = 60 # Her deneme arasında bekleme süresi (saniye)

    for attempt in range(max_retries):
        try:
            # FutureWarning'ı engellemek için auto_adjust=True açıkça belirtildi
            data = yf.download(ticker_symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if not data.empty:
                return data
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] UYARI: {ticker_symbol} için {interval} veri boş geldi. {attempt + 1}. deneme...")
        except yf.YFRateLimitError as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HIZ LİMİTİ! {ticker_symbol} için {interval} veri çekilemedi. ({e}) {retry_delay_seconds} saniye bekleniyor... ({attempt + 1}/{max_retries})")
            time.sleep(retry_delay_seconds)
            retry_delay_seconds *= 1.2 # Bekleme süresini her denemede artır
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker_symbol} için {interval} veri çekilirken beklenmeyen hata: {e}")
            break # Beklenmeyen hatalarda yeniden deneme yapma

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker_symbol} için {interval} veri çekilemedi, tüm denemeler başarısız oldu.")
    return pd.DataFrame() # Başarısız olursa boş DataFrame dön

# --- Teknik Gösterge Ekleme Fonksiyonu ---

def add_technical_indicators(df):
    """
    DataFrame'e çeşitli teknik analiz göstergeleri ekler.
    Her göstergeye veri gönderirken .squeeze() kullanarak 1D formatı garantiler.
    Bollinger Bantları metodları güncel ta kütüphanesine göre ayarlandı.
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
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] UYARI: Teknik göstergeler eklenirken hata oluştu: {e}")
        return pd.DataFrame() # Hata olursa boş DataFrame dön

# --- Mum Grafiği Formasyonları Yardımcı Fonksiyonlar ---
# Bu fonksiyonlar şu an kullanılmasa da ileride kullanılabilir diye tutuluyor.
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

# --- Sinyal ve Pozisyon Yönetimi (Backtest İçin Uyarlanmış) ---

open_backtest_positions = {}
closed_backtest_positions = [] 

def record_signal(timestamp, ticker, signal_type, details):
    pass 

def enter_position(timestamp, ticker, entry_price, signal_type, signal_details):
    if ticker not in open_backtest_positions: 
        open_backtest_positions[ticker] = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'signal_time': timestamp, 
            'signal_type': signal_type,
            'signal_details': signal_details
        }
        # Pozisyon açıldığında daha bilgilendirici bir çıktı
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] POZISYON AÇILDI: {ticker} @ {entry_price:.2f}$ | Sinyal: {signal_type} ({signal_details})")

def exit_position(timestamp, ticker, exit_price, signal_type, signal_details):
    if ticker in open_backtest_positions:
        pos_data = open_backtest_positions.pop(ticker)
        entry_price = pos_data['entry_price']
        
        profit_loss_usd = (exit_price - entry_price)
        profit_loss_pct = (profit_loss_usd / entry_price) * 100 if entry_price != 0 else 0

        closed_backtest_positions.append({
            'ticker': ticker,
            'entry_time': pos_data['entry_time'],
            'entry_price': entry_price,
            'exit_time': timestamp,
            'exit_price': exit_price,
            'profit_loss_usd': profit_loss_usd,
            'profit_loss_pct': profit_loss_pct,
            'signal_type_entry': pos_data['signal_type'],
            'signal_details_entry': pos_data['signal_details'],
            'signal_type_exit': signal_type,
            'signal_details_exit': signal_details
        })
        # Pozisyon kapatıldığında daha bilgilendirici bir çıktı
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] POZISYON KAPATILDI: {ticker} @ {exit_price:.2f}$ | Kar/Zarar: {profit_loss_usd:.2f}$ ({profit_loss_pct:.2}%) | Çıkış Sinyali: {signal_type} ({signal_details})")

# --- Sinyal Kontrol Fonksiyonu (Backtest İçin Uyarlanmış) ---

def check_for_signals_backtest(ticker, current_timestamp, data_1h, data_1d):
    """
    Belirli bir hisse senedi için çoklu zaman dilimi ve hacim onayı ile
    hem alış hem de satış sinyallerini kontrol eder.
    Bu fonksiyon, her zaman dilimindeki veriyi 'o ana kadar' olan veriye göre değerlendirir.
    """
    required_min_bars_1h = 100 
    required_min_bars_1d = 200 

    if data_1h.empty or data_1d.empty or \
       len(data_1h) < required_min_bars_1h or len(data_1d) < required_min_bars_1d:
        return

    latest_1h_data = data_1h.iloc[-1]
    prev_1h_data = data_1h.iloc[-2]
    
    latest_1d_data = data_1d.iloc[-1]
    prev_1d_data = data_1d.iloc[-2]

    indicators_1h_to_check = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'SMA_200', 'BBL', 'BBM', 'BBH', 'OBV', 'PSAR']
    indicators_1d_to_check = ['MACD', 'MACD_Signal', 'SMA_50', 'SMA_200']

    # Her bir göstergenin mevcut olup olmadığını ve NaN olmadığını kontrol et
    for col in indicators_1h_to_check:
        if col not in latest_1h_data or pd.isna(latest_1h_data[col].item()):
            return 
        if col not in prev_1h_data or pd.isna(prev_1h_data[col].item()):
            return

    for col in indicators_1d_to_check:
        if col not in latest_1d_data or pd.isna(latest_1d_data[col].item()):
            return
        if col not in prev_1d_data or pd.isna(prev_1d_data[col].item()):
            return

    # --- YÜKSELİŞ (ALIŞ) SİNYALİ KONTROLÜ ---
    if ticker not in open_backtest_positions: 
        # Kural 1: Saatlik RSI güçlü momentumu gösteriyor (eşik 55)
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
        
        # Kural 5: Hacim Onayı (çarpanlar daha da gevşetildi)
        avg_volume_1h_series = data_1h['Volume'].iloc[-20:]
        max_volume_1h_50_period_series = data_1h['Volume'].iloc[-50:]

        avg_volume_1h = avg_volume_1h_series.mean().item() if not avg_volume_1h_series.empty else 0
        max_volume_1h_50_period = max_volume_1h_50_period_series.max().item() if not max_volume_1h_50_period_series.empty else 0
        
        volume_condition = (latest_1h_data['Volume'].item() > (avg_volume_1h * 1.2) and  # 1.5'ten 1.2'ye
                            latest_1h_data['Volume'].item() > (max_volume_1h_50_period * 0.5)) # 0.6'dan 0.5'e

        # Kural 6 ve 7 (Mum Formasyonu ve Bollinger Bandı) geçici olarak kaldırıldı
        # candlestick_pattern_condition = is_bullish_engulfing(data_1h.iloc[-2:])
        # bollinger_breakout_condition = latest_1h_data['Close'].item() > latest_1h_data['BBH'].item()


        # Tüm koşullar sağlanıyorsa alış sinyali ver
        if (rsi_1h_condition and 
            sma_crossover_1h_condition and 
            macd_1h_condition and 
            daily_trend_condition and 
            volume_condition
            # and candlestick_pattern_condition # Kaldırıldı
            # and bollinger_breakout_condition # Kaldırıldı
            ):

            details = ("BUY: RSI>55, SMA_CO, MACD_POS, DailyTrend, HighVol (Gevşetilmiş)")
            enter_position(current_timestamp, ticker, latest_1h_data['Close'].item(), "BUY", details) 
            return 

    # --- DÜŞÜŞ (SATIŞ/KAR ALMA) SİNYALİ KONTROLÜ ---
    if ticker in open_backtest_positions:
        # Kural 1: Fiyat Parabolik SAR'ın altına düştü
        psar_condition = latest_1h_data['Close'].item() < latest_1h_data['PSAR'].item()

        # Kural 2: Hisse kapanış fiyatı 20 periyotluk Basit Hareketli Ortalamanın (SMA_20) altına düştü
        close_below_sma20 = latest_1h_data['Close'].item() < latest_1h_data['SMA_20'].item()

        # Kural 3: MACD negatif veya MACD hattı sinyal hattının altına düştü
        macd_negative_or_crossover = (latest_1h_data['MACD'].item() < 0 or 
                                      (prev_1h_data['MACD'].item() > prev_1h_data['MACD_Signal'].item() and
                                       latest_1h_data['MACD'].item() < latest_1h_data['MACD_Signal'].item()))
        
        # Kural 4 (Ayı Yutan Mum Formasyonu) geçici olarak kaldırıldı
        # candlestick_pattern_sell_condition = is_bearish_engulfing(data_1h.iloc[-2:])

        if psar_condition or close_below_sma20 or macd_negative_or_crossover: # Mum formasyonu kaldırıldı
            details = "SELL: PSAR/SMA20_Down OR MACD_Bearish (Gevşetilmiş)"
            exit_position(current_timestamp, ticker, latest_1h_data['Close'].item(), "SELL", details)


# --- Ana Geri Test Döngüsü ---

if __name__ == "__main__":
    stocks_to_monitor = ["AAPL", "MSFT", "GOOGL"] 
    
    BACKTEST_PERIOD_1H = "60d" 
    BACKTEST_PERIOD_1D = "5y" 

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test programı başlatıldı.")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Takip edilen hisseler: {', '.join(stocks_to_monitor)}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test periyodu (1 saatlik veri): {BACKTEST_PERIOD_1H}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test periyodu (Günlük veri): {BACKTEST_PERIOD_1D}\n")

    all_1h_data = {}
    all_1d_data = {}
    
    valid_stocks_to_monitor = [] 

    for ticker in stocks_to_monitor:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {ticker} için backtest verileri çekiliyor ve işleniyor...")
        try:
            data_1h = get_stock_data(ticker, period=BACKTEST_PERIOD_1H, interval="1h")
            if data_1h.empty:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker} için 1 saatlik veri boş geldi veya indirilemedi. Bu hisse atlanıyor.")
                continue 
            
            data_1h_processed = add_technical_indicators(data_1h.copy())
            if data_1h_processed.empty:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker} için 1 saatlik veri göstergeler eklenirken boş kaldı. Bu hisse atlanıyor.")
                continue

            data_1d = get_stock_data(ticker, period=BACKTEST_PERIOD_1D, interval="1d")
            if data_1d.empty:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker} için günlük veri boş geldi veya indirilemedi. Bu hisse atlanıyor.")
                continue

            data_1d_processed = add_technical_indicators(data_1d.copy())
            if data_1d_processed.empty:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker} için günlük veri göstergeler eklenirken boş kaldı. Bu hisse atlanıyor.")
                continue

            all_1h_data[ticker] = data_1h_processed.sort_index() 
            all_1d_data[ticker] = data_1d_processed.sort_index()
            valid_stocks_to_monitor.append(ticker)

        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HATA: {ticker} hissesi için veri çekilirken veya işlenirken genel bir sorun oluştu: {e}. Bu hisse atlanıyor.")
        
        # Her hisse senedi veri çekiminden sonra kısa bir bekleme ekle
        time.sleep(2) 

    stocks_to_monitor = valid_stocks_to_monitor 
    
    if not stocks_to_monitor:
        print("Geri test edilecek geçerli hisse senedi kalmadı. Çıkılıyor.")
        exit()

    min_start_time = None
    max_end_time = None

    for ticker in stocks_to_monitor:
        if not all_1h_data[ticker].empty:
            if min_start_time is None or all_1h_data[ticker].index.min() > min_start_time:
                min_start_time = all_1h_data[ticker].index.min()
            if max_end_time is None or all_1h_data[ticker].index.max() < max_end_time:
                max_end_time = all_1h_data[ticker].index.max()

    if min_start_time is None or max_end_time is None:
        print("Ortak bir zaman dilimi belirlenemedi veya yeterli veri yok. Çıkılıyor.")
        exit()

    start_index_offset_1h = 100 

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test simülasyonu başlatılıyor...")
    
    longest_1h_df = pd.DataFrame()
    for ticker in stocks_to_monitor:
        if len(all_1h_data[ticker]) > len(longest_1h_df):
            longest_1h_df = all_1h_data[ticker]
            
    if longest_1h_df.empty or len(longest_1h_df) <= start_index_offset_1h:
        print("Yeterli 1 saatlik veri yok veya başlangıç ofseti çok büyük. Geri test yapılamıyor.")
        exit()

    for i in range(start_index_offset_1h, len(longest_1h_df)):
        current_1h_timestamp = longest_1h_df.index[i]
        
        for ticker in stocks_to_monitor:
            current_1h_data_slice = all_1h_data[ticker].loc[all_1h_data[ticker].index <= current_1h_timestamp]
            current_1d_data_slice = all_1d_data[ticker].loc[all_1d_data[ticker].index.date <= current_1h_timestamp.date()]

            if not current_1h_data_slice.empty and not current_1d_data_slice.empty:
                check_for_signals_backtest(ticker, current_1h_timestamp, current_1h_data_slice, current_1d_data_slice)
        
        if i % 100 == 0 or i == len(longest_1h_df) -1 : 
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] İşlenen mum: {i}/{len(longest_1h_df)} ({current_1h_timestamp.strftime('%Y-%m-%d %H:%M')})")

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test simülasyonu tamamlandı.")

    for ticker, pos_data in list(open_backtest_positions.items()):
        if ticker not in all_1h_data or all_1h_data[ticker].empty:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] UYARI: {ticker} için son 1H veri bulunamadı, açık pozisyon kapatılamıyor.")
            continue
        exit_price = all_1h_data[ticker].iloc[-1]['Close'].item() 
        exit_position(all_1h_data[ticker].index[-1], ticker, exit_price, "END_OF_TEST", "Test sonu açık pozisyon kapatıldı.")

    # --- Sonuçları Kaydet ve Raporla ---
    if closed_backtest_positions:
        results_df = pd.DataFrame(closed_backtest_positions)
        results_df.to_csv(BACKTEST_POSITIONS_FILE, index=False)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Kapatılan pozisyonlar '{BACKTEST_POSITIONS_FILE}' dosyasına kaydedildi.")

        total_profit = results_df['profit_loss_usd'].sum()
        total_trades = len(results_df)
        winning_trades = results_df[results_df['profit_loss_usd'] > 0]
        losing_trades = results_df[results_df['profit_loss_usd'] < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        avg_profit_per_trade = results_df['profit_loss_usd'].mean() if total_trades > 0 else 0
        total_profit_pct = results_df['profit_loss_pct'].sum() 

        summary = {
            "Total Trades": total_trades,
            "Winning Trades": len(winning_trades),
            "Losing Trades": len(losing_trades),
            "Win Rate (%)": f"{win_rate:.2f}%",
            "Total Profit/Loss (USD)": f"{total_profit:.2f}$",
            "Average Profit/Loss Per Trade (USD)": f"{avg_profit_per_trade:.2f}$",
            "Total Profit/Loss (%)": f"{total_profit_pct:.2f}%" 
        }

        with open(BACKTEST_RESULTS_FILE, "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test sonuçları '{BACKTEST_RESULTS_FILE}' dosyasına kaydedildi.")
        print("\n--- Geri Test Özeti ---")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("--------------------")

    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Geri test sırasında hiçbir pozisyon açılıp kapatılmadı.")