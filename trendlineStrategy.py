"""
================================================================================
REAL-TIME SUPPORT & RESISTANCE DETECTION & ALGORITHMIC EXECUTION ENGINE
================================================================================

This program creates an autonomous, real-time algorithmic trading system 
using live market data from the FYERS WebSocket API. It dynamically detects 
support and resistance trendline zones using the `trendln` library and 
executes a rule-based trading strategy with live risk management.

The system continuously streams live tick-by-tick market data to:

1. Build Real-Time OHLCV Candles:
   Processes cumulative session volume into incremental, minute-by-minute 
   candle volume while dynamically updating Open, High, Low, and Close prices.

2. Map Dynamic Market Structure:
   Evaluates finalized candles using local mathematical minima/maxima to 
   calculate, rank, and project standard-deviation-based support and 
   resistance zones instead of static single lines.

3. Track Structural State:
   Monitors whether support/resistance regions are active, valid, or broken 
   (invalidated if price breaches a zone by more than 3 standard deviations).

4. Execute Algorithmic Trade Signals:
   Evaluates setups exactly once per candle close to eliminate mid-candle noise:
   - LONG Entry: Price respects a Support Zone (Low pokes inside, Close holds above).
   - SHORT Entry: Price respects a Resistance Zone (High pokes inside, Close holds below).

5. Manage Active Risk & Live Exits:
   - Queries live Market Depth (Bid/Ask) to simulate execution fills.
   - Implements a dynamic, structure-based Trailing Stop Loss (trailing to the 
     low/high of the previous closed candle) and automatically recalculates 
     Take Profit targets to optimize Risk-to-Reward.
   - Persists all execution metrics chronologically to 'trades_log.csv'.

6. Display Real-Time Visualizations:
   Renders a live, self-updating Matplotlib / mplfinance dual-axis chart 
   displaying candlestick action, volume, and transparently shaded active S/R bands.

The project demonstrates production-grade architecture used in:
- Quantitative Trading & Strategy Automation
- Event-Driven WebSocket Streaming Data Engineering
- Live State Management & Thread-Safe Callbacks
- Algorithmic Risk Containment & Trailing Allocations

Libraries Used:
- FYERS API v3 (fyersModel, FyersWebsocket)
- pandas & numpy
- trendln
- matplotlib & mplfinance
- pytz

Author: Vaibhav Saxena
================================================================================
"""

# ============================================================
# Imports
# ============================================================
from fyers_apiv3.FyersWebsocket import data_ws
from fyers_apiv3 import fyersModel
from credentials import client_id
import datetime as dt
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.animation import FuncAnimation
import trendln
import math
import numpy as np


# ============================================================
# Configuration
# ============================================================
symbol = 'NSE:RELIANCE-EQ'
timeZone = 'Asia/Kolkata'
max_candles = 100
plot_window = 100
resolution = "1"


# ============================================================
# Support and Resistance Data
# ============================================================
def support_resistance(df):
    """
    Detect support and resistance trendlines from market data.

    This function uses the `trendln` library to identify important
    price structures in the market.

    Support lines are created using candle LOW prices.
    Resistance lines are created using candle HIGH prices.

    The library first:
    1. Finds local swing highs and swing lows.
    2. Creates possible trendlines.
    3. Ranks the best trendlines mathematically.

    We only return the latest window because recent market structure
    is more useful for real-time trading than old historical structure.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV candlestick dataframe.

    Returns
    -------
    tuple
        Latest support trendlines and resistance trendlines.
    """
    
    (minimaIdxs, pmin, mintrend, minwindows),(maximaIdxs, pmax, maxtrend, maxwindows) = trendln.calc_support_resistance(
    (df['low'].to_numpy(), df['high'].to_numpy()),
    extmethod=trendln.METHOD_NUMDIFF,
    method=trendln.METHOD_NSQUREDLOGN,
    window=50,
    errpct=0.003)

    minwindows = [item for sublist in minwindows for item in sublist]
    maxwindows = [item for sublist in maxwindows for item in sublist]

    return minwindows, maxwindows


# ============================================================
# Generate Trendline Data
# ============================================================
def generate_trendline_data(df, trend_data, type):
    """
    Converts trendln output into mplfinance-compatible line data.
    
    Parameters
    ----------
    df : DataFrame
        OHLC dataframe
    
    trend_data : list
        mintrend or maxtrend from trendln
    
    Returns
    -------
    list
        List of mplfinance addplot objects
    """

    trendlines = []

    for points, result in trend_data:

        if len(points) < 2:
            continue

        slope = result[0]
        intercept = result[1]

        # trendln returns SSR (Sum of Squared Residuals),
        # which measures how far points are from the trendline.
        #
        # We convert it into a rough standard deviation estimate
        # so we can create a support/resistance "zone"
        # instead of a single thin line.
        sd = math.sqrt(result[2] / (len(points) - 2))

        # We create two arrays:
        #
        # upper -> top boundary of support/resistance zone
        # lower -> bottom boundary of support/resistance zone
        #
        # np.nan means "empty value".
        # This helps matplotlib avoid drawing unwanted lines.
        upper = np.full(len(df), np.nan)
        lower = np.full(len(df), np.nan)

        start = min(points)
        end = len(df) - 1

        # A trendline is considered ACTIVE only while price respects it.
        # 
        # Example:
        # - A support zone should stay below price.
        # - A resistance zone should stay above price.
        #
        # If price strongly breaks through the zone,
        # we mark the trendline as invalid.
        is_active = True

        for i in range(start, end + 1):

            val = slope * i + intercept

            # If price breaks too far beyond the zone,
            # the trendline is no longer valid.
            #
            # Example:
            # - If price falls below support strongly,
            #   support has failed.
            #
            # - If price rises above resistance strongly,
            #   resistance has failed.
            if df['low'].iloc[i] + 3 * sd < val and type == 'support':
                is_active = False
                break
            elif df['high'].iloc[i] - 3 * sd > val and type == 'resistance':
                is_active = False
                break

            upper[i] = val + sd * 3
            lower[i] = val - sd * 3

        trendlines.append((points, upper, lower, is_active))

    return trendlines


# ============================================================
# Update Logic for Live Data
# ============================================================
def update_live_data(data, message, last_total_volume):
    """
    Update the OHLCV dataframe using incoming websocket tick data.

    The websocket provides cumulative traded volume for the session.
    This function converts cumulative volume into incremental candle volume.

    Parameters
    ----------
    data : pandas.DataFrame
        Existing OHLCV dataframe indexed by timestamp.

    message : dict
        Incoming websocket tick message from Fyers.

    last_total_volume : int or None
        Previous cumulative traded volume received from websocket.

    Returns
    -------
    tuple
        Updated dataframe and latest cumulative traded volume.
    """

    # If the message is empty or broken, don't do anything
    if "symbol" not in message:
        return data, last_total_volume, None
    
    ltp = message.get('ltp')                            # LTP = Last Traded Price (The current market price)

    if ltp is None:
        return data, last_total_volume, None
    
    # Cumulative volume is the total shares traded since 9:15 AM. 
    # We want to find out how many were traded just in the last tick.
    total_vol = message.get('vol_traded_today')
    
    # Create a timestamp rounded to the current minute (e.g., 10:05:42 becomes 10:05:00)
    timestamp = pd.Timestamp.now(tz=timeZone).floor('1min')

    if total_vol is None:
        total_vol = last_total_volume if last_total_volume is not None else 0

    # The websocket gives TOTAL traded volume for the entire day.
    #
    # But each candle should only contain volume traded
    # during that specific minute.
    #
    # So:
    #
    # Incremental Volume = Current Total Volume - Previous Total Volume
    if last_total_volume is None:
        incremental_vol = 0
    else:
        incremental_vol = total_vol - last_total_volume

    if incremental_vol < 0:
        incremental_vol = 0

    # If the latest candle already belongs to the current minute,
    # we UPDATE the existing candle.
    #
    # Otherwise:
    # we CREATE a completely new candle.
    if len(data) > 0 and data.index[-1] == timestamp:
        # If we are still in the same minute, we update the existing candle
        data.iloc[-1, 3] = ltp                          # Close price continuously tracks latest traded price
        data.iloc[-1, 1] = max(data.iloc[-1, 1], ltp)   # Update High if price went higher
        data.iloc[-1, 2] = min(data.iloc[-1, 2], ltp)   # Update Low if price went lower
        data.iloc[-1, 4] += incremental_vol             # Add the new volume to the minute's total
        return data, total_vol, None
    else:
        new_candle = pd.DataFrame(
            [{'open': ltp, 'high': ltp, 'low': ltp, 'close': ltp, 'volume': incremental_vol}],
            index=[timestamp]
        )

        # data = data.tail(max_candles).copy()
        data = pd.concat([data, new_candle])

        # Compute S/R on all closed candles (exclude last, which is forming)
        closed = data.iloc[:-1]
        minwindows, maxwindows = support_resistance(df=closed)

        # Build compact zone list — just the boundary values at the LAST closed candle
        # This is O(k) where k = number of trendlines, typically 4-8, not n=100
        # Inside update_live_data, just before calling _build_zone_cache
        print(f"\n{'='*55}")
        print(f"[CANDLE CLOSE] {data.index[-2]}  |  "
            f"O={closed.iloc[-1]['open']} H={closed.iloc[-1]['high']} "
            f"L={closed.iloc[-1]['low']} C={closed.iloc[-1]['close']}")
        print(f"{'='*55}")
        zones = _build_zone_cache(closed, minwindows, maxwindows)

        return data, total_vol, zones

    # return data, total_vol


def _build_zone_cache(df, minwindows, maxwindows):
    """
    Extract only the zone boundary values at the most recent candle.
    Returns a flat list of dicts — O(k) to build, O(1) to query per tick.
    """
    zones = []
    last_idx = len(df) - 1

    for trend_data, zone_type in [(minwindows, 'support'), (maxwindows, 'resistance')]:
        for points, result in trend_data:
            if len(points) < 2:
                continue

            slope = result[0]
            intercept = result[1]
            ssr = result[2]
            n = len(points)

            if n < 3:       # need at least 3 points for n-2 denominator
                continue

            sd = math.sqrt(ssr / (n - 2)) if ssr > 0 else 0
            # min_sd = intercept * 0.003
            # sd = max(sd, min_sd)

            # Check if zone is still active up to the last candle
            is_active = True
            for i in range(min(points), last_idx + 1):
                val = slope * i + intercept
                if df['low'].iloc[i] + 3 * sd < val and zone_type == 'support':
                    is_active = False
                    break
                elif df['high'].iloc[i] - 3 * sd > val and zone_type == 'resistance':
                    is_active = False
                    break

            val_at_last = slope * last_idx + intercept
            upper = val_at_last + sd * 3
            lower = val_at_last - sd * 3

            # ── Debug print ──────────────────────────────────────
            status = "ACTIVE  ✅" if is_active else "BROKEN  ❌"
            print(f"  [{zone_type.upper():<10}] {status} | "
                  f"zone={lower:.2f}-{upper:.2f} | "
                  f"slope={slope:.4f} | points={points}")

            if not is_active:
                continue

            # Only store the boundary values at the last candle index
            # This is the single value we need for entry/exit checks
            # val_at_last = slope * last_idx + intercept
            zones.append({
                'type': zone_type,
                'upper': val_at_last + sd * 3,
                'lower': val_at_last - sd * 3,
                'slope': slope,
                'intercept': intercept,
                'sd': sd,
                'start_idx': min(points),   # trendline starts here in the closed df
                'df_length': last_idx + 1,  # how many candles were in closed df when computed
                'points': points,
            })

    return zones


# ============================================================
# Plotting the Candlestick Chart
# ============================================================
def plot_chart(candlestick):
    """
    This function handles the 'UI' (User Interface). 
    It creates the window and draws the candles.
    """
    # Create a figure with two parts: Ax (Price chart) and Ax_vol (Volume bars at bottom)
    fig, (ax, ax_vol) = plt.subplots(2, 1, figsize=(10, 6),sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    def animate(i):
        ax.clear()
        ax_vol.clear()

        data_to_plot = candlestick.data.tail(plot_window).copy()
        plot_data = data_to_plot[['open', 'high', 'low', 'close', 'volume']]

        mpf.plot(
            plot_data,
            type='candle',
            style='charles',
            ax=ax,
            volume=ax_vol,
            ylabel='Price',
        )

        n = len(plot_data)

        # Total candles in self.data (used to map zone indices to plot positions)
        # Zone indices were computed on data.iloc[:-1], so offset = total_len - 1
        total_len = len(candlestick.data)

        for zone in candlestick.active_zones:
            color = 'green' if zone['type'] == 'support' else 'red'
            slope = zone['slope']
            intercept = zone['intercept']
            sd = zone['sd']
            start_idx = zone['start_idx']   # absolute index in closed df

            upper_vals = np.full(n, np.nan)
            lower_vals = np.full(n, np.nan)

            
            for plot_pos in range(n):
                # Map plot position back to absolute candle index in self.data
                # plot_pos 0 = data_to_plot.index[0] = self.data[-n]
                abs_idx = total_len - n + plot_pos  # -1 because last is forming

                # Only draw from where the trendline starts
                if abs_idx < start_idx:
                    continue

                val = slope * abs_idx + intercept
                upper_vals[plot_pos] = val + sd * 3
                lower_vals[plot_pos] = val - sd * 3

            valid = ~np.isnan(upper_vals) & ~np.isnan(lower_vals)
            if not valid.any():
                continue

            x = np.arange(n)
            ax.fill_between(
                x,
                lower_vals,
                upper_vals,
                where=valid,
                alpha=0.15,
                color=color,
                zorder=2
            )

        ax.set_title(f'{symbol} Real-Time Candlestick Chart')
    
    # Start the animation loop
    ani = FuncAnimation(
        fig,
        animate,
        interval=1000,
        cache_frame_data=False
    )

    plt.show()


# ============================================================
# Historical Data Fetching
# ============================================================
def fetch_historical_data(fyers: fyersModel.FyersModel) -> pd.DataFrame:
    """
    Before starting the live chart, we need to know what happened earlier 
    in the day so the chart doesn't start from a blank screen.
    """
    
    now = dt.datetime.now(pytz.timezone(timeZone))

    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    previous_time = start_of_day.timestamp()
    current_time = now.timestamp()

    nifty_data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "0",
        "range_from": int(previous_time),
        "range_to": int(current_time),
        "cont_flag": "1"
    }

    # Ask Fyers for the data and convert it into an Excel-like table (DataFrame)
    response = fyers.history(data=nifty_data)
    historical_data = response['candles']
    df = pd.DataFrame(historical_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # Fix the time format so humans and computers can read it easily
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert(pytz.timezone(timeZone))
    df.set_index('date', inplace=True)
    return df


# ============================================================
# Log Data
# ============================================================
def log_trade(action, symbol, price):
    """
    Append trade details to a CSV log.

    Parameters:
        action (str): "BUY" or "SELL".
        symbol (str): Symbol.
        price (float): Execution price.
    """
    with open("trades_log.csv", "a") as file:
        file.write(f"{dt.datetime.now(pytz.timezone(timeZone))},{action},trendlines,{symbol},{price}\n")

# ============================================================
# Candlestick Class to Manage State and WebSocket Callbacks
# ============================================================
class Candlestick:
    """
    This class acts like a real-time market data manager.

    Responsibilities:
    - Receives live market ticks from the websocket.
    - Updates candles continuously.
    - Stores latest market state in memory.
    - Maintains live support/resistance structures.

    Think of this class as the "brain"
    controlling the live chart.
    """
    # ============================================================
    # Initialization
    # ============================================================
    def __init__(self, data, fyers):
        """
        Initializes the Candlestick class.
        """
        self.data = data
        self.last_total_volume = None
        self.position = None
        self.sl = None
        self.tp = None
        self.fyers = fyers
        self.trigger = None
        self.last_evaluated_candle = None
        self.active_zones = []


    # ============================================================
    # Cleaner function
    # ============================================================    
    def _clear_position(self):
        self.position = None
        self.sl = None
        self.tp = None
        self.trigger = None


    # ============================================================
    # WebSocket Callback to Handle Incoming Messages
    # ============================================================
    def onmessage(self, message):
        """
        Callback function to handle incoming messages from the FyersDataSocket WebSocket.

        Parameters:
            message (dict): The received message from the WebSocket.

        """
        self.data, self.last_total_volume, new_zones = update_live_data(data=self.data, message=message, last_total_volume=self.last_total_volume)
        print(self.data.tail())

        # --- Guard Conditions ---
        if new_zones is not None:
            self.active_zones = new_zones

        if len(self.data) < 2:
            return
        
        ltp = message.get('ltp')
        if ltp is None:
            return

        # --- Last Closed Candle ---
        closed_candle = self.data.iloc[-2]
        closed_candle_time = self.data.index[-2]

        # Always get fresh zone data from last closed candle
        # (zones are recomputed on every candle close, so this
        #  extract all active support/resistance column names present)
        # zone_cols = [c for c in self.data.columns
        #             if (c.startswith('support:') or c.startswith('resistance:'))
        #             and not pd.isna(closed_candle[c])]

        # --------------------------------------------------------
        # Buy/Sell LOGIC (Evaluates on the last closed candle)
        # --------------------------------------------------------
        if not self.position and closed_candle_time != self.last_evaluated_candle:
            self.last_evaluated_candle = closed_candle_time

            for zone in self.active_zones:
                # --- LONG ENTRY (Price respects Support) ---
                if zone['type'] == 'support':
                    if zone['lower'] <= closed_candle['low'] <= zone['upper'] \
                            and closed_candle['close'] > zone['upper']:
                        ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                        log_trade("Buy", symbol, ask)
                        self.sl = zone['lower']
                        self.tp = ask + (ask - self.sl) * 2
                        self.trigger = closed_candle['high']
                        self.position = 'LONG'
                        print(f"[ENTRY LONG] Zone {zone['lower']:.2f}-{zone['upper']:.2f}, ask={ask}")
                        break  # Exit loop once a signal is taken

                # --- SHORT ENTRY (Price respects Resistance) ---
                elif zone['type'] == 'resistance':
                    if zone['lower'] <= closed_candle['high'] <= zone['upper'] \
                            and closed_candle['close'] < zone['lower']:
                        bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                        log_trade("Sell", symbol, bid)
                        self.sl = zone['upper']
                        self.tp = bid - (self.sl - bid) * 2
                        self.trigger = closed_candle['low']
                        self.position = 'SHORT'
                        print(f"[ENTRY SHORT] Zone {zone['lower']:.2f}-{zone['upper']:.2f}, bid={bid}")
                        break  # Exit loop once a signal is taken


        # --------------------------------------------------------
        # Exit LOGIC (Evaluates on the last closed candle)
        # --------------------------------------------------------
        if self.position == 'LONG':

            # --- Condition 1: Hard TP/SL hit ---
            if ltp >= self.tp or ltp < self.sl:
                print(f"[EXIT LONG] Hard TP/SL hit at {ltp}")
                bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                log_trade("Sell", symbol, bid)
                self._clear_position()

            else:
                # --- Condition 2: Price entering a resistance zone ---
                # If ltp is inside any active resistance band, exit -
                # resistance overhead is a structural reason to close long
                for zone in self.active_zones:
                    if zone['type'] == 'resistance' and zone['lower'] <= ltp <= zone['upper']:
                        bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                        log_trade("Sell", symbol, bid)
                        print(f"[EXIT LONG] Entered resistance {zone['lower']:.2f}-{zone['upper']:.2f}")
                        self._clear_position()
                        break

                # --- Condition 3: Price breaking down through support ---
                # If ltp breaks BELOW a support zone that sits ABOVE sl,
                # that support has failed - exit before hitting hard SL.
                # Only check if still in position (condition 2 may have exited)
                if self.position == 'LONG':
                    for zone in self.active_zones:
                        if zone['type'] == 'support' \
                                and zone['lower'] > self.sl \
                                and ltp < zone['lower']:
                            bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                            log_trade("Sell", symbol, bid)
                            print(f"[EXIT LONG] Support broken {zone['lower']:.2f}")
                            self._clear_position()
                            break

                # --- Trailing SL/TP (only if still in position) ---
                if self.position == 'LONG' and closed_candle['close'] > self.trigger:
                    new_sl = closed_candle['low']
                    if new_sl > self.sl:
                        diff = new_sl - self.sl
                        self.tp += diff
                        self.sl = new_sl
                        self.trigger = closed_candle['high']

        elif self.position == 'SHORT':

            # --- Condition 1: Hard TP/SL hit ---
            if ltp <= self.tp or ltp > self.sl:
                print(f"[EXIT SHORT] Hard TP/SL hit at {ltp}")
                ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                log_trade("Buy", symbol, ask)
                self._clear_position()

            else:
                # --- Condition 2: Price entering a support zone ---
                # Support below price is a structural reason to close short
                for zone in self.active_zones:
                    if zone['type'] == 'support' and zone['lower'] <= ltp <= zone['upper']:
                        ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                        log_trade("Buy", symbol, ask)
                        print(f"[EXIT SHORT] Entered support {zone['lower']:.2f}-{zone['upper']:.2f}")
                        self._clear_position()
                        break

                # --- Condition 3: Price breaking up through resistance ---
                if self.position == 'SHORT':
                    for zone in self.active_zones:
                        if zone['type'] == 'resistance' \
                                and zone['upper'] < self.sl \
                                and ltp > zone['upper']:
                            ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                            log_trade("Buy", symbol, ask)
                            print(f"[EXIT SHORT] Resistance broken {zone['upper']:.2f}")
                            self._clear_position()
                            break

                # --- Trailing SL/TP ---
                if self.position == 'SHORT' and closed_candle['close'] < self.trigger:
                    new_sl = closed_candle['high']
                    if new_sl < self.sl:
                        diff = self.sl - new_sl
                        self.tp -= diff
                        self.sl = new_sl
                        self.trigger = closed_candle['low']


    # ============================================================
    # Websocket Callback to Handle Errors Events
    # ============================================================
    def onerror(self, message):
        """
        Callback function to handle WebSocket errors.

        Parameters:
            message (dict): The error message received from the WebSocket.


        """
        print("Error:", message)


    # ============================================================
    # Websocket Callback to Handle Connection Close Events
    # ============================================================
    def onclose(self, message):
        """
        Callback function to handle WebSocket connection close events.
        """
        print("Connection closed:", message)


    # ============================================================
    # Websocket Callback to Handle Subscription Upon Connection
    # ============================================================
    def onopen(self):
        """
        Callback function to subscribe to data type and symbols upon WebSocket connection.

        """
        # Specify the data type and symbols you want to subscribe to
        data_type = "SymbolUpdate"

        # Subscribe to the specified symbols and data type
        symbols = [symbol]
        fyersSocket.subscribe(symbols=symbols, data_type=data_type)

        # Keep the socket running to receive real-time data
        fyersSocket.keep_running()


# ============================================================
# STARTING THE PROGRAM
# ============================================================
if __name__ == "__main__":
    '''
    ============================================================
    HOW THE SYSTEM WORKS
    ============================================================
    
    WebSocket Tick
           ↓
    Update Current Candle
           ↓
    Candle Closes
           ↓
    Detect Support & Resistance
           ↓
    Validate Active Trendlines
           ↓
    Execute Entry/Exit
    
    ============================================================
    '''
    
    # 1. Read your secret keys (Like a password for the stock market)
    try:
        with open('access_token.txt', 'r') as file:
            access_token = file.read()
    except FileNotFoundError:
        print("Error: access_token.txt not found! Please login first.")
        exit()

    # 2. Fetch data from earlier today
    print("Fetching morning data...")
    fyers_connection = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path='')
    historical_df = fetch_historical_data(fyers=fyers_connection)

    # 3. Initialize our data manager
    candlestick = Candlestick(historical_df, fyers_connection)

    # 4. Create a FyersDataSocket instance with the provided parameters
    fyersSocket = data_ws.FyersDataSocket(
        access_token=access_token,          # Access token in the format "appid:accesstoken"
        log_path="",                        # Path to save logs. Leave empty to auto-create logs in the current directory.
        litemode=False,                     # Lite mode disabled. Set to True if you want a lite response.
        write_to_file=False,                # Save response in a log file instead of printing it.
        reconnect=True,                     # Enable auto-reconnection to WebSocket on disconnection.
        on_connect=candlestick.onopen,      # Callback function to subscribe to data upon connection.
        on_close=candlestick.onclose,       # Callback function to handle WebSocket connection close events.
        on_error=candlestick.onerror,       # Callback function to handle WebSocket errors.
        on_message=candlestick.onmessage    # Callback function to handle incoming messages from the WebSocket.
    )

    # 5. Establish a connection to the Fyers WebSocket
    print("Connecting to live stream...")
    fyersSocket.connect()

    # 6. Plot the chart
    try:
        plot_chart(candlestick)
    finally:
        fyersSocket.close_connection()

# df = [
#     2026-06-01 13:52:02.004667+05:30,Buy,NSE:RELIANCE-EQ,1321
# 2026-06-01 13:52:06.475871+05:30,Sell,NSE:RELIANCE-EQ,1321.6
# ]
