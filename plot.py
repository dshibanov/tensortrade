# all analitical plot functionality is here

import sys, os, time
# from tensortrade.env.generic.multy_symbol_env import *
from tensortrade.env.generic.environment import *
import tensortrade.env.default as default

from pprint import pprint
import finplot as fplt
from PyQt6.QtWidgets import QApplication, QGridLayout, QMainWindow, QGraphicsView, QComboBox, QLabel
from pyqtgraph.dockarea import DockArea, Dock
from functools import lru_cache
from threading import Thread
import yfinance as yf
import subprocess
import shlex

BG_LIGHT='light'
BG_DARK='dark'


# PROVIDERS =[consts.BINANCE]
QUOTES = os.getenv('QUOTES')

def get_quotes(symbol, timeframe, provider ='binance'): #, use_modin=False):
    quotes = pd.read_csv(f'{QUOTES}/{provider}/{symbol}/{timeframe}.csv', index_col='date')
    quotes.index = pd.DatetimeIndex(quotes.index)
    return quotes

def set_gruvbox(bg=BG_LIGHT):
    fplt.legend_border_color = '#777'
    fplt.legend_fill_color   = '#3c3836' if bg == BG_DARK else '#3c3836'
    # legend_text_color   = '#ddd6'
    fplt.legend_text_color   = '#fabd2f'
    # soft_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # hard_colors = ['#000000', '#772211', '#000066', '#555555', '#0022cc', '#ffcc00']
    # colmap_clash = ColorMap([0.0, 0.2, 0.6, 1.0], [[127, 127, 255, 51], [0, 0, 127, 51], [255, 51, 102, 51], [255, 178, 76, 51]])
    # foreground = '#000'
    fplt.background = '#282828' if bg == BG_DARK else '#ebdbb2'
    # fplt.odd_plot_background = '#504945'if bg == BG_DARK else '#669'
    fplt.odd_plot_background = '#3c3836'if bg == BG_DARK else '#ebdbb2'

    # candle_bull_color = '#26a69a'
    # candle_bear_color = '#ef5350'
    # candle_bull_body_color = background
    # volume_bull_color = '#92d2cc'
    # volume_bear_color = '#f7a9a7'
    # volume_bull_body_color = volume_bull_color
    # volume_neutral_color = '#bbb'
    # poc_color = '#006'
    # band_color = '#d2dfe6'
    # cross_hair_color = '#0007'
    # draw_line_color = '#000'
    # draw_done_color = '#555'


# def plot_history(series, trades):
def plot_history(track, trades, save_track=False, suffix="", bg="light", title='', info=''):


    _trades=[]
    for t in trades:
        trade = trades[t]
        _trades.append(trade)
        step = trade[0].step

        if trade[0].side.value == 'buy':
            track.loc[(track.index == trade[0].step), 'buy'] = trade[0].price
        else:
            track.loc[(track.index == trade[0].step), 'sell'] = trade[0].price

    track["buy"].iloc[0:len(track)-1] = track["buy"].iloc[1:len(track)]
    track["sell"].iloc[0:len(track)-1] = track["sell"].iloc[1:len(track)]

    set_gruvbox(bg=bg)
    # set_gruvbox()


    # print("2:")
    # pprint(track)
    # return
    if info != '':
        header, ax1, ax2, ax3, ax4, ax5, ax6 = fplt.create_plot(title ='test plot', rows=7)
        header.grid = False
        fplt.add_legend(f"{info}", ax=header)
    else:
        ax1, ax2, ax3, ax4, ax5, ax6 = fplt.create_plot(title ='test plot', rows=6)

    ax1.grid = False

    track.plot('net_worth', ax=ax2, legend='net_worth', title='some title')
    track.plot('close', ax=ax1, legend='close', linestyle='solid', width=1.5, color='black', grid=False)
    track.plot('reward', ax=ax3, legend='reward')
    track.plot('action', ax=ax4, legend='action')

    # line styles
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    track.plot('end_of_episode', ax=ax5, legend='end_of_episode', color='#cc241d')
    track.plot('symbol_code', ax=ax6, legend='symbol_code', color='#cc241d')
    # ax1.set_visible(xgrid=True, ygrid=True)
    ax1.set_visible(xgrid=False, ygrid=False)

    fplt.add_band((30), (35), color='#076678', ax=ax3)
    fplt.add_rect((100,100), (200,200), color='#cc241d', interactive=False, ax=None)
    fplt.add_text((150,150), 'texttexttext', color='#076678', anchor=(0,0), ax=ax1)

    # make info string here
    # print lines between open and close
    _open = False
    i = 0
    import math
    trades=[]
    last_order_side = ''
    # for index, row in df.iterrows():

    print(" 2 ")
    pprint(track)
    for index, row in track.iterrows():
        if math.isnan(row["buy"]) == False and math.isnan(row["sell"]) == False:
            print("ERROR: simultanious bus/sell .. at #", index)

        if math.isnan(row["buy"]) == False :
            trades.append({"time":index, "price":row["buy"]})
            last_order_side = 'buy'

        if math.isnan(row["sell"]) == False:
            trades.append({"time":index, "price":row["sell"]})
            last_order_side = 'sell'

    if last_order_side == 'buy':
        track["sell"].iloc[-1] = track["close"].iloc[-1]
        trades.append({"time":index, "price":track["close"].iloc[-1]})

    fplt.plot(track['buy'], ax=ax1, color='#076678', style='>', legend='buy', width=2)
    fplt.plot(track['sell'], ax=ax1, color='#cc241d', style='<', legend='sell', width=2)
    while i+1 < len(trades):
        # fplt.add_line((trades[i]["time"], trades[i]["price"]), (trades[i+1]["time"], trades[i+1]["price"]), color='#3c3836', interactive=False, ax=ax1, style='_')
        fplt.add_line((trades[i]["time"], trades[i]["price"]), (trades[i+1]["time"], trades[i+1]["price"]), color='#e3242b', interactive=False, ax=ax1, style='_')
        i += 2

    fplt.show()

    if save_track:
        track.to_csv(f'test_track_{suffix}.csv', index=False)

# def make_symbols(num_symbols=5, length=666):
#     symbols=[]
#     for i in range(num_symbols):
#         spread = 0.01
#         commission=0
#         if i == 2:
#             commission=0
#             spread=1.13
#         elif i == 4:
#             commission=0
#             spread=3.66
#         symbols.append(default.make_sin_symbol("AST"+str(i), i, commission=commission, spread=spread, length=length))
#     return symbols


def make_symbols(num_symbols=5, length=666):
    symbols=[]
    for i in range(num_symbols):
        spread = 0.01
        commission=0
        if i == 2:
            commission=0
            spread=1.13
        elif i == 4:
            commission=0
            spread=3.66
        # symbols.append(default.make_sin_symbol("AST"+str(i), i, commission=commission, spread=spread, length=length))
        #

        config = {"name": "AST"+str(i),
              "spread": spread,
              "commission": commission,
              "code": i,
              "length": length,
              "max_episode_steps": 11,
              # "max_episode_steps": 152,
              # "process": 'flat',
              "process": 'sin',
              "price_value": 100}
        # symbols.append(default.make_synthetic_symbol("AST"+str(i), i, commission=commission, spread=spread, length=length))
        symbols.append(default.make_synthetic_symbol(config))
    return symbols


def test_plot_history():

    track = pd.read_csv('test_track2.csv')
    # pprint(track)
    # return
    symbols = make_symbols()
    config = {"symbols": symbols,
              "reward_window_size": 7,
              "window_size": 1,
              "max_allowed_loss":100,
              "multy_symbol_env": True
             }

    # print(track["buy"])
    # print(track["buy"].iloc[1:len(track)])
    # track["buy"].iloc[0:len(track)-1] = track["buy"].iloc[1:len(track)]
    # track["sell"].iloc[0:len(track)-1] = track["sell"].iloc[1:len(track)]
    # print("good")
    # print(track["buy"])
    # return

    dataset = pd.concat([config["symbols"][i]["feed"] for i in range(len(config["symbols"]))])
    # history = {"series":dataset,
    # history = {"series":track,
               # "trades":[]}

    print('track')
    pprint(track)
    plot_history(track)

def test_layout32_up_to_down():

    symbol = 'LAZIOBTC'
    timeframes=['1w', '1d', '4h', '1h', '15m']
    screens={}
    screens_quotes=[]
    for t in timeframes:
        print(t)
        screens[t] = get_quotes(symbol, t)
        screens_quotes.append(screens[t])
    pprint(screens)

    app = QApplication([])
    win = QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1600,800)
    win.setWindowTitle("Docking charts example for finplot")

    # Set width/height of QSplitter
    win.setStyleSheet("QSplitter { width : 20px; height : 20px; }")

    dock_0 = Dock("1w", size = (100, 100), closable = True)
    dock_1 = Dock("1d", size = (100, 100), closable = True)
    dock_2 = Dock("4h", size = (100, 100), closable = True)
    dock_3 = Dock("1h", size = (100, 100), closable = True)
    dock_4 = Dock("15m", size = (100, 100), closable = True)

    area.addDock(dock_0, position='left')
    area.addDock(dock_1, position='right')
    area.addDock(dock_2, position='right')
    area.addDock(dock_3, position='bottom')
    area.addDock(dock_4, 'right', dock_3)

    # Create example charts
    combo = QComboBox()
    combo.setEditable(True)
    [combo.addItem(i) for i in 'AMRK META REVG TSLA TWTR WMT CT=F GC=F ^GSPC ^FTSE ^N225 EURUSD=X ETH-USD'.split()]
    info = QLabel()


    docks = [dock_0, dock_1, dock_2, dock_3, dock_4]
    axes=[]
    for i in range(len(docks)):
        ax0 = fplt.create_plot_widget(master=area, rows=1, init_zoom_periods=100)
        docks[i].addWidget(ax0.ax_widget, 1, 0, 1, 2)
        print(i)
        df = screens_quotes[i]
        print(df.columns)
        price = df["open close high low".split()]
        print(price)
        volume = df ["open close volume".split()]
        fplt.candlestick_ochl(price, ax = ax0)
        ax0.decouple()
        print(type(ax0))
        axes.append(ax0) #= [ax0, ax1, ax2, ax3, ax4]

    area.axs = axes
    fplt.show(qt_exec = False) # prepares plots when they're all setup
    win.show()
    app.exec()
    return
    # Chart for dock_0
    # ax0,ax1,ax2 = fplt.create_plot_widget(master=area, rows=3, init_zoom_periods=100)
    ax0,ax1,ax2, ax3, ax4 = fplt.create_plot_widget(master=area, rows=5, init_zoom_periods=100)

    # print(type(ax0))
    # return
    # area.axs = [ax0, ax1, ax2]
    area.axs = [ax0, ax1, ax2, ax3, ax4]
    dock_0.addWidget(ax0.ax_widget, 1, 0, 1, 2)
    dock_0.addWidget(ax1.ax_widget, 2, 0, 1, 2)
    dock_0.addWidget(ax2.ax_widget, 3, 0, 1, 2)

    # dock_1.addWidget(ax2.ax_widget, 1, 0, 1, 2)
    dock_2.addWidget(ax3.ax_widget, 1, 0, 1, 2)
    dock_2.addWidget(ax4.ax_widget, 1, 0, 1, 2)

    # Link x-axis
    ax1.setXLink(ax0)
    ax2.setXLink(ax0)
    # win.axs = [ax0,ax1,ax2]

    @lru_cache(maxsize = 15)
    def download(symbol):
        return yf.download(symbol, "2020-01-01")

    @lru_cache(maxsize = 100)
    def get_name(symbol):
        return yf.Ticker(symbol).info.get("shortName") or symbol

    def update(txt):
        df = download(txt)
        if len(df) < 20: # symbol does not exist
            return
        info.setText("Loading symbol name...")
        price = df ["Open Close High Low".split()]
        ma20 = df.Close.rolling(20).mean()
        ma50 = df.Close.rolling(50).mean()
        volume = df ["Open Close Volume".split()]
        ax0.reset() # remove previous plots
        ax1.reset() # remove previous plots
        ax2.reset() # remove previous plots
        fplt.candlestick_ochl(price, ax = ax0)
        fplt.plot(ma20, legend = "MA-20", ax = ax1)
        fplt.plot(ma50, legend = "MA-50", ax = ax1)
        fplt.volume_ocv(volume, ax = ax2)
        fplt.refresh() # refresh autoscaling when all plots complete
        Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

    combo.currentTextChanged.connect(update)
    update(combo.currentText())

    fplt.show(qt_exec = False) # prepares plots when they're all setup
    win.show()
    app.exec()


def test_layout32_left_to_right():

    app = QApplication([])
    win = QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1600,800)
    win.setWindowTitle("Docking charts example for finplot")

    # Set width/height of QSplitter
    win.setStyleSheet("QSplitter { width : 20px; height : 20px; }")

    dock_0 = Dock("dock_0", size = (100, 100), closable = True)
    dock_1 = Dock("dock_1", size = (100, 100), closable = True)
    dock_2 = Dock("dock_2", size = (100, 100), closable = True)
    dock_3 = Dock("dock_3", size = (100, 100), closable = True)
    dock_4 = Dock("dock_4", size = (100, 100), closable = True)
    # area.addDock(dock_0, position='left')
    # area.addDock(dock_1, position='right')
    # area.addDock(dock_2, position='bottom')

# area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
# area.addDock(d2, 'right')     ## place d2 at right edge of dock area
# area.addDock(d3, 'bottom', d1)## place d3 at bottom edge of d1
# area.addDock(d4, 'right')     ## place d4 at right edge of dock area
# area.addDock(d5, 'left', d1)  ## place d5 at left edge of d1
# area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4

    area.addDock(dock_0, position='left')
    # area.addDock(dock_1, position='left')
    area.addDock(dock_1, 'bottom', dock_0)
    area.addDock(dock_2, 'bottom', dock_1)
    # area.addDock(dock_2, position='right')
    # area.addDock(dock_3, position='bottom')
    area.addDock(dock_3, 'right')
    area.addDock(dock_4, 'bottom', dock_3)
    # area.addDock(dock_4, position='bottom')
    # Create example charts
    combo = QComboBox()
    combo.setEditable(True)
    [combo.addItem(i) for i in 'AMRK META REVG TSLA TWTR WMT CT=F GC=F ^GSPC ^FTSE ^N225 EURUSD=X ETH-USD'.split()]
    # dock_0.addWidget(combo, 0, 0, 1, 1)
    info = QLabel()
    # dock_0.addWidget(info, 0, 1, 1, 1)

    # Chart for dock_0
    # ax0,ax1,ax2 = fplt.create_plot_widget(master=area, rows=3, init_zoom_periods=100)
    ax0,ax1,ax2, ax3, ax4 = fplt.create_plot_widget(master=area, rows=5, init_zoom_periods=100)

    # print(type(ax0))
    # return
    # area.axs = [ax0, ax1, ax2]
    area.axs = [ax0, ax1, ax2, ax3, ax4]
    dock_0.addWidget(ax0.ax_widget, 1, 0, 1, 2)
    dock_0.addWidget(ax1.ax_widget, 2, 0, 1, 2)
    dock_0.addWidget(ax2.ax_widget, 3, 0, 1, 2)

    # dock_1.addWidget(ax2.ax_widget, 1, 0, 1, 2)
    dock_2.addWidget(ax3.ax_widget, 1, 0, 1, 2)
    dock_2.addWidget(ax4.ax_widget, 1, 0, 1, 2)

    # Link x-axis
    ax1.setXLink(ax0)
    ax2.setXLink(ax0)
    # win.axs = [ax0,ax1,ax2]

    @lru_cache(maxsize = 15)
    def download(symbol):
        return yf.download(symbol, "2020-01-01")

    @lru_cache(maxsize = 100)
    def get_name(symbol):
        return yf.Ticker(symbol).info.get("shortName") or symbol

    def update(txt):
        df = download(txt)
        if len(df) < 20: # symbol does not exist
            return
        info.setText("Loading symbol name...")
        price = df ["Open Close High Low".split()]
        ma20 = df.Close.rolling(20).mean()
        ma50 = df.Close.rolling(50).mean()
        volume = df ["Open Close Volume".split()]
        ax0.reset() # remove previous plots
        ax1.reset() # remove previous plots
        ax2.reset() # remove previous plots
        fplt.candlestick_ochl(price, ax = ax0)
        fplt.plot(ma20, legend = "MA-20", ax = ax1)
        fplt.plot(ma50, legend = "MA-50", ax = ax1)
        fplt.volume_ocv(volume, ax = ax2)
        fplt.refresh() # refresh autoscaling when all plots complete
        Thread(target=lambda: info.setText(get_name(txt))).start() # slow, so use thread

    combo.currentTextChanged.connect(update)
    update(combo.currentText())

    fplt.show(qt_exec = False) # prepares plots when they're all setup
    win.show()
    app.exec()

def get_last_hash():
    return subprocess.run(shlex.split("git rev-parse --short HEAD"), check=True, stdout=subprocess.PIPE).stdout.decode("utf-8").rstrip()

if __name__ == "__main__":
    print(get_last_hash())
    # test_plot_history()
    # test_layout32_up_to_down()
    # test_layout32_left_to_right()

    #import matplotlib.pyplot as plt
    #import numpy as np

    ##define x and y values
    #x = np.linspace(0, 10, 100)
    #y1 = np.sin(x)*np.exp(-x/3)
    #y2 = np.cos(x)*np.exp(-x/5)

    ##create line plot with multiple lines
    #plt.plot(x, y1, linewidth=3)
    #plt.plot(x, y2, linewidth=1)

    ##display plot
    #plt.show()





