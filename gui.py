import tkinter as tk
from tkinter import ttk

from video_handler import HandsMatcher, HandsCapture, HandsDrawer
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
matplotlib.use("TkAgg")
style.use('bmh')


class GUI:

    def __init__(self, master):
        self.master = master
        master.title("White Board")
        master.geometry("800x600")
        self.vh = None
        self.hc = None
        self.hd = None
        self.master.protocol("WM_DELETE_WINDOW", self.master.quit)
        self.scalar = 1
        self.popup_return = ''
        self.menubar = tk.Menu(self.master)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.operations_menu = tk.Menu(self.menubar, tearoff=0)
        self.init_menubar()
        self.master.config(menu=self.menubar)
        self.path = ''

    def init_menubar(self):
        self.file_menu.add_command(
            label="Capture Hands", command=self.on_capture_hands)

        self.file_menu.add_command(
            label="Match Hands", command=self.on_match_hands)
        self.file_menu.add_command(
            label="Draw with hand", command=self.on_draw)

        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.master.quit)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.menubar.add_cascade(label="Operations", menu=self.operations_menu)

    def on_capture_hands(self):
        self.hc = HandsCapture(0, 'sift', 'Hands Capture')
        self.hc.start()

    def on_match_hands(self):
        self.vh = HandsMatcher(0, 'sift', 'Hands Matcher')
        self.vh.start()

    def on_draw(self):
        self.hd = HandsDrawer(0, (80, 80), 'Draw With Hands', './model')
        self.hd.start()


root = tk.Tk()
gui = GUI(root)

root.mainloop()
