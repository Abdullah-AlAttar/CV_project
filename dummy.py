import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

import cv2


def read_image(path, BGR_to_RGB=False, gray_scale=False):

    img = cv2.imread(path, 0) if gray_scale else cv2.imread(path)

    if BGR_to_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cvimg_to_PIL(img):
    pil_img = Image.fromarray(img)

    pil_img = ImageTk.PhotoImage(pil_img)
    return pil_img


class GUI:

    def __init__(self, master):
        self.master = master
        master.title("Computer Vision")
        master.geometry("1600x900")

        # layout

    def popupmsg(self, msg):
        popup = tk.Tk()
        popup.wm_title("")
        input = ttk.Entry(popup)

        def disable_event():
            pass
        popup.protocol("WM_DELETE_WINDOW", disable_event)

        def on_press():
            self.D_val = int(input.get())
            popup.destroy()
            self.master.quit()

        label = ttk.Label(popup, text=msg)
        label.pack(side="top", fill="x", padx=12)
        b = ttk.Button(popup, text="Submit", command=on_press)
        input.pack()
        b.pack(side='bottom')
        popup.mainloop()

    def init_layout(self):

        self.panel1.grid(row=3, column=0, padx=8, pady=8, ipadx=8, ipady=8)
        self.panel2.grid(row=3, column=1, padx=8, pady=8, ipadx=8, ipady=8)


if __name__ == "__main__":
    root = tk.Tk()

    gui = GUI(root)

    root.mainloop()

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cvimg_to_PIL(img)
        gui.panel2_img = cvimg_to_PIL(res)
        gui.panel2.configure(image=self.panel2_img)
