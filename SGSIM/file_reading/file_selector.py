import tkinter as tk
import os
import pandas as pd

def open_files():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = tk.filedialog.askopenfilenames(initialdir=os.getcwd(), title='Select Files', parent=root)
    root.destroy()
    return file_path

def open_folder():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = tk.filedialog.askdirectory(initialdir=os.getcwd(), title='Select a Folder', parent=root)
    root.destroy()
    return folder_path

def csv_reader():
    files = open_files()
    dfs = [pd.read_csv(file) for file in files]
    return dfs
