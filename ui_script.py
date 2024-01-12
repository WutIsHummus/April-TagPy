import tkinter as tk
from tkinter import ttk
import threading

driveMap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
selection = "horizontal"

def update_drive_map(new_map):
    global driveMap
    driveMap = new_map
def get_drive_map():
    return driveMap
def create_ui():
    window = tk.Tk()
    window.title("AprilTag Pattern Selector")

    patterns = {
        "horizontal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "spiral_inward": [0, 1, 2, 3, 8, 9, 10, 11, 6, 5, 4, 7],
        "vertical": [0, 5, 6, 11, 10, 7, 4, 1, 2, 3, 8, 9],
    }

    for pattern_name, pattern_map in patterns.items():
        button = ttk.Button(window, text=pattern_name, 
                            command=lambda map=pattern_map: update_drive_map(map))
        button.pack(pady=5)

    window.mainloop()

def start_ui_thread():
    ui_thread = threading.Thread(target=create_ui)
    ui_thread.start()
