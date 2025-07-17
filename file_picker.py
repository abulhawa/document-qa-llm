import tkinter as tk
from tkinter import filedialog
import os
import sys
import json

def pick():
    root = tk.Tk()
    root.withdraw()

    # Try picking files first
    file_paths = filedialog.askopenfilenames(
        title="Select Files or Cancel to Pick Folder",
        filetypes=[("Documents", "*.pdf *.docx *.txt")]
    )

    if file_paths:
        print(json.dumps(list(file_paths)))
        return

    # Otherwise fallback to folder
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        print("[]")
        return

    matched = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith((".pdf", ".docx", ".txt")):
                matched.append(os.path.join(dirpath, filename))

    print(json.dumps(matched))

if __name__ == "__main__":
    pick()