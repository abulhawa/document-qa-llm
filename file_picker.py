import tkinter as tk
from tkinter import filedialog
import os
import sys
import json
from typing import List


def pick_files() -> List[str]:
    """
    Open a file dialog to let the user select one or more document files.
    Returns a list of file paths.
    """
    file_paths = filedialog.askopenfilenames(
        title="Select PDF, DOCX, or TXT File(s)",
        filetypes=[("Documents", "*.pdf *.docx *.txt")]
    )
    return list(file_paths)


def pick_folder() -> List[str]:
    """
    Open a folder dialog and recursively collect all document files in it.
    Returns a list of full paths to .pdf, .docx, or .txt files.
    """
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return []

    matched_files: List[str] = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith((".pdf", ".docx", ".txt")):
                full_path = os.path.join(dirpath, filename)
                matched_files.append(full_path)

    return matched_files


def pick_root() -> List[str]:
    """
    Open a folder dialog and return the selected root path.
    """
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return []
    return [folder_path]


if __name__ == "__main__":
    # Hide the root window (we only want the file/folder dialog)
    root = tk.Tk()
    root.withdraw()

    # Mode is passed via CLI: "files" or "folder"
    mode = sys.argv[1] if len(sys.argv) > 1 else "files"

    try:
        if mode == "root":
            result = pick_root()
        elif mode == "folder":
            result = pick_folder()
        else:
            result = pick_files()
        print(json.dumps(result))
    except Exception as e:
        print("[]")  # Return empty list to avoid crashing the parent process
