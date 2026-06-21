import tkinter as tk 
from tkinter import messagebox
import subprocess
import os
import shutil

def delete_user():
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("Missing Name", "Please enter the name to delete.")
        return

    user_dir = os.path.join("dataset", name)

    if not os.path.exists(user_dir):
        log_output.insert(tk.END, f"No folder found for user '{name}'.\n")
        return

    confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete data for '{name}'?")
    if confirm:
        shutil.rmtree(user_dir)
        log_output.insert(tk.END, f"Deleted registered face data for '{name}'.\n")
    else:
        log_output.insert(tk.END, "Deletion cancelled.\n")

# -----------------------------
# Function to clear the log area
# -----------------------------
def clear_logs():
    log_output.delete('1.0', tk.END)

# -----------------------------
# Step 1: Register Faces
# -----------------------------
def register_faces():
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("Missing Name", "Please enter a name before registering.")
        return
    try:
        clear_logs()
        log_output.insert(tk.END, f"Registering face for '{name}'...\n")
        subprocess.run(["python", "new/register_faces.py", name])
        log_output.insert(tk.END, f"Face registration completed for '{name}'.\n")
    except Exception as e:
        log_output.insert(tk.END, f"Error: {str(e)}\n")

# -----------------------------
# Step 2: Train Model
# -----------------------------
def train_model():
    try:
        clear_logs()
        log_output.insert(tk.END, "Training model...\n")
        subprocess.run(["python", "new/train_model.py"])
        log_output.insert(tk.END, "Model training completed.\n")
    except Exception as e:
        log_output.insert(tk.END, f"Error: {str(e)}\n")

# -----------------------------
# Step 3: Take Attendance
# -----------------------------
def take_attendance():
    try:
        clear_logs()
        log_output.insert(tk.END, "Starting attendance system...\n")
        subprocess.run(["python", "new/attendance.py"])
        log_output.insert(tk.END, "Attendance process completed.\n")
    except Exception as e:
        log_output.insert(tk.END, f"Error: {str(e)}\n")

# -----------------------------
# GUI Layout
# -----------------------------
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("480x500")
window.config(bg="#080808")

# Title
tk.Label(window, text="Face Attendance System", font=("Helvetica", 18, "bold"), bg="#0f0f0f").pack(pady=20)

# Input for name
tk.Label(window, text="Enter Name:", bg="#0d0e0e", font=("Helvetica", 12)).pack()
name_entry = tk.Entry(window, font=("Helvetica", 12), width=30)
name_entry.pack(pady=8)

# Buttons for each step
tk.Button(window, text="1. Register Faces", command=register_faces,
          width=30, height=2, bg="#4f46e5", fg="white", font=("Helvetica", 12)).pack(pady=10)

tk.Button(window, text="2. Train Model", command=train_model,
          width=30, height=2, bg="#2563eb", fg="white", font=("Helvetica", 12)).pack(pady=10)

tk.Button(window, text="3. Take Attendance", command=take_attendance,
          width=30, height=2, bg="#10b981", fg="white", font=("Helvetica", 12)).pack(pady=10)

# Log output box
tk.Label(window, text="System Logs:", bg="#f0f4f8", font=("Helvetica", 12)).pack(pady=(20, 5))
log_output = tk.Text(window, height=8, width=55, bg="#e5e7eb", font=("Courier", 10))
log_output.pack()

# Clear log button
tk.Button(window, text="Clear Logs", command=clear_logs,
          bg="gray", fg="white", font=("Helvetica", 10)).pack(pady=10)

# Start the GUI
window.mainloop()

