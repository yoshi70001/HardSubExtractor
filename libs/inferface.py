import os
import json
from tkinter import Tk, Label, Button, filedialog, messagebox, Menu, Toplevel, Entry
from tkinter import ttk  # Importar ttk para la barra de progreso
from libs.videoProcessor import  process_video
from datetime import datetime

# Ruta del archivo JSON para guardar los datos
CONFIG_FILE = "config.json"



# Función para seleccionar una carpeta y procesar los videos
def select_folder():
    folder_path = filedialog.askdirectory(title="Selecciona una carpeta con videos")
    if folder_path:
        # Obtener la lista de archivos de video en la carpeta
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            messagebox.showinfo("Info", "No se encontraron videos en la carpeta seleccionada.")
            return

        total_videos = len(video_files)
        progress_bar['maximum'] = total_videos
        progress_bar['value'] = 0
        progress_label.config(text="0% completado")

        # Procesar cada video
        for i, video_file in enumerate(video_files, start=1):
            video_path = os.path.join(folder_path, video_file)
            print(f"Procesando video: {video_file}")
            print(datetime.now())
            process_video(video_path)
            print(datetime.now())
            # Actualizar la barra de progreso y el label
            progress_bar['value'] = i
            progress_percentage = int((i / total_videos) * 100)
            progress_label.config(text=f"{progress_percentage}% completado")
            root.update_idletasks()  # Actualizar la interfaz gráfica

        messagebox.showinfo("Completado", "Procesamiento de videos finalizado.")

# Función para abrir la ventana de configuración (token y número)
def open_config_window():
    config_window = Toplevel(root)
    config_window.title("Configuración")
    config_window.geometry("300x200")

    # Etiqueta y campo para el token
    token_label = Label(config_window, text="Ingrese su token:", font=("Arial", 12))
    token_label.pack(pady=5)
    token_entry = Entry(config_window, font=("Arial", 12))
    token_entry.pack(pady=5)

    # Etiqueta y campo para el número
    number_label = Label(config_window, text="Modificar número:", font=("Arial", 12))
    number_label.pack(pady=5)
    number_entry = Entry(config_window, font=("Arial", 12))
    number_entry.pack(pady=5)

    # Cargar datos existentes si el archivo JSON existe
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            config_data = json.load(file)
            token_entry.insert(0, config_data.get("token", ""))
            number_entry.insert(0, config_data.get("number", ""))

    # Botón para guardar la configuración
    def save_config():
        token = token_entry.get()
        number = number_entry.get()
        if token and number:
            # Crear un diccionario con los datos
            config_data = {"token": token, "number": number}

            # Guardar los datos en un archivo JSON
            with open(CONFIG_FILE, "w") as file:
                json.dump(config_data, file, indent=4)

            print(f"Token ingresado: {token}")
            print(f"Número modificado: {number}")
            messagebox.showinfo("Guardado", "Configuración guardada correctamente.")
            config_window.destroy()
        else:
            messagebox.showwarning("Error", "Por favor, complete ambos campos.")

    save_button = Button(config_window, text="Guardar", command=save_config, font=("Arial", 12))
    save_button.pack(pady=10)

# Crear la interfaz gráfica
def create_gui():
    global root, progress_bar, progress_label

    root = Tk()
    root.title("Procesador de Videos con Detección de Texto")
    root.geometry("400x200")

    # Crear un menú principal
    menu_bar = Menu(root)
    root.config(menu=menu_bar)

    # Menú de configuración
    config_menu = Menu(menu_bar, tearoff=0)
    config_menu.add_command(label="Configuración", command=open_config_window)
    menu_bar.add_cascade(label="Opciones", menu=config_menu)

    # Contenido principal de la interfaz
    label = Label(root, text="Selecciona una carpeta con videos para procesar", font=("Arial", 12))
    label.pack(pady=20)

    button = Button(root, text="Seleccionar Carpeta", command=select_folder, font=("Arial", 12))
    button.pack(pady=10)

    # Barra de progreso
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=10)

    # Label para mostrar el porcentaje de progreso
    progress_label = Label(root, text="0% completado", font=("Arial", 12))
    progress_label.pack(pady=5)

    root.mainloop()

