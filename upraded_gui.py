import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np

# =============================
# Load trained Keras model
# =============================
# NOTE: Ensure 'traffic_classifier.keras' is in the same directory OR provide full path.
from keras.models import load_model
MODEL_PATH = 'traffic_classifier.keras'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"WARNING: Could not load model '{MODEL_PATH}': {e}")
    model = None  # Allow GUI to start so layout can be tested

# =============================
# Class ID -> Human-readable label mapping
# (GTSRB-style index starting at 1)
# =============================
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons',
}

# =============================================================
# GUI Application
# =============================================================
class TrafficSignApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title('Traffic Sign Classification')
        self.master.geometry('1000x700')  # Larger default window
        self.master.configure(background='#CDCDCD')

        # Make root grid stretchy
        self.master.rowconfigure(1, weight=1)  # image row grows
        self.master.columnconfigure(0, weight=1)

        # State
        self.current_image_pil = None   # Original PIL image user uploaded (full size)
        self.current_file_path = None   # Path of uploaded file
        self.display_image_tk = None    # Cached PhotoImage for display

        # Build frames
        self._build_frames()
        self._build_header()
        self._build_image_area()
        self._build_controls()
        self._build_status()

        # Resize binding (so image scales when window size changes)
        self.master.bind('<Configure>', self._on_window_resize)

    # ---------------------------------------------------------
    # Frame layout
    # ---------------------------------------------------------
    def _build_frames(self):
        # Header frame (title)
        self.header_frame = tk.Frame(self.master, bg='#CDCDCD')
        self.header_frame.grid(row=0, column=0, sticky='ew')
        self.header_frame.columnconfigure(0, weight=1)

        # Image display frame (expands)
        self.image_frame = tk.Frame(self.master, bg='#EEEEEE', bd=2, relief='sunken')
        self.image_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)

        # Controls frame (buttons)
        self.controls_frame = tk.Frame(self.master, bg='#CDCDCD')
        self.controls_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=(0,5))
        for i in range(4):
            self.controls_frame.columnconfigure(i, weight=1)

        # Status frame (prediction text)
        self.status_frame = tk.Frame(self.master, bg='#CDCDCD')
        self.status_frame.grid(row=3, column=0, sticky='ew', padx=10, pady=(0,10))
        self.status_frame.columnconfigure(0, weight=1)

    # ---------------------------------------------------------
    def _build_header(self):
        self.heading_label = tk.Label(
            self.header_frame,
            text='Know Your Traffic Sign',
            bg='#CDCDCD',
            fg='#364156',
            font=('Arial', 24, 'bold'),
            pady=20
        )
        self.heading_label.grid(row=0, column=0, sticky='ew')

    # ---------------------------------------------------------
    def _build_image_area(self):
        # Label that will show the image
        self.image_label = tk.Label(self.image_frame, bg='#EEEEEE')
        self.image_label.grid(row=0, column=0, sticky='nsew')

    # ---------------------------------------------------------
    def _build_controls(self):
        # Upload button
        self.upload_btn = tk.Button(
            self.controls_frame,
            text='Upload Image',
            command=self.upload_image,
            padx=10, pady=5,
            bg='#364156', fg='white',
            font=('Arial', 12, 'bold')
        )
        self.upload_btn.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        # Classify button (disabled until image uploaded)
        self.classify_btn = tk.Button(
            self.controls_frame,
            text='Classify Image',
            command=self.classify_current_image,
            state='disabled',
            padx=10, pady=5,
            bg='#364156', fg='white',
            font=('Arial', 12, 'bold')
        )
        self.classify_btn.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Clear button
        self.clear_btn = tk.Button(
            self.controls_frame,
            text='Clear',
            command=self.clear_image,
            padx=10, pady=5,
            bg='#555555', fg='white',
            font=('Arial', 12)
        )
        self.clear_btn.grid(row=0, column=2, padx=5, pady=5, sticky='ew')

        # Exit button
        self.exit_btn = tk.Button(
            self.controls_frame,
            text='Exit',
            command=self.master.destroy,
            padx=10, pady=5,
            bg='#AA3333', fg='white',
            font=('Arial', 12)
        )
        self.exit_btn.grid(row=0, column=3, padx=5, pady=5, sticky='ew')

    # ---------------------------------------------------------
    def _build_status(self):
        self.prediction_var = tk.StringVar(value='Upload an image to begin.')
        self.prediction_label = tk.Label(
            self.status_frame,
            textvariable=self.prediction_var,
            bg='#CDCDCD', fg='#011638',
            font=('Arial', 16, 'bold'),
            wraplength=900,  # wrap long labels within window
            justify='center'
        )
        self.prediction_label.grid(row=0, column=0, sticky='ew', pady=10)

    # ==========================================================
    # Image Handling
    # ==========================================================
    def upload_image(self):
        """Open file dialog, load image, and display scaled copy."""
        file_path = filedialog.askopenfilename(
            title='Select a Traffic Sign Image',
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff'),
                ('All files', '*.*')
            ]
        )
        if not file_path:
            return  # user cancelled

        try:
            pil_img = Image.open(file_path).convert('RGB')  # ensure 3-channel
        except Exception as e:
            messagebox.showerror('Error', f'Could not open image:\n{e}')
            return

        self.current_file_path = file_path
        self.current_image_pil = pil_img

        # Enable classify
        self.classify_btn.config(state='normal')

        # Update status
        self.prediction_var.set('Image loaded. Click "Classify Image" to predict.')

        # Display scaled image
        self._display_current_image_scaled()

    # ---------------------------------------------------------
    def clear_image(self):
        self.current_image_pil = None
        self.current_file_path = None
        self.display_image_tk = None
        self.image_label.configure(image='')
        self.prediction_var.set('Upload an image to begin.')
        self.classify_btn.config(state='disabled')

    # ---------------------------------------------------------
    def _display_current_image_scaled(self):
        """Scale the currently loaded PIL image to fit the image_frame area and show it."""
        if self.current_image_pil is None:
            return

        # Dimensions of the frame available for the image
        frame_w = self.image_frame.winfo_width()
        frame_h = self.image_frame.winfo_height()

        # If widget not yet drawn, winfo may return 1; fallback to master size minus margins
        if frame_w < 10 or frame_h < 10:
            self.master.update_idletasks()
            frame_w = max(self.image_frame.winfo_width(), int(self.master.winfo_width() * 0.7))
            frame_h = max(self.image_frame.winfo_height(), int(self.master.winfo_height() * 0.6))

        # Copy original so we don't mutate it
        img_copy = self.current_image_pil.copy()
        img_copy.thumbnail((frame_w, frame_h), Image.LANCZOS)  # keeps aspect ratio

        self.display_image_tk = ImageTk.PhotoImage(img_copy)
        self.image_label.configure(image=self.display_image_tk)
        self.image_label.image = self.display_image_tk  # keep ref

    # ---------------------------------------------------------
    def _on_window_resize(self, event):
        """Whenever the window is resized, redraw the image at new scale."""
        # We get many events; only redraw when actual size changed significantly
        # but for simplicity just redraw if we have an image
        if self.current_image_pil is not None:
            self._display_current_image_scaled()

    # ==========================================================
    # Classification
    # ==========================================================
    def classify_current_image(self):
        if self.current_file_path is None or self.current_image_pil is None:
            messagebox.showwarning('No Image', 'Please upload an image first.')
            return
        if model is None:
            messagebox.showerror('Model Not Loaded', 'Model failed to load; cannot classify.')
            return

        try:
            # Preprocess for model: resize to (30,30), convert to array, add batch dim
            img_model = self.current_image_pil.resize((30, 30))  # assume model trained at 30x30x3
            arr = np.array(img_model)
            arr = np.expand_dims(arr, axis=0)

            # If your model expects scaled values (0-1), uncomment next line:
            # arr = arr.astype('float32') / 255.0

            pred = model.predict(arr)
            pred_class_index0 = int(np.argmax(pred, axis=1)[0])  # 0-based
            pred_class_id = pred_class_index0 + 1  # convert to 1-based key
            sign_text = classes.get(pred_class_id, f'Class {pred_class_id}')

            # Show result
            self.prediction_var.set(sign_text)
        except Exception as e:
            messagebox.showerror('Classification Error', f'Error during prediction:\n{e}')


# =============================================================
# Main Entry
# =============================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
