import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import threading
import os

from src.predict_image import get_prediction_data

# --- PALETTE WARNA ---
C_BG_WINDOW   = "#d1fae5" 
C_GLASS_CARD  = "#ecfdf5" # Mint Cream
C_GLASS_INNER = "#f0fdfa" 

C_PRIMARY     = "#047857" # Hijau Hutan
C_ACCENT      = "#f59e0b" # Kuning
C_DANGER      = "#e11d48" # Merah
C_TEXT_DARK   = "#064e3b" 
C_TEXT_MED    = "#374151" 
C_TEXT_LGT    = "#6b7280" 
C_WHITE       = "#FFFFFF"
C_TAG_BG      = "#f8fafc" 

ASSETS = "assets"

class PlantGuardNature(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Plant Disease Prediction - with EfficientNetV2")
        self.geometry("1100x720")
        
        # Min size agar layout aman
        self.minsize(950, 700)
        
        ctk.set_appearance_mode("Light")
        self.configure(fg_color=C_BG_WINDOW) 
        
        self.icons = self.load_icons()

        # --- KARTU KACA UTAMA ---
        self.glass_card = ctk.CTkFrame(
            self, 
            fg_color=C_GLASS_CARD, 
            corner_radius=32,
            border_width=1,
            border_color=C_WHITE
        )
        self.glass_card.pack(expand=True, fill="both", padx=20, pady=20)

        # Grid Layout
        self.glass_card.grid_columnconfigure(0, weight=1) 
        self.glass_card.grid_columnconfigure(1, weight=1) 
        self.glass_card.grid_rowconfigure(1, weight=1)    

        self.setup_header()
        self.setup_left_panel()
        self.setup_right_panel()

        self.current_image_path = None

    def load_icons(self):
        imgs = {}
        def get(name, size):
            p = os.path.join(ASSETS, name)
            return ctk.CTkImage(Image.open(p), size=(size, size)) if os.path.exists(p) else None
        
        imgs["leaf"] = get("leaf_logo.png", 48)
        imgs["cam"]  = get("icon_camera.png", 28)
        imgs["up"]   = get("btn_upload.png", 24)
        imgs["scan"] = get("btn_scan.png", 24)
        return imgs

    def setup_header(self):
        header = ctk.CTkFrame(self.glass_card, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=30, pady=(20, 10))

        logo_box = ctk.CTkFrame(header, fg_color="transparent")
        logo_box.pack(side="left")
        
        if self.icons.get("leaf"):
            ctk.CTkLabel(logo_box, text="", image=self.icons["leaf"]).pack(side="left")
        
        title_box = ctk.CTkFrame(logo_box, fg_color="transparent")
        title_box.pack(side="left", padx=15)
        
        ctk.CTkLabel(title_box, text="Plant Disease Prediction", font=("Segoe UI", 24, "bold"), text_color=C_TEXT_DARK).pack(anchor="w")
        ctk.CTkLabel(title_box, text="with EfficientNetV2-B3", font=("Segoe UI", 12), text_color=C_TEXT_LGT).pack(anchor="w")

    def setup_left_panel(self):
        self.img_frame = ctk.CTkFrame(
            self.glass_card, 
            fg_color=C_WHITE, 
            corner_radius=20,
            border_width=2,
            border_color="#e5e7eb"
        )
        self.img_frame.grid(row=1, column=0, sticky="nsew", padx=(30, 15), pady=(10, 30))
        self.img_frame.grid_propagate(False) 
        self.img_frame.pack_propagate(False)

        # Overlays
        self.info_tag = ctk.CTkLabel(
            self.img_frame, text="Belum ada file", 
            fg_color=C_TAG_BG, text_color=C_TEXT_MED,
            corner_radius=10, font=("Segoe UI", 11, "bold"), height=32, padx=12
        )
        self.info_tag.place(relx=0.04, rely=0.04, anchor="nw")

        self.status_tag = ctk.CTkLabel(
            self.img_frame, text="Menunggu", 
            fg_color=C_TAG_BG, text_color=C_TEXT_LGT,
            corner_radius=10, font=("Segoe UI", 11, "bold"), height=32, padx=12
        )
        self.status_tag.place(relx=0.96, rely=0.04, anchor="ne")

        self.img_label = ctk.CTkLabel(
            self.img_frame, 
            text="Klik area ini untuk\nupload gambar", 
            font=("Segoe UI", 16), text_color=C_TEXT_LGT,
            image=self.icons.get("cam"), compound="top"
        )
        self.img_label.pack(expand=True, fill="both", padx=20, pady=60) 
        self.img_label.bind("<Button-1>", lambda e: self.upload_action())

    def setup_right_panel(self):
        # 1. Container Panel Kanan (Fixed Frame, Tanpa Scroll)
        self.res_panel = ctk.CTkFrame(self.glass_card, fg_color="transparent")
        self.res_panel.grid(row=1, column=1, sticky="nsew", padx=(15, 30), pady=(10, 30))

        # --- 2. BAGIAN FOOTER (Fixed di Bawah Kanan) ---
        self.footer_area = ctk.CTkFrame(self.res_panel, fg_color="transparent")
        self.footer_area.pack(side="bottom", fill="x")

        # Container Tombol (Berdampingan)
        btn_container = ctk.CTkFrame(self.footer_area, fg_color="transparent")
        btn_container.pack(fill="x", side="bottom")

        # Tombol Upload (Putih - Secondary)
        self.btn_up = ctk.CTkButton(
            btn_container, 
            text="Upload Baru", 
            image=self.icons.get("up"),
            font=("Segoe UI", 14, "bold"), 
            height=50, 
            corner_radius=16,
            fg_color=C_WHITE, # Putih
            text_color=C_TEXT_MED,
            border_width=1,
            border_color="#d1d5db",
            hover_color="#f3f4f6",
            command=self.upload_action
        )
        self.btn_up.pack(side="right", fill="x", expand=True, padx=(0, 0))

        # Tombol Pindai Ulang (Hijau - Primary)
        self.btn_rescan = ctk.CTkButton(
            btn_container,
            text="Pindai Ulang",
            image=self.icons.get("scan"),
            font=("Segoe UI", 14, "bold"),
            height=50,
            corner_radius=16,
            fg_color=C_PRIMARY, # Hijau
            text_color=C_WHITE,
            hover_color="#065f46",
            state="disabled", # Default disabled
            command=self.run_prediction
        )
        self.btn_rescan.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Kotak Kontrol (Di atas tombol)
        self.controls_box = ctk.CTkFrame(
            self.footer_area, 
            fg_color=C_GLASS_INNER,
            corner_radius=16,
            border_width=1,
            border_color="#d1fae5"
        )
        self.controls_box.pack(fill="x", side="bottom", pady=(0, 15))

        # Isi Kontrol (Rapat)
        self.gradcam_var = ctk.BooleanVar(value=False)
        self.switch_gradcam = ctk.CTkSwitch(
            self.controls_box, 
            text="Visualisasi AI (Heatmap)", 
            variable=self.gradcam_var,
            font=("Segoe UI", 13, "bold"),
            text_color=C_TEXT_MED,
            progress_color=C_PRIMARY,
            command=self.on_param_change
        )
        self.switch_gradcam.pack(anchor="w", padx=20, pady=(15, 5))

        slider_row = ctk.CTkFrame(self.controls_box, fg_color="transparent")
        slider_row.pack(fill="x", padx=20, pady=(5, 15))
        
        ctk.CTkLabel(slider_row, text="Threshold:", font=("Segoe UI", 12, "bold"), text_color=C_TEXT_LGT).pack(side="left")
        
        self.threshold_val = ctk.DoubleVar(value=0.55)
        self.slider = ctk.CTkSlider(
            slider_row, 
            from_=0.1, to=0.9, 
            variable=self.threshold_val,
            number_of_steps=100,
            button_color=C_PRIMARY,
            progress_color=C_PRIMARY,
            height=16,
            command=self.update_thresh_label
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        
        self.lbl_thresh_val = ctk.CTkLabel(slider_row, text="0.55", font=("Segoe UI", 12, "bold"), text_color=C_PRIMARY, width=35)
        self.lbl_thresh_val.pack(side="right")


        # --- 3. BAGIAN INFO (Di Atas - Mengisi sisa ruang) ---
        self.info_container = ctk.CTkFrame(self.res_panel, fg_color="transparent")
        self.info_container.pack(side="top", fill="both", expand=True)

        ctk.CTkLabel(self.info_container, text="HASIL DIAGNOSA", font=("Segoe UI", 12, "bold"), text_color=C_TEXT_LGT).pack(anchor="w", pady=(10, 5))
        
        self.lbl_disease = ctk.CTkLabel(self.info_container, text="Menunggu...", font=("Segoe UI", 48, "bold"), text_color=C_TEXT_DARK)
        self.lbl_disease.pack(anchor="w")

        self.desc_frame = ctk.CTkFrame(self.info_container, fg_color=C_GLASS_INNER, corner_radius=16, border_width=0)
        self.desc_frame.pack(fill="x", pady=20)
        
        self.lbl_desc = ctk.CTkLabel(
            self.desc_frame, 
            text="Silakan upload gambar daun mangga.", 
            font=("Segoe UI", 14), text_color=C_TEXT_MED,
            wraplength=350, justify="left"
        )
        self.lbl_desc.pack(padx=20, pady=20, anchor="w")

        self.stats_frame = ctk.CTkFrame(self.info_container, fg_color="transparent")
        self.stats_frame.pack(fill="x", pady=5)

    # --- LOGIKA APLIKASI ---

    def update_thresh_label(self, value):
        self.lbl_thresh_val.configure(text=f"{value:.2f}")

    def on_param_change(self):
        if self.current_image_path:
            self.run_prediction()

    def upload_action(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if path:
            self.current_image_path = path
            
            filename = os.path.basename(path)
            if len(filename) > 20: filename = filename[:17] + "..."
            self.info_tag.configure(text=f"📄 {filename}")
            
            self.status_tag.configure(text="Siap", fg_color=C_TAG_BG, text_color=C_TEXT_MED)
            self.lbl_disease.configure(text="...", text_color=C_TEXT_MED)
            self.lbl_desc.configure(text="Klik 'Pindai Ulang' jika ingin menganalisis kembali.")
            
            # Aktifkan tombol rescan karena gambar sudah ada
            self.btn_rescan.configure(state="normal", fg_color=C_PRIMARY)
            
            for w in self.stats_frame.winfo_children(): w.destroy()

            self.display_image(Image.open(path))
            self.run_prediction()

    def display_image(self, pil_img):
        max_w, max_h = 450, 550 
        w, h = pil_img.size
        ratio = min(max_w/w, max_h/h)
        new_size = (int(w*ratio), int(h*ratio))
        
        ctk_img = ctk.CTkImage(pil_img, size=new_size)
        self.img_label.configure(image=ctk_img, text="") 

    def run_prediction(self):
        self.status_tag.configure(text="Menganalisis...", fg_color=C_ACCENT, text_color="white")
        thresh = self.threshold_val.get()
        use_cam = self.gradcam_var.get()
        threading.Thread(target=self._process, args=(thresh, use_cam)).start()

    def _process(self, thresh, use_cam):
        try:
            res = get_prediction_data(
                self.current_image_path, 
                threshold=thresh, 
                enable_gradcam=use_cam
            )
            self.after(0, lambda: self.show_result(res))
        except Exception as e:
            print(f"Error: {e}")
            self.after(0, lambda: self.status_tag.configure(text="Error", fg_color=C_DANGER))

    def show_result(self, res):
        label = res['label']
        top3 = res['top3']
        is_unknown = res['is_unknown']
        gradcam_img = res.get('gradcam_image')

        self.status_tag.configure(text="Selesai", fg_color=C_PRIMARY, text_color="white")

        if gradcam_img:
            self.display_image(gradcam_img)
        else:
            self.display_image(Image.open(self.current_image_path))

        if is_unknown:
            self.lbl_disease.configure(text=f"{label} (?)", text_color=C_ACCENT)
            self.lbl_desc.configure(text=f"Confidence rendah (di bawah {self.threshold_val.get():.2f}). Sistem ragu dengan hasil ini.")
        else:
            self.lbl_disease.configure(text=label, text_color=C_TEXT_DARK)
            if label == "Healthy":
                self.lbl_desc.configure(text="✅ Tanaman sehat. Pertahankan nutrisi dan penyiraman.")
                self.lbl_disease.configure(text_color=C_PRIMARY)
            else:
                self.lbl_desc.configure(text=f"⚠️ Terdeteksi {label}. Cek kondisi fisik tanaman.")
                self.lbl_disease.configure(text_color=C_DANGER)

        for w in self.stats_frame.winfo_children(): w.destroy()
        for item in top3:
            self.add_stat_pill(item['label'], item['score'])

    def add_stat_pill(self, label, score):
        row = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        row.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 13, "bold"), text_color=C_TEXT_MED, width=120, anchor="w").pack(side="left")
        
        color = C_DANGER if score > 0.5 and label != "Healthy" else C_PRIMARY if label=="Healthy" else "#d1d5db"
        progress = ctk.CTkProgressBar(row, height=12, corner_radius=6, progress_color=color, fg_color="#e5e7eb")
        progress.set(score)
        progress.pack(side="left", fill="x", expand=True, padx=10)
        
        ctk.CTkLabel(row, text=f"{int(score*100)}%", font=("Segoe UI", 13, "bold"), text_color=color, width=40, anchor="e").pack(side="right")

if __name__ == "__main__":
    app = PlantGuardNature()
    app.mainloop()