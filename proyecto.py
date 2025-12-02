import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Configuración inicial
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class ProjectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Comunidad 7 - Segmentación")
        
        try:
            self.state('zoomed') 
        except:
            self.geometry("1600x900")

        # --- FUENTES ---
        self.font_title = ctk.CTkFont(family="Arial", size=32, weight="bold")
        self.font_header = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.font_btn = ctk.CTkFont(family="Arial", size=18, weight="bold")
        self.font_combo_selected = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.font_dropdown_list = ctk.CTkFont(family="Arial", size=24) 
        self.font_slider = ctk.CTkFont(family="Arial", size=14, weight="bold")
        self.font_stats = ctk.CTkFont(family="Consolas", size=16, weight="bold") # Fuente monoespaciada para datos

        # --- VARIABLES DE ESTADO ---
        self.original_cv_image = None
        self.img_step_noise = None    
        self.img_step_filter = None   
        self.image_ready_for_segmentation = None 
        self.processed_cv_image = None 
        
        self.current_thresholds = []
        self.last_method_name = ""
        
        self.var_t1 = tk.IntVar(value=50)
        self.var_t2 = tk.IntVar(value=200)
        self.var_noise = tk.DoubleVar(value=0) 
        self.var_filter = tk.DoubleVar(value=0)
        self.var_bright = tk.DoubleVar(value=50)

        # ============ LAYOUT ============
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- BARRA LATERAL ---
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=480, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="COMUNIDAD 7", font=self.font_title)
        self.logo_label.pack(padx=20, pady=(30, 20))

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="📂 ABRIR IMAGEN", command=self.load_image, 
                                      height=60, corner_radius=10, 
                                      fg_color="#00695C", hover_color="#004D40", 
                                      font=self.font_title)
        self.btn_load.pack(padx=20, pady=(0, 10), fill="x")

        # BOTÓN FIJAR CAMBIOS
        self.btn_commit = ctk.CTkButton(self.sidebar_frame, text="💾 FIJAR CAMBIOS (Nueva Original)", 
                                        command=self.commit_changes,
                                        height=40, corner_radius=10, 
                                        fg_color="#455A64", hover_color="#263238",
                                        font=ctk.CTkFont(family="Arial", size=14, weight="bold"))
        self.btn_commit.pack(padx=20, pady=(0, 20), fill="x")

        # 1. RUIDO
        self.create_separator(self.sidebar_frame, "1. RUIDO")
        self.noise_combo = ctk.CTkComboBox(self.sidebar_frame, values=["Gaussiano", "Sal y Pimienta", "Solo Sal", "Solo Pimienta"], 
                                           height=45, font=self.font_combo_selected, dropdown_font=self.font_dropdown_list, 
                                           corner_radius=10, border_color="#5E35B1", dropdown_fg_color="#333333",
                                           command=self.chain_reaction_noise)
        self.noise_combo.pack(padx=20, pady=(0, 10), fill="x")
        self.slider_noise = ctk.CTkSlider(self.sidebar_frame, from_=0, to=100, variable=self.var_noise, 
                                          command=self.chain_reaction_noise, button_color="#5E35B1", progress_color="#5E35B1")
        self.slider_noise.pack(padx=20, pady=(0, 20), fill="x")

        # 2. FILTROS
        self.create_separator(self.sidebar_frame, "2. FILTROS")
        self.filter_combo = ctk.CTkComboBox(self.sidebar_frame, values=["Mediana (Recomendado)", "Bilateral (Bordes)", "Gaussiano", "Promediador", "Máximo", "Mínimo"], 
                                            height=45, font=self.font_combo_selected, dropdown_font=self.font_dropdown_list,
                                            corner_radius=10, border_color="#1E88E5", dropdown_fg_color="#333333",
                                            command=self.chain_reaction_filter)
        self.filter_combo.pack(padx=20, pady=(0, 10), fill="x")
        self.slider_filter = ctk.CTkSlider(self.sidebar_frame, from_=0, to=100, variable=self.var_filter,
                                           command=self.chain_reaction_filter, button_color="#1E88E5", progress_color="#1E88E5")
        self.slider_filter.pack(padx=20, pady=(0, 20), fill="x")

        # 3. BRILLO
        self.create_separator(self.sidebar_frame, "3. BRILLO Y CONTRASTE")
        opciones_brillo = ["1. Corrección Gamma", "2. Función Potencia", "3. Ecualización Logarítmica", 
                           "4. Ecualización Rayleigh", "5. Ecualización Exponencial", "6. Ecualización Uniforme", "7. Ecualización Hipercúbica"]
        self.bright_combo = ctk.CTkComboBox(self.sidebar_frame, values=opciones_brillo, 
                                            height=45, font=self.font_combo_selected, dropdown_font=self.font_dropdown_list,
                                            corner_radius=10, border_color="#00ACC1", dropdown_fg_color="#333333",
                                            command=self.reset_bright_slider)
        self.bright_combo.pack(padx=20, pady=(0, 10), fill="x")
        self.slider_bright = ctk.CTkSlider(self.sidebar_frame, from_=0, to=100, variable=self.var_bright,
                                           command=self.chain_reaction_brightness, button_color="#00ACC1", progress_color="#00ACC1")
        self.slider_bright.pack(padx=20, pady=(0, 20), fill="x")

        # 4. SEGMENTACIÓN
        self.create_separator(self.sidebar_frame, "4. SEGMENTACIÓN")
        opciones_seg = ["Otsu (Automático)", "Umbral Media (Promedio)", "Umbral Banda (Manual T1-T2)", "Kapur (Entropía)", "Mínimo del Histograma", "Multiumbralización (3 clases)"]
        self.seg_combo = ctk.CTkComboBox(self.sidebar_frame, values=opciones_seg, height=45, font=self.font_combo_selected, dropdown_font=self.font_dropdown_list,
                                         corner_radius=10, border_color="#D84315", dropdown_fg_color="#333333", command=self.check_segmentation_mode) 
        self.seg_combo.pack(padx=20, pady=(0, 10), fill="x")

        # Sliders Segmentación
        self.slider_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.lbl_t1 = ctk.CTkLabel(self.slider_frame, text="T1: 50", font=self.font_slider)
        self.lbl_t1.pack(fill="x")
        self.slider_t1 = ctk.CTkSlider(self.slider_frame, from_=0, to=255, variable=self.var_t1, command=self.update_sliders_text, button_color="#D84315")
        self.slider_t1.pack(fill="x", pady=5)
        self.lbl_t2 = ctk.CTkLabel(self.slider_frame, text="T2: 200", font=self.font_slider)
        self.lbl_t2.pack(fill="x")
        self.slider_t2 = ctk.CTkSlider(self.slider_frame, from_=0, to=255, variable=self.var_t2, command=self.update_sliders_text, button_color="#D84315")
        self.slider_t2.pack(fill="x", pady=5)
        
        self.btn_seg = ctk.CTkButton(self.sidebar_frame, text="SEGMENTAR AHORA", command=self.apply_segmentation, 
                                     height=50, corner_radius=10, font=self.font_btn, fg_color="#D84315", hover_color="#BF360C")
        self.btn_seg.pack(padx=20, pady=(0, 10), fill="x")
        
        self.btn_undo_seg = ctk.CTkButton(self.sidebar_frame, text="↩ DESHACER SEGMENTACIÓN", command=self.undo_segmentation, 
                                          height=40, corner_radius=10, font=ctk.CTkFont(family="Arial", size=14, weight="bold"), 
                                          fg_color="#546E7A", hover_color="#37474F")
        self.btn_undo_seg.pack(padx=20, pady=(0, 15), fill="x")

        # --- 5. NUEVA SECCIÓN: ANÁLISIS MÉTRICO ---
        self.create_separator(self.sidebar_frame, "5. ANÁLISIS DE OBJETOS")
        
        self.btn_analyze = ctk.CTkButton(self.sidebar_frame, text="🔍 CONTAR Y MEDIR OBJETOS", command=self.analyze_objects,
                                         height=50, corner_radius=10, font=self.font_btn, 
                                         fg_color="#8E24AA", hover_color="#6A1B9A") # Morado oscuro
        self.btn_analyze.pack(padx=20, pady=(0, 10), fill="x")

        # Label para mostrar resultados
        self.frame_stats = ctk.CTkFrame(self.sidebar_frame, fg_color="#333333", corner_radius=10)
        self.frame_stats.pack(padx=20, pady=(0, 20), fill="x")
        
        self.lbl_stats = ctk.CTkLabel(self.frame_stats, text="Esperando análisis...", 
                                      font=self.font_stats, text_color="#E1BEE7", justify="left")
        self.lbl_stats.pack(padx=10, pady=10, fill="x")


        self.btn_reset = ctk.CTkButton(self.sidebar_frame, text="↺ RESTAURAR ORIGINAL", command=self.reset_image, 
                                       height=50, corner_radius=10, font=self.font_btn, fg_color="#C62828", hover_color="#B71C1C")
        self.btn_reset.pack(padx=20, pady=(20, 30), fill="x")

        # --- ÁREA VISUAL ---
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_view.grid_rowconfigure(0, weight=3) 
        self.main_view.grid_rowconfigure(1, weight=2) 
        self.main_view.grid_columnconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.main_view, text="IMAGEN ORIGINAL", font=self.font_header).grid(row=0, column=0, sticky="s", pady=(0, 5))
        ctk.CTkLabel(self.main_view, text="RESULTADO EN TIEMPO REAL", font=self.font_header, text_color="#64FFDA").grid(row=0, column=1, sticky="s", pady=(0, 5))

        self.frame_orig = ctk.CTkFrame(self.main_view, corner_radius=10, fg_color="#1a1a1a")
        self.frame_orig.grid(row=1, column=0, padx=10, pady=(0, 20), sticky="nsew")
        self.frame_proc = ctk.CTkFrame(self.main_view, corner_radius=10, fg_color="#1a1a1a")
        self.frame_proc.grid(row=1, column=1, padx=10, pady=(0, 20), sticky="nsew")

        self.lbl_img_orig = tk.Label(self.frame_orig, bg="#1a1a1a")
        self.lbl_img_orig.pack(fill="both", expand=True) 
        self.lbl_img_proc = tk.Label(self.frame_proc, bg="#1a1a1a")
        self.lbl_img_proc.pack(fill="both", expand=True) 
        self.frame_orig.bind('<Configure>', lambda event: self.resize_image_event(event, "orig"))
        self.frame_proc.bind('<Configure>', lambda event: self.resize_image_event(event, "proc"))

        self.frame_hist = ctk.CTkFrame(self.main_view, corner_radius=10, fg_color="#2b2b2b")
        self.frame_hist.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        self.setup_projector_histogram()

    # ================= ANÁLISIS MÉTRICO (NUEVO) =================

    def analyze_objects(self):
        """Cuenta objetos y calcula área/perímetro sobre la máscara binaria actual"""
        if self.processed_cv_image is None:
            messagebox.showwarning("Aviso", "Primero debes segmentar la imagen.")
            return

        # 1. Asegurar que tenemos una imagen binaria (blanco y negro)
        img_analysis = self.processed_cv_image.copy()
        if len(img_analysis.shape) == 3:
            img_gray = cv2.cvtColor(img_analysis, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_analysis

        # Umbralizar para asegurar 0 y 255 estrictos
        # Asumimos que los objetos son lo que no es negro (> 10)
        _, binary = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)

        # 2. Encontrar Contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Filtrar ruido pequeño (Objetos menores a 50px se ignoran)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        
        count = len(valid_contours)
        if count == 0:
            self.lbl_stats.configure(text="No se detectaron objetos.\nPrueba ajustar la segmentación.")
            return

        # 4. Calcular Métricas y Dibujar
        # Convertimos a color para poder dibujar líneas verdes/rojas
        vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        total_area = 0
        total_perimeter = 0
        
        for i, cnt in enumerate(valid_contours):
            # Área
            area = cv2.contourArea(cnt)
            total_area += area
            
            # Perímetro
            perimeter = cv2.arcLength(cnt, True)
            total_perimeter += perimeter
            
            # Dibujar Contorno (Verde Neon)
            cv2.drawContours(vis_image, [cnt], -1, (0, 255, 0), 3)
            
            # Poner ID en el centro del objeto
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Dibujar texto ID (Rojo)
                cv2.putText(vis_image, f"#{i+1}", (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)

        # 5. Mostrar Resultados
        stats_text = (f"🔍 RESULTADOS:\n"
                      f"----------------\n"
                      f"• Objetos: {count}\n"
                      f"• Área Total: {int(total_area)} px²\n"
                      f"• Perím. Total: {int(total_perimeter)} px\n"
                      f"• Prom. Área: {int(total_area/count)} px²")
        
        self.lbl_stats.configure(text=stats_text)
        
        # Mostrar la imagen con los dibujos
        self.processed_cv_image = vis_image
        self.show_images()

    # ================= FUNCIONES PREVIAS (RESUMIDAS) =================

    def commit_changes(self):
        if self.processed_cv_image is None: return
        self.original_cv_image = self.processed_cv_image.copy()
        self.img_step_noise = self.original_cv_image.copy()
        self.img_step_filter = self.original_cv_image.copy()
        self.image_ready_for_segmentation = self.original_cv_image.copy()
        self.var_noise.set(0); self.var_filter.set(0); self.var_bright.set(50)
        self.current_thresholds = []
        self.lbl_stats.configure(text="Esperando análisis...") # Reset stats
        self.show_images()
        messagebox.showinfo("Éxito", "Imagen fijada como nueva base.")

    def undo_segmentation(self):
        if self.image_ready_for_segmentation is None: return
        self.processed_cv_image = self.image_ready_for_segmentation.copy()
        self.current_thresholds = []
        self.lbl_stats.configure(text="Segmentación deshecha.") 
        self.show_images()

    def chain_reaction_noise(self, value=None):
        if self.original_cv_image is None: return
        intensity = self.var_noise.get()
        tipo = self.noise_combo.get()
        img = self.original_cv_image.copy()
        if intensity > 0:
            if "Gaussiano" in tipo:
                noise = np.random.normal(0, intensity*0.5, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            else:
                prob = (intensity / 100.0) * 0.30
                rnd = np.random.rand(*img.shape[:2])
                if "Sal" in tipo and "Pimienta" in tipo: img[rnd < prob/2] = 0; img[rnd > 1 - prob/2] = 255
                elif "Sal" in tipo: img[rnd < prob] = 255
                elif "Pimienta" in tipo: img[rnd < prob] = 0
        self.img_step_noise = img 
        self.chain_reaction_filter()

    def chain_reaction_filter(self, value=None):
        if self.img_step_noise is None: return
        intensity = self.var_filter.get()
        tipo = self.filter_combo.get()
        k = int((intensity / 100) * 20); k = k+1 if k%2==0 else k
        img = self.img_step_noise.copy()
        if intensity > 0:
            if "Mediana" in tipo: img = cv2.medianBlur(img, k)
            elif "Gaussiano" in tipo: img = cv2.GaussianBlur(img, (k,k), 0)
            elif "Promediador" in tipo: img = cv2.blur(img, (k,k))
            elif "Máximo" in tipo: img = cv2.dilate(img, np.ones((k,k), np.uint8))
            elif "Mínimo" in tipo: img = cv2.erode(img, np.ones((k,k), np.uint8))
            elif "Bilateral" in tipo: d = int((intensity/100)*15)+1; img = cv2.bilateralFilter(img, d, d*2, d*2)
        self.img_step_filter = img 
        self.chain_reaction_brightness()

    def reset_bright_slider(self, choice):
        self.var_bright.set(50)
        self.chain_reaction_brightness()

    def chain_reaction_brightness(self, value=None):
        if self.img_step_filter is None: return
        intensity_val = self.var_bright.get()
        tipo = self.bright_combo.get()
        img_in = self.img_step_filter.copy()
        img_yuv = cv2.cvtColor(img_in, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        if "Gamma" in tipo or "Potencia" in tipo:
            exponent = 0.1 + (intensity_val / 50.0) * 0.9 if intensity_val <= 50 else 1.0 + ((intensity_val - 50) / 50.0) * 2.0
            if "Gamma" in tipo: y = cv2.LUT(y, np.array([((i/255.0)**exponent)*255 for i in range(256)]).astype("uint8"))
            else: y = np.array(255 * (y/255.0)**exponent, dtype='uint8')
        else:
            opacity = intensity_val / 100.0
            y_orig = y.copy()
            if "Logarítmica" in tipo: y_processed = np.uint8(255 * np.log1p(y.astype(np.float32)) / np.log1p(255))
            elif "Rayleigh" in tipo: y_processed = np.uint8(255 * np.sqrt(y / 255.0))
            elif "Exponencial" in tipo: y_processed = np.uint8(255 * (1 - np.exp(-y / 255.0)))
            elif "Uniforme" in tipo: y_processed = cv2.equalizeHist(y)
            elif "Hipercúbica" in tipo: y_processed = np.uint8(255 * ((y / 255.0) ** 4))
            y = cv2.addWeighted(y_orig, 1.0 - opacity, y_processed, opacity, 0)
        img_out = cv2.cvtColor(cv2.merge((y, u, v)), cv2.COLOR_YUV2RGB)
        self.image_ready_for_segmentation = img_out 
        self.processed_cv_image = img_out
        self.current_thresholds = [] 
        self.show_images()

    def check_segmentation_mode(self, choice):
        if "Banda" in choice: self.slider_frame.pack(padx=20, pady=(0, 10), fill="x", before=self.btn_seg)
        else: self.slider_frame.pack_forget()

    def update_sliders_text(self, value):
        t1, t2 = int(self.var_t1.get()), int(self.var_t2.get())
        if t1 > t2: t1 = t2; self.var_t1.set(t1)
        self.lbl_t1.configure(text=f"T1: {t1}"); self.lbl_t2.configure(text=f"T2: {t2}")
        self.current_thresholds = [t1, t2]
        self.update_histogram() 

    def create_separator(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=self.font_header, text_color="#A0A0A0", anchor="w").pack(fill="x", padx=20, pady=(25, 10))

    def setup_projector_histogram(self):
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_hist)
        self.canvas_plot.get_tk_widget().configure(background='#2b2b2b', highlightthickness=0)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_plot.draw()

    def resize_image_event(self, event, target):
        if target == "orig" and self.original_cv_image is not None: self.display_image(self.original_cv_image, self.lbl_img_orig, event.width, event.height)
        elif target == "proc" and self.processed_cv_image is not None: self.display_image(self.processed_cv_image, self.lbl_img_proc, event.width, event.height)

    def display_image(self, cv_img, label_widget, max_w, max_h):
        if cv_img is None or max_w < 10 or max_h < 10: return
        img_h, img_w = cv_img.shape[:2]
        scale = min(max_w / img_w, max_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2: resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img 

    def show_images(self):
        if self.original_cv_image is None: return
        w_orig, h_orig = self.frame_orig.winfo_width(), self.frame_orig.winfo_height()
        w_proc, h_proc = self.frame_proc.winfo_width(), self.frame_proc.winfo_height()
        self.display_image(self.original_cv_image, self.lbl_img_orig, w_orig, h_orig)
        self.display_image(self.processed_cv_image, self.lbl_img_proc, w_proc, h_proc)
        self.update_histogram()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.tif")])
        if path:
            img = cv2.imread(path)
            self.original_cv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img_step_noise = self.original_cv_image.copy()
            self.img_step_filter = self.original_cv_image.copy()
            self.processed_cv_image = self.original_cv_image.copy()
            self.image_ready_for_segmentation = self.original_cv_image.copy()
            self.current_thresholds = []
            self.var_noise.set(0); self.var_filter.set(0); self.var_bright.set(50)
            self.show_images()

    def reset_image(self):
        if self.original_cv_image is not None:
            self.img_step_noise = self.original_cv_image.copy()
            self.img_step_filter = self.original_cv_image.copy()
            self.processed_cv_image = self.original_cv_image.copy()
            self.image_ready_for_segmentation = self.original_cv_image.copy()
            self.current_thresholds = []
            self.var_noise.set(0); self.var_filter.set(0); self.var_bright.set(50)
            self.show_images()

    def update_histogram(self):
        self.ax.clear()
        if self.processed_cv_image is None or self.original_cv_image is None: return
        img_orig = self.original_cv_image
        if len(img_orig.shape) == 3: img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
        hist_orig = cv2.calcHist([img_orig], [0], None, [256], [0, 256])
        self.ax.fill_between(range(256), hist_orig.ravel(), color='#9E9E9E', alpha=0.3, label='Original')
        img_proc = self.processed_cv_image
        if len(img_proc.shape) == 3: img_proc = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)
        hist_proc = cv2.calcHist([img_proc], [0], None, [256], [0, 256])
        self.ax.plot(hist_proc, color='#00E676', lw=2, label='Modificado')
        self.ax.fill_between(range(256), hist_proc.ravel(), color='#00E676', alpha=0.2)
        self.ax.set_title(f"HISTOGRAMA", color="white", fontsize=14, fontweight='bold')
        self.ax.tick_params(axis='both', which='major', labelsize=12, colors='#B0BEC5')
        self.ax.set_xlim([0, 256]); self.ax.grid(True, linestyle=':', alpha=0.3)
        self.ax.legend(loc='upper right', frameon=False, labelcolor='white')
        if self.current_thresholds:
            for th in self.current_thresholds:
                self.ax.axvline(th, color='#FF5252', linestyle='--', linewidth=3)
                self.ax.text(th + 2, np.max(hist_orig)*0.6, f"{int(th)}", color='#FF5252', fontweight='bold', fontsize=12, rotation=90)
        self.canvas_plot.draw()

    def get_gray(self):
        if self.image_ready_for_segmentation is not None: img = self.image_ready_for_segmentation.copy()
        else: img = self.processed_cv_image.copy()
        if len(img.shape) == 3: return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def apply_segmentation(self, combo=None): 
        if self.processed_cv_image is None: return
        if self.image_ready_for_segmentation is not None: gray = cv2.cvtColor(self.image_ready_for_segmentation, cv2.COLOR_RGB2GRAY)
        else: gray = self.get_gray()
        tipo = self.seg_combo.get()
        self.last_method_name = tipo
        if "Otsu" in tipo: th, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); self.current_thresholds = [th]
        elif "Media" in tipo: th = np.mean(gray); _, res = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY); self.current_thresholds = [th]
        elif "Banda" in tipo: T1, T2 = int(self.var_t1.get()), int(self.var_t2.get()); res = np.zeros_like(gray); res[(gray >= T1) & (gray <= T2)] = 255; self.current_thresholds = [T1, T2]
        elif "Multiumbralización" in tipo:
            try: ths = threshold_multiotsu(gray, classes=3); res = (np.digitize(gray, bins=ths) * 127).astype(np.uint8); self.current_thresholds = ths
            except: return
        elif "Kapur" in tipo: th = self.kapur_threshold(gray); _, res = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY); self.current_thresholds = [th]
        else: 
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            hist_smooth = np.convolve(hist, np.ones(20)/20, mode='same')
            th = np.argmin(hist_smooth[50:200]) + 50
            _, res = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
            self.current_thresholds = [th]
        self.processed_cv_image = res
        self.show_images()

    def kapur_threshold(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-10)
        cdf = hist.cumsum()
        entropy = -hist * np.log(hist + 1e-10)
        cum_entropy = entropy.cumsum()
        max_ent, threshold = -float('inf'), 0
        for t in range(1, 255):
            w0, w1 = cdf[t], 1 - cdf[t]
            if w0 == 0 or w1 == 0: continue
            h0 = cum_entropy[t] / w0 + np.log(w0)
            h1 = (cum_entropy[255] - cum_entropy[t]) / w1 + np.log(w1)
            if h0 + h1 > max_ent: max_ent, threshold = h0 + h1, t
        return threshold

if __name__ == "__main__":
    app = ProjectorApp()
    app.mainloop()
