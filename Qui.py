# -*- coding: utf-8 -*-
"""
EE444 — Image Filtering Lab GUI (Starter)
TODO FUNCTIONS to be implemented by students:
- apply_nonlinear_filter(img_rgb, kind, **params)
- compress_image(img_rgb, codec, quality)
- compute_diff_image(orig_rgb, comp_rgb)
- compute_metrics(orig_rgb, comp_rgb, encoded_bytes, codec)
"""
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

APP_TITLE = "EE444 — Image Filtering & Compression (Starter GUI)"

class ImageCanvas(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#1f2023")))
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pix)
        self._zoom = 1.0
        self._tilt = 0.0

    def has_image(self) -> bool:
        return not self._pix.pixmap().isNull()

    def clear_image(self):
        self._pix.setPixmap(QtGui.QPixmap())
        self._scene.setSceneRect(QtCore.QRectF())
        self._zoom = 1.0
        self._tilt = 0.0

    def set_image_rgb(self, img_rgb: np.ndarray):
        if img_rgb is None:
            self.clear_image()
            return
        h, w, ch = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self._pix.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self._pix.setOffset(-w/2, -h/2)
        self._update_transform()
        self._center_and_fit()

    def _center_and_fit(self):
        self.setSceneRect(self._pix.boundingRect())
        self.centerOn(self._pix)

    def set_zoom_percent(self, percent: int):
        percent = max(10, min(400, int(percent)))
        self._zoom = percent / 100.0
        self._update_transform()

    def set_tilt_deg(self, degrees: int):
        self._tilt = float(max(-180, min(180, degrees)))
        self._update_transform()

    def _update_transform(self):
        t = QtGui.QTransform()
        t.rotate(self._tilt)
        t.scale(self._zoom, self._zoom)
        self._pix.setTransform(t)

# ---------- TODO STUBS ----------
def apply_nonlinear_filter(img_rgb: np.ndarray, kind: str, **params) -> np.ndarray:
    """
    Apply non-linear filters to an RGB image.
    
    Parameters:
    -----------
    img_rgb : np.ndarray
        Input RGB image (H×W×3, uint8)
    kind : str
        Type of filter: "Median", "Bilateral", or "NLMeans"
    **params : dict
        ksize : int - kernel size for Median filter (must be odd)
        bilateral_d : int - diameter for Bilateral filter
        bilateral_sigma_color : float - color sigma for Bilateral
        bilateral_sigma_space : float - space sigma for Bilateral
    
    Returns:
    --------
    np.ndarray
        Filtered image (H×W×3, uint8)
    """
    img = img_rgb.copy()
    
    if kind == "Median":
        ksize = params.get('ksize', 5)
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1
        img = cv2.medianBlur(img, ksize)
    
    elif kind == "Bilateral":
        d = params.get('bilateral_d', 7)
        sigma_color = params.get('bilateral_sigma_color', 75.0)
        sigma_space = params.get('bilateral_sigma_space', 75.0)
        img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    elif kind == "NLMeans":
        # Non-local means denoising
        h = params.get('ksize', 5)  # Filter strength
        img = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    
    else:
        raise ValueError(f"Unknown filter type: {kind}")
    
    return img


def compress_image(img_rgb: np.ndarray, codec: str, quality: int) -> tuple:
    """
    Compress an RGB image using the specified codec.
    
    Parameters:
    -----------
    img_rgb : np.ndarray
        Input RGB image (H×W×3, uint8)
    codec : str
        Compression codec: "JPEG", "PNG", "WebP", or "TIFF-LZW"
    quality : int
        Quality parameter (1-100, higher is better)
    
    Returns:
    --------
    tuple: (compressed_rgb, encoded_bytes)
        compressed_rgb : np.ndarray - Decompressed image (H×W×3, uint8)
        encoded_bytes : bytes - Compressed data
    """
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Encode based on codec
    if codec == "JPEG":
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, encoded = cv2.imencode('.jpg', img_bgr, encode_param)
    
    elif codec == "PNG":
        # PNG compression level: 0-9 (we map quality 1-100 to compression 9-0)
        compression = int(9 - (quality - 1) * 9 / 99)
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        success, encoded = cv2.imencode('.png', img_bgr, encode_param)
    
    elif codec == "WebP":
        encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
        success, encoded = cv2.imencode('.webp', img_bgr, encode_param)
    
    elif codec == "TIFF-LZW":
        # TIFF with LZW compression
        encode_param = [cv2.IMWRITE_TIFF_COMPRESSION, 5]  # 5 = LZW
        success, encoded = cv2.imencode('.tiff', img_bgr, encode_param)
    
    else:
        raise ValueError(f"Unknown codec: {codec}")
    
    if not success:
        raise RuntimeError(f"Failed to encode image with {codec}")
    
    # Decode the compressed image to get the actual result
    encoded_bytes = encoded.tobytes()
    decoded_bgr = cv2.imdecode(np.frombuffer(encoded_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if decoded_bgr is None:
        raise RuntimeError(f"Failed to decode image with {codec}")
    
    # Convert back to RGB
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    
    return decoded_rgb, encoded_bytes


def compute_diff_image(orig_rgb: np.ndarray, comp_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a visual difference image between original and compressed images.
    
    Parameters:
    -----------
    orig_rgb : np.ndarray
        Original RGB image (H×W×3, uint8)
    comp_rgb : np.ndarray
        Compressed RGB image (H×W×3, uint8)
    
    Returns:
    --------
    np.ndarray
        Difference image (H×W×3, uint8) - amplified for visualization
    """
    # Ensure both images have the same shape
    if orig_rgb.shape != comp_rgb.shape:
        comp_rgb = cv2.resize(comp_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
    
    # Compute absolute difference
    diff = cv2.absdiff(orig_rgb, comp_rgb)
    
    # Amplify the difference for better visualization (multiply by 10)
    diff_amplified = np.clip(diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
    
    return diff_amplified


def compute_metrics(orig_rgb: np.ndarray, comp_rgb: np.ndarray, encoded_bytes: bytes, codec: str) -> str:
    """
    Compute quality metrics between original and compressed images.
    
    Parameters:
    -----------
    orig_rgb : np.ndarray
        Original RGB image (H×W×3, uint8)
    comp_rgb : np.ndarray
        Compressed RGB image (H×W×3, uint8)
    encoded_bytes : bytes
        Compressed image data
    codec : str
        Compression codec used
    
    Returns:
    --------
    str
        HTML-formatted string with quality metrics
    """
    # Ensure both images have the same shape
    if orig_rgb.shape != comp_rgb.shape:
        comp_rgb = cv2.resize(comp_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
    
    # Convert to float for calculations
    orig = orig_rgb.astype(np.float64)
    comp = comp_rgb.astype(np.float64)
    
    # 1. Mean Squared Error (MSE)
    mse = np.mean((orig - comp) ** 2)
    
    # 2. Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    
    # 3. Structural Similarity Index (SSIM) - simplified version
    # Using OpenCV's quality module if available, otherwise compute manually
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale for SSIM
        orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        comp_gray = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2GRAY)
        ssim_value = ssim(orig_gray, comp_gray, data_range=255)
    except ImportError:
        # Simplified SSIM calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu1 = cv2.GaussianBlur(orig, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(comp, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(orig ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(comp ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(orig * comp, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_value = np.mean(ssim_map)
    
    # 4. File sizes and compression ratio
    orig_size = orig_rgb.nbytes
    comp_size = len(encoded_bytes)
    compression_ratio = orig_size / comp_size if comp_size > 0 else 0
    
    # 5. Bits per pixel
    h, w, c = orig_rgb.shape
    total_pixels = h * w
    bpp = (comp_size * 8) / total_pixels if total_pixels > 0 else 0
    
    # Format results as HTML
    html = f"""
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f1c57a; color: #1f2023; font-weight: bold; }}
        .good {{ color: #4CAF50; font-weight: bold; }}
        .bad {{ color: #f44336; font-weight: bold; }}
    </style>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td><b>Codec</b></td><td>{codec}</td></tr>
        <tr><td><b>Image Size</b></td><td>{w} × {h} pixels</td></tr>
        <tr><td><b>Original Size</b></td><td>{orig_size:,} bytes ({orig_size/1024:.2f} KB)</td></tr>
        <tr><td><b>Compressed Size</b></td><td>{comp_size:,} bytes ({comp_size/1024:.2f} KB)</td></tr>
        <tr><td><b>Compression Ratio</b></td><td class="good">{compression_ratio:.2f}:1</td></tr>
        <tr><td><b>Bits per Pixel (BPP)</b></td><td>{bpp:.3f}</td></tr>
        <tr><td><b>MSE</b></td><td>{mse:.4f}</td></tr>
        <tr><td><b>PSNR</b></td><td class="{'good' if psnr > 30 else 'bad'}">{psnr:.2f} dB</td></tr>
        <tr><td><b>SSIM</b></td><td class="{'good' if ssim_value > 0.9 else 'bad'}">{ssim_value:.4f}</td></tr>
    </table>
    <br>
    <p><b>Quality Assessment:</b></p>
    <ul>
        <li>PSNR {'> 40 dB: Excellent' if psnr > 40 else '> 30 dB: Good' if psnr > 30 else '> 20 dB: Fair' if psnr > 20 else '< 20 dB: Poor'}</li>
        <li>SSIM {'> 0.95: Excellent' if ssim_value > 0.95 else '> 0.90: Good' if ssim_value > 0.90 else '> 0.80: Fair' if ssim_value > 0.80 else '< 0.80: Poor'}</li>
        <li>Compression: {100 * (1 - 1/compression_ratio):.1f}% space saved</li>
    </ul>
    """
    
    return html

# ---------- Main Window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 820)
        self._orig_rgb = None
        self._filtered_rgb = None
        self._comp_rgb = None
        self._comp_bytes = b""

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(12,12,12,12)

        top = QtWidgets.QHBoxLayout()
        self.btnOpen = QtWidgets.QPushButton("Load Image")
        self.btnSave = QtWidgets.QPushButton("Save Filtered")
        self.btnSave.setEnabled(False)
        self.btnSaveComp = QtWidgets.QPushButton("Save Compressed")
        self.btnSaveComp.setEnabled(False)
        top.addWidget(self.btnOpen)
        top.addStretch(1)
        top.addWidget(self.btnSave)
        top.addWidget(self.btnSaveComp)
        root_layout.addLayout(top)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter, 1)

        left = QtWidgets.QFrame(objectName="SidePanel")
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(10,10,10,10)
        left_lay.setSpacing(8)

        logo = QtWidgets.QLabel("EE444")
        logo.setAlignment(QtCore.Qt.AlignCenter)
        logo.setStyleSheet("font-weight:800; font-size: 16pt; color:#1f2023; background:#f1c57a; padding:6px")
        left_lay.addWidget(logo)

        self.stepList = QtWidgets.QListWidget()
        self.stepList.addItems(["1. Φόρτωση Εικόνας","2. Γραμμικά Φίλτρα","3. Μη Γραμμικά Φίλτρα","4. Συμπίεση","5. Αλλαγή Μεγέθους"])
        self.stepList.setFixedWidth(280)
        left_lay.addWidget(self.stepList, 1)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.edW = QtWidgets.QLineEdit()
        self.edH = QtWidgets.QLineEdit()
        self.edW.setPlaceholderText("πλάτος ή κενό")
        self.edH.setPlaceholderText("ύψος ή κενό")

        # Linear Filters Section
        form.addRow(QtWidgets.QLabel("— Γραμμικά Φίλτρα —"))
        
        self.cmbLinear = QtWidgets.QComboBox()
        self.cmbLinear.addItems(["Box/Average", "Gaussian", "Sobel X", "Sobel Y", "Laplacian", "Unsharp (α)"])
        
        self.spnLinearK = QtWidgets.QSpinBox()
        self.spnLinearK.setRange(3, 101)
        self.spnLinearK.setSingleStep(2)
        self.spnLinearK.setValue(5)
        
        self.dspSigma = QtWidgets.QDoubleSpinBox()
        self.dspSigma.setRange(0.0, 25.0)
        self.dspSigma.setValue(1.0)
        self.dspSigma.setSingleStep(0.1)
        
        self.cboBorder = QtWidgets.QComboBox()
        self.cboBorder.addItems(["reflect", "replicate", "constant"])
        
        self.chkGray = QtWidgets.QCheckBox("Grayscale processing")
        
        self.spnIter = QtWidgets.QSpinBox()
        self.spnIter.setRange(1, 50)
        self.spnIter.setValue(1)
        
        # Alpha slider for Unsharp
        alphaRow = QtWidgets.QHBoxLayout()
        self.sldAlpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldAlpha.setRange(0, 300)
        self.sldAlpha.setValue(100)
        self.lblAlpha = QtWidgets.QLabel("α = 1.00")
        alphaRow.addWidget(self.sldAlpha)
        alphaRow.addWidget(self.lblAlpha)
        
        self.btnApplyLinearFilter = QtWidgets.QPushButton("✓ Apply Linear Filter")
        self.btnApplyLinearFilter.setStyleSheet("""
            QPushButton {
                font-weight: bold; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb85c, stop:1 #4cae4c);
                color: white;
                border: 2px solid #4cae4c;
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6fc56f, stop:1 #5cb85c);
                border: 2px solid #5cb85c;
            }
            QPushButton:pressed {
                background: #449d44;
                border: 2px solid #398439;
            }
        """)
        
        form.addRow("Filter:", self.cmbLinear)
        form.addRow("Kernel (odd):", self.spnLinearK)
        form.addRow("σ (Gaussian):", self.dspSigma)
        form.addRow("Border:", self.cboBorder)
        form.addRow("", self.chkGray)
        form.addRow("Iterations:", self.spnIter)
        form.addRow("Unsharp α:", alphaRow)
        form.addRow("", self.btnApplyLinearFilter)

        # Non-Linear Filters Section
        form.addRow(QtWidgets.QLabel("— Μη Γραμμικά Φίλτρα —"))
        
        self.cmbNL = QtWidgets.QComboBox()
        self.cmbNL.addItems(["Median", "Bilateral", "NLMeans"])

        self.spnK = QtWidgets.QSpinBox()
        self.spnK.setRange(1, 99)
        self.spnK.setSingleStep(2)
        self.spnK.setValue(5)

        self.spnB_d = QtWidgets.QSpinBox()
        self.spnB_d.setRange(1, 25)
        self.spnB_d.setValue(7)

        self.spnB_sc = QtWidgets.QDoubleSpinBox()
        self.spnB_sc.setRange(1.0, 250.0)
        self.spnB_sc.setValue(75.0)

        self.spnB_ss = QtWidgets.QDoubleSpinBox()
        self.spnB_ss.setRange(1.0, 250.0)
        self.spnB_ss.setValue(75.0)

        self.btnApplyNL = QtWidgets.QPushButton("✓ Apply Non-Linear Filter")
        self.btnApplyNL.setStyleSheet("""
            QPushButton {
                font-weight: bold; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5bc0de, stop:1 #46b8da);
                color: white;
                border: 2px solid #46b8da;
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6fd4ed, stop:1 #5bc0de);
                border: 2px solid #5bc0de;
            }
            QPushButton:pressed {
                background: #31b0d5;
                border: 2px solid #269abc;
            }
        """)

        form.addRow("Επιλογή:", self.cmbNL)
        form.addRow("Kernel (odd):", self.spnK)
        form.addRow("Bilateral d:", self.spnB_d)
        form.addRow("σ_color:", self.spnB_sc)
        form.addRow("σ_space:", self.spnB_ss)
        form.addRow("", self.btnApplyNL)
        
        # Resize Section
        form.addRow(QtWidgets.QLabel("— Αλλαγή Μεγέθους —"))
        form.addRow("resize width:", self.edW)
        form.addRow("resize height:", self.edH)
        self.btnApplyLinear = QtWidgets.QPushButton("↔ Apply Resize")
        self.btnApplyLinear.setStyleSheet("""
            QPushButton {
                font-weight: bold; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f0ad4e, stop:1 #ec971f);
                color: white;
                border: 2px solid #ec971f;
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f5c26b, stop:1 #f0ad4e);
                border: 2px solid #f0ad4e;
            }
            QPushButton:pressed {
                background: #d58512;
                border: 2px solid #c77c11;
            }
        """)
        form.addRow("", self.btnApplyLinear)

        left_lay.addLayout(form)
        splitter.addWidget(left)

        self.tabs = QtWidgets.QTabWidget()

        self.tabOrig = QtWidgets.QWidget()
        oLay = QtWidgets.QVBoxLayout(self.tabOrig)
        oLay.setContentsMargins(0,0,0,0)
        self.viewOrig = ImageCanvas()
        self.viewOrig.setMinimumSize(560, 420)
        oLay.addWidget(self.viewOrig, 1)
        self.tabs.addTab(self.tabOrig, "Original")

        self.tabFilt = QtWidgets.QWidget()
        fLay = QtWidgets.QVBoxLayout(self.tabFilt)
        fLay.setContentsMargins(0,0,0,0)
        self.viewFilt = ImageCanvas()
        self.viewFilt.setMinimumSize(560, 420)
        fLay.addWidget(self.viewFilt, 1)
        self.tabs.addTab(self.tabFilt, "Filtered")

        self.tabComp = QtWidgets.QWidget()
        cLay = QtWidgets.QVBoxLayout(self.tabComp)
        cLay.setContentsMargins(8,8,8,8)

        ctrlRow = QtWidgets.QHBoxLayout()
        self.cmbCodec = QtWidgets.QComboBox()
        self.cmbCodec.addItems(["JPEG","PNG","WebP","TIFF-LZW"])

        self.sldQ = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldQ.setRange(1,100)
        self.sldQ.setValue(80)

        self.lblQ = QtWidgets.QLabel("Q=80")
        self.btnCompress = QtWidgets.QPushButton("⚙ Compress Image")
        self.btnCompress.setStyleSheet("""
            QPushButton {
                font-weight: bold; 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d9534f, stop:1 #c9302c);
                color: white;
                border: 2px solid #c9302c;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e57373, stop:1 #d9534f);
                border: 2px solid #d9534f;
            }
            QPushButton:pressed {
                background: #ac2925;
                border: 2px solid #8b211e;
            }
        """)

        ctrlRow.addWidget(QtWidgets.QLabel("Codec:"))
        ctrlRow.addWidget(self.cmbCodec)
        ctrlRow.addSpacing(16)
        ctrlRow.addWidget(QtWidgets.QLabel("Quality:"))
        ctrlRow.addWidget(self.sldQ,1)
        ctrlRow.addWidget(self.lblQ)
        ctrlRow.addWidget(self.btnCompress)
        cLay.addLayout(ctrlRow)

        viewers = QtWidgets.QHBoxLayout()
        self.viewCompOrig = ImageCanvas()
        self.viewComp = ImageCanvas()
        self.viewDiff = ImageCanvas()

        viewers.addWidget(self._boxed("Original", self.viewCompOrig), 1)
        viewers.addWidget(self._boxed("Compressed", self.viewComp), 1)
        viewers.addWidget(self._boxed("Diff", self.viewDiff), 1)
        cLay.addLayout(viewers, 1)

        self.txtMetrics = QtWidgets.QTextEdit()
        self.txtMetrics.setReadOnly(True)
        self.txtMetrics.setMinimumHeight(100)
        cLay.addWidget(self._boxed("Μετρικές Ποιότητας (TODO από φοιτητές)", self.txtMetrics))

        self.tabs.addTab(self.tabComp, "Compression")
        splitter.addWidget(self.tabs)

        right = QtWidgets.QTextEdit()
        right.setReadOnly(True)
        right.setHtml("<h3>Οδηγίες</h3><ul><li>Υλοποιήστε apply_nonlinear_filter</li><li>Υλοποιήστε compress_image</li><li>Υλοποιήστε compute_diff_image</li><li>Υλοποιήστε compute_metrics</li></ul>")
        splitter.addWidget(right)
        splitter.setSizes([320, 760, 300])

        self.btnOpen.clicked.connect(self.on_open)
        self.btnSave.clicked.connect(self.on_save_filtered)
        self.btnSaveComp.clicked.connect(self.on_save_compressed)
        self.btnApplyLinearFilter.clicked.connect(self.on_apply_linear_filter)
        self.btnApplyLinear.clicked.connect(self.on_apply_linear)
        self.btnApplyNL.clicked.connect(self.on_apply_nl)
        self.btnCompress.clicked.connect(self.on_compress)
        self.sldQ.valueChanged.connect(lambda v: self.lblQ.setText(f"Q={v}"))
        self.sldAlpha.valueChanged.connect(lambda v: self.lblAlpha.setText(f"α = {v/100:.2f}"))

    def _boxed(self, title: str, widget: QtWidgets.QWidget):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(8,8,8,8)
        lay.addWidget(widget)
        return box

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Επιλογή Εικόνας", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.critical(self, "Σφάλμα", "Αποτυχία φόρτωσης εικόνας.")
            return
        self._orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.viewOrig.set_image_rgb(self._orig_rgb)
        self._filtered_rgb = None
        self.viewFilt.clear_image()
        self._comp_rgb, self._comp_bytes = None, b""
        self.viewCompOrig.set_image_rgb(self._orig_rgb)
        self.viewComp.clear_image()
        self.viewDiff.clear_image()
        self.btnSave.setEnabled(False)
        self.btnSaveComp.setEnabled(False)
        self.tabs.setCurrentWidget(self.tabOrig)

    def on_save_filtered(self):
        if self._filtered_rgb is None:
            QtWidgets.QMessageBox.information(self, "Info", "Δεν υπάρχει φιλτραρισμένη εικόνα.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Αποθήκευση", "", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        cv2.imwrite(path, cv2.cvtColor(self._filtered_rgb, cv2.COLOR_RGB2BGR))
        QtWidgets.QMessageBox.information(self, "OK", f"Αποθηκεύτηκε: {path}")

    def on_save_compressed(self):
        if not self._comp_bytes:
            QtWidgets.QMessageBox.information(self, "Info", "Δεν υπάρχει συμπιεσμένη εικόνα.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Αποθήκευση Συμπιεσμένου", "", "All (*.*)")
        if not path:
            return
        with open(path, "wb") as f:
            f.write(self._comp_bytes)
        QtWidgets.QMessageBox.information(self, "OK", f"Αποθηκεύτηκε: {path}")

    def on_apply_linear_filter(self):
        if self._orig_rgb is None:
            QtWidgets.QMessageBox.information(self, "Info", "Φόρτωσε πρώτα μια εικόνα.")
            return
        
        filter_name = self.cmbLinear.currentText()
        ksize = int(self.spnLinearK.value())
        sigma = float(self.dspSigma.value())
        border_mode_str = self.cboBorder.currentText()
        grayscale_only = self.chkGray.isChecked()
        iterations = int(self.spnIter.value())
        unsharp_alpha = float(self.sldAlpha.value()) / 100.0
        
        try:
            from linear_filters import apply_linear_filter
        except ImportError as e:
            QtWidgets.QMessageBox.warning(
                self, "Import Error",
                f"Could not import linear_filters module:\n{e}\n\nMake sure linear_filters.py is in the same directory."
            )
            return
        
        try:
            result = apply_linear_filter(
                self._orig_rgb,
                filter_name,
                ksize,
                sigma=sigma,
                border_mode_str=border_mode_str,
                grayscale_only=grayscale_only,
                iterations=iterations,
                unsharp_alpha=unsharp_alpha
            )
            self._filtered_rgb = result
            self.viewFilt.set_image_rgb(self._filtered_rgb)
            self.btnSave.setEnabled(True)
            self.tabs.setCurrentWidget(self.tabFilt)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Σφάλμα", f"Error applying linear filter:\n{e}")

    def on_apply_linear(self):
        if self._orig_rgb is None:
            QtWidgets.QMessageBox.information(self, "Info", "Φόρτωσε πρώτα μια εικόνα.")
            return
        wtxt = self.edW.text().strip()
        htxt = self.edH.text().strip()
        w = int(wtxt) if wtxt else None
        h = int(htxt) if htxt else None
        img = self._orig_rgb.copy()
        if w or h:
            if w is None:
                scale = h / img.shape[0]
                w = int(round(img.shape[1]*scale))
            elif h is None:
                scale = w / img.shape[1]
                h = int(round(img.shape[0]*scale))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        self._filtered_rgb = img
        self.viewFilt.set_image_rgb(self._filtered_rgb)
        self.btnSave.setEnabled(True)
        self.tabs.setCurrentWidget(self.tabFilt)

    def on_apply_nl(self):
        if self._orig_rgb is None:
            QtWidgets.QMessageBox.information(self, "Info", "Φόρτωσε πρώτα μια εικόνα.")
            return
        kind = self.cmbNL.currentText()
        params = dict(
            ksize=int(self.spnK.value()),
            bilateral_d=int(self.spnB_d.value()),
            bilateral_sigma_color=float(self.spnB_sc.value()),
            bilateral_sigma_space=float(self.spnB_ss.value())
        )
        try:
            out = apply_nonlinear_filter(self._orig_rgb, kind, **params)
            self._filtered_rgb = out
            self.viewFilt.set_image_rgb(out)
            self.btnSave.setEnabled(True)
            self.tabs.setCurrentWidget(self.tabFilt)
        except NotImplementedError as e:
            QtWidgets.QMessageBox.information(self, "TODO", str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Σφάλμα", str(e))

    def on_compress(self):
        if self._orig_rgb is None:
            QtWidgets.QMessageBox.information(self, "Info", "Φόρτωσε πρώτα μια εικόνα.")
            return
        codec = self.cmbCodec.currentText()
        quality = int(self.sldQ.value())
        try:
            comp_rgb, enc = compress_image(self._orig_rgb, codec, quality)
            self._comp_rgb = comp_rgb
            self._comp_bytes = enc
            self.viewCompOrig.set_image_rgb(self._orig_rgb)
            self.viewComp.set_image_rgb(comp_rgb)
            diff_rgb = compute_diff_image(self._orig_rgb, comp_rgb)
            self.viewDiff.set_image_rgb(diff_rgb)
            html = compute_metrics(self._orig_rgb, comp_rgb, enc, codec)
            self.txtMetrics.setHtml(html)
            self.btnSaveComp.setEnabled(True)
            self.tabs.setCurrentWidget(self.tabComp)
        except NotImplementedError as e:
            QtWidgets.QMessageBox.information(self, "TODO", str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Σφάλμα", str(e))

if __name__ == "__main__":
    import sys
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
