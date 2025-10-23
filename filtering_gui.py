# filtering_gui.py
# -*- coding: utf-8 -*-
"""
Image Filtering — Linear (GUI only)
-----------------------------------
This is a ready-to-run GUI skeleton for linear spatial filtering.
Students will implement the actual filtering logic separately.
"""
import os
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

APP_TITLE = "Image Filtering — Linear"

DARK_QSS = """
* { font-family: 'Segoe UI', Arial; font-size: 10.5pt; color: #E6E6E6; }
QWidget { background-color: #1f2023; }
QFrame#SidePanel { background-color: #2b2d31; border: 1px solid #2b2d31; border-radius: 8px; }
QFrame#TopBar { background-color: #2b2d31; border: 1px solid #2b2d31; border-radius: 8px; }
QLabel#Banner { color: #f1c57a; font-weight: 700; }
QPushButton { background-color: #f1c57a; color: #2b2d31; border: 0px; padding: 8px 14px; border-radius: 6px; }
QPushButton:hover { background-color: #ffd28e; }
QPushButton:pressed { background-color: #e7b86a; }
QPushButton#Secondary { background: #3a3c42; color: #E6E6E6; font-weight: 500; }
QPushButton#Secondary:hover { background: #44474e; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QSlider {
    background: #1c1d21; border: 1px solid #3a3c42; padding: 6px; border-radius: 6px;
}
QSlider::groove:horizontal { height: 6px; background: #3a3c42; border-radius: 3px; }
QSlider::handle:horizontal { background: #f1c57a; width: 14px; border-radius: 7px; margin: -4px 0; }
"""

def np_rgb_to_qpixmap(img_rgb: np.ndarray, target_size: QtCore.QSize) -> QtGui.QPixmap:
    if img_rgb is None:
        return QtGui.QPixmap()
    h, w, ch = img_rgb.shape
    qimg = QtGui.QImage(img_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg)
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

class ScaledImageLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_rgb = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(560, 420)
        self.setStyleSheet("border: 1px dashed #3a3c42;")
    
    def set_image_rgb(self, img_rgb: np.ndarray):
        self._last_rgb = img_rgb
        self._update_pix()
    
    def clear_image(self, placeholder="— no image —"):
        self._last_rgb = None
        self.setText(placeholder)
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_pix()
    
    def _update_pix(self):
        if self._last_rgb is None:
            return
        pix = np_rgb_to_qpixmap(self._last_rgb, self.size())
        self.setPixmap(pix)
        self.setText("")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 720)
        self._orig = None
        self._filtered = None
        
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)
        
        # Top bar
        self.topBar = QtWidgets.QFrame(objectName="TopBar")
        tlay = QtWidgets.QHBoxLayout(self.topBar)
        tlay.setContentsMargins(12, 8, 12, 8)
        self.title = QtWidgets.QLabel(APP_TITLE, objectName="Banner")
        self.btnOpen = QtWidgets.QPushButton("Load Image")
        self.btnOpen.setObjectName("Secondary")
        self.btnSave = QtWidgets.QPushButton("Save Result")
        self.btnSave.setObjectName("Secondary")
        
        tlay.addWidget(self.title)
        tlay.addStretch(1)
        tlay.addWidget(self.btnOpen)
        tlay.addWidget(self.btnSave)
        root_layout.addWidget(self.topBar)
        
        # Main splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter, 1)
        
        # Side panel
        self.side = QtWidgets.QFrame(objectName="SidePanel")
        side_layout = QtWidgets.QVBoxLayout(self.side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(16)
        
        side_layout.addWidget(QtWidgets.QLabel("DIP LAB — Linear Filters", objectName="Banner", alignment=QtCore.Qt.AlignCenter))
        
        # Form layout for controls
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.cboFilter = QtWidgets.QComboBox()
        self.cboFilter.addItems(["Box/Average", "Gaussian", "Sobel X", "Sobel Y", "Laplacian", "Unsharp (α)"])
        
        self.spnK = QtWidgets.QSpinBox()
        self.spnK.setRange(3, 101)
        self.spnK.setSingleStep(2)
        self.spnK.setValue(5)
        
        self.dspSigma = QtWidgets.QDoubleSpinBox()
        self.dspSigma.setRange(0.0, 25.0)
        self.dspSigma.setValue(1.0)
        self.dspSigma.setSingleStep(0.1)
        
        self.spnIter = QtWidgets.QSpinBox()
        self.spnIter.setRange(1, 50)
        self.spnIter.setValue(1)
        
        self.cboBorder = QtWidgets.QComboBox()
        self.cboBorder.addItems(["reflect", "replicate", "constant"])
        
        self.chkGray = QtWidgets.QCheckBox("Grayscale processing")
        
        # Alpha slider for Unsharp
        alphaRow = QtWidgets.QHBoxLayout()
        self.sldAlpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldAlpha.setRange(0, 300)
        self.sldAlpha.setValue(100)
        self.lblAlpha = QtWidgets.QLabel("α = 1.00")
        alphaRow.addWidget(self.sldAlpha)
        alphaRow.addWidget(self.lblAlpha)
        
        form.addRow("Filter:", self.cboFilter)
        form.addRow("Kernel size (odd):", self.spnK)
        form.addRow("σ (Gaussian):", self.dspSigma)
        form.addRow("Iterations:", self.spnIter)
        form.addRow("Border:", self.cboBorder)
        form.addRow("", self.chkGray)
        form.addRow("Unsharp α:", alphaRow)
        
        side_layout.addLayout(form)
        
        # Buttons
        btnRow = QtWidgets.QHBoxLayout()
        self.btnApply = QtWidgets.QPushButton("Apply")
        self.btnReset = QtWidgets.QPushButton("Reset")
        btnRow.addWidget(self.btnApply)
        btnRow.addWidget(self.btnReset)
        side_layout.addLayout(btnRow)
        
        side_layout.addStretch()
        splitter.addWidget(self.side)
        
        # Image tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabOrig = QtWidgets.QWidget()
        self.tabFilt = QtWidgets.QWidget()
        self.tabs.addTab(self.tabOrig, "Original")
        self.tabs.addTab(self.tabFilt, "Filtered")
        
        # Original image tab
        oLay = QtWidgets.QVBoxLayout(self.tabOrig)
        oLay.setContentsMargins(8, 8, 8, 8)
        self.viewOrig = ScaledImageLabel("— Load an image —")
        oLay.addWidget(self.viewOrig, 1)
        
        # Filtered image tab
        fLay = QtWidgets.QVBoxLayout(self.tabFilt)
        fLay.setContentsMargins(8, 8, 8, 8)
        self.viewFilt = ScaledImageLabel("— Apply a filter —")
        fLay.addWidget(self.viewFilt, 1)
        
        splitter.addWidget(self.tabs)
        splitter.setSizes([360, 800])
        
        # Connect signals
        self.btnOpen.clicked.connect(self.on_open)
        self.btnSave.clicked.connect(self.on_save)
        self.btnApply.clicked.connect(self.on_apply)
        self.btnReset.clicked.connect(self.on_reset)
        self.sldAlpha.valueChanged.connect(lambda v: self.lblAlpha.setText(f"α = {v/100:.2f}"))
        
        # Apply style
        self.setStyleSheet(DARK_QSS)
    
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return
        
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Could not load image.")
            return
        
        self._orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._filtered = None
        self.viewOrig.set_image_rgb(self._orig)
        self.viewFilt.clear_image("— Apply a filter —")
        self.tabs.setCurrentWidget(self.tabOrig)
    
    def on_apply(self):
        if self._orig is None:
            QtWidgets.QMessageBox.information(self, "Info", "Load an image first.")
            return
        
        params = dict(
            img_rgb=self._orig,
            filter_name=self.cboFilter.currentText(),
            ksize=int(self.spnK.value()),
            sigma=float(self.dspSigma.value()),
            border_mode_str=self.cboBorder.currentText(),
            grayscale_only=self.chkGray.isChecked(),
            iterations=int(self.spnIter.value()),
            unsharp_alpha=float(self.sldAlpha.value())/100.0
        )
        
        try:
            from linear_filters import apply_linear_filter
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Not implemented",
                "Λείπει το αρχείο ή η συνάρτηση: linear_filters.apply_linear_filter(...)\n\n" + str(e)
            )
            return
        
        try:
            result = apply_linear_filter(**params)
        except NotImplementedError:
            QtWidgets.QMessageBox.information(
                self, "TODO", "Υλοποιήστε τη συνάρτηση apply_linear_filter"
            )
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Σφάλμα κατά το φιλτράρισμα:\n{e}")
            return
        
        self._filtered = result.astype(np.uint8)
        self.viewFilt.set_image_rgb(self._filtered)
        self.tabs.setCurrentWidget(self.tabFilt)
    
    def on_reset(self):
        if self._orig is not None:
            self.viewOrig.set_image_rgb(self._orig)
        self._filtered = None
        self.viewFilt.clear_image("— Apply a filter —")
        self.tabs.setCurrentWidget(self.tabOrig)
    
    def on_save(self):
        if self._filtered is None:
            QtWidgets.QMessageBox.information(self, "Info", "No result to save.")
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", "", "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        
        bgr = cv2.cvtColor(self._filtered, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        QtWidgets.QMessageBox.information(self, "OK", f"Saved: {path}")

if __name__ == "__main__":
    import sys
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())