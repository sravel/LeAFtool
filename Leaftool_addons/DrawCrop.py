from PyQt5.QtCore import Qt
import PyQt5.QtWidgets as qt
import UsefullVariables as vrb
from Leaftool_addons.commonWidget import NumberLineEditLabel, FileSelectorLeaftool, style, get_files_ext, allow_ext


class DrawCropMargin(qt.QGroupBox):
    def __init__(self):
        super().__init__()

        # Layout Style
        self.setTitle("Margins")
        self.setStyleSheet(style)
        self.setFixedSize(int(250 * vrb.ratio), int(110 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(5, 10, 5, 10)
        self.setLayout(self.layout)

        # Widgets
        self.left = NumberLineEditLabel(constraint="Natural", text="0", label="Left:")
        self.top = NumberLineEditLabel(constraint="Natural", text="0", label="Top:")
        self.right = NumberLineEditLabel(constraint="Natural", text="0", label="Right:")
        self.bottom = NumberLineEditLabel(constraint="Natural", text="0", label="Bottom:")

        # Position widgets
        self.layout.addWidget(self.left, 0, 0)
        self.layout.addWidget(self.top, 0, 1)
        self.layout.addWidget(self.right, 1, 0)
        self.layout.addWidget(self.bottom, 1, 1)


class DrawCropCutPart(qt.QGroupBox):
    def __init__(self):
        super().__init__()

        # Layout Style
        self.setTitle("Cut part")
        self.setStyleSheet(style)
        self.setFixedSize(int(140 * vrb.ratio), int(110 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(5, 10, 5, 10)
        self.setLayout(self.layout)

        # Widgets
        self.x_pieces = NumberLineEditLabel(constraint="Natural", text="0", label="X parts:")
        self.y_pieces = NumberLineEditLabel(constraint="Natural", text="0", label="Y parts:")

        # Position widgets
        self.layout.addWidget(self.x_pieces, 0, 0)
        self.layout.addWidget(self.y_pieces, 1, 0)


class DrawCropParams(qt.QGroupBox):
    """params for draw and/or crop option"""

    def __init__(self, parent):
        super().__init__()
        self.csv_path = None
        self.parent = parent
        self.loading = False

        # Layout Style
        self.setTitle("Draw or/and Crop options")
        self.setStyleSheet(style)
        self.setFixedSize(int(498 * vrb.ratio), int(330 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 20, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.images_path = FileSelectorLeaftool(label="Images Path:")
        self.images_ext = qt.QComboBox()
        # self.images_ext.setVisible(False)
        self.images_ext.setFixedSize(int(45 * vrb.ratio), int(25 * vrb.ratio))
        self.out_draw_dir = FileSelectorLeaftool(label="Output Draw:")
        self.out_cut_dir = FileSelectorLeaftool(label="Output Crop:")
        self.cut_part = DrawCropCutPart()
        self.margin = DrawCropMargin()

        self.noise_remove = qt.QCheckBox()
        self.noise_remove.setText("Noise Remove")
        self.noise_remove.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.force_rerun = qt.QCheckBox()
        self.force_rerun.setText("Force rerun")
        self.force_rerun.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.numbering = qt.QComboBox()
        self.numbering.addItems(["Bottom", "Right"])
        self.numbering.setFixedSize(int(100 * vrb.ratio), int(25 * vrb.ratio))

        # Position widgets
        spacerItem = qt.QSpacerItem(10, 10, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.layout.addWidget(self.images_path, 0, 0, 1, 2)
        self.layout.addWidget(self.images_ext, 0, 3)
        self.layout.addWidget(self.out_draw_dir, 1, 0, 1, 2)
        self.layout.addWidget(self.out_cut_dir, 2, 0, 1, 2)

        self.layout.addItem(spacerItem, 4, 0, 1, 1)
        self.layout.addWidget(self.cut_part, 5, 0, 1, 1)
        self.layout.addWidget(self.margin, 5, 1, 1, 1)
        self.layout.addWidget(self.force_rerun, 6, 0, 1, 1)
        self.layout.addWidget(self.noise_remove, 6, 1, 1, 1)
        self.layout.addWidget(self.numbering, 6, 2, 1, 1)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 2)
        self.layout.setColumnStretch(2, 3)

        # size policy
        not_resize = self.out_draw_dir.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.out_draw_dir.setSizePolicy(not_resize)
        not_resize = self.out_cut_dir.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.out_cut_dir.setSizePolicy(not_resize)
        not_resize = self.noise_remove.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.noise_remove.setSizePolicy(not_resize)

        not_resize = self.numbering.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.numbering.setSizePolicy(not_resize)

        # Connection
        self.images_path.lineEditFile.editingFinished.connect(self.update_draw_crop_params)
        self.images_path.lineEditFile.editingFinished.connect(self.update_ext)
        self.out_cut_dir.lineEditFile.editingFinished.connect(self.update_draw_crop_params)
        self.out_draw_dir.lineEditFile.editingFinished.connect(self.update_draw_crop_params)
        self.cut_part.x_pieces.lineEdit.editingFinished.connect(self.update_draw_crop_params)
        self.cut_part.y_pieces.lineEdit.editingFinished.connect(self.update_draw_crop_params)
        self.margin.left.lineEdit.editingFinished.connect(self.update_draw_crop_params)
        self.margin.top.lineEdit.editingFinished.connect(self.update_draw_crop_params)
        self.margin.right.lineEdit.editingFinished.connect(self.update_draw_crop_params)
        self.margin.bottom.lineEdit.editingFinished.connect(self.update_draw_crop_params)
        self.noise_remove.stateChanged.connect(self.update_draw_crop_params)
        self.force_rerun.stateChanged.connect(self.update_draw_crop_params)
        self.images_ext.currentIndexChanged.connect(self.update_draw_crop_params)
        self.numbering.currentIndexChanged.connect(self.update_draw_crop_params)

    def update_draw_crop_params(self):
        # try:
        if not self.loading:
            # from pprint import pprint as pp
            # pp(self.parent.dict_for_yaml)
            self.parent.dict_for_yaml["DRAWCROP"]["images_path"] = self.images_path.lineEditFile.text()
            self.parent.dict_for_yaml["DRAWCROP"]["out_draw_dir"] = self.out_draw_dir.lineEditFile.text()
            self.parent.dict_for_yaml["DRAWCROP"]["out_cut_dir"] = self.out_cut_dir.lineEditFile.text()
            self.parent.dict_for_yaml["log_path"] = self.out_cut_dir.lineEditFile.text()
            if (self.parent.dict_for_yaml["RUNSTEP"]["ML"] or self.parent.dict_for_yaml["RUNSTEP"]["merge"]) and self.out_cut_dir.lineEditFile.text() != "":
                self.parent.dict_for_yaml["ML"]["images_path"] = self.out_cut_dir.lineEditFile.text()
                self.parent.layer_ml_merge.images_path.lineEditFile.setText(self.parent.dict_for_yaml["DRAWCROP"]["out_cut_dir"])
            self.parent.dict_for_yaml["DRAWCROP"]["x_pieces"] = int(self.cut_part.x_pieces.lineEdit.text())
            self.parent.dict_for_yaml["DRAWCROP"]["y_pieces"] = int(self.cut_part.y_pieces.lineEdit.text())
            self.parent.dict_for_yaml["DRAWCROP"]["left"] = int(self.margin.left.lineEdit.text())
            self.parent.dict_for_yaml["DRAWCROP"]["top"] = int(self.margin.top.lineEdit.text())
            self.parent.dict_for_yaml["DRAWCROP"]["right"] = int(self.margin.right.lineEdit.text())
            self.parent.dict_for_yaml["DRAWCROP"]["bottom"] = int(self.margin.bottom.lineEdit.text())
            self.parent.dict_for_yaml["DRAWCROP"]["noise_remove"] = bool(self.noise_remove.isChecked())
            self.parent.dict_for_yaml["DRAWCROP"]["force_rerun"] = bool(self.force_rerun.isChecked())
            self.parent.dict_for_yaml["DRAWCROP"]["extension"] = self.images_ext.currentText()
            self.parent.dict_for_yaml["DRAWCROP"]["numbering"] = self.numbering.currentText()
            self.parent.preview_config.setText(self.parent.export_use_yaml)
        # except Exception as e:
        #     print(f"WARNING update_draw_crop_params: {e}")
        #     pass

    def upload_draw_crop_params(self):
        try:
            self.loading = True
            if self.parent.dict_for_yaml["RUNSTEP"]["draw"] or self.parent.dict_for_yaml["RUNSTEP"]["crop"]:
                self.images_path.lineEditFile.setText(self.parent.dict_for_yaml["DRAWCROP"]["images_path"])
                self.out_draw_dir.lineEditFile.setText(self.parent.dict_for_yaml["DRAWCROP"]["out_draw_dir"])
                self.out_cut_dir.lineEditFile.setText(self.parent.dict_for_yaml["DRAWCROP"]["out_cut_dir"])
                self.cut_part.x_pieces.lineEdit.setText(str(self.parent.dict_for_yaml["DRAWCROP"]["x_pieces"]))
                self.cut_part.y_pieces.lineEdit.setText(str(self.parent.dict_for_yaml["DRAWCROP"]["y_pieces"]))
                self.margin.left.lineEdit.setText(str(self.parent.dict_for_yaml["DRAWCROP"]["left"]))
                self.margin.top.lineEdit.setText(str(self.parent.dict_for_yaml["DRAWCROP"]["top"]))
                self.margin.right.lineEdit.setText(str(self.parent.dict_for_yaml["DRAWCROP"]["right"]))
                self.margin.bottom.lineEdit.setText(str(self.parent.dict_for_yaml["DRAWCROP"]["bottom"]))
                self.noise_remove.setChecked(bool(self.parent.dict_for_yaml["DRAWCROP"]["noise_remove"]))
                self.force_rerun.setChecked(bool(self.parent.dict_for_yaml["DRAWCROP"]["force_rerun"]))
                self.images_ext.addItem(self.parent.dict_for_yaml["DRAWCROP"]["extension"])
                self.images_ext.setCurrentText(self.parent.dict_for_yaml["DRAWCROP"]["extension"])
                self.update_ext()
            self.show_draw_params()
            self.loading = False
        except KeyError as e:
            self.parent.logger.error(f"ERROR: Key {e} is not found on file")
            self.parent.dict_for_yaml = self.parent.dict_backup

    def show_draw_params(self):
        if self.parent.layer_tools.draw_checkbox.isChecked() or self.parent.layer_tools.crop_checkbox.isChecked():
            self.setVisible(True)
            self.out_draw_dir.setVisible(self.parent.layer_tools.draw_checkbox.isChecked())
            self.out_cut_dir.setVisible(self.parent.layer_tools.crop_checkbox.isChecked())
            self.noise_remove.setVisible(self.parent.layer_tools.crop_checkbox.isChecked())
            self.numbering.setVisible(self.parent.layer_tools.crop_checkbox.isChecked())
        else:
            self.setVisible(False)

    def update_ext(self):
        files, ext_list = get_files_ext(self.images_path.lineEditFile.text(), allow_ext)
        self.images_ext.clear()
        if ext_list:
            self.images_ext.addItems(ext_list)
            self.images_ext.setCurrentIndex(0)
            self.images_ext.setVisible(True)


if __name__ == '__main__':
    from PyQt5.QtCore import Qt, QCoreApplication
    import sys

    app = QCoreApplication.instance()
    if app is None:
        app = qt.QApplication([])

    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook

    foo = DrawCropParams(parent=None)

    # foo.showMaximized()
    foo.show()
    app.exec_()
