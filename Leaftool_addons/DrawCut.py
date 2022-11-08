from functools import partial
import PyQt5.QtWidgets as qt
import UsefullVariables as vrb
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5 import QtGui

from Leaftool_addons.commonWidget import NumberLineEditLabel, FileSelectorLeaftool, SpinBoxLabel, style, get_files_ext, allow_ext, int_validator, check_values


class DrawCutMargin(qt.QGroupBox):
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
        self.left.lineEdit.setValidator(int_validator)
        self.top = NumberLineEditLabel(constraint="Natural", text="0", label="Top:")
        self.top.lineEdit.setValidator(int_validator)
        self.right = NumberLineEditLabel(constraint="Natural", text="0", label="Right:")
        self.right.lineEdit.setValidator(int_validator)
        self.bottom = NumberLineEditLabel(constraint="Natural", text="0", label="Bottom:")
        self.bottom.lineEdit.setValidator(int_validator)

        # Position widgets
        self.layout.addWidget(self.left, 0, 0)
        self.layout.addWidget(self.top, 0, 1)
        self.layout.addWidget(self.right, 1, 0)
        self.layout.addWidget(self.bottom, 1, 1)


class DrawCutPart(qt.QGroupBox):
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
        self.x_pieces = SpinBoxLabel(min_v=1, max_v=10, step=1, value=1, label="X parts:")
        self.y_pieces = SpinBoxLabel(min_v=1, max_v=10, step=1, value=1, label="Y parts:")

        # Position widgets
        self.layout.addWidget(self.x_pieces, 0, 0)
        self.layout.addWidget(self.y_pieces, 1, 0)


class DrawCutParams(qt.QGroupBox):
    """params for draw and/or cut option"""

    def __init__(self, parent):
        super().__init__()
        self.csv_path = None
        self.parent = parent
        self.loading = False

        # Layout Style
        self.setTitle("Draw or/and Cut options")
        self.setStyleSheet(style)
        self.setFixedSize(int(498 * vrb.ratio), int(320 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 20, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.images_path = FileSelectorLeaftool(label="Images Path:", )
        self.images_path.setWhatsThis(self.parent.parent.dico_doc["images_path"])
        self.images_path.setStatusTip(self.parent.parent.dico_doc_str["images_path"])
        self.images_ext = qt.QComboBox()
        self.images_ext.setWhatsThis(self.parent.parent.dico_doc["extension"])
        self.images_ext.setStatusTip(self.parent.parent.dico_doc_str["extension"])
        # self.images_ext.setVisible(False)
        self.images_ext.setFixedSize(int(45 * vrb.ratio), int(25 * vrb.ratio))
        self.out_draw_dir = FileSelectorLeaftool(label="Output Draw:")
        self.out_draw_dir.setWhatsThis(self.parent.parent.dico_doc["out_draw_dir"])
        self.out_draw_dir.setStatusTip(self.parent.parent.dico_doc_str["out_draw_dir"])
        self.out_cut_dir = FileSelectorLeaftool(label="Output Cut:")
        self.out_cut_dir.setWhatsThis(self.parent.parent.dico_doc["out_cut_dir"])
        self.out_cut_dir.setStatusTip(self.parent.parent.dico_doc_str["out_cut_dir"])
        self.cut_part = DrawCutPart()
        self.cut_part.setWhatsThis(self.parent.parent.dico_doc["cut_part"])
        self.cut_part.setStatusTip(self.parent.parent.dico_doc_str["cut_part"])
        self.cut_part.x_pieces.setWhatsThis(self.parent.parent.dico_doc["x_pieces"])
        self.cut_part.y_pieces.setWhatsThis(self.parent.parent.dico_doc["y_pieces"])
        self.cut_part.x_pieces.setStatusTip(self.parent.parent.dico_doc_str["x_pieces"])
        self.cut_part.y_pieces.setStatusTip(self.parent.parent.dico_doc_str["y_pieces"])

        self.margin = DrawCutMargin()
        self.margin.setWhatsThis(self.parent.parent.dico_doc["margin"])
        self.margin.top.setWhatsThis(self.parent.parent.dico_doc["top"])
        self.margin.left.setWhatsThis(self.parent.parent.dico_doc["left"])
        self.margin.bottom.setWhatsThis(self.parent.parent.dico_doc["bottom"])
        self.margin.right.setWhatsThis(self.parent.parent.dico_doc["right"])
        self.margin.setStatusTip(self.parent.parent.dico_doc_str["margin"])
        self.margin.top.setStatusTip(self.parent.parent.dico_doc_str["top"])
        self.margin.left.setStatusTip(self.parent.parent.dico_doc_str["left"])
        self.margin.bottom.setStatusTip(self.parent.parent.dico_doc_str["bottom"])
        self.margin.right.setStatusTip(self.parent.parent.dico_doc_str["right"])

        self.noise_remove = qt.QCheckBox()
        self.noise_remove.setText("Noise Remove")
        self.noise_remove.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))
        self.noise_remove.setWhatsThis(self.parent.parent.dico_doc["noise_remove"])
        self.noise_remove.setStatusTip(self.parent.parent.dico_doc_str["noise_remove"])

        self.force_rerun = qt.QCheckBox()
        self.force_rerun.setText("Force rerun")
        self.force_rerun.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))
        self.force_rerun.setWhatsThis(self.parent.parent.dico_doc["force_rerun"])
        self.force_rerun.setStatusTip(self.parent.parent.dico_doc_str["force_rerun"])

        self.numbering = qt.QComboBox()
        self.numbering.addItems(["Bottom", "Right"])
        self.numbering.setFixedSize(int(100 * vrb.ratio), int(25 * vrb.ratio))
        self.numbering.setWhatsThis(self.parent.parent.dico_doc["numbering"])
        self.numbering.setStatusTip(self.parent.parent.dico_doc_str["numbering"])

        # Position widgets
        spacerItem = qt.QSpacerItem(10, 10, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.layout.addWidget(self.images_path, 0, 0, 1, 2)
        self.layout.addWidget(self.images_ext, 0, 3, Qt.AlignLeft)
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

        # Connection , keys_list
        self.images_path.lineEditFile.textChanged.connect(partial(self.parent.check_path, from_object=self.images_path.lineEditFile, keys_list="'DRAW-CUT' 'images_path'"))
        self.images_path.lineEditFile.editingFinished.connect(self.update_ext)
        self.out_cut_dir.lineEditFile.textChanged.connect(partial(self.parent.check_path, from_object=self.out_cut_dir.lineEditFile, keys_list="'DRAW-CUT' 'out_draw_dir'"))
        self.out_draw_dir.lineEditFile.textChanged.connect(partial(self.parent.check_path, from_object=self.out_draw_dir.lineEditFile, keys_list="'DRAW-CUT' 'out_cut_dir'"))

        self.cut_part.x_pieces.lineEdit.valueChanged.connect(self.update_draw_cut_params)
        self.cut_part.y_pieces.lineEdit.valueChanged.connect(self.update_draw_cut_params)
        self.margin.left.lineEdit.editingFinished.connect(self.update_draw_cut_params)
        self.margin.top.lineEdit.editingFinished.connect(self.update_draw_cut_params)
        self.margin.right.lineEdit.editingFinished.connect(self.update_draw_cut_params)
        self.margin.bottom.lineEdit.editingFinished.connect(self.update_draw_cut_params)
        self.noise_remove.stateChanged.connect(self.update_draw_cut_params)
        self.force_rerun.stateChanged.connect(self.update_draw_cut_params)
        self.images_ext.currentIndexChanged.connect(self.update_draw_cut_params)
        self.numbering.currentIndexChanged.connect(self.update_draw_cut_params)

    def update_draw_cut_params(self):
        # try:
        if not self.loading:
            # from pprint import pprint as pp
            # pp(self.parent.dict_for_yaml)
            self.parent.dict_for_yaml["DRAW-CUT"]["images_path"] = self.images_path.lineEditFile.text()
            self.parent.dict_for_yaml["DRAW-CUT"]["out_draw_dir"] = self.out_draw_dir.lineEditFile.text()
            self.parent.dict_for_yaml["DRAW-CUT"]["out_cut_dir"] = self.out_cut_dir.lineEditFile.text()
            self.parent.dict_for_yaml["log_path"] = self.out_cut_dir.lineEditFile.text()
            if (self.parent.dict_for_yaml["RUNSTEP"]["ML"] or self.parent.dict_for_yaml["RUNSTEP"]["merge"]) and self.out_cut_dir.lineEditFile.text() != "":
                self.parent.dict_for_yaml["ML"]["images_path"] = self.out_cut_dir.lineEditFile.text()
                self.parent.layer_ml_merge.images_path.lineEditFile.setText(self.parent.dict_for_yaml["DRAW-CUT"]["out_cut_dir"])
            self.parent.dict_for_yaml["DRAW-CUT"]["x_pieces"] = int(self.cut_part.x_pieces.lineEdit.value())
            self.parent.dict_for_yaml["DRAW-CUT"]["y_pieces"] = int(self.cut_part.y_pieces.lineEdit.value())
            self.parent.dict_for_yaml["DRAW-CUT"]["left"] = int(self.margin.left.lineEdit.text())
            self.parent.dict_for_yaml["DRAW-CUT"]["top"] = int(self.margin.top.lineEdit.text())
            self.parent.dict_for_yaml["DRAW-CUT"]["right"] = int(self.margin.right.lineEdit.text())
            self.parent.dict_for_yaml["DRAW-CUT"]["bottom"] = int(self.margin.bottom.lineEdit.text())
            self.parent.dict_for_yaml["DRAW-CUT"]["noise_remove"] = bool(self.noise_remove.isChecked())
            self.parent.dict_for_yaml["DRAW-CUT"]["force_rerun"] = bool(self.force_rerun.isChecked())
            self.parent.dict_for_yaml["DRAW-CUT"]["extension"] = self.images_ext.currentText()
            self.parent.dict_for_yaml["DRAW-CUT"]["numbering"] = self.numbering.currentText()
            self.parent.preview_config.setText(self.parent.export_use_yaml)
        # except Exception as e:
        #     print(f"WARNING update_draw_cut_params: {e}")
        #     pass

    def upload_draw_cut_params(self):
        try:
            self.loading = True
            # if self.parent.dict_for_yaml["RUNSTEP"]["draw"] or self.parent.dict_for_yaml["RUNSTEP"]["cut"]:
            self.images_path.lineEditFile.setText(self.parent.dict_for_yaml["DRAW-CUT"]["images_path"])
            self.out_draw_dir.lineEditFile.setText(self.parent.dict_for_yaml["DRAW-CUT"]["out_draw_dir"])
            self.out_cut_dir.lineEditFile.setText(self.parent.dict_for_yaml["DRAW-CUT"]["out_cut_dir"])

            # check all integer values
            message = check_values(dico_params=self.parent.dict_for_yaml,
                                   primary_key="DRAW-CUT",
                                   secondary_key_list=["x_pieces", "y_pieces", "left", "top", "right", "bottom"],
                                   type_value=int,
                                   default_error=1)
            if message:
                self.parent.logger.error(message)

            self.cut_part.x_pieces.lineEdit.setValue(int(self.parent.dict_for_yaml["DRAW-CUT"]["x_pieces"]))
            self.cut_part.y_pieces.lineEdit.setValue(int(self.parent.dict_for_yaml["DRAW-CUT"]["y_pieces"]))
            self.margin.left.lineEdit.setText(str(self.parent.dict_for_yaml["DRAW-CUT"]["left"]))
            self.margin.top.lineEdit.setText(str(self.parent.dict_for_yaml["DRAW-CUT"]["top"]))
            self.margin.right.lineEdit.setText(str(self.parent.dict_for_yaml["DRAW-CUT"]["right"]))
            self.margin.bottom.lineEdit.setText(str(self.parent.dict_for_yaml["DRAW-CUT"]["bottom"]))

            # check all boolean values
            message = check_values(dico_params=self.parent.dict_for_yaml,
                                   primary_key="DRAW-CUT",
                                   secondary_key_list=["noise_remove", "force_rerun"],
                                   type_value=bool,
                                   default_error=False)
            if message:
                self.parent.logger.error(message)

            self.noise_remove.setChecked(bool(self.parent.dict_for_yaml["DRAW-CUT"]["noise_remove"]))
            self.force_rerun.setChecked(bool(self.parent.dict_for_yaml["DRAW-CUT"]["force_rerun"]))

            self.images_ext.addItem(self.parent.dict_for_yaml["DRAW-CUT"]["extension"])
            self.images_ext.setCurrentText(self.parent.dict_for_yaml["DRAW-CUT"]["extension"])
            # check numbering is valid value
            if self.parent.dict_for_yaml["DRAW-CUT"]["numbering"].capitalize() not in ["Right", "Bottom"]:
                self.parent.logger.error(f"ERROR: 'DRAW-CUT' 'numbering': '{self.parent.dict_for_yaml['DRAW-CUT']['numbering'].capitalize()}' not a valid value, allow only 'Right' or 'Bottom' (not case sensitive), please reload valid file")
            else:
                self.numbering.setCurrentText(self.parent.dict_for_yaml["DRAW-CUT"]["numbering"].capitalize())

            self.update_ext()
            self.show_draw_params()
            self.loading = False
        except KeyError as e:
            self.parent.logger.error(f"ERROR: Key {e} is not found on file")
            self.parent.dict_for_yaml = self.parent.dict_backup

    def show_draw_params(self):
        if self.parent.layer_tools.draw_checkbox.isChecked() or self.parent.layer_tools.cut_checkbox.isChecked():
            self.setVisible(True)
            self.out_draw_dir.setVisible(self.parent.layer_tools.draw_checkbox.isChecked())
            self.out_cut_dir.setVisible(self.parent.layer_tools.cut_checkbox.isChecked())
            self.noise_remove.setVisible(self.parent.layer_tools.cut_checkbox.isChecked())
            self.numbering.setVisible(self.parent.layer_tools.cut_checkbox.isChecked())
        else:
            self.setVisible(False)

    def update_ext(self):
        files, ext_list = get_files_ext(self.images_path.lineEditFile.text(), allow_ext)
        self.images_ext.clear()
        if ext_list:
            self.images_ext.addItems(ext_list)
            self.images_ext.setCurrentIndex(0)
            self.images_ext.setVisible(True)


###############################################################################################
# MAIN
if __name__ == '__main__':
    app = QCoreApplication.instance()
    if app is None:
        app = qt.QApplication([])
    foo = DrawCutParams(parent=None)
    foo.show()
    app.exec_()
