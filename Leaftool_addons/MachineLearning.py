import os
from PyQt5.QtCore import Qt
import PyQt5.QtWidgets as qt
import UsefullVariables as vrb
from Leaftool_addons.commonWidget import style, FileSelector, NumberLineEditLabel, allow_ext

class MLParams(qt.QGroupBox):
    """params for machine learning option"""

    def __init__(self):
        super().__init__()

        # Layout Style
        self.setTitle("Machine Learning options")
        self.setStyleSheet(style)
        self.setFixedSize(int(420 * vrb.ratio), int(180 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 15, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.label_ML = qt.QLabel("Model:")
        self.label_ML.setFixedSize(int(100 * vrb.ratio), int(30 * vrb.ratio))
        self.comboBoxModel = qt.QComboBox()
        self.comboBoxModel.setFixedSize(int(200 * vrb.ratio), int(25 * vrb.ratio))

        self.split_ML = qt.QCheckBox()
        self.split_ML.setText("Split ML")
        self.split_ML.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.calibration = NumberLineEditLabel(constraint="Real", text="1", label="Calibration:", size=80)
        self.small_object = NumberLineEditLabel(constraint="Real", text="100", label="Small object:", size=80)
        self.alpha = NumberLineEditLabel(constraint="Real", text="0.8", label="Alpha:", size=80)
        self.leaf_border = NumberLineEditLabel(constraint="Real", text="0", label="Leaf Border:", size=80)

        self.noise_remove = qt.QCheckBox()
        self.noise_remove.setText("Noise Remove")
        self.noise_remove.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.force_rerun = qt.QCheckBox()
        self.force_rerun.setText("Force rerun")
        self.force_rerun.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.draw_ML_image = qt.QCheckBox()
        self.draw_ML_image.setText("Draw ML")
        self.draw_ML_image.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        # Position Widgets
        self.layout.addWidget(self.label_ML, 0, 0, 1, 1, Qt.AlignLeft)
        self.layout.addWidget(self.comboBoxModel, 0, 0, 1, 2, Qt.AlignRight)
        self.layout.addWidget(self.split_ML, 0, 2, Qt.AlignRight)
        self.layout.addWidget(self.calibration, 2, 0, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.small_object, 2, 1, 1, 2, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.alpha, 3, 0, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.leaf_border, 3, 1, 1, 2, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.noise_remove, 4, 0, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.force_rerun, 4, 1, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.draw_ML_image, 4, 2, (Qt.AlignLeft | Qt.AlignVCenter))


class MergeParams(qt.QGroupBox):
    """params for machine learning option"""

    def __init__(self):
        super().__init__()

        # Layout Style
        self.setTitle("Merge options")
        self.setStyleSheet(style)
        # self.setFixedSize(int(420 * vrb.ratio), int(70 * vrb.ratio))
        self.setMinimumSize(int(420 * vrb.ratio), int(70 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 15, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.label_images_ext = qt.QLabel("Output extension:")
        self.label_images_ext.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))
        self.images_ext = qt.QComboBox()
        self.images_ext.setFixedSize(int(70 * vrb.ratio), int(25 * vrb.ratio))
        self.images_ext.addItems(allow_ext)
        self.rm_original = qt.QCheckBox()
        self.rm_original.setText("Remove original")
        self.rm_original.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        # Position widgets
        self.layout.addWidget(self.label_images_ext, 0, 0, Qt.AlignLeft)
        self.layout.addWidget(self.images_ext, 0, 1, Qt.AlignLeft)
        self.layout.addWidget(self.rm_original, 0, 2, Qt.AlignLeft)


class MachineLearningParams(qt.QGroupBox):
    """params for machine learning option"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.loading = True

        # Layout Style
        self.setTitle("Machine Learning and Merge Params")
        self.setStyleSheet(style)
        self.setFixedSize(int(450 * vrb.ratio), int(330 * vrb.ratio))
        # self.setMinimumSize(int(450 * vrb.ratio), int(310 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(self.layout)

        # Widgets
        self.images_path = FileSelector(label="Images Path:")
        self.ml_params = MLParams()
        self.merge_params = MergeParams()
        self.avail_models = []
        spacerItem = qt.QSpacerItem(20, 40, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.layout.addItem(spacerItem, 3, 0, 1, 1)
        # Position Widgets
        self.layout.addWidget(self.images_path, 1, 0, Qt.AlignTop)
        self.layout.addWidget(self.ml_params, 2, 0, Qt.AlignTop)
        self.layout.addWidget(self.merge_params, 4, 0, Qt.AlignBottom)

        # INIT
        self.loading_models()

        # connection
        # common
        self.images_path.lineEditFile.editingFinished.connect(self.update_ml_params)
        # ML
        self.ml_params.calibration.lineEdit.editingFinished.connect(self.update_ml_params)
        self.ml_params.alpha.lineEdit.editingFinished.connect(self.update_ml_params)
        self.ml_params.small_object.lineEdit.editingFinished.connect(self.update_ml_params)
        self.ml_params.leaf_border.lineEdit.editingFinished.connect(self.update_ml_params)
        self.ml_params.comboBoxModel.currentIndexChanged.connect(self.update_ml_params)
        self.ml_params.noise_remove.stateChanged.connect(self.update_ml_params)
        self.ml_params.force_rerun.stateChanged.connect(self.update_ml_params)
        self.ml_params.draw_ML_image.stateChanged.connect(self.update_ml_params)
        self.ml_params.split_ML.stateChanged.connect(self.update_ml_params)
        #Merge
        self.merge_params.images_ext.currentIndexChanged.connect(self.update_ml_params)
        self.merge_params.rm_original.stateChanged.connect(self.update_ml_params)

    def loading_models(self):
        self.ml_params.comboBoxModel.clear()
        folder = vrb.folderPixelClassification
        for modelName in sorted(os.listdir(folder), key=str.casefold):
            filePath = folder + '/' + modelName
            if os.path.isdir(filePath):
                self.avail_models.append(modelName)
                self.ml_params.comboBoxModel.addItem(modelName)

    def update_ml_params(self):
        try:
            if not self.loading and (self.parent.dict_for_yaml["RUNSTEP"]["ML"] or self.parent.dict_for_yaml["RUNSTEP"]["merge"]):
                self.parent.dict_for_yaml["ML"]["model_name"] = self.ml_params.comboBoxModel.currentText()
                self.parent.dict_for_yaml["ML"]["images_path"] = self.images_path.lineEditFile.text()
                self.parent.dict_for_yaml["log_path"] = self.images_path.lineEditFile.text()
                self.parent.dict_for_yaml["ML"]["calibration_value"] = float(self.ml_params.calibration.lineEdit.text())
                self.parent.dict_for_yaml["ML"]["small_object"] = int(self.ml_params.small_object.lineEdit.text())
                self.parent.dict_for_yaml["ML"]["alpha"] = float(self.ml_params.alpha.lineEdit.text())
                self.parent.dict_for_yaml["ML"]["leaf_border"] = int(self.ml_params.leaf_border.lineEdit.text())
                self.parent.dict_for_yaml["ML"]["noise_remove"] = bool(self.ml_params.noise_remove.isChecked())
                self.parent.dict_for_yaml["ML"]["force_rerun"] = bool(self.ml_params.force_rerun.isChecked())
                self.parent.dict_for_yaml["ML"]["draw_ML_image"] = bool(self.ml_params.draw_ML_image.isChecked())
                self.parent.dict_for_yaml["ML"]["split_ML"] = bool(self.ml_params.split_ML.isChecked())
                if self.parent.dict_for_yaml["RUNSTEP"]["merge"]:
                    self.parent.dict_for_yaml["MERGE"]["rm_original"] = bool(self.merge_params.rm_original.isChecked())
                    self.parent.dict_for_yaml["MERGE"]["extension"] = self.merge_params.images_ext.currentText()
                self.parent.preview_config.setText(self.parent.export_use_yaml)
        except Exception as e:
            print(e)
            pass

    def upload_ml_params(self):
        self.loading = True
        if (self.parent.dict_for_yaml["RUNSTEP"]["ML"]) or (self.parent.dict_for_yaml["RUNSTEP"]["merge"]):
            if self.parent.dict_for_yaml["ML"]["model_name"] and self.parent.dict_for_yaml["ML"]["model_name"] not in self.avail_models:
                self.parent.logger.warning(f"Warning: arguments 'ML''model_name':'{self.parent.dict_for_yaml['ML']['model_name']}' is not avail, please use only {self.avail_models}")
            else:
                self.parent.layer_tools.plant_model.setCurrentText(self.parent.dict_for_yaml["PLANT_MODEL"])
            self.ml_params.comboBoxModel.setCurrentText(self.parent.dict_for_yaml["ML"]["model_name"])
            self.images_path.lineEditFile.setText(self.parent.dict_for_yaml["ML"]["images_path"])
            self.ml_params.calibration.lineEdit.setText(str(self.parent.dict_for_yaml["ML"]["calibration_value"]))
            self.ml_params.small_object.lineEdit.setText(str(self.parent.dict_for_yaml["ML"]["small_object"]))
            self.ml_params.alpha.lineEdit.setText(str(self.parent.dict_for_yaml["ML"]["alpha"]))
            self.ml_params.leaf_border.lineEdit.setText(str(self.parent.dict_for_yaml["ML"]["leaf_border"]))
            self.ml_params.split_ML.setChecked(bool(self.parent.dict_for_yaml["ML"]["split_ML"]))
            self.ml_params.noise_remove.setChecked(bool(self.parent.dict_for_yaml["ML"]["noise_remove"]))
            self.ml_params.force_rerun.setChecked(bool(self.parent.dict_for_yaml["ML"]["force_rerun"]))
            self.ml_params.draw_ML_image.setChecked(bool(self.parent.dict_for_yaml["ML"]["draw_ML_image"]))
            self.loading = False
        if self.parent.dict_for_yaml["RUNSTEP"]["merge"]:
            self.merge_params.rm_original.setChecked(bool(self.parent.dict_for_yaml["MERGE"]["rm_original"]))
            self.merge_params.images_ext.addItem(self.parent.dict_for_yaml["MERGE"]["extension"])
            self.merge_params.images_ext.setCurrentText(self.parent.dict_for_yaml["MERGE"]["extension"])


    def show_ml_merge_params(self):
        if self.parent.layer_tools.ml_checkbox.isChecked() or self.parent.layer_tools.merge_checkbox.isChecked():
            self.setVisible(True)
            self.ml_params.setVisible(self.parent.layer_tools.ml_checkbox.isChecked())
            self.merge_params.setVisible(self.parent.layer_tools.merge_checkbox.isChecked())
        else:
            self.setVisible(False)
