import os
from functools import partial
from pathlib import Path
import PyQt5.QtWidgets as qt
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap

from Leaftool_addons.commonWidget import style, FileSelectorLeaftool, NumberLineEditLabel, SpinBoxLabel, allow_ext, int_validator, check_values
import UsefullVariables as vrb
import DatabaseFunction as Dfct
import xml.etree.ElementTree as xmlet


class MLParams(qt.QGroupBox):
    """params for machine learning option"""

    def __init__(self):
        super().__init__()

        # Layout Style
        self.setTitle("Machine Learning options")
        self.setStyleSheet(style)
        self.setFixedSize(int(480 * vrb.ratio), int(200 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 15, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.label_ML = qt.QLabel("Model segmentation:")
        self.label_ML.setFixedSize(int(200 * vrb.ratio), int(30 * vrb.ratio))
        self.comboBoxModel = qt.QComboBox()
        self.comboBoxModel.setFixedSize(int(200 * vrb.ratio), int(25 * vrb.ratio))

        self.label_ML_classification = qt.QLabel("Model classification:")
        self.label_ML_classification.setFixedSize(int(200 * vrb.ratio), int(30 * vrb.ratio))
        self.comboBoxModel_classification = qt.QComboBox()
        self.comboBoxModel_classification.setFixedSize(int(200 * vrb.ratio), int(25 * vrb.ratio))

        self.split_ML = qt.QCheckBox()
        self.split_ML.setText("Split ML")
        self.split_ML.setFixedSize(int(80 * vrb.ratio), int(30 * vrb.ratio))

        self.labelUserCalibration = qt.QLabel("Calibration:")
        self.labelUserCalibration.setFixedSize(int(200*vrb.ratio), int(30*vrb.ratio))
        self.comboBoxCalibration = qt.QComboBox()
        self.comboBoxCalibration.setFixedSize(int(200 * vrb.ratio), int(25 * vrb.ratio))
        # self.calibration = Calibration()

        self.small_object = NumberLineEditLabel(constraint="Real", text="100", label="Small object:", size=80)
        self.small_object.lineEdit.setValidator(int_validator)
        self.alpha = SpinBoxLabel(min_v=0.0, max_v=1.0, step=0.1, value=0.8, label="Alpha:", size=80)
        self.leaf_border = NumberLineEditLabel(constraint="Real", text="0", label="Leaf Border:", size=80)
        self.leaf_border.lineEdit.setValidator(int_validator)

        self.noise_remove = qt.QCheckBox()
        self.noise_remove.setText("Noise Remove")
        self.noise_remove.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.force_rerun = qt.QCheckBox()
        self.force_rerun.setText("Force rerun")
        self.force_rerun.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.draw_ML_image = qt.QCheckBox()
        self.draw_ML_image.setText("Draw ML")
        self.draw_ML_image.setFixedSize(int(80 * vrb.ratio), int(30 * vrb.ratio))

        self.color_lesion_individual = qt.QCheckBox()
        self.color_lesion_individual.setText("color lesion individual")
        self.color_lesion_individual.setFixedSize(int(150 * vrb.ratio), int(30 * vrb.ratio))

        # Position Widgets
        self.layout.addWidget(self.label_ML, 0, 0, 1, 1, Qt.AlignLeft)
        self.layout.addWidget(self.comboBoxModel, 0, 1, 1, 1, Qt.AlignRight)
        self.layout.addWidget(self.split_ML, 0, 2, Qt.AlignRight)
        self.layout.addWidget(self.label_ML_classification, 1, 0, 1, 1, Qt.AlignLeft)
        self.layout.addWidget(self.comboBoxModel_classification, 1, 1, 1, 1, Qt.AlignRight)
        self.layout.addWidget(self.draw_ML_image, 1, 2, Qt.AlignRight)
        self.layout.addWidget(self.labelUserCalibration, 2, 0, 1, 1, Qt.AlignLeft)
        self.layout.addWidget(self.comboBoxCalibration, 2, 1, 1, 1, Qt.AlignRight)
        self.layout.addWidget(self.small_object, 3, 0, 1, 1, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.alpha, 3, 1, (Qt.AlignCenter | Qt.AlignVCenter))
        self.layout.addWidget(self.leaf_border, 3, 2, 1, 1, (Qt.AlignRight | Qt.AlignVCenter))
        self.layout.addWidget(self.color_lesion_individual, 4, 0, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.noise_remove, 4, 1, (Qt.AlignLeft | Qt.AlignVCenter))
        self.layout.addWidget(self.force_rerun, 4, 2, (Qt.AlignLeft | Qt.AlignVCenter))


class MergeParams(qt.QGroupBox):
    """params for machine learning option"""

    def __init__(self):
        super().__init__()

        # Layout Style
        self.setTitle("Merge options")
        self.setStyleSheet(style)
        # self.setFixedSize(int(420 * vrb.ratio), int(70 * vrb.ratio))
        self.setMinimumSize(int(440 * vrb.ratio), int(50 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.label_images_ext = qt.QLabel("Output extension:")
        self.label_images_ext.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))
        self.images_ext = qt.QComboBox()
        self.images_ext.setFixedSize(int(70 * vrb.ratio), int(25 * vrb.ratio))
        self.images_ext.addItems(allow_ext)
        self.rm_original = qt.QCheckBox()
        self.rm_original.setText("Remove original")
        self.rm_original.setFixedSize(int(120 * vrb.ratio), int(25 * vrb.ratio))

        # Position widgets
        self.layout.addWidget(self.label_images_ext, 0, 0, Qt.AlignLeft)
        self.layout.addWidget(self.images_ext, 0, 1, Qt.AlignLeft)
        self.layout.addWidget(self.rm_original, 0, 2, Qt.AlignLeft)


class Calibration(qt.QGroupBox):

    def __init__(self):
        super().__init__()
        self.comboBoxCalibration = None
        self.calibrationsElement = None
        self.calibration_avail = []
        self.loading = False
        self.load_calibration()
        self.fill_combobox_with_element()

    def load_calibration(self):
        try:
            file = xmlet.parse(vrb.folderInformation + "/UserCalibrations.mho")
            self.calibrationsElement = file.getroot()
        except:
            self.calibrationsElement = xmlet.Element('Calibrations')
            newCalibration = xmlet.SubElement(self.calibrationsElement, "Calibration")
            self.create_empty_calibration(newCalibration, name="No calibration")
        self.loading = False

    @staticmethod
    def create_empty_calibration(element, name="New calibration"):
        Dfct.SubElement(element, "Name").text = name
        Dfct.SubElement(element, "Choice").text = "0"
        Dfct.SubElement(element, "Unit").text = "px"
        Dfct.SubElement(element, "SquarePixel").text = "1"
        Dfct.SubElement(element, "SquareValue").text = "1"
        Dfct.SubElement(element, "XPixel").text = "1"
        Dfct.SubElement(element, "XValue").text = "1"
        Dfct.SubElement(element, "YPixel").text = "1"
        Dfct.SubElement(element, "YValue").text = "1"
        Dfct.SubElement(element, "ZPixel").text = "1"

    def fill_combobox_with_element(self):
        self.comboBoxCalibration.clear()
        for child in self.calibrationsElement:
            self.comboBoxCalibration.addItem(Dfct.childText(child, "Name"), child)
            self.calibration_avail.append(Dfct.childText(child, "Name"))


class MachineLearningParams(qt.QGroupBox):
    """params for machine learning option"""

    def __init__(self, parent):
        super().__init__()
        self.calibrationsElement = None
        self.parent = parent
        self.loading = True
        self.avail_models = []
        self.avail_models_shapes = []
        self.calibration_avail = []

        # Layout Style
        self.setTitle("Machine Learning and Merge Params")
        self.setStyleSheet(style)
        self.setFixedSize(int(498 * vrb.ratio), int(320 * vrb.ratio))
        # self.setMinimumSize(int(450 * vrb.ratio), int(310 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(self.layout)

        # Widgets
        self.images_path = FileSelectorLeaftool(label="Images Path:")
        self.images_path.setWhatsThis(self.parent.parent.dico_doc["images_path_ml"])
        self.images_path.setStatusTip(self.parent.parent.dico_doc_str["images_path_ml"])
        self.ml_params = MLParams()
        self.ml_params.comboBoxModel.setWhatsThis(self.parent.parent.dico_doc["model_name"])
        self.ml_params.comboBoxModel.setStatusTip(self.parent.parent.dico_doc_str["model_name"])
        self.ml_params.comboBoxModel_classification.setWhatsThis(self.parent.parent.dico_doc["model_name_classification"])
        self.ml_params.comboBoxModel_classification.setStatusTip(self.parent.parent.dico_doc_str["model_name_classification"])
        self.ml_params.comboBoxCalibration.setWhatsThis(self.parent.parent.dico_doc["calibration_name"])
        self.ml_params.comboBoxCalibration.setStatusTip(self.parent.parent.dico_doc_str["calibration_name"])
        self.ml_params.small_object.setWhatsThis(self.parent.parent.dico_doc["small_object"])
        self.ml_params.small_object.setStatusTip(self.parent.parent.dico_doc_str["small_object"])
        self.ml_params.alpha.setWhatsThis(self.parent.parent.dico_doc["alpha"])
        self.ml_params.alpha.setStatusTip(self.parent.parent.dico_doc_str["alpha"])
        self.ml_params.leaf_border.setWhatsThis(self.parent.parent.dico_doc["leaf_border"])
        self.ml_params.leaf_border.setStatusTip(self.parent.parent.dico_doc_str["leaf_border"])
        self.ml_params.color_lesion_individual.setWhatsThis(self.parent.parent.dico_doc["color_lesion_individual"])
        self.ml_params.color_lesion_individual.setStatusTip(self.parent.parent.dico_doc_str["color_lesion_individual"])
        self.ml_params.noise_remove.setWhatsThis(self.parent.parent.dico_doc["noise_remove"])
        self.ml_params.noise_remove.setStatusTip(self.parent.parent.dico_doc_str["noise_remove"])
        self.ml_params.force_rerun.setWhatsThis(self.parent.parent.dico_doc["force_rerun"])
        self.ml_params.force_rerun.setStatusTip(self.parent.parent.dico_doc_str["force_rerun"])

        self.merge_params = MergeParams()
        self.merge_params.images_ext.setWhatsThis(self.parent.parent.dico_doc["extension"])
        self.merge_params.images_ext.setStatusTip(self.parent.parent.dico_doc_str["extension"])
        self.merge_params.rm_original.setWhatsThis(self.parent.parent.dico_doc["rm_original"])
        self.merge_params.rm_original.setStatusTip(self.parent.parent.dico_doc_str["rm_original"])

        spacerItem = qt.QSpacerItem(20, 40, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.layout.addItem(spacerItem, 3, 0, 1, 1)
        # Position Widgets
        self.layout.addWidget(self.images_path, 1, 0, Qt.AlignTop)
        self.layout.addWidget(self.ml_params, 2, 0, Qt.AlignTop)
        self.layout.addWidget(self.merge_params, 4, 0, Qt.AlignBottom)

        # INIT
        self.loading_models()
        self.loading_models_classification()
        self.loading_calibration()

        # connection
        # common
        self.images_path.lineEditFile.textChanged.connect(partial(self.parent.check_path, from_object=self.images_path.lineEditFile, keys_list="'ML' 'images_path'"))
        # self.images_path.lineEditFile.editingFinished.connect(self.update_load)
        # ML
        self.ml_params.comboBoxCalibration.currentIndexChanged.connect(self.update_ml_params)
        self.ml_params.alpha.lineEdit.valueChanged.connect(self.update_ml_params)
        self.ml_params.small_object.lineEdit.editingFinished.connect(self.update_ml_params)
        self.ml_params.leaf_border.lineEdit.editingFinished.connect(self.update_ml_params)
        self.ml_params.comboBoxModel.currentIndexChanged.connect(self.update_ml_params)
        self.ml_params.comboBoxModel_classification.currentIndexChanged.connect(self.update_ml_params)
        self.ml_params.noise_remove.stateChanged.connect(self.update_ml_params)
        self.ml_params.force_rerun.stateChanged.connect(self.update_ml_params)
        self.ml_params.draw_ML_image.stateChanged.connect(self.update_ml_params)
        self.ml_params.split_ML.stateChanged.connect(self.update_ml_params)
        self.ml_params.color_lesion_individual.stateChanged.connect(self.update_ml_params)
        # Merge
        self.merge_params.images_ext.currentIndexChanged.connect(self.update_ml_params)
        self.merge_params.rm_original.stateChanged.connect(self.update_ml_params)

    def update_load(self):
        self.loading_models()
        self.loading_models_classification()
        self.loading_calibration()

    def loading_models(self):
        self.ml_params.comboBoxModel.clear()
        folder = vrb.folderPixelClassification
        for modelName in sorted(os.listdir(folder), key=str.casefold):
            filePath = folder + '/' + modelName
            if os.path.isdir(filePath):
                self.avail_models.append(modelName)
                self.ml_params.comboBoxModel.addItem(modelName)

    def loading_models_classification(self):
        self.ml_params.comboBoxModel_classification.clear()
        self.ml_params.comboBoxModel_classification.addItem("")
        folder = vrb.folderShapeClassification
        for modelName in sorted(os.listdir(folder), key=str.casefold):
            filePath = folder + '/' + modelName
            if os.path.isdir(filePath):
                self.avail_models_shapes.append(modelName)
                self.ml_params.comboBoxModel_classification.addItem(modelName)

    def loading_calibration(self):
        def create_empty_calibration(element, name="New calibration"):
            Dfct.SubElement(element, "Name").text = name
            Dfct.SubElement(element, "Choice").text = "0"
            Dfct.SubElement(element, "Unit").text = "px"
            Dfct.SubElement(element, "SquarePixel").text = "1"
            Dfct.SubElement(element, "SquareValue").text = "1"
            Dfct.SubElement(element, "XPixel").text = "1"
            Dfct.SubElement(element, "XValue").text = "1"
            Dfct.SubElement(element, "YPixel").text = "1"
            Dfct.SubElement(element, "YValue").text = "1"
            Dfct.SubElement(element, "ZPixel").text = "1"
        calibrationsElement = None
        try:
            file = xmlet.parse(vrb.folderInformation + "/UserCalibrations.mho")
            calibrationsElement = file.getroot()
        except:
            calibrationsElement = xmlet.Element('Calibrations')
            newCalibration = xmlet.SubElement(self.calibrationsElement, "Calibration")
            create_empty_calibration(newCalibration, name="No calibration")
        self.ml_params.comboBoxCalibration.clear()
        for child in calibrationsElement:
            self.ml_params.comboBoxCalibration.addItem(Dfct.childText(child, "Name"), child)
            self.calibration_avail.append(Dfct.childText(child, "Name"))

    # CHECK all PARAMS


    def update_ml_params(self):
        try:
            if not self.loading and (self.parent.dict_for_yaml["RUNSTEP"]["ML"] or self.parent.dict_for_yaml["RUNSTEP"]["merge"]):
                self.parent.dict_for_yaml["ML"]["model_name"] = self.ml_params.comboBoxModel.currentText()
                self.parent.dict_for_yaml["ML"]["model_name_classification"] = self.ml_params.comboBoxModel_classification.currentText()

                self.parent.dict_for_yaml["ML"]["images_path"] = self.images_path.lineEditFile.text()

                self.parent.dict_for_yaml["log_path"] = self.images_path.lineEditFile.text()
                self.parent.dict_for_yaml["ML"]["calibration_name"] = self.ml_params.comboBoxCalibration.currentText()
                self.parent.dict_for_yaml["ML"]["small_object"] = int(self.ml_params.small_object.lineEdit.text())
                self.parent.dict_for_yaml["ML"]["alpha"] = round(self.ml_params.alpha.lineEdit.value(), 2)
                self.parent.dict_for_yaml["ML"]["leaf_border"] = int(self.ml_params.leaf_border.lineEdit.text())
                self.parent.dict_for_yaml["ML"]["noise_remove"] = bool(self.ml_params.noise_remove.isChecked())
                self.parent.dict_for_yaml["ML"]["force_rerun"] = bool(self.ml_params.force_rerun.isChecked())
                self.parent.dict_for_yaml["ML"]["draw_ML_image"] = bool(self.ml_params.draw_ML_image.isChecked())
                self.parent.dict_for_yaml["ML"]["split_ML"] = bool(self.ml_params.split_ML.isChecked())
                self.parent.dict_for_yaml["ML"]["color_lesion_individual"] = bool(self.ml_params.color_lesion_individual.isChecked())
                if self.parent.dict_for_yaml["RUNSTEP"]["merge"]:
                    self.parent.dict_for_yaml["MERGE"]["rm_original"] = bool(self.merge_params.rm_original.isChecked())
                    self.parent.dict_for_yaml["MERGE"]["extension"] = self.merge_params.images_ext.currentText()
                self.parent.preview_config.setText(self.parent.export_use_yaml)
        except Exception as e:
            self.parent.logger.warning(e)
            pass

    def upload_ml_params(self):
        try:
            self.loading = True
            # Check image Path
            if self.parent.dict_for_yaml["ML"]["images_path"] and Path(self.parent.dict_for_yaml["ML"]["images_path"]).exists():
                self.images_path.lineEditFile.setText(self.parent.dict_for_yaml["ML"]["images_path"])
            elif self.parent.dict_for_yaml["ML"]["images_path"]:
                self.parent.logger.warning(f"Warning: arguments 'ML''images_path':'{Path(self.parent.dict_for_yaml['ML']['images_path'])}' doesn't exist")

            # check ML model_name segmentation
            if self.parent.dict_for_yaml["ML"]["model_name"]:
                if self.parent.dict_for_yaml["ML"]["model_name"] not in self.avail_models:
                    self.parent.logger.warning(f"Warning: arguments 'ML''model_name':'{self.parent.dict_for_yaml['ML']['model_name']}' is not avail, please use only {self.avail_models}")
                else:
                    self.ml_params.comboBoxModel.setCurrentText(self.parent.dict_for_yaml["ML"]["model_name"])
            elif self.parent.dict_for_yaml["RUNSTEP"]["ML"]:
                self.parent.logger.warning(f"Warning: arguments 'ML''model_name' is empty is not but ML is activated, please add correct model name (load with {self.ml_params.comboBoxModel.currentText()})")
                self.ml_params.comboBoxModel.setCurrentIndex(0)
            else:
                self.ml_params.comboBoxModel.setCurrentIndex(0)

            # check ML model_name_classification
            if self.parent.dict_for_yaml["ML"]["model_name_classification"]:
                if self.parent.dict_for_yaml["ML"]["model_name_classification"] not in self.avail_models_shapes:
                    self.parent.logger.warning(f"Warning: arguments 'ML''model_name_classification':'{self.parent.dict_for_yaml['ML']['model_name_classification']}' is not avail, please use only {self.avail_models_shapes}")
                else:
                    self.ml_params.comboBoxModel_classification.setCurrentText(self.parent.dict_for_yaml["ML"]["model_name_classification"])
            else:
                self.ml_params.comboBoxModel_classification.setCurrentIndex(0)

            # check calibration_name
            if self.parent.dict_for_yaml["ML"]["calibration_name"]:
                if self.parent.dict_for_yaml["ML"]["calibration_name"] not in self.calibration_avail:
                    self.parent.logger.warning(f"Warning: arguments 'ML''calibration_name':'{self.parent.dict_for_yaml['ML']['calibration_name']}' is not avail, please use only {self.calibration_avail}")
                    self.ml_params.comboBoxCalibration.setCurrentIndex(0)
                else:
                    self.ml_params.comboBoxCalibration.setCurrentText(str(self.parent.dict_for_yaml["ML"]["calibration_name"]))
            else:
                self.ml_params.comboBoxCalibration.setCurrentIndex(0)

            # check all integer values
            message = check_values(dico_params=self.parent.dict_for_yaml,
                                   primary_key="ML",
                                   secondary_key_list=["small_object", "leaf_border"],
                                   type_value=int,
                                   default_error=0)
            if message:
                self.parent.logger.error(message)
            self.ml_params.small_object.lineEdit.setText(str(self.parent.dict_for_yaml["ML"]["small_object"]))
            self.ml_params.leaf_border.lineEdit.setText(str(self.parent.dict_for_yaml["ML"]["leaf_border"]))

            # check float value
            if isinstance(self.parent.dict_for_yaml["ML"]["alpha"], int):
                if self.parent.dict_for_yaml["ML"]["alpha"] == 0:
                    self.parent.dict_for_yaml["ML"]["alpha"] = 0.0
                if self.parent.dict_for_yaml["ML"]["alpha"] != 1:
                    self.parent.dict_for_yaml["ML"]["alpha"] = 1.0
            if not isinstance(self.parent.dict_for_yaml["ML"]["alpha"], float):
                self.parent.logger.error(f"ERROR: 'ML' 'alpha' is not a valid float value, please reload valid file")
            elif 0 <= float(self.parent.dict_for_yaml["ML"]["alpha"]) <= 1:
                self.ml_params.alpha.lineEdit.setValue(float(self.parent.dict_for_yaml["ML"]["alpha"]))
            else:
                self.parent.logger.error(f"ERROR: 'ML' 'alpha' {float(self.parent.dict_for_yaml['ML']['alpha'])} is not a between 0 <= alpha <= 1, please reload valid file")

            # check all boolean values
            message = check_values(dico_params=self.parent.dict_for_yaml,
                                   primary_key="ML",
                                   secondary_key_list=["split_ML", "noise_remove", "force_rerun", "draw_ML_image", "color_lesion_individual"],
                                   type_value=bool,
                                   default_error=False)
            if message:
                self.parent.logger.error(message)
            self.ml_params.split_ML.setChecked(bool(self.parent.dict_for_yaml["ML"]["split_ML"]))
            self.ml_params.noise_remove.setChecked(bool(self.parent.dict_for_yaml["ML"]["noise_remove"]))
            self.ml_params.force_rerun.setChecked(bool(self.parent.dict_for_yaml["ML"]["force_rerun"]))
            self.ml_params.draw_ML_image.setChecked(bool(self.parent.dict_for_yaml["ML"]["draw_ML_image"]))
            self.ml_params.color_lesion_individual.setChecked(bool(self.parent.dict_for_yaml["ML"]["color_lesion_individual"]))

            if self.parent.dict_for_yaml["RUNSTEP"]["merge"]:
                message = check_values(dico_params=self.parent.dict_for_yaml,
                                       primary_key="MERGE",
                                       secondary_key_list=["rm_original"],
                                       type_value=bool,
                                       default_error=False)
                if message:
                    self.parent.logger.error(message)
                self.merge_params.rm_original.setChecked(bool(self.parent.dict_for_yaml["MERGE"]["rm_original"]))
                self.merge_params.images_ext.addItem(self.parent.dict_for_yaml["MERGE"]["extension"])
                self.merge_params.images_ext.setCurrentText(self.parent.dict_for_yaml["MERGE"]["extension"])
            self.loading = False
        except KeyError as e:
            self.parent.logger.error(f"ERROR: Key {e} is not found on file")
            self.parent.dict_for_yaml = self.parent.dict_backup
            self.loading = False

    def show_ml_merge_params(self):
        if self.parent.layer_tools.ml.isChecked() or self.parent.layer_tools.merge.isChecked():
            self.setVisible(True)
            self.ml_params.setVisible(self.parent.layer_tools.ml.isChecked())
            self.merge_params.setVisible(self.parent.layer_tools.merge.isChecked())
        else:
            self.setVisible(False)
        if not self.parent.layer_tools.ml.isChecked():
            self.parent.layer_tools.ml_label.setText("Not ML")
            self.parent.layer_tools.ml.pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/unml.png")
            self.parent.layer_tools.ml.resizeEvent(None)
            self.parent.layer_tools.ml.setChecked(False)
        else:
            self.parent.layer_tools.ml_label.setText("ML")
            self.parent.layer_tools.ml.pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/ml.png")
            self.parent.layer_tools.ml.resizeEvent(None)
            self.parent.layer_tools.ml.setChecked(True)

        if not self.parent.layer_tools.merge.isChecked():
            self.parent.layer_tools.merge_label.setText("Not merge")
            self.parent.layer_tools.merge.pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/unmerge.png")
            self.parent.layer_tools.merge.resizeEvent(None)
            self.parent.layer_tools.merge.setChecked(False)
        else:
            self.parent.layer_tools.merge_label.setText("Merge")
            self.parent.layer_tools.merge.pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/merge.png")
            self.parent.layer_tools.merge.resizeEvent(None)
            self.parent.layer_tools.merge.setChecked(True)

###############################################################################################
# MAIN
if __name__ == '__main__':
    app = QCoreApplication.instance()
    if app is None:
        app = qt.QApplication([])
    foo = MachineLearningParams(parent=None)
    foo = MLParams()
    # foo = DoubleSpinBoxLabel(value=0.8, label="Alpha:", size=80)
    foo.showMaximized()
    foo.show()
    app.exec_()
