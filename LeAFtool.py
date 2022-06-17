import logging.config
import sys
from pathlib import Path
from PyQt5.QtCore import Qt, QCoreApplication, pyqtSlot
from PyQt5 import QtGui, Qsci
import PyQt5.QtWidgets as qt
from PyQt5.QtGui import QIcon

import UsefullVariables as vrb
import UsefullWidgets as wgt
import UsefullFunctions as fct
import DatabaseFunction as Dfct

sys.path.insert(0, Path(vrb.folderMacroInterface + "/LeAFtool/").as_posix())

# Import LeAFtool class
from Leaftool_addons.DrawCrop import DrawCropParams
from Leaftool_addons.MachineLearning import MachineLearningParams
from Leaftool_addons.commonWidget import style, return_default_folder, TableWidget, TwoListSelection, FileSelectorLeaftool
from Leaftool_addons.cmd_LeAFtool import *


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'handlers': [],
            'level': 'DEBUG',
            'propagate': True,
        }
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
logger = logging.getLogger('LeAFtool GUI')

try:
    mainWindow = vrb.mainWindow
    groupMenu = mainWindow.groupMenu
    button = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/favicon.png")
    button.setFixedSize(30 * vrb.ratio, 30 * vrb.ratio)
    groupMenu.layoutBar1.addWidget(button, 0, vrb.numMacro, Qt.AlignVCenter)
    vrb.numMacro += 1
except:
    # traceback.print_exc(file=sys.stderr)
    pass


class OutLog:
    def __init__(self, edit, out=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        """
        self.edit = edit
        self.out = out
        self.edit.setStyleSheet('background-color: #a3a0a0;')

    def write(self, message):
        if "DEBUG" in message.upper():
            self.edit.setTextColor(QtGui.QColor('white'))
        elif "INFO" in message.upper():
            self.edit.setTextColor(QtGui.QColor('cyan'))
        elif "WARNING" in message.upper():
            self.edit.setTextColor(QtGui.QColor('yellow'))
        elif "ERROR" in message.upper():
            self.edit.setTextColor(QtGui.QColor('red'))
        elif "CRITICAL" in message.upper():
            self.edit.setTextColor(QtGui.QColor('darkred'))
        if "[" in message[:5]:
            self.edit.append(message.rstrip()[5:-4])
        else:
            self.edit.append(message.rstrip())
        qt.QApplication.processEvents()
        # win = qt.QApplication.focusWindow()
        # print(win.title())

        if self.out:
            self.out.write(message)


class YAMLEditor(Qsci.QsciScintilla):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLexer(Qsci.QsciLexerYAML(self))
        self.setReadOnly(True)
        # Set the zoom factor, the factor is in points.
        self.zoomTo(2)
        # line numbers margin colors
        self.setMarginsBackgroundColor(QtGui.QColor("#323232"))
        self.setMarginsForegroundColor(QtGui.QColor("#323232"))
        # Use boxes as folding visual
        self.setFolding(self.BoxedTreeFoldStyle)
        # Braces matching
        self.setBraceMatching(self.SloppyBraceMatch)
        # folding margin colors (foreground,background)
        self.setFoldMarginColors(QtGui.QColor("#929292"),
                                 QtGui.QColor("#323232"))
        # Show whitespace to help detect whitespace errors
        self.setWhitespaceVisibility(True)
        self.setIndentationGuides(True)


class ToolsActivation(qt.QGroupBox):
    """add tools selection layer"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.loading = False

        # Layout Style
        self.setTitle("Globals Parameters")
        self.setStyleSheet(style)
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        # Widgets
        self.plant_model_label = qt.QLabel()
        self.plant_model_label.setText("Plant model:")
        self.plant_model_label.setFixedHeight(int(30 * vrb.ratio))
        self.plant_model = qt.QComboBox()
        self.plant_model.addItems(["banana", "rice"])
        self.plant_model.setFixedSize(int(100 * vrb.ratio), int(25 * vrb.ratio))

        self.show_meta_checkbox = qt.QCheckBox()
        self.show_meta_checkbox.setText("Show Meta Section")
        self.show_meta_checkbox.setChecked(True)
        self.show_meta_checkbox.setFixedWidth(int(150 * vrb.ratio))

        self.draw_checkbox = qt.QCheckBox()
        self.draw_checkbox.setText("Draw")
        self.draw_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))

        self.crop_checkbox = qt.QCheckBox()
        self.crop_checkbox.setText("Crop")
        self.crop_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))

        self.ml_checkbox = qt.QCheckBox()
        self.ml_checkbox.setText("ML")
        self.ml_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))

        self.merge_checkbox = qt.QCheckBox()
        self.merge_checkbox.setText("Merge")
        self.merge_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))

        self.csv_file = FileSelectorLeaftool(label="CSV file:", file=True)
        not_resize = self.csv_file.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.csv_file.setSizePolicy(not_resize)

        self.list_selection = TwoListSelection()
        self.list_selection.setFixedSize(int(700 * vrb.ratio), int(80 * vrb.ratio))

        # Position widgets
        self.layout.addWidget(self.plant_model_label, 0, 0, Qt.AlignLeft)
        self.layout.addWidget(self.plant_model, 0, 1, Qt.AlignLeft)
        self.layout.addWidget(self.show_meta_checkbox, 0, 2, Qt.AlignLeft)

        # Layout Style
        tools_group = qt.QGroupBox()
        tools_group.setTitle("Tools activation")
        tools_group.setStyleSheet(style)
        tools_group.layout = qt.QHBoxLayout()
        tools_group.layout.setContentsMargins(5, 5, 5, 5)
        tools_group.setLayout(tools_group.layout)
        tools_group.layout.addStretch()
        tools_group.layout.addWidget(self.draw_checkbox)
        tools_group.layout.addWidget(self.crop_checkbox)
        tools_group.layout.addWidget(self.ml_checkbox)
        tools_group.layout.addWidget(self.merge_checkbox)
        tools_group.layout.addStretch()
        tools_group.setFixedWidth(int(500 * vrb.ratio))
        self.layout.addWidget(tools_group, 0, 3, Qt.AlignLeft)

        # Meta group
        self.meta_group = qt.QGroupBox()
        self.meta_group.setTitle("Meta Infos")
        self.meta_group.setStyleSheet(style)
        self.meta_group.layout = qt.QVBoxLayout()
        self.meta_group.layout.setContentsMargins(10, 10, 10, 10)
        self.meta_group.setLayout(self.meta_group.layout)
        self.meta_group.layout.addStretch()
        self.meta_group.layout.addWidget(self.csv_file)
        self.meta_group.layout.addWidget(self.list_selection)
        self.meta_group.layout.addStretch()
        self.layout.addWidget(self.meta_group, 1, 0, 1, 4)

        # connections
        self.draw_checkbox.stateChanged.connect(self.update_activation_tools)
        self.crop_checkbox.stateChanged.connect(self.update_activation_tools)
        self.ml_checkbox.stateChanged.connect(self.update_activation_tools)
        self.merge_checkbox.stateChanged.connect(self.update_activation_tools)
        self.plant_model.currentIndexChanged.connect(self.update_activation_tools)
        self.csv_file.lineEditFile.textChanged.connect(self.update_activation_tools)
        # self.list_selection.mInput.itemSelectionChanged.connect(self.update_activation_tools)
        # self.list_selection.mOuput.itemSelectionChanged.connect(self.update_activation_tools)
        self.list_selection.mBtnMoveToAvailable.clicked.connect(self.update_activation_tools)
        self.list_selection.mBtnMoveToSelected.clicked.connect(self.update_activation_tools)
        self.list_selection.mButtonToAvailable.clicked.connect(self.update_activation_tools)
        self.list_selection.mButtonToSelected.clicked.connect(self.update_activation_tools)
        self.list_selection.mBtnUp.clicked.connect(self.update_activation_tools)
        self.list_selection.mBtnDown.clicked.connect(self.update_activation_tools)

        self.show_meta_checkbox.stateChanged.connect(self.show_meta_section)
        self.show_meta_section()

    def show_meta_section(self):
        self.meta_group.setVisible(self.show_meta_checkbox.isChecked())

    def upload_activation_tools(self):

        self.loading = True
        if self.parent.dict_for_yaml["PLANT_MODEL"] not in ["banana", "rice"]:
            self.parent.logger.warning(
                f"Warning: arguments PLANT_MODEL:'{self.parent.dict_for_yaml['PLANT_MODEL']}' is not allow, please use only 'banana' or 'rice'")
        else:
            self.plant_model.setCurrentText(self.parent.dict_for_yaml["PLANT_MODEL"])
        self.draw_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["draw"]))
        self.crop_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["crop"]))
        self.ml_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["ML"]))
        self.merge_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["merge"]))
        self.csv_file.lineEditFile.setText(self.parent.dict_for_yaml["csv_file"])
        self.list_selection.clean_list()
        self.list_selection.add_right_elements(self.parent.dict_for_yaml["rename"])
        self.loading = False

    def update_activation_tools(self):
        if not self.loading:

            self.parent.dict_for_yaml["RUNSTEP"]["crop"] = self.crop_checkbox.isChecked()
            self.parent.dict_for_yaml["RUNSTEP"]["draw"] = self.draw_checkbox.isChecked()
            self.parent.layer_draw_crop.show_draw_params()
            self.parent.dict_for_yaml["RUNSTEP"]["ML"] = self.ml_checkbox.isChecked()
            self.parent.dict_for_yaml["RUNSTEP"]["merge"] = self.merge_checkbox.isChecked()
            self.parent.layer_ml_merge.show_ml_merge_params()
            self.parent.dict_for_yaml["PLANT_MODEL"] = self.plant_model.currentText()
            self.parent.dict_for_yaml["csv_file"] = self.csv_file.lineEditFile.text()
            self.update_header_csv()
            self.parent.dict_for_yaml["rename"] = self.list_selection.get_right_elements()
            self.parent.preview_config.setText(self.parent.export_use_yaml)

            # if all disable, disable run
            if not self.parent.layer_tools.draw_checkbox.isChecked() \
                    and not self.parent.layer_tools.crop_checkbox.isChecked() \
                    and not self.parent.layer_tools.ml_checkbox.isChecked() \
                    and not self.parent.layer_tools.merge_checkbox.isChecked():
                self.parent.layer_leaftool_params.run.setDisabled(True)
                self.parent.layer_leaftool_params.save.setDisabled(True)
            else:
                self.parent.layer_leaftool_params.run.setDisabled(False)
                self.parent.layer_leaftool_params.save.setDisabled(False)
                # self.csv_file.setVisible(self.parent.layer_tools.draw_checkbox.isChecked())
                # self.csv_file.setVisible(self.parent.layer_tools.crop_checkbox.isChecked())
                # self.csv_file.setVisible(self.parent.layer_tools.ml_checkbox.isChecked())

    def update_header_csv(self):
        if self.parent.dict_for_yaml["csv_file"] and Path(self.parent.dict_for_yaml["csv_file"]).exists():
            header_txt = Path(self.parent.dict_for_yaml["csv_file"]).open("r").readline().strip()
            sep_dict = {",": header_txt.count(","),
                        ";": header_txt.count(";"),
                        ".": header_txt.count("."),
                        "\t": header_txt.count("\t")
                        }
            csv_separator = max(sep_dict, key=sep_dict.get)
            header_list = header_txt.split(csv_separator)
            left_list = self.list_selection.get_left_elements()
            right_list = self.list_selection.get_right_elements()
            # if header_list not equal left_list+right_list
            if len(header_list) != len(left_list+right_list):
                if len(left_list) == 0:
                    self.list_selection.add_left_elements(list(set(header_list) - set(right_list)))
                else:
                    self.list_selection.add_left_elements(header_list)
        else:
            self.list_selection.clean_list()


class LeaftoolParams(qt.QGroupBox):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Layout Style
        self.setTitle("LeAFtool params")
        self.setStyleSheet(style)
        self.setMaximumSize(int(980 * vrb.ratio), int(70 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(10, 20, 10, 10)
        self.setLayout(self.layout)
        self.loading = False

        # Widgets
        icon_size = 30
        self.save_label = qt.QLabel()
        self.save_label.setText("Save YAML:")
        self.save = wgt.PushButtonImage(vrb.folderImages + "/Save_As.png")
        self.save.setFixedSize(int(icon_size * vrb.ratio), int(icon_size * vrb.ratio))

        self.upload_label = qt.QLabel()
        self.upload_label.setText("Upload YAML:")
        self.upload = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/upload.png")
        self.upload.setFixedSize(int(icon_size * vrb.ratio), int(icon_size * vrb.ratio))

        self.run_label = qt.QLabel()
        self.run_label.setText("Run LeAFtool:")
        self.run = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/run.png")
        self.run.setFixedSize(int(icon_size * vrb.ratio), int(icon_size * vrb.ratio))

        self.preview_yaml_checkbox = qt.QCheckBox()
        self.preview_yaml_checkbox.setChecked(True)
        self.preview_yaml_checkbox.setText("Preview YAML")
        self.preview_yaml_checkbox.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        self.debug_checkbox = qt.QCheckBox()
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.setText("Debug")
        self.debug_checkbox.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))

        # Position Widgets
        self.layout.addWidget(self.upload_label, 0, 0, Qt.AlignRight)
        self.layout.addWidget(self.upload, 0, 1, Qt.AlignLeft)

        self.layout.addWidget(self.save_label, 0, 2, Qt.AlignRight)
        self.layout.addWidget(self.save, 0, 3, Qt.AlignLeft)

        self.layout.addWidget(self.run_label, 0, 4, Qt.AlignRight)
        self.layout.addWidget(self.run, 0, 5, Qt.AlignLeft)

        self.layout.addWidget(self.preview_yaml_checkbox, 0, 6, Qt.AlignCenter)
        self.layout.addWidget(self.debug_checkbox, 0, 7, Qt.AlignCenter)

        # Init state
        self.hide_preview_yaml()

        # Connections
        self.preview_yaml_checkbox.clicked.connect(self.hide_preview_yaml)
        self.debug_checkbox.clicked.connect(self.update_debug)

    def hide_preview_yaml(self):
        if self.preview_yaml_checkbox.isChecked():
            self.parent.preview_config.setVisible(True)
        else:
            self.parent.preview_config.setVisible(False)

    def update_debug(self):
        try:
            if not self.loading:
                self.parent.dict_for_yaml["debug"] = bool(self.debug_checkbox.isChecked())
                self.parent.preview_config.setText(self.parent.export_use_yaml)
        except Exception as e:
            print(f"WARNING update_debug: {e}")
            pass

    def upload_debug(self):
        self.loading = True
        self.debug_checkbox.setChecked(bool(self.parent.dict_for_yaml["debug"]))
        self.loading = False


class RunLeAFtool(qt.QWidget):
    """
    """
    def __init__(self):
        super().__init__()
        self.dict_for_yaml = {}
        self.dict_backup = {}
        self.logger = logger
        self.leaftool = None
        self.connect = None

        # Layout Style
        self.layout = qt.QGridLayout()
        self.layout.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.setLayout(self.layout)
        self.setContentsMargins(5, 5, 5, 5)
        style_global = fct.getStyleSheet()
        self.setStyleSheet(style_global)
        self.setMaximumWidth(1920)

        # Create the text output widget.
        self.process = qt.QTextEdit(self, readOnly=True)
        self.process.setMinimumHeight(50)
        self.process.setMaximumSize(int(980 * vrb.ratio), 200)
        sys.stdout = OutLog(self.process, sys.stdout)
        sys.stderr = OutLog(self.process, sys.stderr)

        # add preview of yaml file
        self.preview_config = qt.QPlainTextEdit()
        self.preview_config = YAMLEditor()
        self.preview_config.setMinimumWidth(500)

        # add path config file
        self.yaml_path = vrb.folderMacroInterface + "/LeAFtool/config.yaml"
        self.default_yaml_path = vrb.folderMacroInterface + "/LeAFtool/config.yaml"

        # Add layer part
        self.layer_tools = ToolsActivation(parent=self)
        self.layer_draw_crop = DrawCropParams(parent=self)
        self.layer_leaftool_params = LeaftoolParams(parent=self)
        self.layer_ml_merge = MachineLearningParams(parent=self)

        # # size policy
        not_resize = self.layer_ml_merge.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.layer_ml_merge.setSizePolicy(not_resize)
        not_resize = self.layer_draw_crop.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.layer_draw_crop.setSizePolicy(not_resize)

        self.layout.addWidget(self.layer_tools, 0, 0, 1, 2)
        self.layout.addWidget(self.preview_config, 0, 3, 4, 1)
        self.layout.addWidget(self.layer_draw_crop, 1, 0)
        self.layout.addWidget(self.layer_ml_merge, 1, 1)
        self.layout.addWidget(self.layer_leaftool_params, 2, 0, 1, 2)
        self.layout.addWidget(self.process, 3, 0, 1, 2)
        # self.layout.setColumnMinimumWidth(460, 0)
        # self.layout.setColumnStretch(0, 1)
        # self.layout.setColumnStretch(1, 2)
        # self.layout.setColumnStretch(2, 3)
        # # size policy
        # not_resize = self.preview_config.sizePolicy()
        # not_resize.setRetainSizeWhenHidden(True)
        # self.preview_config.setSizePolicy(not_resize)

        ## INIT states:
        ##### connection
        self.load_yaml()

        ## Layer leaftool params connection
        self.layer_leaftool_params.upload.clicked.connect(self.upload_yaml)
        self.layer_leaftool_params.save.clicked.connect(self.save_yaml)
        self.layer_leaftool_params.run.clicked.connect(self.run_leaftool)

    def load_yaml(self):
        with open(self.yaml_path, "r") as file_config:
            self.dict_for_yaml = yaml.load(file_config, Loader=yaml.Loader)
        with open(self.default_yaml_path, "r") as file_config:
            self.dict_backup = yaml.load(file_config, Loader=yaml.Loader)
        # self.mask_yaml()
        # from pprint import pprint as pp
        # pp(self.dict_for_yaml)
        # print(f"{'#'*15}\ndict_backup\n{'#'*15}\n{self.dict_backup}")
        self.upload_all()

    @pyqtSlot("QWidget*", "QWidget*")
    def on_focus_changed(self):
        self.update_all()

    def update_all(self):

        self.layer_tools.update_activation_tools()
        self.layer_draw_crop.update_draw_crop_params()
        self.layer_ml_merge.update_ml_params()
        self.layer_leaftool_params.update_debug()
        self.preview_config.setText(self.export_use_yaml)

    def upload_all(self):
        self.layer_tools.upload_activation_tools()
        self.layer_draw_crop.upload_draw_crop_params()
        self.layer_ml_merge.upload_ml_params()
        self.layer_leaftool_params.upload_debug()
        self.update_all()

    def mask_yaml(self):
        # Remove section on YAML if not activated
        # For ML/merge
        # If not ML AND not merge delete
        if not self.dict_for_yaml["RUNSTEP"]["ML"] and not self.dict_for_yaml["RUNSTEP"]["merge"] and "ML" in self.dict_for_yaml:
            self.dict_backup.update({"ML": self.dict_for_yaml["ML"]})
            del self.dict_for_yaml["ML"]


        # if key ML not in dict and ML or MERGE is TRUE
        elif "ML" not in self.dict_for_yaml and (self.dict_for_yaml["RUNSTEP"]["ML"] or self.dict_for_yaml["RUNSTEP"]["merge"]):
            self.dict_for_yaml.update({"ML": self.dict_backup["ML"]})

        if not self.dict_for_yaml["RUNSTEP"]["merge"] and "MERGE" in self.dict_for_yaml:
            self.dict_backup.update({"MERGE": self.dict_for_yaml["MERGE"]})
            del self.dict_for_yaml["MERGE"]
        elif self.dict_for_yaml["RUNSTEP"]["merge"] and not "MERGE" in self.dict_for_yaml:
            self.dict_for_yaml.update({"MERGE": self.dict_backup["MERGE"]})

        # For Draw/crop
        if not self.dict_for_yaml["RUNSTEP"]["draw"] and not self.dict_for_yaml["RUNSTEP"]["crop"] and "DRAWCROP" in self.dict_for_yaml:
            self.dict_backup.update({"DRAWCROP": self.dict_for_yaml["DRAWCROP"]})
            del self.dict_for_yaml["DRAWCROP"]
        elif "DRAWCROP" not in self.dict_for_yaml and (self.dict_for_yaml["RUNSTEP"]["draw"] or self.dict_for_yaml["RUNSTEP"]["crop"]):
            self.dict_for_yaml.update({"DRAWCROP": self.dict_backup["DRAWCROP"]})

    @property
    def export_use_yaml(self):
        """Use to print a dump config.yaml with corrected parameters"""

        def represent_dictionary_order(self, dict_data):
            return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())

        def setup_yaml():
            yaml.add_representer(OrderedDict, represent_dictionary_order)
        # self.mask_yaml()
        setup_yaml()
        return yaml.dump(self.dict_for_yaml, default_flow_style=False, sort_keys=False, indent=4)

    def upload_yaml(self):
        defaultFolder = return_default_folder()
        filename = qt.QFileDialog.getOpenFileName(self, "Select your file", defaultFolder, "yaml file (*.yaml)")
        filename = filename[0]
        if filename != "" and filename:
            self.yaml_path = filename
            self.load_yaml()
            Dfct.SubElement(vrb.userPathElement, "ImportImages").text = os.path.dirname(filename)
            Dfct.saveXmlElement(vrb.userPathElement, vrb.folderInformation + "/UserPath.mho", forceSave=True)
        else:
            self.logger.error("Error: Please select file")

    def save_yaml(self):
        defaultFolder = return_default_folder()
        filename = qt.QFileDialog.getSaveFileName(self, "Save config yaml file", defaultFolder, "yaml file(*.yaml)")
        filename = filename[0]
        if filename != "" and filename:
            self.yaml_path = filename
            with open(self.yaml_path, "w") as write_yaml:
                write_yaml.write(self.export_use_yaml)
            Dfct.SubElement(vrb.userPathElement, "ImportImages").text = os.path.dirname(filename)
            Dfct.saveXmlElement(vrb.userPathElement, vrb.folderInformation + "/UserPath.mho", forceSave=True)
            return True
        else:
            return False

    def run_leaftool(self):
        self.process.clear()
        if self.save_yaml():
            self.layer_leaftool_params.run.setDisabled(True)
            with Timer() as timer:
                self.leaftool = LeAFtool(config_file=self.yaml_path)
            # if leaftool and leaftool.analysis.csv_path_merge:
            #     print(leaftool.analysis.csv_path_merge)
            self.logger.info(f'Total time in seconds:{timer.interval}')
            qt.QApplication.processEvents()

            self.layer_leaftool_params.run.setDisabled(False)


class MainInterface(qt.QWidget):
    """
    """
    def __init__(self):
        super().__init__()
        # Layout Style
        self.layout = qt.QGridLayout()
        # self.layout.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.setLayout(self.layout)
        self.setContentsMargins(5, 5, 5, 5)
        self.setAutoFillBackground(True)
        # windows size
        # self.setMinimumWidth(int(1920))
        self.setMaximumSize(int(1800), int(900))
        style_global = fct.getStyleSheet()
        self.setStyleSheet(style_global)


        # Add title and logo
        self.setWindowTitle("LeAFtool")
        self.setWindowIcon(QIcon(vrb.folderMacroInterface + "/LeAFtool/Images/favicon.png"))
        self.logo_label = qt.QLabel(self)
        self.logo_img = QtGui.QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/LeAFtool-long.png")

        self.logo_img = self.logo_img.scaledToHeight(60, mode=Qt.FastTransformation)
        self.logo_label.setPixmap(self.logo_img)
        self.logo_label.setAlignment(Qt.AlignVCenter)
        self.logo_label.setMaximumHeight(60)

        # Initialize tab screen
        self.tabs = qt.QTabWidget()
        self.tabs.setTabPosition(qt.QTabWidget.West)
        self.tab1 = RunLeAFtool()
        #
        # if self.tab1.leaftool:
        #     csv_file = self.tab1.leaftool.analysis.csv_path_merge
        #     path_images = self.tab1.layer_ml_merge.images_path
        #     print(csv_file)
        #     print(path_images)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding))

        self.csv_file = FileSelectorLeaftool(label="CSV file:", file=True)
        not_resize = self.csv_file.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.csv_file.setSizePolicy(not_resize)

        self.table_final = TableWidget()
        # Layout Style
        table_group = qt.QGroupBox()
        table_group.setTitle("Merge Results")
        table_group.setStyleSheet(style)
        table_group.layout = qt.QVBoxLayout()
        table_group.layout.setContentsMargins(5, 5, 5, 5)
        table_group.setLayout(table_group.layout)
        table_group.layout.addWidget(self.csv_file)
        table_group.layout.addWidget(self.table_final)
        self.layout.addWidget(table_group, 0, 3, Qt.AlignLeft)

        self.tab2 = table_group

        # Add tabs
        self.tabs.addTab(self.tab1, "Run LeAFtool")
        self.tabs.addTab(self.tab2, "Explore Results")
        self.layout.addWidget(self.logo_label, 0, 0, Qt.AlignCenter)
        self.layout.addWidget(self.tabs, 1, 0, Qt.AlignLeft)

        self.csv_file.lineEditFile.textChanged.connect(self.update_table)

    def update_table(self):
        csv_file = self.csv_file.lineEditFile.text()
        if csv_file and Path(csv_file).exists():
            path_images = Path(csv_file).parent
            df = pd.read_csv(csv_file, index_col=None, header=[0], sep="\t")
            ddict = df.to_dict(orient='list')
            self.table_final.loadDictionary(ddict, path_images)
        else:
            self.table_final.clear()


main_interface = MainInterface()


def openWidget():
    main_interface.showMaximized()
    fct.showWidget(main_interface)


try:
    button.clicked.connect(openWidget)
except:
    pass

if __name__ == '__main__':

    app = QCoreApplication.instance()
    if app is None:
        app = qt.QApplication([])

    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook

    # foo = FileSelectorLeaftool("test")
    # foo = DataExplorer()
    # foo = DrawCropParams(parent=RunLeAFtool)
    foo = MainInterface()
    # foo = RunLeAFtool()
    # foo = NumberLineEditLabel(constraint="Natural", text="0", label="Y pieces:")
    foo.showMaximized()
    foo.show()
    app.exec_()
