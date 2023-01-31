import logging.config
import os, sys
from sys import path as syspath, executable
import subprocess
from pathlib import Path
import yaml
import pandas as pd
from collections import OrderedDict
from PyQt5.QtCore import Qt, QCoreApplication, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.Qsci import QsciScintilla, QsciLexerYAML
import PyQt5.QtWidgets as qt
from functools import partial

import PyIPSDK

python_exe = sys.executable

# auto add Explorer in PYTHONPATH
explorer_path = Path(PyIPSDK.getPyIPSDKDir()).parent.parent.joinpath("Explorer", "Interface")
syspath.insert(0, explorer_path.as_posix())

# import explorer variables/functions
import UsefullVariables as vrb
import UsefullWidgets as wgt
import UsefullFunctions as fct
import DatabaseFunction as Dfct

# add plugin LeAFtool to PYTHONPATH
syspath.insert(0, Path(vrb.folderMacroInterface + "/LeAFtool/").as_posix())

# Import LeAFtool class
from Leaftool_addons.DrawCut import DrawCutParams
from Leaftool_addons.MachineLearning import MachineLearningParams
from Leaftool_addons.commonWidget import style, scroll_style, return_default_folder, TableWidget, TwoListSelection, FileSelectorLeaftool, Documentator, check_values
# from Leaftool_addons.cmd_LeAFtool import LeAFtool

# configure logger
logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'handlers': {
                'stdout_handler': {
                    'level': f'{"DEBUG"}',
                    'class': "logging.StreamHandler",
                    'formatter': '',
                    'stream': 'ext://sys.stdout',
                },
            },
            'loggers': {
                "": {
                    'handlers': ['stdout_handler'],
                    'propagate': True,
                },
            }
        })
logger = logging.getLogger('LeAFtool GUI')


class QTextEditLogger(logging.Handler, QObject):
    appendPlainText = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = qt.QPlainTextEdit(parent)
        self.widget.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
        self.widget.setReadOnly(True)
        self.widget.setStyleSheet('background-color:#4c4c4c;border:1px;border-style:solid;border-color:#999;')
        # self.widget.setFixedWidth(int(1000 * vrb.ratio))
        self.appendPlainText.connect(self.widget.appendPlainText)

    def emit(self, record):
        message = self.format(record)
        if "DEBUG" in message.upper():
            self.widget.setStyleSheet('background-color:#4c4c4c;border:1px;border-style:solid;border-color:#999;color: #ffffff;')
        elif "INFO" in message.upper():
            self.widget.setStyleSheet('background-color:#4c4c4c;border:1px;border-style:solid;border-color:#999;color: #48dada;')
        elif "WARNING" in message.upper():
            self.widget.setStyleSheet('background-color:#4c4c4c;border:1px;border-style:solid;border-color:#999;color: #f5f500;')
        elif "ERROR" in message.upper():
            self.widget.setStyleSheet('background-color:#4c4c4c;border:1px;border-style:solid;border-color:#999;color: #ff0000;')
        elif "CRITICAL" in message.upper():
            self.widget.setStyleSheet('background-color:#4c4c4c;border:1px;border-style:solid;border-color:#999;color: #aa0000;')
        if "[" in message[:5]:
            self.appendPlainText.emit(f"{message.rstrip()[5:-4]}")
        elif "\n" in message:
            self.appendPlainText.emit(f"{message.rstrip()}")
        else:
            self.appendPlainText.emit(f"{message}")
        qt.QApplication.processEvents()

    def clear(self):
        self.widget.clear()


class YAMLEditor(QsciScintilla):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setLexer(QsciLexerYAML(self))
        self.setReadOnly(True)
        # Set the zoom factor, the factor is in points.
        self.zoomTo(2)
        # line numbers margin colors
        self.setMarginsBackgroundColor(QColor("#323232"))
        self.setMarginsForegroundColor(QColor("#323232"))
        # Use boxes as folding visual
        self.setFolding(self.BoxedTreeFoldStyle)
        # Braces matching
        self.setBraceMatching(self.SloppyBraceMatch)
        # folding margin colors (foreground,background)
        self.setFoldMarginColors(QColor("#929292"),
                                 QColor("#323232"))
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
        # self.layout.setSizeConstraint(qt.QVBoxLayout.SetDefaultConstraint)
        self.layout.setContentsMargins(10, 0, 10, 5)
        self.setLayout(self.layout)
        self.setFixedWidth(int(1000 * vrb.ratio))
        # self.setAutoFillBackground(True)

        # Widgets
        self.plant_model_label = qt.QLabel()
        self.plant_model_label.setText("Plant model:")
        self.plant_model_label.setFixedHeight(int(30 * vrb.ratio))
        self.plant_model_label.setWhatsThis(self.parent.parent.dico_doc["PLANT_MODEL"])
        self.plant_model_label.setStatusTip(self.parent.parent.dico_doc_str["PLANT_MODEL"])
        self.plant_model = qt.QComboBox()
        self.plant_model.addItems(["banana", "rice"])
        self.plant_model.setFixedSize(int(100 * vrb.ratio), int(25 * vrb.ratio))
        self.plant_model.setWhatsThis(self.parent.parent.dico_doc["PLANT_MODEL"])
        self.plant_model.setStatusTip(self.parent.parent.dico_doc_str["PLANT_MODEL"])

        self.show_meta_checkbox = qt.QCheckBox()
        self.show_meta_checkbox.setText("Show Meta Section")
        self.show_meta_checkbox.setChecked(True)
        self.show_meta_checkbox.setFixedWidth(int(150 * vrb.ratio))
        self.show_meta_checkbox.setWhatsThis(self.parent.parent.dico_doc["show_meta"])
        self.show_meta_checkbox.setStatusTip(self.parent.parent.dico_doc_str["show_meta"])

        self.draw_checkbox = qt.QCheckBox()
        self.draw_checkbox.setText("Draw")
        self.draw_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))
        self.draw_checkbox.setWhatsThis(self.parent.parent.dico_doc["draw"])
        self.draw_checkbox.setStatusTip(self.parent.parent.dico_doc_str["draw"])

        self.cut_checkbox = qt.QCheckBox()
        self.cut_checkbox.setText("Cut")
        self.cut_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))
        self.cut_checkbox.setWhatsThis(self.parent.parent.dico_doc["cut"])
        self.cut_checkbox.setStatusTip(self.parent.parent.dico_doc_str["cut"])

        self.ml_checkbox = qt.QCheckBox()
        self.ml_checkbox.setText("ML")
        self.ml_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))
        self.ml_checkbox.setWhatsThis(self.parent.parent.dico_doc["ml"])
        self.ml_checkbox.setStatusTip(self.parent.parent.dico_doc_str["ml"])

        self.merge_checkbox = qt.QCheckBox()
        self.merge_checkbox.setText("Merge")
        self.merge_checkbox.setFixedSize(int(60 * vrb.ratio), int(30 * vrb.ratio))
        self.merge_checkbox.setWhatsThis(self.parent.parent.dico_doc["merge"])
        self.merge_checkbox.setStatusTip(self.parent.parent.dico_doc_str["merge"])

        self.csv_file = FileSelectorLeaftool(label="CSV file:", file=True)
        self.csv_file.setWhatsThis(self.parent.parent.dico_doc["csv_file"])
        self.csv_file.setStatusTip(self.parent.parent.dico_doc_str["csv_file"])

        self.list_selection = TwoListSelection()
        self.list_selection.setWhatsThis(self.parent.parent.dico_doc["rename"])
        self.list_selection.setStatusTip(self.parent.parent.dico_doc_str["rename"])
        # self.list_selection.setAutoFillBackground(True)
        self.list_selection.setMaximumHeight(int(65 * vrb.ratio))

        # Position widgets
        self.layout.addWidget(self.plant_model_label, 0, 0, Qt.AlignLeft)
        self.layout.addWidget(self.plant_model, 0, 1, Qt.AlignLeft)
        self.layout.addWidget(self.show_meta_checkbox, 0, 2, Qt.AlignLeft)
        self.layout.addWidget(self.parent.parent.whatsThisButton, 0, 4, Qt.AlignBottom)

        # Layout Style
        tools_group = qt.QGroupBox()
        tools_group.setTitle("Tools activation")
        tools_group.setStyleSheet(style)
        tools_group.layout = qt.QHBoxLayout()
        tools_group.layout.setContentsMargins(5, 10, 0, 0)
        tools_group.setLayout(tools_group.layout)
        tools_group.layout.addStretch()
        tools_group.layout.addWidget(self.draw_checkbox)
        tools_group.layout.addWidget(self.cut_checkbox)
        tools_group.layout.addWidget(self.ml_checkbox)
        tools_group.layout.addWidget(self.merge_checkbox)
        tools_group.layout.addStretch()
        # tools_group.setAutoFillBackground(True)
        # tools_group.setFixedWidth(int(500 * vrb.ratio))
        self.layout.addWidget(tools_group, 0, 3, Qt.AlignTop)

        # Meta group
        self.meta_group = qt.QGroupBox()
        self.meta_group.setTitle("Meta Infos")
        self.meta_group.setStyleSheet(style)
        self.meta_group.layout = qt.QVBoxLayout()
        # self.meta_group.layout.setSizeConstraint(qt.QVBoxLayout.SetDefaultConstraint)
        self.meta_group.layout.setContentsMargins(10, 10, 10, 10)
        self.meta_group.setLayout(self.meta_group.layout)
        self.meta_group.layout.addWidget(self.csv_file)
        self.meta_group.layout.addWidget(self.list_selection)
        self.meta_group.setFixedWidth(int(1000 * vrb.ratio))
        # self.meta_group.setAutoFillBackground(True)
        # self.layout.addWidget(self.meta_group, 1, 0, 1, 4)

        # connections
        self.draw_checkbox.stateChanged.connect(self.update_activation_tools)
        self.cut_checkbox.stateChanged.connect(self.update_activation_tools)
        self.ml_checkbox.stateChanged.connect(self.update_activation_tools)
        self.merge_checkbox.stateChanged.connect(self.update_activation_tools)
        self.plant_model.currentIndexChanged.connect(self.update_activation_tools)
        self.csv_file.lineEditFile.textChanged.connect(partial(self.parent.check_path, from_object=self.csv_file.lineEditFile, keys_list="'csv_file'"))
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
        try:
            self.loading = True
            if self.parent.dict_for_yaml["PLANT_MODEL"] not in ["banana", "rice"]:
                self.parent.logger.error(
                    f"Error: arguments PLANT_MODEL:'{self.parent.dict_for_yaml['PLANT_MODEL']}' is not allow, please use only 'banana' or 'rice', please reload valid file")
            else:
                self.plant_model.setCurrentText(self.parent.dict_for_yaml["PLANT_MODEL"])

             # check all boolean values
            message = check_values(dico_params=self.parent.dict_for_yaml,
                                   primary_key="RUNSTEP",
                                   secondary_key_list=["draw", "cut", "ML", "merge"],
                                   type_value=bool,
                                   default_error=False)
            if message:
                self.parent.logger.error(message)
            self.draw_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["draw"]))
            self.cut_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["cut"]))
            self.ml_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["ML"]))
            self.merge_checkbox.setChecked(bool(self.parent.dict_for_yaml["RUNSTEP"]["merge"]))

            self.csv_file.lineEditFile.setText(self.parent.dict_for_yaml["csv_file"])
            self.list_selection.clean_list()
            self.list_selection.add_right_elements(self.parent.dict_for_yaml["rename"])
            self.loading = False
        except KeyError as e:
            self.parent.logger.error(f"ERROR: Key {e} is not found on file")
            self.parent.dict_for_yaml = self.parent.dict_backup

    def update_activation_tools(self):
        if not self.loading:
            self.parent.dict_for_yaml["RUNSTEP"]["cut"] = self.cut_checkbox.isChecked()
            self.parent.dict_for_yaml["RUNSTEP"]["draw"] = self.draw_checkbox.isChecked()
            self.parent.layer_draw_cut.show_draw_params()
            self.parent.dict_for_yaml["RUNSTEP"]["ML"] = self.ml_checkbox.isChecked()
            self.parent.dict_for_yaml["RUNSTEP"]["merge"] = self.merge_checkbox.isChecked()
            self.parent.layer_ml_merge.show_ml_merge_params()
            self.parent.dict_for_yaml["PLANT_MODEL"] = self.plant_model.currentText()
            self.parent.dict_for_yaml["csv_file"] = self.csv_file.lineEditFile.text()
            self.update_header_csv()
            self.parent.dict_for_yaml["rename"] = self.list_selection.get_right_elements()
            self.parent.preview_config.setText(self.parent.export_use_yaml)

    def disable_running_bottom(self):
        # if all disable, disable run

        if (not self.parent.layer_tools.draw_checkbox.isChecked() \
                and not self.parent.layer_tools.cut_checkbox.isChecked() \
                and not self.parent.layer_tools.ml_checkbox.isChecked() \
                and not self.parent.layer_tools.merge_checkbox.isChecked()) or self.parent.warning_found:
            self.parent.layer_leaftool_params.run.setDisabled(True)
            self.parent.layer_leaftool_params.save.setDisabled(True)
        else:
            self.parent.layer_leaftool_params.run.setDisabled(False)
            self.parent.layer_leaftool_params.save.setDisabled(False)
                # self.csv_file.setVisible(self.parent.layer_tools.draw_checkbox.isChecked())
                # self.csv_file.setVisible(self.parent.layer_tools.cut_checkbox.isChecked())
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
        self.setTitle("LeAFtool")
        self.setStyleSheet(style)
        # self.setAutoFillBackground(True)
        self.layout = qt.QGridLayout()
        # self.layout.setSizeConstraint(qt.QVBoxLayout.SetMinAndMaxSize)
        self.setMaximumHeight(int(80*vrb.ratio))
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)
        self.setFixedWidth(int(1000 * vrb.ratio))
        self.loading = False

        # Widgets
        icon_size = 30
        self.save_label = qt.QLabel()
        self.save_label.setText("Save YAML:")
        self.save = wgt.PushButtonImage(vrb.folderImages + "/Save_As.png")
        self.save.setFixedSize(int(icon_size * vrb.ratio), int(icon_size * vrb.ratio))
        self.save.setWhatsThis(self.parent.parent.dico_doc["save"])
        self.save.setStatusTip(self.parent.parent.dico_doc_str["save"])

        self.upload_label = qt.QLabel()
        self.upload_label.setText("Upload YAML:")
        self.upload = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/upload.png")
        self.upload.setFixedSize(int(icon_size * vrb.ratio), int(icon_size * vrb.ratio))
        self.upload.setWhatsThis(self.parent.parent.dico_doc["upload"])
        self.upload.setStatusTip(self.parent.parent.dico_doc_str["upload"])

        self.run_label = qt.QLabel()
        self.run_label.setText("Run:")
        self.run = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/run.png")
        self.run.setCheckable(True)
        self.run.setFixedSize(int(icon_size * vrb.ratio), int(icon_size * vrb.ratio))
        self.run.setWhatsThis(self.parent.parent.dico_doc["run"])
        self.run.setStatusTip(self.parent.parent.dico_doc_str["run"])

        self.preview_yaml_checkbox = qt.QCheckBox()
        self.preview_yaml_checkbox.setChecked(False)
        self.preview_yaml_checkbox.setText("Preview YAML")
        self.preview_yaml_checkbox.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))
        self.preview_yaml_checkbox.setWhatsThis(self.parent.parent.dico_doc["preview"])
        self.preview_yaml_checkbox.setStatusTip(self.parent.parent.dico_doc_str["preview"])

        self.debug_checkbox = qt.QCheckBox()
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.setText("Debug")
        self.debug_checkbox.setFixedSize(int(120 * vrb.ratio), int(30 * vrb.ratio))
        self.debug_checkbox.setWhatsThis(self.parent.parent.dico_doc["debug"])
        self.debug_checkbox.setStatusTip(self.parent.parent.dico_doc_str["debug"])

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
        except KeyError as e:
            self.parent.logger.error(f"ERROR: Key {e} is not found on file")
            self.parent.dict_for_yaml = self.parent.dict_backup

    def upload_debug(self):
        self.loading = True
        self.debug_checkbox.setChecked(bool(self.parent.dict_for_yaml["debug"]))
        self.loading = False


class RunLeAFtool(qt.QWidget):
    """"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.dict_for_yaml = {}
        self.dict_backup = {}
        self.logger = logger
        self.leaftool = None
        self.connect = None
        self.running_process = None
        self.warning_found = False

        # Layout Style
        self.layout = qt.QGridLayout()
        self.layout.setSizeConstraint(qt.QVBoxLayout.SetMinAndMaxSize)
        self.setLayout(self.layout)
        self.setContentsMargins(0, 0, 0, 0)
        style_global = fct.getStyleSheet()
        self.setStyleSheet(style_global)
        self.setStyleSheet(scroll_style)

        # Create the text output widget.
        self.process = QTextEditLogger(self)
        self.logger.addHandler(self.process)
        self.logger.setLevel(logging.DEBUG)

        # add preview of yaml file
        self.preview_config = qt.QPlainTextEdit()
        self.preview_config = YAMLEditor()
        self.preview_config.setMinimumWidth(500)
        self.preview_config.setAutoFillBackground(True)
        self.preview_config.setWhatsThis(self.parent.dico_doc["preview_edit"])
        self.preview_config.setStatusTip(self.parent.dico_doc_str["preview_edit"])

        # add path config file
        self.yaml_path = vrb.folderMacroInterface + "/LeAFtool/config.yaml"
        self.default_yaml_path = vrb.folderMacroInterface + "/LeAFtool/config.yaml"

        # Add layer part
        self.layer_tools = ToolsActivation(parent=self)
        self.layer_draw_cut = DrawCutParams(parent=self)
        self.layer_leaftool_params = LeaftoolParams(parent=self)
        self.layer_ml_merge = MachineLearningParams(parent=self)

        # # size policy
        not_resize = self.layer_ml_merge.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.layer_ml_merge.setSizePolicy(not_resize)
        not_resize = self.layer_draw_cut.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.layer_draw_cut.setSizePolicy(not_resize)

        self.layout.addWidget(self.layer_tools, 0, 0, 1, 2)
        self.layout.addWidget(self.layer_tools.meta_group, 1, 0, 1, 2)
        self.layout.addWidget(self.preview_config, 0, 3, 5, 1)
        self.layout.addWidget(self.layer_draw_cut, 2, 0)
        self.layout.addWidget(self.layer_ml_merge, 2, 1)
        self.layout.addWidget(self.layer_leaftool_params, 3, 0, 1, 2)
        self.layout.addWidget(self.process.widget, 4, 0, 1, 2)
        not_resize = self.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.setSizePolicy(not_resize)
        self.load_yaml()
        # self.setAutoFillBackground(True)

        # Layer leaftool params connection
        self.layer_leaftool_params.upload.clicked.connect(self.upload_yaml)
        self.layer_leaftool_params.save.clicked.connect(self.save_yaml)
        self.layer_leaftool_params.run.clicked.connect(self.change_run_state)
        qt.QApplication.instance().focusChanged.connect(self.on_focus_changed)

    def change_run_state(self):
        if self.layer_leaftool_params.run.isChecked():
            if self.save_yaml():
                self.layer_leaftool_params.run_label.setText("Stop:")
                self.layer_leaftool_params.run.pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/stop.png")
                self.layer_leaftool_params.run.resizeEvent(None)
                self.start_threads()
            else:
                self.layer_leaftool_params.run.setChecked(False)
        else:
            self.layer_leaftool_params.run_label.setText("Run:")
            self.layer_leaftool_params.run.pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/run.png")
            self.layer_leaftool_params.run.resizeEvent(None)
            self.layer_leaftool_params.run.setChecked(False)
            self.abort_workers()

    def start_threads(self):
        self.process.clear()
        # self.leaftool = LeAFtool(config_file=self.yaml_path)
        # cmd = f"{python_exe} {Path(__file__).parent.joinpath('Leaftool_addons', 'cmd_LeAFtool.py')} -c {Path(self.yaml_path).resolve()}"
        # self.running_process = subprocess.Popen("exec " + cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ)
        cmd = f"source /etc/profile; {executable} {Path(__file__).parent.joinpath('Leaftool_addons', 'cmd_LeAFtool.py')} -c {Path(self.yaml_path).resolve()}"
        self.logger.info(cmd)
        self.running_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable="/bin/bash")
        with self.running_process:
            for line in iter(self.running_process.stdout.readline, b''):
                line = line.decode("utf-8").rstrip().rstrip()
                if line:
                    self.logger.info(line)
        self.layer_leaftool_params.run.setChecked(False)
        del self.running_process
        self.running_process = None
        self.change_run_state()

    def abort_workers(self):
        if self.running_process:
            self.logger.info('Asking to abort')
            self.running_process.kill()
            del self.running_process
            self.running_process = None
            self.logger.info("kill Thread success")

    def load_yaml(self):
        self.process.clear()
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
    def on_focus_changed(self, old, now):
        self.update_all()
        self.get_warning()
        self.layer_tools.disable_running_bottom()
        # pass

    def get_warning(self):
        """Get warning on txt"""

        txt = self.process.widget.toPlainText()
        if "warning" in txt.lower() or "error" in txt.lower():
            self.warning_found = True
        else:
            self.warning_found = False
            self.process.clear()

    def update_all(self):
        self.layer_tools.update_activation_tools()
        self.layer_draw_cut.update_draw_cut_params()
        self.layer_ml_merge.update_ml_params()
        self.layer_leaftool_params.update_debug()
        self.preview_config.setText(self.export_use_yaml)
        self.dict_backup = self.dict_for_yaml.copy()

    def upload_all(self):
        self.layer_tools.upload_activation_tools()
        self.layer_draw_cut.upload_draw_cut_params()
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
        elif self.dict_for_yaml["RUNSTEP"]["merge"] and "MERGE" not in self.dict_for_yaml:
            self.dict_for_yaml.update({"MERGE": self.dict_backup["MERGE"]})

        # For Draw/cut
        if not self.dict_for_yaml["RUNSTEP"]["draw"] and not self.dict_for_yaml["RUNSTEP"]["cut"] and "DRAW-CUT" in self.dict_for_yaml:
            self.dict_backup.update({"DRAW-CUT": self.dict_for_yaml["DRAW-CUT"]})
            del self.dict_for_yaml["DRAW-CUT"]
        elif "DRAW-CUT" not in self.dict_for_yaml and (self.dict_for_yaml["RUNSTEP"]["draw"] or self.dict_for_yaml["RUNSTEP"]["cut"]):
            self.dict_for_yaml.update({"DRAW-CUT": self.dict_backup["DRAW-CUT"]})

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
        try:
            default_filename = list(Path(defaultFolder).glob("*.yaml"))[0].as_posix()
        except IndexError:
            default_filename = defaultFolder
        filename = qt.QFileDialog.getOpenFileName(self, "Select your file", default_filename, "yaml file (*.yaml)")
        filename = filename[0]
        if filename != "" and filename:
            self.yaml_path = filename
            self.load_yaml()
            Dfct.SubElement(vrb.userPathElement, "ImportImages").text = Path(filename).parent.as_posix()
            Dfct.saveXmlElement(vrb.userPathElement, vrb.folderInformation + "/UserPath.mho", forceSave=True)
        else:
            self.logger.error("Error: Please select file")

    def save_yaml(self):
        defaultFolder = return_default_folder()
        default_filename = Path(defaultFolder).joinpath("config.yaml").as_posix()
        filename = qt.QFileDialog.getSaveFileName(self, "Save config yaml file", default_filename, "yaml file(*.yaml)")
        filename = filename[0]
        if filename != "" and filename:
            self.yaml_path = filename
            with open(self.yaml_path, "w") as write_yaml:
                write_yaml.write(self.export_use_yaml)
            Dfct.SubElement(vrb.userPathElement, "ImportImages").text = Path(filename).parent.as_posix()
            Dfct.saveXmlElement(vrb.userPathElement, vrb.folderInformation + "/UserPath.mho", forceSave=True)
            return True
        else:
            return False

    def check_path(self, from_object, keys_list):
        def clean_warning(keys_list):
            txt = self.process.widget.toPlainText()
            if keys_list in txt:
                txt = "\n".join([elm for elm in txt.split("\n") if keys_list not in elm])
                self.process.clear()
                self.logger.warning(txt)

        path_str = from_object.text()
        if path_str:
            if Path(path_str).exists():
                from_object.setStyleSheet("background-color: #606060;")
                clean_warning(keys_list)
            else:
                self.logger.warning(f"Warning: arguments {keys_list}: '{path_str}' doesn't exist")
                from_object.setStyleSheet("background-color: darkRed;")
        elif path_str == "":
            from_object.setStyleSheet("background-color: #606060;")
            clean_warning(keys_list)


class MainInterface(qt.QMainWindow):
    """
    """
    def __init__(self):
        super().__init__()
        self.main = qt.QWidget()
        self.setCentralWidget(self.main)
        # Add documentation mode:
        self.documentator = Documentator()
        self.dico_doc = self.documentator.dico_doc
        self.dico_doc_str = self.documentator.dico_doc_str
        self.whatsThisButton = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/help.png")
        self.whatsThisButton.setFixedSize(int(40 * vrb.ratio), int(40 * vrb.ratio))
        self.whatsThisButton.clicked.connect(qt.QWhatsThis.enterWhatsThisMode)
        self.whatsThisButton.setToolTip("Click on this button and then on another object to get the documentation")
        self.statusbar = self.statusBar()
        self.statusbar.setSizeGripEnabled(False)

        # Layout Style
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.layout = qt.QVBoxLayout(self.main)
        # self.layout.setSizeConstraint(1)
        self.main.setContentsMargins(0, 5, 5, 5)
        self.main.setAutoFillBackground(True)
        style_global = fct.getStyleSheet()
        self.setStyleSheet(style_global)

        # Add title and logo
        self.setWindowTitle("LeAFtool")
        self.setWindowIcon(QIcon(vrb.folderMacroInterface + "/LeAFtool/Images/favicon.png"))
        self.logo_label = qt.QLabel(self)
        self.logo_img = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/LeAFtool-long.png")

        self.logo_img = self.logo_img.scaledToHeight(80, mode=Qt.FastTransformation)
        self.logo_label.setPixmap(self.logo_img)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setMaximumHeight(80)

        # Initialize the 2 tab screen
        self.tabs = qt.QTabWidget()
        self.tabs.setTabPosition(qt.QTabWidget.West)

        # FIRST TAB PAGE
        self.tab1 = RunLeAFtool(parent=self)

        # SECOND TAB PAGE
        self.csv_file = FileSelectorLeaftool(label="CSV file:", file=True)
        self.table_final = TableWidget()
        self.table_final.setAutoFillBackground(True)
        # SECOND TAB Layout Style
        table_group = qt.QGroupBox()
        table_group.setTitle("Merge Results")
        table_group.setStyleSheet(style)
        table_group.layout = qt.QVBoxLayout()
        table_group.layout.setContentsMargins(10, 10, 10, 10)
        table_group.setLayout(table_group.layout)
        table_group.layout.addWidget(self.csv_file)
        table_group.layout.addWidget(self.table_final)
        # table_group.setAutoFillBackground(True)
        self.tab2 = table_group

        # Add th 2 tables to object TABS
        self.tabs.addTab(self.tab1, "   Run LeAFtool   ")
        self.tabs.addTab(self.tab2, "   Explore Results   ")

        # add TABS to layout
        self.layout.addWidget(self.logo_label, Qt.AlignCenter)
        self.layout.addWidget(self.tabs)

        # Edit connection
        self.csv_file.lineEditFile.textChanged.connect(self.update_table)

    def update_table(self):
        csv_file = self.csv_file.lineEditFile.text()
        if csv_file and Path(csv_file).exists():
            with open(csv_file, "r") as csv:
                header_txt = csv.readline().rstrip()
            sep_dict = {",": header_txt.count(","),
                        ";": header_txt.count(";"),
                        # ".": header_txt.count("."),
                        "\t": header_txt.count("\t")
                        }
            csv_separator = max(sep_dict, key=sep_dict.get)
            path_images = Path(csv_file).parent
            df = pd.read_csv(csv_file, index_col=None, header=[0], sep=csv_separator)
            ddict = df.to_dict(orient='list')
            self.table_final.loadDictionary(ddict, path_images)
        else:
            self.table_final.clear()


def openWidget():
    app = QCoreApplication.instance()
    name = app.applicationName()
    app.setObjectName("LeAFtool")
    app.setApplicationName("LeAFtool")
    app.setApplicationDisplayName("LeAFtool")
    # SplashScreen
    pixmap = QPixmap(vrb.folderMacroInterface + "/LeAFtool/Images/LeAFtool-long.png")
    pixmap = pixmap.scaled(700, 700, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
    splashScreen = qt.QSplashScreen(pixmap)
    splashScreen.setFixedSize(700, 700)
    splashScreen.show()
    # Load App interface
    main_interface = MainInterface()
    main_interface.setWindowIcon(QIcon(vrb.folderMacroInterface + "/LeAFtool/Images/favicon.png"))
    splashScreen.finish(main_interface)
    main_interface.show()
    if name == "LeAFtool.py":
        app.exec()


###############################################################################################
# For adding button on Explorer
mainWindow = vrb.mainWindow
if mainWindow:
    groupMenu = mainWindow.groupMenu
    button = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/favicon.png")
    button.setFixedSize(30 * vrb.ratio, 30 * vrb.ratio)
    groupMenu.layoutBar1.addWidget(button, 0, vrb.numMacro, Qt.AlignVCenter)
    vrb.numMacro += 1
    button.clicked.connect(openWidget)


###############################################################################################
# MAIN
if __name__ == '__main__':

    openWidget()
    # app = QCoreApplication.instance()
    # if app is None:
    #     app = qt.QApplication([])
    #
    # splashScreen = seeSplashScreen()
    # app.processEvents()
    # app.setWindowIcon(QIcon(vrb.folderMacroInterface + "/LeAFtool/Images/favicon.png"))
    #
    # qt.QApplication.setStyle(qt.QStyleFactory.create('Fusion'))  # <- Choose the style
    # main_interface = MainInterface()
    #
    # # main_interface = FileSelectorLeaftool("test")
    # # main_interface = DataExplorer()
    # # main_interface = ToolsActivation(parent=RunLeAFtool)
    # # main_interface = RunLeAFtool()
    # # main_interface = NumberLineEditLabel(constraint="Natural", text="0", label="Y pieces:")
    # # main_interface.showFullScreen()
    # splashScreen.finish(main_interface)
    # main_interface.show()
    # app.exec_()
