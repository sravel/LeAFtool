from sys import path as syspath
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt, QCoreApplication, pyqtSlot
import PyQt5.QtWidgets as qt

from pathlib import Path
from pandas import read_csv
from re import match
from docutils.core import publish_parts

syspath.insert(0, Path("../").as_posix())

import PyIPSDK
import UsefullVariables as vrb
import UsefullWidgets as wgt
import UsefullFunctions as fct
import DatabaseFunction as Dfct


style = """QGroupBox:title {left: 20px ;padding-left: 10px;padding-right: 10px; padding-top: -12px; color:rgb(129, 184, 114)} 
QGroupBox {font: bold; border: 1px solid gray; margin-top:12 px; margin-bottom: 0px}
QDoubleSpinBox:focus {
border-color: #81b872;
}
QDoubleSpinBox {
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0 #646464, stop: 1 #5d5d5d);
}
QSpinBox:focus {
border-color: #81b872;
}
QSpinBox {
    background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #4d4d4d, stop: 0 #646464, stop: 1 #5d5d5d);
}
"""
allow_ext = ["tif", "tiff", "TIF", "TIFF", "Tif", "Tiff", "im6", "IM6", "jpg", "JPG", "PNG", "png", "BMP", "bmp"]

scroll_style = """QScrollBar:horizontal {
     border: 1px solid #222222;
     background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);
     height: 20px;
     margin: 0px 16px 0 16px;
}
QScrollBar:vertical
{
      background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0, stop: 0.0 #121212, stop: 0.2 #282828, stop: 1 #484848);
      width: 20px;
      margin: 16px 0 16px 0;
      border: 1px solid #222222;
}"""

int_validator = QIntValidator(0, 9999)  # Create validator.


def return_default_folder():
    defaultFolder = Dfct.childText(vrb.userPathElement, "ImportImages")
    if defaultFolder is None or defaultFolder == "" or not Path(defaultFolder).exists():
        defaultFolder = Path(vrb.folderExplorer).parent.joinpath("images")
        if not Path(defaultFolder).exists():
            defaultFolder = Path(vrb.folderExplorer).parent.joinpath("data", "Explorer", "images")
    return defaultFolder


def get_files_ext(path, extensions, add_ext=True):
    """List of files with specify extension included on folder

    Arguments:
        path (str): a path to folder
        extensions (list or tuple): a list or tuple of extension like (".py")
        add_ext (bool): if True (default), file have extension

    Returns:
        :class:`list`: List of files name with or without extension , with specify extension include on folder
        :class:`list`: List of  all extension found
     """
    if not (extensions, (list, tuple)) or not extensions:
        raise ValueError(f'ERROR LeAFtool: "extensions" must be a list or tuple not "{type(extensions)}"')
    tmp_all_files = []
    all_files = []
    files_ext = []

    for ext in extensions:
        tmp_all_files.extend(Path(path).glob(f"*{ext}"))

    for elm in tmp_all_files:
        ext = "".join(elm.suffixes).replace(".", "")
        if ext not in files_ext:
            files_ext.append(ext)
        if add_ext:
            all_files.append(elm.as_posix())
        else:
            if len(elm.suffixes) > 1:
                all_files.append(Path(elm.stem).stem)
            else:
                all_files.append(elm.stem)
    return all_files, files_ext


def check_values(dico_params, primary_key, secondary_key_list, type_value, default_error):
    # check all boolean values
    error_keys = []
    error_message = ""
    for key in secondary_key_list:
        if not isinstance(dico_params[primary_key][key], type_value):
            error_keys.append(key)
            dico_params[primary_key][key] = default_error
    if len(error_keys) > 0:
        error_message = f"ERROR: 'ML' '{' and '.join(error_keys)}' {'is' if len(error_keys)==1 else 'are'} not a valid {type_value} value, please reload valid file"
    return error_message


class Documentator:
    def __init__(self):
        self.dico_doc_rst = {}
        self.dico_doc_str = {}
        key = None
        with open(f"{vrb.folderMacroInterface}/LeAFtool/docs/params.rst", "r") as docs:
            for line in docs:
                if match("^- \*\*", line):
                    key, value = line.rstrip().split(" ")[1].replace("*", ""), line.rstrip().replace("- ", "")
                    self.dico_doc_rst[key] = value
                    self.dico_doc_str[key] = value.replace("*", "")
                elif line.rstrip() == "":
                    key = None
                elif key:
                    self.dico_doc_rst[key] = self.dico_doc_rst[key]+"\n\n\n"+line
        self.dico_doc = {}
        for key, value in self.dico_doc_rst.items():
            self.dico_doc[key] = publish_parts(value.replace("./docs/", f"{vrb.folderMacroInterface}/LeAFtool/docs/"), writer_name='html')['html_body'].replace("<img", "<p><img").replace("</div>", "</p></div>")


class FileSelectorLeaftool(qt.QGroupBox):
    """add file selector button"""

    def __init__(self, label, title=None, style="minimal", file=False):
        super().__init__()

        self.labelFile = qt.QLabel(label)
        self.labelFile.setMinimumWidth(int(83 * vrb.ratio))
        self.lineEditFile = qt.QLineEdit()
        self.lineEditFile.setFixedHeight(int(25 * vrb.ratio))
        self.lineEditFile.setReadOnly(True)

        if file:
            self.buttonOpen = wgt.PushButtonImage(vrb.folderMacroInterface + "/LeAFtool/Images/table.png")
        else:
            self.buttonOpen = wgt.PushButtonImage(vrb.folderImages + "/Folder.png")
        self.buttonOpen.setFixedSize(int(25 * vrb.ratio), int(25 * vrb.ratio))
        self.layout = qt.QGridLayout()
        self.layout.addWidget(self.labelFile, 0, 0)
        self.layout.addWidget(self.buttonOpen, 0, 1)
        self.layout.addWidget(self.lineEditFile, 0, 2)
        self.is_file = file

        self.layout.setContentsMargins(0, 2, 10, 2)
        self.setLayout(self.layout)
        self.buttonOpen.clicked.connect(self.openQDialog)
        self.setMinimumSize(int(400 * vrb.ratio), int(30 * vrb.ratio))
        self.setMaximumHeight(int(30 * vrb.ratio))
        if title:
            self.setTitle(title)
        if style == "minimal":
            self.setStyleSheet('QGroupBox {font: bold; border: 0px ; margin-top: 0 px}')
        if style == "all":
            self.setStyleSheet(style)
        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)

    def openQDialog(self):
        dlg = qt.QFileDialog()
        dlg.setOption(dlg.DontUseNativeDialog, True)
        # dlg.setOption(dlg.HideNameFilterDetails, False)
        # dlg.setFilter(dlg.filter() | QDir.Hidden)
        defaultFolder = return_default_folder()
        if self.is_file:
            filename = qt.QFileDialog.getOpenFileName(self, caption="Select your file",
                                                      directory=defaultFolder,
                                                      filter="table file (*.csv *.tsv)")
            # options=qt.QFileDialog.DontUseNativeDialog)
            filename = filename[0]
        else:
            dlg.setOption(dlg.ShowDirsOnly, True)
            dlg.setFileMode(dlg.Directory)
            filename = dlg.getExistingDirectory(dlg, caption="Select your folder",
                                                directory=defaultFolder)

        if filename != "" and filename:
            self.lineEditFile.setText(filename)
            self.lineEditFile.setFocus()
            Dfct.SubElement(vrb.userPathElement, "ImportImages").text = Path(filename).parent.as_posix()
            Dfct.saveXmlElement(vrb.userPathElement, vrb.folderInformation + "/UserPath.mho", forceSave=True)


class NumberLineEditLabel(qt.QGroupBox):
    def __init__(self, constraint, text, label, size=65):
        super().__init__()
        self.layout = qt.QGridLayout()
        self.lineEdit = wgt.NumberLineEdit(constraint=constraint, align=Qt.AlignLeft)
        self.lineEdit.setText(text)
        self.lineEdit.setPlaceholderText(text)
        self.lineEdit.setMinimumWidth(int(50 * vrb.ratio))
        self.lineEdit.setMaximumWidth(int(60 * vrb.ratio))
        self.lineEdit.setFixedHeight(int(25 * vrb.ratio))
        self.lineEdit.setMaxLength(5)
        self.label_lineEdit = qt.QLabel(label)
        self.label_lineEdit.setFixedWidth(int(size * vrb.ratio))
        self.label_lineEdit.setFixedHeight(int(25 * vrb.ratio))

        self.layout.addWidget(self.label_lineEdit, 0, 0)
        self.layout.addWidget(self.lineEdit, 0, 1, Qt.AlignLeft)

        # self.setMinimumSize(int(110 * vrb.ratio), int(35 * vrb.ratio))
        self.setLayout(self.layout)
        # self.layout.setContentsMargins(2, 2, 2, 2)
        self.setStyleSheet('QGroupBox {font: bold; border: 0px ; margin-top: 0 px}')


class SpinBoxLabel(qt.QGroupBox):
    def __init__(self, min_v, max_v, step, value, label, size=65):
        super().__init__()
        self.layout = qt.QGridLayout()

        if isinstance(value,float):
            self.lineEdit = qt.QDoubleSpinBox()
        else:
            self.lineEdit = qt.QSpinBox()
        self.lineEdit.setRange(min_v, max_v)
        self.lineEdit.setSingleStep(step)
        self.lineEdit.setValue(value)
        self.lineEdit.setMinimumWidth(int(50 * vrb.ratio))
        self.lineEdit.setMaximumWidth(int(60 * vrb.ratio))
        self.lineEdit.setFixedHeight(int(25 * vrb.ratio))
        self.label_lineEdit = qt.QLabel(label)
        self.label_lineEdit.setFixedWidth(int(size * vrb.ratio))
        self.label_lineEdit.setFixedHeight(int(25 * vrb.ratio))

        self.layout.addWidget(self.label_lineEdit, 0, 0)
        self.layout.addWidget(self.lineEdit, 0, 1, Qt.AlignLeft)

        # self.setMinimumSize(int(110 * vrb.ratio), int(35 * vrb.ratio))
        self.setLayout(self.layout)
        # self.layout.setContentsMargins(2, 2, 2, 2)
        self.setStyleSheet('QGroupBox {font: bold; border: 0px ; margin-top: 0 px}')


def randomPastelColor(i):
    # colors = [171, 54, 114, 5, 231, 187, 254, 7, 214, 192, 50, 258, 48, 56, 107, 255, 33, 34, 177, 246]
    colors = [10, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 255, 33, 34, 177, 246, 171, 54, 114, 5, 231, 187, 254, 7, 214, 192, 50, 258, 48, 56, 107, 255, 33, 34, 177, 246]
    saturation = 90
    light = 90
    return QColor.fromHsl(colors[i], int(saturation * 255 / 100), int(light * 255 / 100), 100)


class TableWidget(qt.QTableWidget):
    def __init__(self, ddict=None, path_images=None):
        super(TableWidget, self).__init__()

        self.ddict = ddict
        self.indice_cut_name = None
        self.indice_class_name = None
        self.old_selection = [None, None]
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding))
        self.path_images = path_images

        self.horizontalHeader().setVisible(True)
        self.verticalHeader().setVisible(False)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSortingEnabled(True)
        # self.loadDictionary()
        self.horizontalHeader().setStretchLastSection(True)
        self.setAlternatingRowColors(True)

        self.itemPressed.connect(self.cellClick)
        self.display()

    def loadDictionary(self, ddict, path_images):
        try:
            self.ddict = ddict
            self.path_images = path_images
            self.setColumnCount(len(self.ddict))
            col = 0
            for indice, name in enumerate(self.ddict):
                if name in ["cut_name", "crop_name"]:
                    self.indice_cut_name = indice
                if name == "Class":
                    self.indice_class_name = indice
                self.setRowCount(len(self.ddict[name]))
                cell = qt.QTableWidgetItem(name)
                self.setHorizontalHeaderItem(col, cell)
                columnColor = randomPastelColor(col)
                cell.setBackground(columnColor.darker(100))
                columnColorDarker = columnColor.darker(40)
                row = 0
                for i in range(len(self.ddict[name])):
                    if isinstance(self.ddict[name][i], type(True)):
                        cell = qt.QTableWidgetItem(str(self.ddict[name][i]))
                    elif isinstance(self.ddict[name][i], type(float)):
                        cell = qt.QTableWidgetItem(fct.numberCalibration(self.ddict[name][i], 3))
                    else:
                        cell = qt.QTableWidgetItem(str(self.ddict[name][i]))
                    cell.setBackground((columnColor, columnColorDarker)[row % 2 == 0])
                    self.setItem(row, col, cell)
                    row += 1
                self.resizeColumnToContents(col)
                col += 1
        except Exception as e:
            print(e)
            pass

    def display(self, boolShow=True):
        self.setWindowTitle("Global Merge")
        if boolShow:
            if self.isMaximized():
                self.showMaximized()
            else:
                self.showNormal()
            self.window().raise_()
            self.window().activateWindow()

    @staticmethod
    def clearAllImages():
        while vrb.mainWindow.widgetLabelImage.layout.count() != 0:
            itemImage = vrb.mainWindow.widgetLabelImage.layout.itemAt(0)
            if itemImage is not None:
                vrb.mainWindow.widgetLabelImage.deleteLabel(itemImage.widget(), actualize=False)

    def cellClick(self, cellItem):
        if vrb.mainWindow:
            rowValue, columnValue = cellItem.row(), cellItem.column()
            text = self.item(rowValue, self.indice_cut_name).text()  # On peut récupérer les coordonnées et le texte de la cellule sur laquelle on a cliqué
            if rowValue != self.old_selection[0] and text != self.old_selection[1]:
                self.old_selection = [rowValue, text]
                if vrb.mainWindow.widgetLabelImage.layout.count() > 0:
                    self.clearAllImages()

                # header = self.horizontalHeaderItem(cellItem.column()).text()
                # if "_" in header:
                #     class_label = header.split("_")[0]
                # elif self.indice_class_name:
                #     class_label = self.item(rowValue, self.indice_class_name).text()
                # else:
                class_label = "lesion"
                nameImageInput = f"{self.path_images}/{text}.tif"
                imageInput = PyIPSDK.loadTiffImageFile(nameImageInput)

                vrb.mainWindow.widgetLabelImage.addNewImage(text, imageInput)

                all_overlay_path = Path(self.path_images).glob(f'*{text}*_overlay_ipsdk.tif')
                for file in all_overlay_path:
                    label = file.stem.split("_")[-3]
                    image = PyIPSDK.loadTiffImageFile(file.as_posix())
                    vrb.mainWindow.widgetLabelImage.addNewImage(f"{label}_overlay", image)

                for num in range(
                        vrb.mainWindow.widgetLabelImage.layout.count()):  # Boucle pour afficher l'image "Image" et l'image "Result" en overlay
                    try:
                        item = vrb.mainWindow.widgetLabelImage.layout.itemAt(num)
                        if item is not None:
                            label = item.widget()
                            if label.name == text:
                                vrb.mainWindow.changeCurrentXmlElement(label)
                                vrb.mainWindow.widgetImage.groupBoxOverlay.checkBoxOverlay.setChecked(True)
                                vrb.mainWindow.widgetImage.groupBoxOverlay.comboBoxOverlay.setCurrentText(f"{class_label}_overlay")
                    except Exception as e:
                        pass


class TwoListSelection(qt.QWidget):
    def __init__(self, parent=None):
        super(TwoListSelection, self).__init__(parent)
        self.mButtonToAvailable = None
        self.mBtnMoveToSelected = None
        self.mBtnMoveToAvailable = None
        self.mButtonToSelected = None
        self.label = None
        self.mOuput = None
        self.mInput = None
        self.mBtnDown = None
        self.mBtnUp = None
        self.setup_layout()

    def setup_layout(self):
        lay = qt.QGridLayout(self)
        self.mInput = qt.QListWidget()
        self.mOuput = qt.QListWidget()

        self.label = qt.QLabel("Rename order:")
        self.label.setWordWrap(True)
        self.label.setFixedWidth(int(83 * vrb.ratio))
        self.mButtonToSelected = qt.QPushButton(">>")
        self.mBtnMoveToAvailable = qt.QPushButton(">")
        self.mBtnMoveToSelected = qt.QPushButton("<")
        self.mButtonToAvailable = qt.QPushButton("<<")

        vlay = qt.QVBoxLayout()
        vlay.addStretch()
        vlay.addWidget(self.mButtonToSelected)
        vlay.addWidget(self.mBtnMoveToAvailable)
        vlay.addWidget(self.mBtnMoveToSelected)
        vlay.addWidget(self.mButtonToAvailable)
        vlay.addStretch()
        vlay.setContentsMargins(0, 0, 0, 0)

        self.mBtnUp = qt.QPushButton("Up")
        self.mBtnDown = qt.QPushButton("Down")

        vlay2 = qt.QVBoxLayout()
        vlay2.addStretch()
        vlay2.addWidget(self.mBtnUp)
        vlay2.addWidget(self.mBtnDown)
        vlay2.addStretch()
        vlay2.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self.label, 0, 0)
        lay.addWidget(self.mInput, 0, 1)
        lay.addLayout(vlay, 0, 2)
        lay.addWidget(self.mOuput, 0, 3)
        lay.addLayout(vlay2, 0, 4)
        lay.setContentsMargins(0, 0, 0, 0)

        self.update_buttons_status()
        self.connections()

    @pyqtSlot()
    def update_buttons_status(self):
        self.mBtnUp.setDisabled(not bool(self.mOuput.selectedItems()) or self.mOuput.currentRow() == 0)
        self.mBtnDown.setDisabled(
            not bool(self.mOuput.selectedItems()) or self.mOuput.currentRow() == (self.mOuput.count() - 1))
        self.mBtnMoveToAvailable.setDisabled(not bool(self.mInput.selectedItems()) or self.mOuput.currentRow() == 0)
        self.mBtnMoveToSelected.setDisabled(not bool(self.mOuput.selectedItems()))

    def connections(self):
        self.mInput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mOuput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mBtnMoveToAvailable.clicked.connect(self.on_mBtnMoveToAvailable_clicked)
        self.mBtnMoveToSelected.clicked.connect(self.on_mBtnMoveToSelected_clicked)
        self.mButtonToAvailable.clicked.connect(self.on_mButtonToAvailable_clicked)
        self.mButtonToSelected.clicked.connect(self.on_mButtonToSelected_clicked)
        self.mBtnUp.clicked.connect(self.on_mBtnUp_clicked)
        self.mBtnDown.clicked.connect(self.on_mBtnDown_clicked)

    @pyqtSlot()
    def on_mBtnMoveToAvailable_clicked(self):
        self.mOuput.addItem(self.mInput.takeItem(self.mInput.currentRow()))

    @pyqtSlot()
    def on_mBtnMoveToSelected_clicked(self):
        self.mInput.addItem(self.mOuput.takeItem(self.mOuput.currentRow()))

    @pyqtSlot()
    def on_mButtonToAvailable_clicked(self):
        while self.mOuput.count() > 0:
            self.mInput.addItem(self.mOuput.takeItem(0))

    @pyqtSlot()
    def on_mButtonToSelected_clicked(self):
        while self.mInput.count() > 0:
            self.mOuput.addItem(self.mInput.takeItem(0))

    @pyqtSlot()
    def on_mBtnUp_clicked(self):
        row = self.mOuput.currentRow()
        currentItem = self.mOuput.takeItem(row)
        self.mOuput.insertItem(row - 1, currentItem)
        self.mOuput.setCurrentRow(row - 1)

    @pyqtSlot()
    def on_mBtnDown_clicked(self):
        row = self.mOuput.currentRow()
        currentItem = self.mOuput.takeItem(row)
        self.mOuput.insertItem(row + 1, currentItem)
        self.mOuput.setCurrentRow(row + 1)

    def add_left_elements(self, items):
        if items:
            self.mInput.addItems(items)

    def add_right_elements(self, items):
        if items:
            self.mOuput.addItems(items)

    def get_left_elements(self):
        return [self.mInput.item(i).text() for i in range(self.mInput.count())]

    def get_right_elements(self):
        return [self.mOuput.item(i).text() for i in range(self.mOuput.count())]

    def clean_list(self):
        self.mInput.clear()
        self.mOuput.clear()


###############################################################################################
# MAIN
if __name__ == '__main__':

    app = QCoreApplication.instance()
    if app is None:
        app = qt.QApplication([])
    csv_file = "/home/sebastien/Documents/IPSDK/IMAGE/bug_francoise/cut_images/global-merge.csv"
    df = read_csv(csv_file, index_col=None, header=[0], squeeze=True, sep="\t")
    ddict = df.to_dict(orient='list')
    foo = TableWidget(ddict, path_images="/home/sebastien/Documents/IPSDK/IMAGE/bug_francoise/cut_images/")
    foo.showMaximized()
    app.exec_()