import sys, os, traceback

from PyQt5.QtCore import Qt, QCoreApplication, QDir
import PyQt5.QtWidgets as qt
from PyQt5 import QtGui
from pathlib import Path
import pandas as pd

sys.path.insert(0, Path("../").as_posix())

import PyIPSDK
import UsefullVariables as vrb
import UsefullWidgets as wgt
import UsefullFunctions as fct
import DatabaseFunction as Dfct

from pathlib import Path

style = 'QGroupBox:title {left: 20px ;padding-left: 10px;padding-right: 10px; padding-top: -12px; color:rgb(6, 115, 186)} QGroupBox {font: bold; border: 1px solid gray; margin-top: 12 px}'
allow_ext = ["tif", "tiff", "TIF", "TIFF", "Tif", "Tiff", "im6", "IM6", "jpg", "JPG", "PNG", "png", "BMP", "bmp"]


def return_default_folder():
    defaultFolder = Dfct.childText(vrb.userPathElement, "ImportImages")
    if defaultFolder is None or defaultFolder == "" or os.path.exists(defaultFolder) is False:
        defaultFolder = os.path.dirname(vrb.folderExplorer) + "/images"
        if not os.path.exists(defaultFolder):
            defaultFolder = os.path.dirname(vrb.folderExplorer) + "/data/Explorer/images"
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


class FileSelector(qt.QGroupBox):
    """add file selector buttom"""

    def __init__(self, label, title=None, style="minimal", file=False):
        super().__init__()

        self.labelFile = qt.QLabel(label)
        self.labelFile.setMinimumWidth(int(83 * vrb.ratio))
        self.lineEditFile = qt.QLineEdit()
        self.lineEditFile.setFixedHeight(int(25 * vrb.ratio))

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
        dlg.setOption(dlg.DontUseNativeDialog, False)
        dlg.setOption(dlg.HideNameFilterDetails, False)
        dlg.setFilter(dlg.filter() | QDir.Hidden)
        defaultFolder = return_default_folder()
        if self.is_file:
            filename = dlg.getOpenFileName(dlg, caption="Select your file",
                                              directory=defaultFolder,
                                              filter="table file (*.csv *.tsv)")
            filename = filename[0]
        else:
            dlg.setOption(dlg.ShowDirsOnly, True)
            dlg.setFileMode(dlg.Directory)
            filename = dlg.getExistingDirectory(dlg, caption="Select your folder",
                                                   directory=defaultFolder)

        if filename != "" and filename:
            try:
                self.lineEditFile.setText(filename)
                self.lineEditFile.setFocus()
                Dfct.SubElement(vrb.userPathElement, "ImportImages").text = os.path.dirname(filename)
                Dfct.saveXmlElement(vrb.userPathElement, vrb.folderInformation + "/UserPath.mho", forceSave=True)
            except:
                traceback.print_exc(file=sys.stderr)


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


def randomPastelColor(i):
    # colors = [171, 54, 114, 5, 231, 187, 254, 7, 214, 192, 50, 258, 48, 56, 107, 255, 33, 34, 177, 246]
    colors = [10, 30, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 255, 33, 34, 177, 246]
    s = 90
    l = 90
    return QtGui.QColor.fromHsl(colors[i], int(s * 255 / 100), int(l * 255 / 100), 100)


class TableWidget(qt.QTableWidget):
    def __init__(self, ddict=None, path_images=None):
        super(TableWidget, self).__init__()

        self.ddict = ddict
        self.path_images = path_images

        self.horizontalHeader().setVisible(True)
        self.verticalHeader().setVisible(False)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSortingEnabled(True)
        self.loadDictionary()
        # self.setMaximumHeight(self.rowHeight(0) * self.rowCount() + self.horizontalScrollBar().height())
        self.horizontalHeader().setStretchLastSection(True)
        self.setAlternatingRowColors(True)
        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)

        self.itemPressed.connect(self.cellClick)
        self.display()

    def loadDictionary(self):
        try:
            self.setColumnCount(len(self.ddict))
            col = 0
            for name in self.ddict:
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
        except:
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

    def cellClick(self, cellItem):

        rowValue, columnValue = cellItem.row(), cellItem.column()
        text = self.item(rowValue,0).text()  # On peut récupérer les coordonnées et le texte de la cellule sur laquelle on a cliqué

        if vrb.mainWindow:
            try:
                nameImageInput = f"{self.path_images}{text}.tif"
                imageInput = PyIPSDK.loadTiffImageFile(nameImageInput)

                nameImageOverlay = f"{self.path_images}{text}_mask_overlay.tif"
                imageOverlay = PyIPSDK.loadTiffImageFile(nameImageOverlay)

                vrb.mainWindow.widgetLabelImage.addNewImage("Image",imageInput)
                vrb.mainWindow.widgetLabelImage.addNewImage("Result",imageOverlay)

                for num in range(vrb.mainWindow.widgetLabelImage.layout.count()): # Boucle pour afficher l'image "Image" et l'image "Result" en overlay
                    try:
                        item = vrb.mainWindow.widgetLabelImage.layout.itemAt(num)
                        if item is not None:
                            label = item.widget()
                            if label.name == "Image":
                                vrb.mainWindow.changeCurrentXmlElement(label)
                                vrb.mainWindow.widgetImage.groupBoxOverlay.checkBoxOverlay.setChecked(True)
                                vrb.mainWindow.widgetImage.groupBoxOverlay.comboBoxOverlay.setCurrentText("Result")
                    except:
                        pass

            except:
                traceback.print_exc(file=sys.stderr)


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

    csv_file = "/home/sebastien/Documents/IPSDK/IMAGE/bug_francoise/cut_images/global-merge.csv"
    df = pd.read_csv(csv_file, index_col=None, header=[0], squeeze=True, sep="\t")
    print(df)
    ddict = df.to_dict(orient='list')
    print(ddict)
    # exit()
    foo = TableWidget(ddict, path_images="/home/sebastien/Documents/IPSDK/IMAGE/bug_francoise/cut_images/")
    foo.showMaximized()
    app.exec_()
