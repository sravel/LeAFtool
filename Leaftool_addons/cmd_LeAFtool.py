#!/opt/programmes/Explorer_3_2_1_2_linux/Miniconda/bin/python3.8
import sys
import PyIPSDK
import PyIPSDK.IPSDKUI as ui
import PyIPSDK.IPSDKIPLGlobalMeasure as glbmsr
import PyIPSDK.IPSDKIPLAdvancedMorphology as advmorpho
import PyIPSDK.IPSDKIPLBinarization as bin
import PyIPSDK.IPSDKIPLClassification as classif
import PyIPSDK.IPSDKIPLShapeAnalysis as shapeanalysis
import PyIPSDK.IPSDKIPLBasicMorphology as morpho
import PyIPSDK.IPSDKIPLUtility as util
import PyIPSDK.IPSDKIPLMachineLearning as ml
import PyIPSDK.IPSDKIPLLogical as logic
import PyIPSDK.IPSDKIPLArithmetic as arithm
import PyIPSDK.IPSDKIPLIntensityTransform as itrans
import PyIPSDK.IPSDKFunctionsMachineLearning as fctML

import xml.etree.ElementTree as xmlet
import cv2
from pathlib import Path
from re import compile
import numpy as np
import pandas as pd
from collections import OrderedDict
from pprint import pprint as pp
import logging
import logging.config
from sys import path as syspath
import yaml
import colorlog
import click

# sys.tracebacklimit = 1
# auto add Explorer in PYTHONPATH
explorer_path = Path(PyIPSDK.getPyIPSDKDir()).parent.parent.joinpath("Explorer", "Interface")
syspath.insert(0, explorer_path.as_posix())

import DatabaseFunction as Dfct
import UsefullFunctions as fct
import UsefullVariables as vrb
import UsefullWidgets as wgt

# environment settings:
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 30000)
pd.set_option('expand_frame_repr', True)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.set_option('display.precision', 9)

# COLORS
# import seaborn as sns
# colors = sns.color_palette("tab10")
# colors65535 = [(int(r*65535), int(g*65535), int(b*65535)) for r, g, b in colors]
# color_label = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "gold", "turquoise"]
# color_label_to_RGB_Uint16 = dict([(i, elm) for i, elm in zip(color_label, colors65535)])
# print(color_label_to_RGB_Uint16)

# color_label_to_RGB_Uint16 = {'blue': (7967, 30583, 46260),
#                              'orange': (65535, 32639, 3598),
#                              'green': (11308, 41120, 11308),
#                              'red': (54998, 10023, 10280),
#                              'purple': (38036, 26471, 48573),
#                              'brown': (35980, 22102, 19275),
#                              'pink': (58339, 30583, 49858),
#                              'gray': (32639, 32639, 32639),
#                              'gold': (48316, 48573, 8738),
#                              'turquoise': (5911, 48830, 53199),
#                              'black': (0, 0, 0),
#                              'white': (65535, 65535, 65535)
#                              }


# pp(color_label_to_RGB_Uint16)

def read_image_UINT16(path_image):
    extension = Path(path_image).suffix[1:]

    if extension in ["tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]:
        imageIP = PyIPSDK.loadTiffImageFile(path_image)
        range_UInt16 = PyIPSDK.createRange(0, 65535)
        imageIP = util.convertImg(imageIP, PyIPSDK.eImageBufferType.eIBT_UInt16)
        imageIP = itrans.normalizeImg(imageIP, range_UInt16)
    elif extension in ["im6", "IM6"]:
        imageIP = PyIPSDK.loadIm6ImageFile(path_image)
    elif extension in ["jpg", "JPG", "PNG", "png", "BMP", "bmp"]:
        image_load = cv2.imread(path_image, -1)
        image = (image_load * 256).astype('uint16')
        b, g, r = 0, 0, 0
        if len(image.shape) >= 3:
            if image.shape[2] >= 3:
                if image.shape[2] == 3:
                    b, g, r = cv2.split(image)
                if image.shape[2] == 4:
                    b, g, r, a = cv2.split(image)
                allChannels = [PyIPSDK.fromArray(r), PyIPSDK.fromArray(g), PyIPSDK.fromArray(b)]
                imageIP = PyIPSDK.createImageRgb(PyIPSDK.eImageBufferType.eIBT_UInt16, image.shape[1], image.shape[0])
                util.eraseImg(imageIP, 0)
                for c in range(3):
                    plan = PyIPSDK.extractPlan(0, c, 0, imageIP)
                    util.copyImg(allChannels[c], plan)
            else:
                imageIP = PyIPSDK.fromArray(image)
        else:
            imageIP = PyIPSDK.fromArray(image)
    imageIP = util.convertImg(imageIP, PyIPSDK.eImageBufferType.eIBT_UInt16)
    return imageIP


def outset_to_df(infoSet):
    dictResult = {}
    allMeasures = []
    for msr in infoSet.getMeasureInfoSet().getMeasureInfoColl():
        key = msr.key()
        userName, isFound = fct.findUserName(key)  # Passage du nom de la mesure IPSDK à un nom plus simple
        if "->" not in userName:
            allMeasures.append((key, userName))
    for callName, userName in allMeasures:
        try:
            values = list(infoSet.getMeasure(callName).getMeasureResult().getColl(0))[1:]
            if infoSet.getMeasure(callName).getMsrUnitStr() != "" and infoSet.getMeasure(
                    callName).getMsrUnitStr() is not None:
                unit = f" ({infoSet.getMeasure(callName).getMsrUnitStr()})".replace('^2', '²')
            else:
                unit = ""
            dictResult[userName + unit] = values
        except:
            print("Error with : " + userName)
    df = pd.DataFrame.from_dict(dictResult, orient='index')
    df_t = df.T
    return df_t


class MetaInfo:
    """Object to read csv table to build name, keep meta info"""

    def __init__(self, csv_path, rename_oder=None):
        """
        Args:
            csv_path: (:obj:`str`): Path to csv with meta-info
        """
        self.path_csv = csv_path
        self.rename_order = rename_oder
        self.header = None
        self.dataframe = None
        self.dataframe_with_cut_name = None
        self.__dict_names_pos = None
        self.__list_filenames = None
        self.rename_to_df = {}
        self.exit_status = False

        self.__load_metadata_csv_to_dict()
        self.__build_meta_to_rename()
        self.__build_dataframe_with_cut_name()

    def __load_metadata_csv_to_dict(self):
        """ Load csv file into dict use for rename scan
            The dataframe must contain header for columns
            The 2 first columns are use as dict key (tuple key with scan name and cut position)

        Args:
            csv_file (:obj:`str`): Path to csv file
        """
        if not Path(self.path_csv).exists():
            raise FileNotFoundError(f"CSV file '{self.path_csv}' doesn't exist !!! exit")

        with open(self.path_csv, "r") as csv:
            header_txt = csv.readline().rstrip()
        sep_dict = {",": header_txt.count(","),
                    ";": header_txt.count(";"),
                    ".": header_txt.count("."),
                    "\t": header_txt.count("\t")
                    }
        csv_separator = max(sep_dict, key=sep_dict.get)
        self.header = header_txt.split(csv_separator)
        # check rename_order on header
        if not self.rename_order:
            self.rename_order = self.header[:3]
        for elm in self.rename_order:
            if elm not in self.header:
                raise ValueError(f"Value '{elm}' in not on the header file {self.path_csv}, found: {self.header}")
        try:
            df = pd.read_csv(self.path_csv, index_col=[0, 1], header=0, sep=csv_separator)
            self.dataframe = pd.read_csv(self.path_csv, header=0, sep=csv_separator).reset_index(drop=True)
            self.__dict_names_pos = df.to_dict('index')
            self.__list_filenames = [str(key1) for (key1, key2) in self.__dict_names_pos.keys()]
        except ValueError:
            df = pd.read_csv(self.path_csv, index_col=False, header=0, sep=csv_separator)
            duplicate = df.iloc[:, 0:2].duplicated()
            raise ValueError(f"Found {duplicate.sum()} duplicate lines:\n\n{df[duplicate].iloc[:, 0:2]}\n\n on file {self.path_csv}")

    def check_corresponding(self, files_list):
        """test if all scan file have name on csv file"""
        not_found_list = []
        for img_file in files_list:
            basename = img_file.stem
            if basename not in self.__list_filenames:
                not_found_list.append(img_file.name)
        if not_found_list:
            self.exit_status = False
            txt_list = '\n - '.join([""] + not_found_list)
            raise NameError(f"Not found corresponding MetaInfo for scan:{txt_list}")
        self.exit_status = True

    def __build_meta_to_rename(self):
        if not self.rename_to_df:
            for scan_name, pos in self.__dict_names_pos:
                df = self.dataframe.query(f'{self.header[0]}=="{scan_name}" & {self.header[1]}=={pos}').copy(deep=True)
                rename = "_".join([str(df[elm].values[0]).replace("/", "-") for elm in self.rename_order])
                df["cut_name"] = rename
                self.__dict_names_pos[(scan_name, pos)].update({"cut_name": rename})
                self.rename_to_df[rename] = df.reset_index(drop=True)

    def __build_dataframe_with_cut_name(self):
        df_list = [v for k, v in self.rename_to_df.items()]
        self.dataframe_with_cut_name = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)

    def check_correspondingML(self, files_list):
        """test if all scan file have name on csv file"""
        not_found_list = []
        for img_file in files_list:
            basename = img_file.stem
            if not self.rename_to_meta(basename):
                not_found_list.append(img_file.name)
        if not_found_list:
            self.exit_status = False
            txt_list = '\n - '.join([""] + not_found_list)
            raise NameError(f"Not found corresponding MetaInfo for scan:{txt_list}")
        self.exit_status = True

    def rename_to_meta(self, scan_name):
        try:
            return self.rename_to_df[scan_name].iloc[0][self.header[0]]
        except KeyError:
            return None

    def meta_to_cut_rename(self, scan_name, pos):
        if (scan_name, pos) in self.__dict_names_pos:
            return self.__dict_names_pos[(scan_name, pos)]["cut_name"]
        else:
            return None

    # def __repr__(self):
    #     return f"{self.__class__}({pp(self.__dict__)})"
        # return f"{self.dataframe_with_cut_name}"


class ParseDataframe(MetaInfo):
    """
    Object to merge raw dataframe with results
    Build aggregated leaves and by leaves results
    """
    def __init__(self, csv_path=None, rename_oder=None, basedir=None, calibration_unit=None):
        super().__init__(csv_path=csv_path, rename_oder=rename_oder)
        self.basedir = Path(basedir)
        self.big_frame = None
        self.calibration = calibration_unit+"²"
        self.df_leaves = None
        self.ml_classes = None
        self.sep = ","
        self.logger = logging.getLogger('ParseDataframe')

    def generate(self, sep=","):
        self.sep = sep
        filenames = self.basedir.glob("*_split-info.csv")
        dfs = [pd.read_csv(filename, sep="\t") for filename in filenames]
        self.big_frame = pd.concat(dfs, ignore_index=True)
        self.big_frame.astype({'Number of pixels': 'float'})
        self.big_frame.to_csv(self.basedir.joinpath("global_merge_split_info.csv"), index=False, sep=",", float_format="%.6f")
        self.ml_classes = [elm for elm in self.big_frame['Class'].unique().tolist() if elm != "leaf"]
        self.get_all_merge(group=[self.header[0], self.header[1], "cut_name", "leaf_ID"], aggregated_leaves=False)
        self.get_all_merge(group=[self.header[0], self.header[1], "cut_name"], aggregated_leaves=True)

    def get_all_merge(self, group=None, aggregated_leaves=False):
        self.get_leaves_dataframe(group=group)
        if aggregated_leaves:
            on_list = [self.header[0], self.header[1], "cut_name", "number_of_leaves", f"leaves_area_{self.calibration}", "leaves_number_pixels"]
            csv_path_file = self.basedir.joinpath(f"global-merge-ALL_aggregated_leaves.csv").as_posix()
            self.logger.info(f"Merge all files aggregated leaves to {csv_path_file}")
        else:
            on_list = [self.header[0], self.header[1], "cut_name", "number_of_leaves", "leaf_ID", f"leaves_area_{self.calibration}", "leaves_number_pixels"]
            csv_path_file = self.basedir.joinpath(f"global-merge-ALL_by_leaves.csv").as_posix()
            self.logger.info(f"Merge all files by leaves to {csv_path_file}")

        all_merge = []
        for class_ml in self.ml_classes:
            all_merge.append(self.df_to_stats_class(ml_class=class_ml, group=group))
        all_merge_df = all_merge[0]
        for df_ in all_merge[1:]:
            all_merge_df = all_merge_df.merge(df_, on=on_list)
        all_merge_df = pd.merge(self.dataframe_with_cut_name, all_merge_df, on=[self.header[0], self.header[1], "cut_name"], how='inner')
        all_merge_df.sort_values([self.header[0], self.header[1]], ascending=(True, True)).fillna(0, inplace=True)
        all_merge_df.to_csv(csv_path_file, index=False, sep=",", float_format='%.6f', na_rep=0)

    @staticmethod
    def flatten_columns(df, sep='.', ml_class=None):
        def _remove_empty(column_name):
            return tuple(element for element in column_name if element)

        def _join(column_name):
            if len(column_name) == 2 and "count" == column_name[1]:
                return f"{ml_class}_nb"
            elif ml_class and len(column_name) == 2 and "Area" in column_name[0]:
                return f"{ml_class}_area_{column_name[1]}"
            elif ml_class and len(column_name) == 2 and "pixels" in column_name[0]:
                return f"{ml_class}_pixels_{column_name[1]}"
            else:
                return sep.join(column_name)

        new_columns = [_join(_remove_empty(column)) for column in df.columns.values]
        return new_columns

    def get_leaves_dataframe(self, group=None):
        self.df_leaves = self.big_frame.query("Class=='leaf'").groupby(group).agg(
            {f"Area 2D ({self.calibration})": [("number_of_leaves", "count"),
                                               (f"leaves_area_{self.calibration}", "sum")],
             "Number of pixels": [("leaves_number_pixels", "sum")],
             }).droplevel(0, axis=1).reset_index()

        self.df_leaves.columns = self.flatten_columns(self.df_leaves, sep="")

    def df_to_stats_class(self, ml_class='lesion', group=None):
        df_tmp = self.big_frame.query(f"Class=='{ml_class}'").groupby(group).agg({
            "leaf_ID": [("count", "count")],
            f"Area 2D ({self.calibration})": [(f"sum_{self.calibration}", "sum"),
                                              (f"mean_{self.calibration}", "mean"),
                                              (f"median_{self.calibration}", "median"),
                                              (f"std_{self.calibration}", "std"),
                                              (f"min_{self.calibration}", "min"),
                                              (f"max_{self.calibration}", "max")],
            "Number of pixels": ["sum", "mean"],
        }).reset_index()
        df_tmp.columns = self.flatten_columns(df_tmp, ml_class=ml_class)
        df_merge = self.df_leaves.merge(df_tmp)

        df_merge.loc[df_merge[f'{ml_class}_nb'] == df_merge['number_of_leaves'], f'{ml_class}_nb'] = 0
        df_merge[f"{ml_class}_percent"] = (df_merge[f"{ml_class}_area_sum_{self.calibration}"] / df_merge[
            f"leaves_area_{self.calibration}"]) * 100
        df_merge.insert(7, f"{ml_class}_percent", df_merge.pop(f"{ml_class}_percent"))
        return df_merge

    def __repr__(self):
        return f"""{self.__class__}
        {pp(self.basedir)}
        {pp(self.calibration)}
        {pp(self.df_leaves)}
        {pp(self.ml_classes)}
"""
        # return f"{self.__class__}({pp(self.__dict__)})"


class DrawAndCutImages:
    """
    Object to cut and cut scan images on folder.
    There are able to draw lines to show the cut result before real cut.
    When cut use dataframe to rename scan images
    """

    def __init__(self, scan_folder, rename=None, extension='jpg', x_pieces=2, y_pieces=2, top=0, left=0, bottom=0,
                 right=0,
                 noise_remove=False, numbering="right", plant_model=None, force_rerun=False):
        """Created the objet

        Args:
            scan_folder (:obj:`str`): Path to scan images
            rename (:obj:`list`): List of columns header used to rename cut image (default first 2 columns)
            extension (:obj:`str`): The scan images extension, must be the same for all scan. allow extension are:[
            "jpg", "JPG", "PNG", "png", "BMP", "bmp", "tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]
            x_pieces (:obj:`int`): Number of vertical cut
            y_pieces (:obj:`int`): Number of horizontal cut
            top (:obj:`int`): The top marge to remove before cut
            left (:obj:`int`): The left marge to remove before cut
            bottom (:obj:`int`): The bottom marge to remove before cut
            right (:obj:`int`): The right marge to remove before cut
            noise_remove (:obj:`boolean`): use IPSDK unionLinearOpening2dImg function to remove small objet noise (
            default value 3)
            numbering (:obj:`str`): if right (default), the output order cut is left to right, if bottom,
            the output order is top to bottom then left
            plant_model (:obj:`str`): The plant model name (rice or banana)
            force_rerun (:obj:`boolean`): even files existed, rerun draw and/or cut
        """
        self.logger = logging.getLogger('DrawAndCutImages')
        # input
        self.__scan_folder = Path(scan_folder)
        self.rename_order = rename
        self.extension = extension

        # params
        self.params = {"x_pieces": x_pieces,
                       "y_pieces": y_pieces,
                       "top": top,
                       "left": left,
                       "bottom": bottom,
                       "right": right
                       }
        self.noise_rm = noise_remove
        self.numbering = numbering.lower()
        self.plant_model = plant_model
        self.force_rerun = force_rerun
        # others
        self.meta_info = None

        self.__allow_ext = ["jpg", "JPG", "PNG", "png", "BMP", "bmp", "tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]
        self.exit_status = False  # to now exit status if fail some step

        # call functions
        self.__check_input()
        message = ", ".join([f"{key}:{value}" for key, value in self.params.items()])
        self.logger.info(f"CutImage parameters: {message}")

    def __check_input(self):
        """ check inputs values"""
        try:
            if not self.__scan_folder.exists():
                raise NotADirectoryError(f"folder '{self.__scan_folder}' doesn't exist !!! exit")
            for key, value in self.params.items():
                if not 0 <= value < 100000:
                    raise ValueError(f"'{key}': '{value}' is not a valid value. 0 < {key} < 100000!!! exit")
                if not isinstance(value, int):
                    self.logger.warning(f"'{key}': '{value}' is not a valid int. {int(value)} will be use instead")
                    self.params[key] = int(value)
            if "." in self.extension:
                self.extension = self.extension[1:]
            if self.extension not in self.__allow_ext:
                raise NotImplementedError(
                    f"'extension': '{self.extension}' is not allow use only {self.__allow_ext} value!!! exit")
            if self.numbering not in ["right", "bottom"]:
                raise ValueError(
                    f"'numbering' value: '{self.numbering}' is not a valid value. Only use 'right' or 'bottom' !!! "
                    f"exit")
        except Exception as e:
            self.logger.error(e)

    def loop_cut(self, cutdir_name, csv_file):
        """Run cut on images files

        Args:
            cutdir_name (:obj:`str`): the output directory to store cut images
            csv_file (:obj:`str`): The file use to rename images
        """
        def save_image(img, pos):
            basename = self.meta_info.meta_to_cut_rename(scan_name=img, pos=pos)
            file_name = cut_dir_path.joinpath(f"{basename}.tif")
            if basename and (not Path(file_name).exists() or self.force_rerun):
                imageIP = util.getROI2dImg(im_borderless, *box)
                if self.noise_rm:
                    # image noise removal
                    imageIP = morpho.unionLinearOpening2dImg(imageIP, 3.0, PyIPSDK.eBEP_Disable)
                PyIPSDK.saveTiffImageFile(file_name.as_posix(), imageIP)
                self.logger.info(f" - {file_name.name}")
            elif not basename:
                self.logger.warning(f" - Not metainfo for  {img} at position {pos}")
            else:
                self.logger.warning(f" - {file_name.name} already cut")

        self.logger.info("~~~~~~~~~ START STEP CUT ~~~~~~~~~")
        self.logger.info(f"LOAD CSV FILE: {csv_file}")
        files = self.__scan_folder.glob(f"*.{self.extension}")
        self.meta_info = MetaInfo(csv_path=csv_file, rename_oder=self.rename_order)
        self.meta_info.check_corresponding(files_list=files)
        self.exit_status = self.meta_info.exit_status
        cut_dir_path = self.__scan_folder.joinpath(cutdir_name)
        cut_dir_path.mkdir(exist_ok=True)
        self.logger.info(f"CUT dir path is: {cut_dir_path}")
        nb_files = len(list(self.__scan_folder.glob(f"*.{self.extension}")))
        if nb_files == 0:
            raise FileNotFoundError(
                f"Not found file extension with'.{self.extension}' on folder: {self.__scan_folder.as_posix()}")
        if self.meta_info.exit_status:
            for scan_num, img_file in enumerate(self.__scan_folder.glob(f"*.{self.extension}"), 1):
                self.logger.info(f"CROP IMAGE FILE {scan_num}/{nb_files}:\t{img_file.name} to ")
                # image = read_image_UINT16(img_file.as_posix())
                image = fct.urlToPythonImage(img_file.as_posix())
                x_size = image.getGeometry().getSizeX()
                y_size = image.getGeometry().getSizeY()
                w = x_size - (self.params["left"] + self.params["right"])
                h = y_size - (self.params["top"] + self.params["bottom"])

                im_borderless = util.getROI2dImg(image, self.params["left"], self.params["top"], w, h)
                img_width = im_borderless.getGeometry().getSizeX()
                img_height = im_borderless.getGeometry().getSizeY()

                height = img_height // self.params["y_pieces"]
                width = img_width // self.params["x_pieces"]
                position = 1
                # print(f"img_width:{img_width}\timg_height:{img_height}\twidth:{width}\theight:{height}")
                if self.numbering == "right":
                    for i in range(0, self.params["y_pieces"]):
                        for j in range(0, self.params["x_pieces"]):
                            box = (j * width, i * height, width, height)
                            # print(f"box: {box}")
                            save_image(img_file.stem, position)
                            position += 1
                elif self.numbering == "bottom":
                    for i in range(0, self.params["x_pieces"]):
                        for j in range(0, self.params["y_pieces"]):
                            box = (i * width, j * height, width, height)
                            # print(f"box: {box}")
                            save_image(img_file.stem, position)
                            position += 1
        self.logger.info("~~~~~~~~~ END STEP CUT ~~~~~~~~~")

    def loop_draw(self, draw_dir_name):
        """draw lines before cut

        Args:
            draw_dir_name (:obj:`str`): the output directory to store draw images
        """
        self.logger.info("~~~~~~~~~ START STEP DRAW ~~~~~~~~~")
        draw_dir_path = self.__scan_folder.joinpath(draw_dir_name)
        draw_dir_path.mkdir(exist_ok=True)
        self.logger.info(f"OUTPUT DRAW directory is: {draw_dir_path}")
        nb_files = len(list(self.__scan_folder.glob(f"*.{self.extension}")))
        for scan_num, img_file in enumerate(self.__scan_folder.glob(f"*.{self.extension}"),1):
            outname = f"{draw_dir_path}/{img_file.stem}_draw.tif"
            self.logger.info(f"DRAW IMAGE FILE {scan_num}/{nb_files}: {img_file.name} to {Path(outname).name}")

            if not Path(outname).exists() or self.force_rerun:
                # print(f"\n####### DRAW IMAGE FILE:\n{img_file.name}")
                im_draw = cv2.imread(img_file.as_posix())
                p1 = (self.params["left"], self.params["top"])
                p2 = (im_draw.shape[1] - self.params["right"], im_draw.shape[0] - self.params["bottom"])
                cv2.rectangle(im_draw, p1, p2, (255, 0, 0), 4)

                im_borderless = im_draw[self.params["top"]:im_draw.shape[0] - self.params["bottom"],
                                self.params["left"]:im_draw.shape[1] - self.params["right"]]
                img_width, img_height = im_borderless.shape[1], im_borderless.shape[0]
                height = img_height // self.params["y_pieces"]
                width = img_width // self.params["x_pieces"]
                for i in range(1, self.params["y_pieces"]):
                    for j in range(1, self.params["x_pieces"]):
                        cv2.line(im_draw, pt1=((width * j + self.params["left"]), self.params["top"]),
                                 pt2=((width * j + self.params["left"]), im_draw.shape[0] - self.params["bottom"]),
                                 color=(0, 255, 0),
                                 thickness=4)
                        cv2.line(im_draw, pt1=(self.params["left"], (height * i + self.params["top"])),
                                 pt2=(im_draw.shape[1] - self.params["right"], (height * i + self.params["top"])),
                                 color=(0, 255, 0),
                                 thickness=4)
                cv2.imwrite(outname, im_draw)
                del im_draw
            else:
                self.logger.warning(f" - {Path(outname).name} Already draw")
        self.logger.info("~~~~~~~~~ END STEP DRAW ~~~~~~~~~")


class AnalysisImages:
    """
    Object to perform ML analysis on images.
    """

    def __init__(self, scan_folder, model_name, csv_file, rename, calibration_name=None, small_object=100, border=0,
                 alpha=0.5,
                 noise_remove=False, split_ML=False, force_rerun=True, draw_ML_image=False, plant_model=None,
                 model_name_classification=None, color_lesion_individual=None):
        """
        Args:
            scan_folder (:obj:`str`): Path to scan images
            model_name (:obj:`int`): The IPSDK PixelClassification model name build with Explorer
            model_name_classification (:obj:`int`): The IPSDK Classification model name build with Explorer
            csv_file (:obj:`str`): The file use to rename images
            rename (:obj:`list`): List of columns header used to rename cut image (default first 2 columns)
            calibration_name (:obj:`str`): Name of Explorer calibration, no calibration if empty
            small_object (:obj:`int`): The minimum area of class, to remove small noise detect object Default: 100
            border (:obj:`int`): The diameter of the brush (in pixels) used to erode the leaf Default: 0
            alpha (:obj:`float`): The degree of transparency of overlay color label. Must float 0 <= alpha <= 1
            Default: 0.5
            noise_remove (:obj:`boolean`): Use IPSDK unionLinearOpening2dImg function with param 3 pixels Default: False
            split_ML (:obj:`boolean`): Use machine learning to split leaves instead of: False
            force_rerun (:obj:`boolean`): If True, rerun all images, else only not run Default: True
            draw_ML_image (:obj:`boolean`): If True, add rectangle overlay corresponding to image use for apply ML
            Default: False
            plant_model (:obj:`boolean`): The plant model (rice or banana)
            color_lesion_individual (:obj:`boolean`): If true make random color for separated lesion else use model color for all Default: True
        """
        self.logger = logging.getLogger('AnalysisImages')
        self.plant_model = plant_model
        self.split_ML = split_ML
        self.meta_file = csv_file
        self.rename_order = rename
        self.meta_info = MetaInfo(csv_path=csv_file, rename_oder=rename)

        # Load machine learning model
        self.model_name = model_name
        self.model_path = None
        self.calibration_obj = Calibration(calibration_name=calibration_name)
        self.small_object = small_object
        self.border_leaf = border
        self.alpha = alpha
        self.noise_remove = noise_remove
        self.force_rerun = force_rerun
        self.basedir = Path(scan_folder)
        self.parseDataframes = ParseDataframe(csv_path=csv_file, rename_oder=rename, basedir=self.basedir, calibration_unit=self.calibration_obj.dico_info['unit'])

        self.full_leaves_ipsdk_img = None
        self.full_leaves_label_img = None
        self.full_files = []
        self.mask_overlay_files = []
        self.files_to_run = []
        self.table_leaves = []
        self.draw_ML_image = draw_ML_image
        self.model_name_classification = model_name_classification
        self.color_lesion_individual = color_lesion_individual
        self.__check_inputs()
        # If model exist load for all images
        self.model_load = PyIPSDK.readRandomForestModel(self.model_path)

        # get class on machine learning model
        self.model_to_label_dict = self.__model_to_label_dict()
        self.model_classification_to_label_dict = self.__model_classification_to_label_dict()
        self.all_ml_labels = set(
            list(self.model_to_label_dict.keys()) + list(self.model_classification_to_label_dict.keys()))
        self.__build_already_run_list()

    @staticmethod
    def __validate_number(key, value, type_value, min_value=None, max_value=None):
        if not isinstance(value, type_value):
            raise TypeError(f"'{key}': '{value}' is not a valid {type_value}.")
        if not (min_value is None) and not (max_value is None):
            if min_value < max_value:
                if not min_value <= value <= max_value:
                    raise ValueError(
                        f"'{key}': '{value}' is not a valid value. {min_value} <= {key} <= {max_value}!!! exit")
            elif max_value < min_value:
                raise ValueError(f"'{key}':  max_value:{max_value} if greater than min_value:{min_value} !!! exit")

    def __check_inputs(self):
        """to check inputs values"""
        if Path(vrb.folderPixelClassification).joinpath(f"{self.model_name}/ModelIPSDK.bin").as_posix():
            self.model_path = Path(vrb.folderPixelClassification).joinpath(
                f"{self.model_name}/ModelIPSDK.bin").as_posix()
        elif Path(vrb.folderPixelClassification).joinpath(f"{self.model_name}/ModelIPSDK.xml").as_posix():
            self.model_path = Path(vrb.folderPixelClassification).joinpath(
                f"{self.model_name}/ModelIPSDK.xml").as_posix()
        else:
            raise FileNotFoundError(
                f"The model '{self.model_name}' doesn't exist on IPSDK, please check name with explorer!!!!")
        self.__validate_number(key="small_object", value=self.small_object, type_value=int, min_value=0,
                               max_value=100000000)
        self.__validate_number(key="border_leaf", value=self.border_leaf, type_value=int, min_value=0, max_value=1000)
        self.__validate_number(key="alpha", value=self.alpha, type_value=float, min_value=0, max_value=1)

    def __model_classification_to_label_dict(self):
        """retrieve the name of the machine learning model labels"""
        dict_label = {}
        mho_path = Path(vrb.folderShapeClassification).joinpath(f"{self.model_name_classification}/Settings.mho")
        if mho_path.exists():
            file = xmlet.parse(mho_path.as_posix())
            xmlElement = file.getroot()
            label_classes_element = Dfct.SubElement(xmlElement, "LabelClasses")
            nb_label = int(Dfct.SubElement(label_classes_element, "NumberLabels").text)
            for numLabel in range(nb_label):
                label_element = Dfct.SubElement(label_classes_element, "Label_" + str(numLabel))
                color_element = Dfct.SubElement(label_element, "Color").text
                name_element = Dfct.SubElement(label_element, "Name").text
                try:
                    name_element = Dfct.convertTextFromAscii(name_element).lower()
                except:
                    pass
                dict_label[name_element] = {"color": [int(elm) for elm in color_element.split(',')],
                                            "value": int(numLabel)}
        return dict_label

    def __model_to_label_dict(self):
        """retrieve the name of the machine learning model labels"""
        dict_label = {}
        mho_path = Path(vrb.folderPixelClassification).joinpath(f"{self.model_name}/Settings.mho").as_posix()
        file = xmlet.parse(mho_path)
        xmlElement = file.getroot()
        label_classes_element = Dfct.SubElement(xmlElement, "LabelClasses")
        nb_label = int(Dfct.SubElement(label_classes_element, "NumberLabels").text)
        for numLabel in range(nb_label):
            label_element = Dfct.SubElement(label_classes_element, "Label_" + str(numLabel))
            color_element = Dfct.SubElement(label_element, "Color").text
            name_element = Dfct.SubElement(label_element, "Name").text
            try:
                name_element = Dfct.convertTextFromAscii(name_element).lower()
            except:
                pass
            value_element = Dfct.SubElement(label_element, "Value").text
            dict_label[name_element] = {"color": [int(elm) for elm in color_element.split(',')],
                                        "value": int(value_element)}
        # print(dict_label)
        found_lesion = False
        found_leaf = False
        for key in dict_label.keys():
            if "leaf" in key:
                found_leaf = True
            if "lesion" in key:
                found_lesion = True
        if not found_lesion or not found_leaf:
            raise ValueError(
                f"ML MODEL CHECKING FAIL : The model '{self.model_name}' must have label 'leaf' and 'lesion', "
                f"found [{','.join(dict_label.keys())}] (not case sensitive)")
        return dict_label

    def __build_already_run_list(self):
        """analyze the inputs and outputs to build the list of images already analyzed"""

        def glob_re(pattern, strings):
            return filter(compile(pattern).match, strings)

        # glob scan file to analysis
        # full_files_filter = glob_re(r'^(.(?!(_overlay)))*.tif$', os.listdir(self.basedir.as_posix()))
        full_files_filter = glob_re(r'^(.(?!(_overlay)))*.tif$', [path.as_posix() for path in self.basedir.glob("*.tif")])
        self.full_files = sorted([self.basedir.joinpath(path) for path in full_files_filter])
        full_files_set = set(sorted(f"{path.stem}" for path in self.full_files))
        # pp(f"full_files_filter: {list(full_files_filter)}")
        # pp(f"full_files: {self.full_files}")
        # if force_rerun load already file run
        # mask_overlay_files_filter = glob_re(r'.*_mask_overlay\.tif$', os.listdir(self.basedir.as_posix()))
        mask_overlay_files_filter = glob_re(r'.*_mask_overlay\.tif$', [path.as_posix() for path in self.basedir.glob("*.tif")])
        self.mask_overlay_files = sorted([self.basedir.joinpath(path) for path in mask_overlay_files_filter])
        mask_overlay_files_filter_set = set(
            sorted(f"{path.stem.replace('_mask_overlay', '')}" for path in self.mask_overlay_files))

        basename_files_to_run = list(full_files_set - mask_overlay_files_filter_set)
        if self.force_rerun:
            self.files_to_run = self.full_files
        else:
            self.files_to_run = sorted([file for file in self.full_files if f"{file.stem}" in basename_files_to_run])
        # pp(self.meta_info)
        self.meta_info.check_correspondingML(files_list=self.files_to_run)
        # pp(f"files_to_run: {self.files_to_run}")

    def run_ML(self):
        """loop to apply ML on all images"""
        self.logger.info("~~~~~~~~~ START STEP MACHINE LEARNING ~~~~~~~~~")
        if self.force_rerun:
            self.logger.info(f"Force rerun analysis for all scans")
        if not self.files_to_run and self.full_files:
            self.logger.info(f"All files already run")
        elif not self.files_to_run and not self.full_files:
            raise FileNotFoundError(f"Not found file extension '.tif' on folder: {self.basedir.as_posix()}")
        nb_scan = len(self.full_files)
        nb_scan_to_run = len(self.files_to_run)
        nb_scan_runned = nb_scan-nb_scan_to_run
        self.logger.info(f"There are {nb_scan_runned}/{nb_scan} scan file already analysis")
        for indice, img_file_path in enumerate(self.files_to_run, 1):
            self.logger.info(f"Analyse scan file {indice}/{nb_scan_to_run}: {img_file_path.name}")
            self.analyse_leaves(image_path=img_file_path.as_posix())
        self.logger.info("~~~~~~~~~ END STEP MACHINE LEARNING ~~~~~~~~~")
        self.logger.info("~~~~~~~~~ START MERGE CSV ~~~~~~~~~")
        self.parseDataframes.generate(sep=",")
        self.logger.info("~~~~~~~~~ END MERGE CSV ~~~~~~~~~")

    def get_dataframe_size_object(self, label_mask):
        """return dataframe with the size of all object"""
        calibration = PyIPSDK.createGeometricCalibration2d(1, 1, 'px')
        inMeasureInfoSet2d = PyIPSDK.createMeasureInfoSet2d(calibration)
        PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "NbPixels2dMsr")
        outset_size = shapeanalysis.labelAnalysis2d(label_mask, label_mask, inMeasureInfoSet2d)
        df = outset_to_df(outset_size)
        print(df.sort_values(by="Number of pixels"))
        return df

    def analyse_leaves(self, image_path):
        # extract path/name from image path
        path_img = Path(image_path)
        basename = path_img.stem

        # load full image (ie with all leaves)
        self.full_leaves_ipsdk_img = PyIPSDK.loadTiffImageFile(image_path)

        x_size = self.full_leaves_ipsdk_img.getGeometry().getSizeX()
        y_size = self.full_leaves_ipsdk_img.getGeometry().getSizeY()
        if self.noise_remove:
            # image noise removal
            self.full_leaves_ipsdk_img = morpho.unionLinearOpening2dImg(self.full_leaves_ipsdk_img, 3.0,
                                                                        PyIPSDK.eBEP_Disable)

        # call function to get leaf position on full images
        self.table_leaves = self.__split_leaves(image_path, self.full_leaves_ipsdk_img)
        # if leaves on image
        if self.table_leaves:
            # created dict to add all pandas dataframe for each leaf
            dict_frames_separated_leaves = {}

            # loop for cut image and apply machine learning
            list_leaves_overlay_IPSDK = []
            for leaf in self.table_leaves:
                self.logger.info(f" - Read and extract lesion on leaf {leaf.leaf_id}/{len(self.table_leaves)}")
                leaf.analysis(model_load=self.model_load,
                              model_to_label_dict=self.model_to_label_dict,
                              model_classification_to_label_dict=self.model_classification_to_label_dict,
                              small_object=self.small_object,
                              calibration_obj=self.calibration_obj,
                              model_name_classification=self.model_name_classification
                              )
                dict_frames_separated_leaves.update(leaf.dico_frames_separated)
                list_leaves_overlay_IPSDK.append(leaf)
            # build IPSDK overlay
            # loop for all label to extract IPSDK label image:
            dico_label_overlay_IPSDK = {}
            for leaf in list_leaves_overlay_IPSDK:
                # self.logger.debug(f"LEAF ID! {leaf.leaf_id}")
                for label, img in leaf.image_ipsdk_blend_dict_class.items():
                    if label == "proba":
                        geometryRgb2_label = PyIPSDK.geometryRgb2d(PyIPSDK.eImageBufferType.eIBT_UInt16, x_size, y_size)
                    else:
                        geometryRgb2_label = PyIPSDK.geometry2d(PyIPSDK.eImageBufferType.eIBT_Label16, x_size, y_size)
                    if label not in dico_label_overlay_IPSDK:
                        dico_label_overlay_IPSDK[label] = PyIPSDK.createImage(geometryRgb2_label)
                        util.eraseImg(dico_label_overlay_IPSDK[label], 0)

                    temp_img = PyIPSDK.createImage(geometryRgb2_label)
                    util.eraseImg(temp_img, 0)
                    util.putROI2dImg(temp_img, img, leaf.x_position, leaf.y_position, temp_img)
                    binary_label_leaves = bin.thresholdImg(self.full_leaves_label_img, leaf.leaf_id, leaf.leaf_id)
                    temp_img = logic.maskImg(temp_img, binary_label_leaves)
                    # ui.displayImg(temp_img, pause=False, title=f"temp_img LABEL:{label} {leaf.leaf_id}")
                    dico_label_overlay_IPSDK[label] = arithm.addImgImg(temp_img, dico_label_overlay_IPSDK[label])
                    if label == "proba":
                        dico_label_overlay_IPSDK[label] = util.convertImg(dico_label_overlay_IPSDK[label], PyIPSDK.eImageBufferType.eIBT_UInt16)
                    else:
                        dico_label_overlay_IPSDK[label] = util.convertImg(dico_label_overlay_IPSDK[label], PyIPSDK.eImageBufferType.eIBT_Label16)
                    # ui.displayImg(dico_label_overlay_IPSDK[label], pause=True, title=f"LABEL:{label} {leaf.leaf_id}, after addImgImg")

                    # ui.displayImg(dico_label_overlay_IPSDK[label], pause=True)
            dico_label_overlay_IPSDK["leaf"] = self.full_leaves_label_img
            for label, img_overlay in dico_label_overlay_IPSDK.items():
                # ui.displayImg(img_overlay, pause=True, title=f" {basename}_{label}_overlay_ipsdk.tif   {label}")
                PyIPSDK.saveTiffImageFile(self.basedir.joinpath(f"{basename}_{label}_overlay_ipsdk.tif").as_posix(),
                                              img_overlay)

            # # build all csv tables
            result_separated = pd.concat(dict_frames_separated_leaves.values(),
                                         keys=dict_frames_separated_leaves.keys(), ignore_index=True)

            self.__build_df_split(basename, result_separated)

            # call blend to build mask overlay
            self.__blend_overlay(basename, dico_label_overlay_IPSDK)

    @staticmethod
    def __bary_sort(list_to_order):
        x_min, y_min, x_size, y_size, x_bary, y_bary = list_to_order
        if x_size < 1000:
            return (x_bary * 3 + y_bary * 5) / 2
        else:
            return (x_bary * 6 + y_bary * 3) / (2.5 * x_size)

    def __blend_overlay(self, basename, dico_label_overlay_IPSDK):
        color_label_to_RGB_blend = {}
        all_labels_image = None
        # loop of all label_16 image by class (leaf, lesion, proba, others)
        dico_label_overlay_IPSDK.pop('proba')
        dico_label_overlay_IPSDK.pop('leaf')
        for label, img_label in dico_label_overlay_IPSDK.items():
            indice_label = self.model_to_label_dict[label]["value"]
            colors_label = self.model_to_label_dict[label]["color"]
            colors_UINT = [int(i) for i in colors_label]
            color_label_to_RGB_blend[indice_label] = colors_UINT

            # convert split labels to 1 value to apply good colorLUT
            if label == "lesion" and self.color_lesion_individual:
                # ui.displayImg(img_label, pause=True, title=f"img_label for class: {label}")
                label_value = util.convertImg(img_label, PyIPSDK.eImageBufferType.eIBT_Label8)
            else:
                bin_labels = bin.thresholdImg(img_label, 0, 0)
                logic.logicalNotImg(bin_labels, bin_labels)
                int_label_value = arithm.multiplyScalarImg(bin_labels, indice_label)
                label_value = util.convertImg(int_label_value, PyIPSDK.eImageBufferType.eIBT_Label8)
            # ui.displayImg(label_value, pause=True, title=f"label_value for class: {label}")
            if not all_labels_image:
                all_labels_image = label_value
            else:
                all_labels_image = arithm.addImgImg(all_labels_image, label_value)
            all_labels_image = util.convertImg(all_labels_image, PyIPSDK.eImageBufferType.eIBT_Label8)
        color_label_to_RGB_blend[0] = [0, 0, 0]
        # ui.displayImg(all_labels_image, pause=True, title=f"all_labels_image")

        # Convert the input image if necessary
        im_UInt8 = self.full_leaves_ipsdk_img
        if im_UInt8.getBufferType() != PyIPSDK.eIBT_UInt8:
            range_UInt8 = PyIPSDK.createRange(0, 255)
            im_normalized = itrans.normalizeImg(self.full_leaves_ipsdk_img, range_UInt8)
            im_UInt8 = util.convertImg(im_normalized, PyIPSDK.eIBT_UInt8)
        # Count the number of labels
        statsRes = glbmsr.statsMsr2d(all_labels_image)
        nbLabels = statsRes.max

        # Create 3 random LUTs (one per channel)
        randValues = np.random.rand(3, int(nbLabels + 1)) * 255
        for i in color_label_to_RGB_blend:
            if i <= nbLabels:
                for c in range(3):
                    randValues[c][i] = color_label_to_RGB_blend[i][c]

        lutR = PyIPSDK.createIntensityLUT(0, 1, randValues[0, :])
        lutG = PyIPSDK.createIntensityLUT(0, 1, randValues[1, :])
        lutB = PyIPSDK.createIntensityLUT(0, 1, randValues[2, :])
        colorLut = [lutR, lutG, lutB]

        # Convert the label image to a color image
        overlayImage = PyIPSDK.createImage(im_UInt8.getGeometry())

        for c in range(0, 3):
            plan = PyIPSDK.extractPlan(0, c, 0, overlayImage)
            itrans.lutTransform2dImg(all_labels_image, colorLut[c], plan)

        # Blending
        blend = arithm.blendImgImg(im_UInt8, overlayImage, 1-self.alpha)
        # change alpha blending see https://fr.wikipedia.org/wiki/Alpha_blending
        blend = itrans.normalizeImg(blend, PyIPSDK.createRange(0, 255))
        blend = util.convertImg(blend, PyIPSDK.eIBT_UInt8)

        mask = bin.lightThresholdImg(all_labels_image, 1)
        binaryGeometry = PyIPSDK.geometryRgb2d(PyIPSDK.eIBT_Binary, im_UInt8.getSizeX(), im_UInt8.getSizeY())
        maskImage = PyIPSDK.createImage(binaryGeometry)

        for c in range(0, 3):
            plan = PyIPSDK.extractPlan(0, c, 0, maskImage)
            util.copyImg(mask, plan)

        logic.maskImgImg(blend, im_UInt8, maskImage, blend)

        # loop to add leaf if draw True
        if self.draw_ML_image:
            leaf_color = [int(i) for i in self.model_to_label_dict["leaf"]["color"]]
            for leaf in self.table_leaves:
                self.__drawRectangle(image=blend,
                                     x=leaf.x_position,
                                     y=leaf.y_position,
                                     w=leaf.x_size,
                                     h=leaf.y_size,
                                     color=leaf_color,
                                     e=3
                                     )
        # Save an image
        PyIPSDK.saveTiffImageFile(self.basedir.joinpath(f"{basename}_mask_overlay.tif").as_posix(), blend)
        self.mask_overlay_files.append(self.basedir.joinpath(f"{basename}_mask_overlay.tif"))

    @staticmethod
    def __drawRectangle(image, x, y, w, h, e, color):
        image.array[0, y - e:y + e, x - e:x + w + e] = color[0]
        image.array[1, y - e:y + e, x - e:x + w + e] = color[1]
        image.array[2, y - e:y + e, x - e:x + w + e] = color[2]

        image.array[0, y:y + h, x - e:x + e] = color[0]
        image.array[1, y:y + h, x - e:x + e] = color[1]
        image.array[2, y:y + h, x - e:x + e] = color[2]

        image.array[0, y + h - e:y + h + e, x - e:x + w + e] = color[0]
        image.array[1, y + h - e:y + h + e, x - e:x + w + e] = color[1]
        image.array[2, y + h - e:y + h + e, x - e:x + w + e] = color[2]

        image.array[0, y:y + h, x + w - e:x + w + e] = color[0]
        image.array[1, y:y + h, x + w - e:x + w + e] = color[1]
        image.array[2, y:y + h, x + w - e:x + w + e] = color[2]

    def __append_col_df(self, basename, df):
        # print(f"APPEND DF {basename} {df}")
        df.insert(0, "cut_name", basename)
        df_merge = pd.merge(self.meta_info.dataframe_with_cut_name, df, on="cut_name")  # ,how="outer")
        return df_merge

    def __build_df_split(self, basename, result_separated=None):
        # all leaf and all lesions
        result_separated = self.__append_col_df(basename, result_separated)
        # save results to csv format
        csv_path_file = self.basedir.joinpath(f"{basename}_split-info.csv").as_posix()
        # print(f"CSV SAVE AT {csv_path_file}")
        result_separated.to_csv(csv_path_file, index=False, sep="\t", float_format='%.6f')

    def __split_leaves(self, image_path, loaded_image):
        # extract path/name from image path
        path_img = Path(image_path)
        basename = path_img.stem
        small_size = 60000
        ##############################################
        # If machine learning for extract
        if self.split_ML:
            all_mask_label = ml.pixelClassificationRFImg(loaded_image, self.model_load)
            leaf_indice = self.model_to_label_dict["leaf"]["value"]
            nbLabels = glbmsr.statsMsr2d(all_mask_label).max
            all_mask = bin.thresholdImg(all_mask_label, leaf_indice, nbLabels)
        ##############################################
        # If not machine learning for extract
        else:
            # split PCA RGB
            imagePCA, eigenValues, eigenVectors, matrixRank = classif.pcaReductionImg(loaded_image)
            img1 = util.copyImg(PyIPSDK.extractPlan(0, 0, 1, imagePCA))
            # ui.displayImg(img1, pause=True)

            # clusterisation and binarisation
            value5 = bin.otsuThreshold(img1)
            all_mask = bin.darkThresholdImg(img1, value5)

        # ui.displayImg(all_mask, pause=True)
        # suppression des artefacts pour obtenir le mask des feuilles
        # ui.displayImg(all_mask, pause=True, title="all_maskLeaf")
        all_mask_filter = advmorpho.removeSmallShape2dImg(all_mask, 6000)
        structuringElement = PyIPSDK.circularSEXYInfo(5)
        all_mask_filter = morpho.closing2dImg(all_mask_filter, structuringElement, PyIPSDK.eBEP_Disable)
        all_mask_filter = advmorpho.fillHole2dImg(all_mask_filter)
        # ui.displayImg(all_mask_filter, pause=True, title="all_mask_filter")

        # trim the edge of the leaf to remove the plastic cover
        structuringElement = PyIPSDK.circularSEXYInfo(int(self.border_leaf))
        all_mask_filter_bin = morpho.erode2dImg(all_mask_filter, structuringElement)
        # ui.displayImg(all_mask_filter_bin, pause=True, title="all_mask_filter_bin erode")

        # split to separated labels (creation mask)
        # TODO: use adaptativeWatershed
        # all_mask_label = advmorpho.connectedComponent2dImg(all_mask_filter_bin,
        #                                                    PyIPSDK.eNeighborhood2dType.eN2T_8Connexity)
        all_mask_label = advmorpho.watershedBinarySeparation2dImg(all_mask_filter_bin, 150,
                                                           PyIPSDK.eWatershedSeparationMode.eWSM_SplitLabel)

        # remove small objects (bad leaves)
        # TODO: get median leaf size as small size
        # df_size = self.get_dataframe_size_object(all_mask_label)
        # if len(df_size) > 1:
        #     Q1 = np.percentile(df_size["Number of pixels"], 25)
        #     Q2 = np.percentile(df_size["Number of pixels"], 50)
        #     Q3 = np.percentile(df_size["Number of pixels"], 75)
        #     IQR = Q3 - Q1
        #     # small_size = df_size['Number of pixels'].mean() - Q3
        #     print(f"mean:\t{df_size['Number of pixels'].mean()}")
        #     print(f"median:\t{df_size['Number of pixels'].median()}")
        #     print(f"std:\t{df_size['Number of pixels'].std()}")
        #     print(f"Q1:\t{Q1}")
        #     print(f"Q2:\t{Q2}")
        #     print(f"Q3:\t{Q3}")
        #     print(f"IQR:\t{IQR}")
        #     print(f"Q1-IQR:\t{Q1 - IQR}")
        #     print(f"mean-Q3:\t{df_size['Number of pixels'].mean() - Q3}")
        #
        # print(f"small_size:\t{small_size}")
        self.full_leaves_label_img = advmorpho.removeSmallShape2dImg(all_mask_label, small_size)
        # df_size = self.get_dataframe_size_object(label_img)
        # ui.displayImg(self.full_leaves_label_img, pause=False, title="label_img")

        # check if mask is empty after remove small elements
        nbLabels = glbmsr.statsMsr2d(self.full_leaves_label_img).max
        if nbLabels > 0:
            # cut the leaves according to the mask
            calibration = PyIPSDK.createGeometricCalibration2d(1, 1, 'px')
            inMeasureInfoSet2d = PyIPSDK.createMeasureInfoSet2d(calibration)
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "BoundingBoxMinXMsr")
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "BoundingBoxMinYMsr")
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "BoundingBoxSizeXMsr")
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "BoundingBoxSizeYMsr")
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "BoundingBoxCenterXMsr")
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "BoundingBoxCenterYMsr")
            outMeasureSet1 = shapeanalysis.labelAnalysis2d(loaded_image, self.full_leaves_label_img, inMeasureInfoSet2d)
            x_min_array = outMeasureSet1.getMeasure("BoundingBoxMinXMsr").getMeasureResult().getColl(0)[1:]
            y_min_array = outMeasureSet1.getMeasure("BoundingBoxMinYMsr").getMeasureResult().getColl(0)[1:]
            x_size_array = outMeasureSet1.getMeasure("BoundingBoxSizeXMsr").getMeasureResult().getColl(0)[1:]
            y_size_array = outMeasureSet1.getMeasure("BoundingBoxSizeYMsr").getMeasureResult().getColl(0)[1:]
            x_barycentre = outMeasureSet1.getMeasure("BoundingBoxCenterXMsr").getMeasureResult().getColl(0)[1:]
            y_barycentre = outMeasureSet1.getMeasure("BoundingBoxCenterYMsr").getMeasureResult().getColl(0)[1:]

            order_list_pos = zip(x_min_array, y_min_array, x_size_array, y_size_array, x_barycentre, y_barycentre)
            table_leaves = []
            for id, tuple_values in enumerate(order_list_pos, 1):
                xmin, ymin, xsize, ysize, xbary, ybary = tuple_values
                leaf_label = bin.thresholdImg(self.full_leaves_label_img, id, id)

                tmp = util.copyImg(loaded_image)
                imWhite = PyIPSDK.createImage(tmp)
                if imWhite.getBufferType() == PyIPSDK.eIBT_UInt8:
                    util.eraseImg(imWhite, 255.0)
                elif imWhite.getBufferType() == PyIPSDK.eIBT_UInt16:
                    util.eraseImg(imWhite, 65025.0)
                original_mask_leaf = logic.maskImgImg(loaded_image, imWhite , leaf_label)
                # ui.displayImg(original_mask_leaf, pause=True, title=f"leaf_id {id}")
                table_leaves.append(Leaf(basename=basename,
                                         basedir=self.basedir,
                                         leaf_id=id,
                                         x_pos=xmin,
                                         y_pos=ymin,
                                         x_size=xsize,
                                         y_size=ysize,
                                         full_leaves_ipsdk_img=loaded_image,
                                         original_mask_leaf=original_mask_leaf
                                         )
                                    )
            return table_leaves
        else:
            self.logger.warning(f" - fail to found leaf for file {basename}")
            return None

    def merge_images(self, rm_original=False, extension="jpg"):
        self.logger.info("~~~~~~~~~ START STEP MERGE IMAGES ~~~~~~~~~")
        from PIL import Image
        merge_dir_path = self.basedir.joinpath("merge_images")
        merge_dir_path.mkdir(exist_ok=True)
        if len(self.full_files) == 0:
            raise FileNotFoundError(f"Files not found")
        if len(self.mask_overlay_files) == 0:
            raise FileNotFoundError(f"Not found overlay images")

        com, u1, u2 = compare_list(self.full_files, self.mask_overlay_files)
        for img1 in self.full_files:
            basename = Path(img1).name
            if basename in com:
                img2 = Path(img1).as_posix().replace(".tif", "_mask_overlay.tif")
                image1 = Image.open(img1)
                image2 = Image.open(img2)
                images_comb = Image.new('RGB', (image1.width + image2.width, min(image1.height, image2.height)))
                images_comb.paste(image1, (0, 0))
                images_comb.paste(image2, (image1.width, 0))
                outname = merge_dir_path.joinpath(f"{img1.stem}.{extension}")
                self.logger.info(f" - MERGE files  {Path(img1).name} and {Path(img2).name} to {Path(outname).name}")
                images_comb.save(outname)
                if rm_original:
                    # Path(img1).unlink(missing_ok=True)
                    Path(img2).unlink(missing_ok=True)
            else:
                self.logger.warning(f"File {img1} doesn't have overlay")
        self.logger.info("~~~~~~~~~ END STEP MERGE IMAGES ~~~~~~~~~")


class Calibration:
    def __init__(self, calibration_name):
        self.calibration_available = []
        self.calibration_name = calibration_name
        self.dico_info = {"unit": "px", "value": 1}
        try:
            file = xmlet.parse(vrb.folderInformation + "/UserCalibrations.mho")
            calibrationsElement = file.getroot()
        except:
            calibrationsElement = xmlet.Element('Calibrations')
            newCalibration = xmlet.SubElement(calibrationsElement, "Calibration")
            self.create_empty_calibration(newCalibration, name="None")
        for element in calibrationsElement:
            self.calibration_available.append(Dfct.SubElement(element, "Name").text)
            if Dfct.SubElement(element, "Name").text == calibration_name:
                # print(Dfct.SubElement(element, "Name").text)
                # print(Dfct.SubElement(element, "Unit").text)
                # print(Dfct.SubElement(element, "SquarePixel").text)
                # print(Dfct.SubElement(element, "SquareValue").text)
                self.dico_info = {"unit": Dfct.SubElement(element, "Unit").text,
                                  "value": float(Dfct.SubElement(element, "SquareValue").text) /
                                           float(Dfct.SubElement(element, "SquarePixel").text),
                                  }
        if self.calibration_name and self.calibration_name not in self.calibration_available:
            raise ValueError(
                f"Calibration '{self.calibration_name}' is not on available value: {self.calibration_available}")

    @staticmethod
    def create_empty_calibration(element, name="None"):
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


class Leaf:
    def __init__(self, basename, basedir, leaf_id, x_pos, y_pos, x_size, y_size, full_leaves_ipsdk_img, original_mask_leaf):
        self.logger = logging.getLogger('Leaf')
        self.basename = basename
        self.basedir = basedir
        self.leaf_id = leaf_id
        self.x_position = int(x_pos)
        self.y_position = int(y_pos)
        self.x_size = int(x_size)
        self.y_size = int(y_size)

        self.full_leaves_ipsdk_img = full_leaves_ipsdk_img
        self.original_mask_leaf = original_mask_leaf

        self.image_ipsdk_blend_dict_class = {}

        self.dico_frames_separated = {}
        self.model_name_classification = None

    def label_image_to_df(self, split_mask_separated_filter, calibration_obj, label):
        # check if mask is empty after remove small elements
        nbLabels = glbmsr.statsMsr2d(split_mask_separated_filter).max
        if nbLabels >= 0:
            calibration = PyIPSDK.createGeometricCalibration2d(calibration_obj.dico_info["value"],
                                                               calibration_obj.dico_info["value"],
                                                               calibration_obj.dico_info["unit"])

            inMeasureInfoSet2d = PyIPSDK.createMeasureInfoSet2d(calibration)
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Area2dMsr",
                                      shapeanalysis.createHolesBasicPolicyMsrParams(False))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "NbPixels2dMsr")
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Perimeter2dMsr",
                                      shapeanalysis.createHolesBasicPolicyMsrParams(False))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dDiameterLengthMsr",
                                      shapeanalysis.createSkeleton2dDiameterLengthMsrParams(
                                          PyIPSDK.eSHP_Ignored))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dDiameterMeanCurvatureMsr",
                                      shapeanalysis.createSkeleton2dDiameterMeanCurvatureMsrParams(
                                          PyIPSDK.eSHP_Ignored))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dDiameterTortuosityMsr",
                                      shapeanalysis.createSkeleton2dDiameterTortuosityMsrParams(
                                          PyIPSDK.eSHP_Ignored))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dLengthMsr",
                                      shapeanalysis.createSkeleton2dLengthMsrParams(PyIPSDK.eSHP_Ignored,
                                                                                    PyIPSDK.eSEC_Leaf))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dMaxThicknessMsr",
                                      shapeanalysis.createSkeleton2dMaxThicknessMsrParams(PyIPSDK.eSHP_Ignored))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dMeanEdgeLengthMsr",
                                      shapeanalysis.createSkeleton2dMeanEdgeLengthMsrParams(
                                          PyIPSDK.eSHP_Ignored, PyIPSDK.eSEC_Leaf))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dMeanThicknessMsr",
                                      shapeanalysis.createSkeleton2dMeanThicknessMsrParams(
                                          PyIPSDK.eSHP_Ignored))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dMinThicknessMsr",
                                      shapeanalysis.createSkeleton2dMinThicknessMsrParams(PyIPSDK.eSHP_Ignored))
            PyIPSDK.createMeasureInfo(inMeasureInfoSet2d, "Skeleton2dNbVertexMsr",
                                      shapeanalysis.createSkeleton2dNbVertexMsrParams(PyIPSDK.eSHP_Ignored,
                                                                                      PyIPSDK.eSVC_Internal))

            # call to separated object of the class
            outMeasureSet1 = shapeanalysis.labelAnalysis2d(split_mask_separated_filter,
                                                           split_mask_separated_filter,
                                                           inMeasureInfoSet2d)

            # ui.displayImg(split_mask_separated, pause=True)
            # ui.displayImg(split_mask_separated_filter, pause=True)
            # convert to panda dataframe
            df = outset_to_df(outMeasureSet1)
            if df.empty:
                df = pd.DataFrame(columns=df.columns, index=[0])
                df.fillna(0, inplace=True)
            df.insert(0, "Class", label)
            df.insert(0, "leaf_ID", self.leaf_id)
            self.dico_frames_separated[f"{label}-{self.leaf_id}"] = df

    def analysis(self, model_load, model_to_label_dict, model_classification_to_label_dict, small_object,
                 calibration_obj, save_cut=False,
                 model_name_classification=None):
        self.model_name_classification = model_name_classification

        # ui.displayImg(self.full_leaves_ipsdk_img, pause=True, title="self.full_leaves_ipsdk_img")
        # ui.displayImg(self.original_mask_leaf, pause=True, title="self.original_mask_leaf")

        ipsdk_img = util.getROI2dImg(self.original_mask_leaf, self.x_position, self.y_position, self.x_size,
                                     self.y_size)
        # ui.displayImg(ipsdk_img, pause=True, title="ipsdk_img")
        if save_cut:
            Path(self.basedir.joinpath("leaf_cut_only")).mkdir(exist_ok=True)
            outimgname = self.basedir.joinpath("leaf_cut_only", f"{self.basename}_{self.leaf_id}.tif")
            self.logger.info(f'Save file leaf: {outimgname.as_posix()}')
            PyIPSDK.saveTiffImageFile(outimgname.as_posix(), ipsdk_img)

        # apply smart segmentation machine learning
        all_masks, imageProbabilities = ml.pixelClassificationRFWithProbabilitiesImg(ipsdk_img, model_load)
        # ui.displayImg(all_masks, pause=True, title="all_masks")

        def save_proba(imageProbabilities):
            outImage = util.copyImg(imageProbabilities)
            outImage = util.convertImg(outImage, PyIPSDK.eIBT_UInt8)
            valueMin = 0.25
            imageOverlay = arithm.addScalarImg(imageProbabilities, - valueMin)
            imageOverlay = arithm.multiplyScalarImg(imageOverlay, 255 / (1 - valueMin))
            imageOverlay = util.convertImg(imageOverlay, PyIPSDK.eIBT_UInt8)
            currentLut = [[255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 0, 1, 2, 4, 5, 6, 7, 8, 8, 9, 10, 12, 13, 14, 15, 16, 16, 17, 18, 20, 21, 22, 23, 24, 24, 25, 26, 28, 29, 30, 31, 32, 32, 33, 34, 36, 37, 38, 39, 40, 40, 41, 42, 44, 45, 46, 47, 48, 48, 49, 50, 52, 53, 54, 55, 56, 56, 57, 58, 60, 61, 62, 63, 64, 65, 65, 66, 68, 69, 70, 71, 72, 73, 73, 74, 76, 77, 78, 79, 80, 81, 81, 82, 84, 85, 86, 87, 88, 89, 89, 90, 92, 93, 94, 95, 96, 97, 97, 98, 100, 101, 102, 103, 104, 105, 105, 106, 108, 109, 110, 111, 112, 113, 113, 114, 116, 117, 118, 119, 120, 121, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 180, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            imageOverlay = fct.applyLut(imageOverlay, currentLut)
            outImage = wgt.customBlending(imageOverlay, outImage, 1)
            outImage = util.convertImg(outImage, PyIPSDK.eIBT_UInt16)
            outImage = itrans.normalizeImg(outImage, PyIPSDK.createRange(0, 65535))
            self.image_ipsdk_blend_dict_class["proba"] = outImage
            # self.image_ipsdk_blend_probabilities = outImage
            # ui.displayImg(outImage, pause=True, title="outImage")
            # outimgname = self.basedir.joinpath(f"{self.basename}_{self.leaf_id}_proba.tif")
            # PyIPSDK.saveTiffImageFile(outimgname.as_posix(), outImage)
        save_proba(imageProbabilities)
        # create empty overlay for label color with original size
        self.image_label_blend = PyIPSDK.createImage(all_masks, PyIPSDK.eImageBufferType.eIBT_UInt16)
        util.eraseImg(self.image_label_blend, 0)

        # create empty overlay for split lesions image with original size
        self.image_ipsdk_blend = PyIPSDK.createImage(all_masks, PyIPSDK.eImageBufferType.eIBT_Label16)
        util.eraseImg(self.image_ipsdk_blend, 0)
        # loop for label found on ML
        for label in model_to_label_dict.keys():
            indice_class = int(model_to_label_dict[label]["value"])
            if indice_class != 0:  # remove first class ie background
                nbLabels = glbmsr.statsMsr2d(all_masks).max  # count nb label for extract complet leaf
                if label.lower() in ["leaf"]:  # if leaf
                    split_mask = bin.thresholdImg(all_masks, 1, nbLabels)
                    # ui.displayImg(split_mask, pause=True, title="split_mask on LOOP")
                    split_mask_separated = advmorpho.connectedComponent2dImg(split_mask,
                                                                             PyIPSDK.eNeighborhood2dType.eN2T_8Connexity)
                    # ui.displayImg(split_mask_separated, pause=True, title="split_mask_separated on LOOP")
                    nb = glbmsr.statsMsr2d(split_mask_separated).max
                    if nb != 1:
                        split_mask_separated_filter = advmorpho.keepBigShape2dImg(split_mask_separated, 1)
                    else:
                        split_mask_separated_filter = split_mask_separated
                    # ui.displayImg(split_mask_separated_filter, pause=True, title="split_mask_separated_filter on
                    # LOOP")
                    self.image_ipsdk_blend_dict_class[label] = split_mask_separated_filter
                    self.label_image_to_df(split_mask_separated_filter, calibration_obj, label)
                else:
                    split_mask = bin.thresholdImg(all_masks, indice_class, indice_class)
                    # remove small elements if < x px on connected
                    split_mask_filter = advmorpho.removeSmallShape2dImg(split_mask, small_object)
                    # split mask to individual label
                    # TODO use watershedBinarySeparation2dImg or adaptiveBinaryWatershed2dImg
                    split_mask_separated_filter = advmorpho.watershedBinarySeparation2dImg(split_mask_filter, 4,
                                                                                           PyIPSDK.eWatershedSeparationMode.eWSM_SplitLabel)
                    # split_mask_separated_filter = advmorpho.adaptiveBinaryWatershed2dImg(split_mask_filter, 0.5,
                    #                                                                      PyIPSDK.eWatershedSeparationMode.eWSM_SplitLabel)
                    # ui.displayImg(split_mask, pause=True)

                    if self.model_name_classification and label.lower() in ["lesion"]:
                        # ui.displayImg(split_mask_separated_filter, pause=True, title="split_mask_filter_label on
                        # LEAF")
                        img3 = fctML.applySmartClassification(split_mask_separated_filter, ipsdk_img,
                                                           Path(vrb.folderShapeClassification).joinpath(
                                                               f"{self.model_name_classification}").as_posix())
                        # ui.displayImg(img3, pause=True, title="img3 on LEAF")
                        for label_classification in model_classification_to_label_dict.keys():
                            indice = int(model_classification_to_label_dict[label_classification]["value"])
                            split_mask_filter = bin.thresholdImg(img3, indice + 1, indice + 1)
                            # ui.displayImg(split_mask_filter, pause=True, title="split_mask_filter on LEAF")
                            split_mask_separated_filter = advmorpho.watershedBinarySeparation2dImg(split_mask_filter, 4,
                                                                                                   PyIPSDK.eWatershedSeparationMode.eWSM_SplitLabel)
                            # split_mask_separated_filter = advmorpho.adaptiveBinaryWatershed2dImg(split_mask_filter, 0.5,
                            #                                                                      PyIPSDK.eWatershedSeparationMode.eWSM_SplitLabel)
                            # ui.displayImg(split_mask_separated_filter, pause=True,
                            # title=f"split_mask_separated_filter on LEAF for label {label_classification}")
                            self.label_image_to_df(split_mask_separated_filter, calibration_obj, label_classification)
                            # ui.displayImg(split_mask_separated_filter, pause=True, title=f"split_mask_separated_filter on {label_classification} LEAF")
                            self.image_ipsdk_blend_dict_class[label_classification] = split_mask_separated_filter

                    if label.lower() in ["lesion"] and not self.model_name_classification:
                        self.image_ipsdk_blend_dict_class[label] = split_mask_separated_filter
                        self.label_image_to_df(split_mask_separated_filter, calibration_obj, label)
                    if label.lower() not in ["lesion"]:

                        self.image_ipsdk_blend_dict_class[label] = split_mask_separated_filter
                        self.label_image_to_df(split_mask_separated_filter, calibration_obj, label)
        for img in self.image_ipsdk_blend_dict_class.values():
            util.convertImg(img, PyIPSDK.eImageBufferType.eIBT_Label16)

    def __repr__(self):
        return f"{self.__class__}({pp(self.__dict__)})"


class LeAFtool:
    def __init__(self, config_file=None, debug=None):

        self.AVAIL_TOOLS = ["draw", "cut", "ML", "merge"]
        self.__allow_ext = ["jpg", "JPG", "PNG", "png", "BMP", "bmp", "tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]
        self.cut_obj = None
        self.analysis = None
        self.plant_model = None
        self.__allow_plant_model = ["banana", "rice"]

        with open(config_file, "r") as file_config:
            self.config = yaml.load(file_config, Loader=yaml.Loader)

        # self.__check_dir(section="log_path")
        self.log_path = self.get_config_value(section="log_path")
        if not self.log_path:
            self.log_path = Path(__file__).parent.as_posix()
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        if "debug" in self.config and not debug:
            self.debug = self.get_config_value(section="debug")
        else:
            self.debug = debug

        self.logger = self.configure_logger('LeAFtool')
        # for printing in logFile
        # cmd_line = " ".join(sys.argv)
        cmd_line = f"{sys.executable} {__file__} -c {Path(config_file).resolve()}"
        self.logger.info(f"{' LeAFtool analysis start ':#^80s}")
        self.logger.info(f"Command line : {cmd_line}")
        self.logger.debug("DEBUG MODE")
        self.logger.info(f"Your Logs folder is : {self.log_path}")
        self.logger.info(f"Your input file config YAML is : {config_file}")
        try:
            self.__check_config_dic()
        except Exception as e:
            if self.debug:
                self.logger.exception(e)
            else:
                self.logger.error(e)

    def configure_logger(self, name):
        basenameLog = Path(self.log_path).joinpath("LeAFtool_log")
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': f'%(asctime)s | {"%(name)-16s | %(funcName)-15s |" if self.debug else ""} %(levelname)-8s | %(message)s',
                    # 'datefmt': '%Y-%m-%d %H:%M',
                    'datefmt': '%m-%d %H:%M',
                },
                'colored': {
                    '()': 'colorlog.ColoredFormatter',
                    'format': f'%(log_color)s %(asctime)s | {"%(name)-16s | %(funcName)-15s |" if self.debug else ""} %(levelname)-8s | %(message)s',
                    'datefmt': '%m-%d %H:%M',
                },
            },
            'handlers': {
                'stdout_handler': {
                    'level': f'{"DEBUG" if self.debug else "INFO"}',
                    'class': "logging.StreamHandler",
                    'formatter': 'colored',
                    'stream': 'ext://sys.stdout',
                },
                'logHandler': {
                    'level': f'{"DEBUG" if self.debug else "INFO"}',
                    'filename': f"{basenameLog}.o",
                    'class': 'logging.FileHandler',
                    'formatter': 'standard',
                    'mode': 'w',
                },
                'errorLogHandler': {
                    'level': 'WARNING',
                    'filename': f"{basenameLog}.e",
                    'class': 'logging.FileHandler',
                    'formatter': 'standard',
                    'mode': 'w',
                },
            },
            'loggers': {
                "": {
                    'handlers': ['stdout_handler', 'logHandler', 'errorLogHandler'],
                    'level': f'{"DEBUG" if self.debug else "INFO"}',
                    'propagate': True,
                },
            }
        })
        return logging.getLogger(name)

    def get_config_value(self, section, key=None, subsection=None):
        if not key and not subsection:
            return self.config[section]
        if subsection:
            return self.config[section][subsection][key]
        else:
            return self.config[section][key]

    def set_config_value(self, section, value, key=None, subsection=None):
        if not key and not subsection:
            self.config[section] = value
        elif subsection:
            self.config[section][subsection][key] = value
        else:
            self.config[section][key] = value

    @property
    def export_use_yaml(self):
        """Use to print a dump config.yaml with corrected parameters"""

        def represent_dictionary_order(self, dict_data):
            return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())

        def setup_yaml():
            yaml.add_representer(OrderedDict, represent_dictionary_order)

        setup_yaml()
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False, indent=4)

    def __check_dir(self, section, key=None, mandatory=[], subsection=None, makedir=False):
        """Check if path is a directory if not empty
            resolve path on config

        Arguments:
            section (str): the first level on config.yaml
            key (str): the second level on config.yaml
            mandatory (list tuple): a list or tuple with tools want mandatory info
            subsection (str): the second level on config.yaml (ie 3 level)
            makedir (boolean): if not exist create directory

        Raises:
            NotADirectoryError: If config.yaml data `path` does not exist.
        """
        path_value = self.get_config_value(section=section, key=key, subsection=subsection)
        if path_value:
            path = Path(path_value).resolve().as_posix() + "/"
            if (not Path(path).exists() or not Path(path).is_dir()) and not makedir:
                raise NotADirectoryError(
                    f'CONFIG FILE CHECKING FAIL : in the "{section}"'
                    f' section, {f"subsection {subsection}" if subsection else ""}, {key} directory "{path}" '
                    f'{"does not exist" if not Path(path).exists() else "is not a valid directory"}')
            else:
                Path(path).mkdir(exist_ok=True)
                self.set_config_value(section=section, key=key, value=path, subsection=subsection)
        elif len(mandatory) > 0:
            raise NotADirectoryError(
                f'CONFIG FILE CHECKING FAIL : in the "{section}" section, '
                f'{f"subsection {subsection}" if subsection else ""}, {key} directory "{path_value}" '
                f'{"does not exist" if not Path(path_value).exists() else "is not a valid directory"} but is '
                f'mandatory for tool: {" ".join(mandatory)}')

    def __check_file(self, section, key=None, mandatory=[], subsection=None):
        """Check if path is a file if not empty
        :return absolute path file"""
        path_value = self.get_config_value(section=section, key=key, subsection=subsection)
        path = Path(path_value).resolve().as_posix()
        if path:
            if not Path(path).exists() or not Path(path).is_file():
                raise FileNotFoundError(
                    f'CONFIG FILE CHECKING FAIL : in the {section} section, '
                    f'{f"subsection {subsection}" if subsection else ""},{key} file "{path}" '
                    f'{"does not exist" if not Path(path).exists() else "is not a valid file"}')
            else:
                self.set_config_value(section=section, key=key, value=path, subsection=subsection)
        elif len(mandatory) > 0:
            raise FileNotFoundError(
                f'CONFIG FILE CHECKING FAIL : in the "{section}" section, '
                f'{f"subsection {subsection}" if subsection else ""},{key} file "{path_value}" '
                f'{"does not exist" if not Path(path_value).exists() else "is not a valid file"} but is mandatory for '
                f'tool: {" ".join(mandatory)}')

    @staticmethod
    def __var_2_bool(key, tool, to_convert):
        """convert to boolean"""
        if isinstance(type(to_convert), bool):
            return to_convert
        elif f"{to_convert}".lower() in ("yes", "true", "t"):
            return True
        elif f"{to_convert}".lower() in ("no", "false", "f"):
            return False
        else:
            raise TypeError(
                f'CONFIG FILE CHECKING FAIL : in the "{key}" section, "{tool}" key: "{to_convert}" is not a valid '
                f'boolean')

    def __build_tools_activated(self, key, allow, mandatory=False):
        tools_activate = []
        for tool, activated in self.config[key].items():
            if tool in allow:
                boolean_activated = self.__var_2_bool(key=key, tool=tool, to_convert=activated)
                if boolean_activated:
                    tools_activate.append(tool)
                    self.config[key][tool] = boolean_activated
            else:
                raise ValueError(
                    f'CONFIG FILE CHECKING FAIL : On section "{key}", tool: "{tool}" not allow on LeAFtool, '
                    f'select from {allow}')
        if len(tools_activate) == 0 and mandatory:
            raise ValueError(f"CONFIG FILE CHECKING FAIL : you need to set True for at least one {key} from {allow}")
        return tools_activate

    def __get_allow_extension(self, section, key):
        ext = self.get_config_value(section=section, key=key)
        if "." in ext:
            ext = ext.replace(".", "")
            self.set_config_value(section=section, key=key, value=ext)
        if ext not in self.__allow_ext:
            raise NotImplementedError(
                f"'extension': '{ext}' is not allow use only {self.__allow_ext} value for section {section}, key, "
                f"{key}!!! exit")

    def __check_config_dic(self):
        """Configuration file checking"""
        # get model name
        self.plant_model = self.get_config_value(section="PLANT_MODEL")
        if not self.plant_model or self.plant_model not in self.__allow_plant_model:
            raise NameError(
                f"{self.plant_model} is not a valid model to work on LeAFtool only use {self.__allow_plant_model}")
        # check tools activation
        self.tools = self.__build_tools_activated("RUNSTEP", self.AVAIL_TOOLS, True)

        # if Draw or Cut
        if "draw" in self.tools or "cut" in self.tools:
            self.__check_dir(section="DRAW-CUT", key="images_path")
            self.__check_dir(section="DRAW-CUT", key="out_draw_dir", makedir=True)
            self.__get_allow_extension(section="DRAW-CUT", key="extension")
            self.cut_obj = DrawAndCutImages(scan_folder=self.get_config_value(section="DRAW-CUT", key="images_path"),
                                             rename=self.get_config_value(section="rename"),
                                             extension=self.get_config_value(section="DRAW-CUT", key="extension"),
                                             x_pieces=self.get_config_value(section="DRAW-CUT", key="x_pieces"),
                                             y_pieces=self.get_config_value(section="DRAW-CUT", key="y_pieces"),
                                             top=self.get_config_value(section="DRAW-CUT", key="top"),
                                             left=self.get_config_value(section="DRAW-CUT", key="left"),
                                             bottom=self.get_config_value(section="DRAW-CUT", key="bottom"),
                                             right=self.get_config_value(section="DRAW-CUT", key="right"),
                                             noise_remove=self.__var_2_bool("DRAW-CUT", "noise_remove",
                                                                            to_convert=self.get_config_value(
                                                                                section="DRAW-CUT",
                                                                                key="noise_remove")),
                                             numbering=self.get_config_value(section="DRAW-CUT", key="numbering"),
                                             plant_model=self.plant_model,
                                             force_rerun=self.__var_2_bool("DRAW-CUT", "force_rerun",
                                                                           to_convert=self.get_config_value(
                                                                               section="DRAW-CUT", key="force_rerun")),
                                             )
            if self.config["RUNSTEP"]["draw"]:
                self.cut_obj.loop_draw(draw_dir_name=self.get_config_value(section="DRAW-CUT", key="out_draw_dir"))
            if self.config["RUNSTEP"]["cut"]:
                self.cut_obj.loop_cut(cutdir_name=self.get_config_value(section="DRAW-CUT", key="out_cut_dir"),
                                        csv_file=self.get_config_value(section="csv_file")
                                        )
        if ("ML" in self.tools or "merge" in self.tools) and (
                not self.cut_obj or (self.cut_obj and self.cut_obj.exit_status)):
            self.__check_dir(section="ML", key="images_path")
            self.analysis = AnalysisImages(scan_folder=self.get_config_value(section="ML", key="images_path"),
                                           model_name=self.get_config_value(section="ML", key="model_name"),
                                           csv_file=self.get_config_value(section="csv_file"),
                                           rename=self.get_config_value(section="rename"),
                                           calibration_name=self.get_config_value(section="ML",
                                                                                  key="calibration_name"),
                                           small_object=self.get_config_value(section="ML", key="small_object"),
                                           alpha=self.get_config_value(section="ML", key="alpha"),
                                           border=self.get_config_value(section="ML", key="leaf_border"),
                                           noise_remove=self.__var_2_bool("ML", "noise_remove",
                                                                          to_convert=self.get_config_value(section="ML",
                                                                                                           key="noise_remove")),
                                           force_rerun=self.__var_2_bool("ML", "force_rerun",
                                                                         to_convert=self.get_config_value(section="ML",
                                                                                                          key="force_rerun")),
                                           draw_ML_image=self.__var_2_bool("ML", "draw_ML_image",
                                                                           to_convert=self.get_config_value(
                                                                               section="ML", key="draw_ML_image")),
                                           split_ML=self.__var_2_bool("ML", "split_ML",
                                                                      to_convert=self.get_config_value(section="ML",
                                                                                                       key="split_ML")),
                                           plant_model=self.plant_model,
                                           model_name_classification=self.get_config_value(section="ML",
                                                                                           key="model_name_classification"),
                                           color_lesion_individual=self.__var_2_bool("ML", "color_lesion_individual",
                                                                      to_convert=self.get_config_value(section="ML",
                                                                                                       key="color_lesion_individual"))
                                           )
            if self.config["RUNSTEP"]["ML"]:
                self.analysis.run_ML()
            if self.config["RUNSTEP"]["merge"]:
                self.__get_allow_extension(section="MERGE", key="extension")
                self.analysis.merge_images(rm_original=self.__var_2_bool("MERGE", "rm_original",
                                                                         to_convert=self.get_config_value(
                                                                             section="MERGE", key="rm_original")),
                                           extension=self.get_config_value(section="MERGE", key="extension"))

    def __repr__(self):
        return f"{self.__class__}({pp(self.__dict__)})"


#####################################################
# Generic code
#####################################################

def sort_human(in_list, _nsre=None):
    """
    Sort a :class:`list` with alpha/digit on the way that humans expect,\n
    use list.sort(key=sort_human) or\n
    sorted(list, key=sort_human)).

    Arguments:
        in_list (:obj:`list`): a python :class:`list`
        _nsre (:obj:`re.compil`, optional): re expression use for compare , defaults re.compile('([0-9]+)'

    Returns:
        list: sorted with human sort number

    Example:
        >>> list_to_sorted = ["something1","something32","something17","something2","something29","something24"]
        >>> print(sorted(list_to_sorted, key=sort_human))
        ['something1', 'something2', 'something17', 'something24', 'something29', 'something32']
        >>> list_to_sorted.sort(key=sort_human)
        >>> print(list_to_sorted)
        ['something1', 'something2', 'something17', 'something24', 'something29', 'something32']

    """
    from warnings import warn
    from re import compile, split
    if not _nsre:
        _nsre = compile('([0-9]+)')
    try:
        return [int(text) if text.isdigit() else f"{text}".lower() for text in split(_nsre, in_list)]
    except TypeError:
        if not isinstance(in_list, int):
            warn(
                f"Yoda_powers::sort_human : element '{in_list}' on the list not understand so don't sort this "
                f"element\n",
                SyntaxWarning, stacklevel=2)
            return in_list


def compare_list(list1, list2):
    """
    Function to compare two list and return common, uniq1 and uniq2

    Arguments:
        list1 (list): the first python :class:`list`
        list2 (list): the second python :class:`list`

    Returns:
        list: common, u1, u2
        common: the common elements of the 2 list,
        u1: uniq to list1,
        u2: uniq to list2

    Notes:
        ens1 = set([1, 2, 3, 4, 5, 6])\n
        ens2 = set([2, 3, 4])\n
        ens3 = set([6, 7, 8, 9])\n
        print(ens1 & ens2) set([2, 3, 4]) car ce sont les seuls à être en même temps dans ens1 et ens2\n
        print(ens1 | ens3) set([1, 2, 3, 4, 5, 6, 7, 8, 9]), les deux réunis\n
        print(ens1 & ens3) set([6]), même raison que deux lignes au dessus\n
        print(ens1 ^ ens3) set([1, 2, 3, 4, 5, 7, 8, 9]), l'union moins les éléments communs\n
        print(ens1 - ens2) set([1, 5, 6]), on enlève les éléments de ens2
    """
    list1 = [Path(elm).name for elm in list1]
    list2 = [Path(elm).name.replace('_mask_overlay', "") for elm in list2]
    ens1 = set(list1)
    ens2 = set(list2)
    common = list(ens1 & ens2)
    uniq1 = list(ens1 - ens2)
    uniq2 = list(ens2 - ens1)
    return sorted(common, key=sort_human), sorted(uniq1, key=sort_human), sorted(uniq2, key=sort_human)





    parser_mandatory = argparse.ArgumentParser(add_help=False)

    mandatory = parser_mandatory.add_argument_group('Input mandatory infos for running')
    mandatory.add_argument('-c', '--config', metavar="path/to/file/", type=existent_file, required=True,
                           dest='config_file', help='path to config file YAML')
    parser_other = argparse.ArgumentParser(
        parents=[parser_mandatory],
        add_help=False,
        prog=Path(__file__).name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description_tools,
        epilog=epilog_tools
    )


def get_last_version(git_url, version):
    """Function for know the last version of Git repo in website"""
    try:
        from urllib.request import urlopen
        from re import search
        import click
        soft_name = "LeAFtool"
        try:
            HTML = urlopen(f"{git_url}/tags").read().decode('utf-8')
            str_search = f"{git_url.replace('https://github.com', '')}/releases/tag/.*"
            lastRelease = search(str_search, HTML).group(0).split("/")[-1].split('"')[0]
        except Exception as e:
            print(e)
            lastRelease = "There aren’t any releases"
        epilogTools = "\n"
        if str(version) != lastRelease:
            if lastRelease < str(version):
                epilogTools = click.style(f"\n    ** NOTE: This {soft_name} version ({version}) is higher than the production version ({lastRelease}), you are using a dev version\n\n", fg="yellow", bold=True)
            elif lastRelease > str(version) and lastRelease != "There aren’t any releases":
                epilogTools = click.style(f"\n    ** NOTE: The Latest version of {soft_name} {lastRelease} is available at {git_url}\n\n", fg="yellow", underline=True)
            elif lastRelease == "There aren’t any releases":
                epilogTools = click.style(f"\n    ** NOTE: There aren’t any releases at the moment\n\n", fg="red", underline=False)
            else:
                epilogTools = click.style(f"\n    ** NOTE: Can't check if new release are available\n\n", fg="red", underline=False)
        return epilogTools
    except Exception as e:
        epilogTools = click.style(f"\n    ** ENABLE TO GET LAST VERSION, check internet connection\n{e}\n\n", fg="red")
        return epilogTools


#####################################################
# MAIN
#####################################################
DOCS = "https://github.com/sravel/LeAFtool/blob/main/README.rst"
GIT_URL = "https://github.com/sravel/LeAFtool"
__version__ = "0.0.1"
description_tools = f"""
    Welcome to LeAFtool version: {__version__}! Created on September 2021
    @author: Sebastien Ravel (CIRAD)
    @email: sebastien.ravel@cirad.fr

    #             .=:                                                                                       
    #          :--%@*:                                                                                      
    #        ::=-#@**#.                                                                                     
    #       -+-:*@**+=.                                                                                     
    #      -=-:+@+=*=+.            :=              .....:.                                        -=. 
    #     :--:-@#-=*##.           -@@%         =#@%*******-         **                           .@%: 
    #    -=--:%%+=**#+           :@- %#         #@.                -@-                           *@   
    #   .=:::*@+#++==           .@+   @*       :@-                 @*                            @#   
    #    -=:-@*=+:==  .:        %%    .@+      #@  ..:-=+.       =*@*+++.   .=+-       .=+-     :@=   
    #     ::#@+#+=- :%#*%+     +@      :@-     @%*#***+=-        =@*---+.  *%*+#@=   .##*+#@-   -@-   
    #      .@*::.  .@*  %@ .:-=@%=++++==@@.   :@:                =@       %%    *@   %#    #@   :@-   
    #      -@=     =@=+#*  -+=@@---::--==@#   *@.                @*       @+   .@#  .@=   :@*   -@-   
    #      .@@*++*##%@*.      @*         =@-  @@:               :@-       -%*+*%*    =%*+*%+    -@+   
    #        =++=-.  =**##*   =:          *+  -%:               .*-         ---        ---       -- 

    Please cite our github: {GIT_URL}
    Licencied under CeCill-C (http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html)
    and GPLv3 Intellectual property belongs to CIRAD and authors.
    Documentation avail at: {DOCS}
    {get_last_version(git_url=GIT_URL, version=__version__)}"""


@click.command("cmd_LeAFtool", short_help=click.secho(description_tools, fg='green', nl=False),
               context_settings={'help_option_names': ('-h', '--help'), "max_content_width": 800}, no_args_is_help=True)
@click.option('--config', '-c',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
              required=True, show_default=False, help='path to config file YAML')
@click.option('--debug', '-d', is_flag=True, required=False, default=False, show_default=True,
              help='enter verbose/debug mode')
@click.version_option(__version__, '--version', '-v')
@click.argument('other', nargs=-1, type=click.UNPROCESSED)
def main(config, debug, other):
    """
    \b
    Run image scan analysis with paramters load on config file YAML

    Example:
        cmd_LeAFtool.py -c config.yaml
    """
    #####################################################
    # EDIT ONLY THIS LINE TO CHANGE CONFIG FILE
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/test-henri/config-henri.yaml"
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/Marie/config-marie.yaml"
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/bug_francoise/config.yaml"
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/bug_marie/config.yaml"
    #####################################################
    LeAFtool(config_file=config, debug=debug)


if __name__ == '__main__':
    main()
