#!/home/sebastien/Documents/IPSDK/Explorer_3_1_0_3_linux/Miniconda/bin/python3.8
import PyIPSDK
import PyIPSDK.IPSDKUI as ui
import PyIPSDK.IPSDKIPLGlobalMeasure as glbmsr
import PyIPSDK.IPSDKIPLAdvancedMorphology as advmorpho
import PyIPSDK.IPSDKIPLBinarization as bin
import PyIPSDK.IPSDKIPLClassification as classif
import PyIPSDK.IPSDKIPLShapeAnalysis as shapeanalysis
import PyIPSDK.IPSDKIPLMorphology as morpho
import PyIPSDK.IPSDKIPLUtility as util
import PyIPSDK.IPSDKIPLMachineLearning as ml
import PyIPSDK.IPSDKIPLLogical as logic
import PyIPSDK.IPSDKIPLArithmetic as arithm
import PyIPSDK.IPSDKIPLIntensityTransform as itrans

import xml.etree.ElementTree as xmlet
import cv2

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from pprint import pprint as pp
import logging
import logging.config
import sys
import time
import yaml
import colorlog
import argparse

# sys.tracebacklimit = 1
# auto add Explorer in PYTHONPATH
for path in sys.path:
    if "/bin/Release_linux_x64" in path:
        explorer_path = Path(path.split('/bin/Release_linux_x64')[0]).joinpath("Explorer/Interface").as_posix()
        break
sys.path.insert(0, explorer_path)

import DatabaseFunction as Dfct
import UsefullFunctions as fct
import UsefullVariables as vrb

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

color_label_to_RGB_Uint16 = {'blue': (7967, 30583, 46260),
                             'orange': (65535, 32639, 3598),
                             'green': (11308, 41120, 11308),
                             'red': (54998, 10023, 10280),
                             'purple': (38036, 26471, 48573),
                             'brown': (35980, 22102, 19275),
                             'pink': (58339, 30583, 49858),
                             'gray': (32639, 32639, 32639),
                             'gold': (48316, 48573, 8738),
                             'turquoise': (5911, 48830, 53199),
                             'black': (0, 0, 0),
                             'white': (65535, 65535, 65535)
                             }


# pp(color_label_to_RGB_Uint16)


class Timer(object):
    def __enter__(self):
        self.start()
        # __enter__ must return an instance bound with the "as" keyword
        return self

        # There are other arguments to __exit__ but we don't care here

    def __exit__(self, *args, **kwargs):
        self.stop()

    def start(self):
        if hasattr(self, 'interval'):
            del self.interval
        self.start_time = time.time()

    def stop(self):
        if hasattr(self, 'start_time'):
            self.interval = time.time() - self.start_time
            del self.start_time  # Force timer reinit


def read_image_UINT16(path_image):
    extension = Path(path_image).suffix[1:]
    if extension in ["tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]:
        imageIP = PyIPSDK.loadTiffImageFile(path_image)
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
        self.dataframe_with_crop_name = None
        self.__dict_names_pos = None
        self.__list_filenames = None
        self.rename_to_df = {}
        self.exit_status = False

        self.__load_metadata_csv_to_dict()
        self.__build_meta_to_rename()
        self.build_dataframe_with_crop_name()

    def __load_metadata_csv_to_dict(self):
        """ Load csv file into dict use for rename scan
            The dataframe must contain header for columns
            The 2 first columns are use as dict key (tuple key with scan name and crop position)

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

        df = pd.read_csv(self.path_csv, index_col=[0, 1], header=0, sep=csv_separator)
        self.dataframe = pd.read_csv(self.path_csv, header=0, sep=csv_separator).reset_index(drop=True)
        self.__dict_names_pos = df.to_dict('index')
        self.__list_filenames = [str(key1) for (key1, key2) in self.__dict_names_pos.keys()]

    def check_corresponding(self, files_list):
        """test if all scan file have name on csv file"""
        # try:
        not_found_list = []
        for img_file in files_list:
            basename = img_file.stem
            if basename not in self.__list_filenames:
                not_found_list.append(img_file.name)
        if not_found_list:
            self.exit_status = False
            txt_list = '\n - '.join([""] + not_found_list)
            raise NameError(f"Not found corresponding name for scan:{txt_list}")
        self.exit_status = True
        # except NameError as e:
        #     print(e)
        #     self.logger.error(e)

    def __build_meta_to_rename(self):
        if not self.rename_to_df:
            for scan_name, pos in self.__dict_names_pos:
                df = self.dataframe.query(f'{self.header[0]}=="{scan_name}" & {self.header[1]}=={pos}').copy(deep=True)
                rename = "_".join([str(df[elm].values[0]).replace("/", "-") for elm in self.rename_order])
                df["crop_name"] = rename
                self.__dict_names_pos[(scan_name, pos)].update({"crop_name": rename})
                self.rename_to_df[rename] = df.reset_index(drop=True)

    def build_dataframe_with_crop_name(self):
        df_list = [v for k, v in self.rename_to_df.items()]
        self.dataframe_with_crop_name = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)
        # self.dataframe_with_crop_name = pd.concat([pd.concat(v) for k,v in self.rename_to_df.items()])

    def check_correspondingML(self, files_list):
        """test if all scan file have name on csv file"""
        # TODO: add test
        pass
        # try:
        # not_found_list = []
        # for img_file in files_list:
        #     basename = img_file.stem
        #     if basename not in self.__list_filenames:
        #         not_found_list.append(img_file.name)
        # if not_found_list:
        #     self.exit_status = False
        #     txt_list = '\n - '.join([""]+not_found_list)
        #     raise NameError(f"Not found corresponding name for scan:{txt_list}")
        # self.exit_status = True
        # # except NameError as e:
        # #     print(e)
        # #     self.logger.error(e)

    def rename_to_meta(self, scan_name):
        return self.rename_to_df[scan_name]

    def meta_to_crop_rename(self, scan_name, pos):
        if (scan_name, pos) in self.__dict_names_pos:
            return self.__dict_names_pos[(scan_name, pos)]["crop_name"]
        else:
            return None

    def __repr__(self):
        # return f"{self.__class__}({pp(self.__dict__)})"
        return f"{self.dataframe_with_crop_name}"


class CropAndCutImages:
    """
    Object to crop and cut scan images on folder.
    There are able to draw lines to show the cut result before real cut.
    When cut use dataframe to rename scan images
    """

    def __init__(self, scan_folder, rename=None, extension='jpg', x_pieces=2, y_pieces=2, top=0, left=0, bottom=0,
                 right=0,
                 noise_remove=False, numbering="right", plant_model=None, force_rerun=False):
        """Created the objet

        Args:
            scan_folder (:obj:`str`): Path to scan images
            rename (:obj:`list`): List of columns header used to rename crop image (default first 2 columns)
            extension (:obj:`str`): The scan images extension, must be the same for all scan. allow extension are:[
            "jpg", "JPG", "PNG", "png", "BMP", "bmp", "tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]
            x_pieces (:obj:`int`): Number of vertical crop
            y_pieces (:obj:`int`): Number of horizontal crop
            top (:obj:`int`): The top marge to remove before cut
            left (:obj:`int`): The left marge to remove before cut
            bottom (:obj:`int`): The bottom marge to remove before cut
            right (:obj:`int`): The right marge to remove before cut
            noise_remove (:obj:`boolean`): use IPSDK unionLinearOpening2dImg function to remove small objet noise (
            default value 3)
            numbering (:obj:`str`): if right (default), the output order crop is left to right, if bottom,
            the output order is top to bottom then left
            plant_model (:obj:`str`): The plant model name (rice or banana)
            force_rerun (:obj:`boolean`): even files existed, rerun draw and/or cut
        """
        self.logger = logging.getLogger('CropAndCutImages')
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
        self.logger.info(f"CropImage parameters: {message}")

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

    @staticmethod
    def __opencv_to_IPSDK(image):
        """convert opencv image to IPSDK image"""
        if len(image.shape) >= 3:
            if image.shape[2] >= 3:
                if image.shape[2] == 3:
                    b, g, r = cv2.split(image)
                if image.shape[2] == 4:
                    b, g, r, a = cv2.split(image)
                allChannels = [PyIPSDK.fromArray(r), PyIPSDK.fromArray(g), PyIPSDK.fromArray(b)]
                imageIP = PyIPSDK.createImageRgb(PyIPSDK.eImageBufferType.eIBT_UInt16, image.shape[1],
                                                 image.shape[0])
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

    def loop_crop(self, cutdir_name, csv_file):
        """Run crop on images files

        Args:
            cutdir_name (:obj:`str`): the output directory to store crop images
            csv_file (:obj:`str`): The file use to rename images
        """

        def save_image(img, pos):
            basename = self.meta_info.meta_to_crop_rename(scan_name=img, pos=pos)
            file_name = cut_dir_path.joinpath(f"{basename}.tif")
            if basename and not Path(file_name).exists():
                x, y, w, h = box
                im_crop = im_borderless[y: h, x: w].copy()
                img = (im_crop * 256).astype('uint16')
                imageIP = self.__opencv_to_IPSDK(img)
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
                image = cv2.imread(img_file.as_posix())
                im_borderless = image[self.params["top"]:image.shape[0] - self.params["bottom"],
                                self.params["left"]:image.shape[1] - self.params["right"]]
                img_width, img_height = im_borderless.shape[1], im_borderless.shape[0]
                height = img_height // self.params["y_pieces"]
                width = img_width // self.params["x_pieces"]
                position = 1

                if self.numbering == "right":
                    for i in range(0, self.params["y_pieces"]):
                        for j in range(0, self.params["x_pieces"]):
                            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                            save_image(img_file.stem, position)
                            position += 1
                elif self.numbering == "bottom":
                    for i in range(0, self.params["x_pieces"]):
                        for j in range(0, self.params["y_pieces"]):
                            box = (i * width, j * height, (i + 1) * width, (j + 1) * height)
                            save_image(img_file.stem, position)
                            position += 1
        self.logger.info("~~~~~~~~~ END STEP CUT ~~~~~~~~~")

    def loop_draw(self, draw_dir_name):
        """draw lines before crop

        Args:
            draw_dir_name (:obj:`str`): the output directory to store draw images
        """
        self.logger.info("~~~~~~~~~ START STEP DRAW ~~~~~~~~~")
        draw_dir_path = self.__scan_folder.joinpath(draw_dir_name)
        draw_dir_path.mkdir(exist_ok=True)
        self.logger.info(f"OUTPUT DRAW directory is: {draw_dir_path}")

        for img_file in self.__scan_folder.glob(f"*.{self.extension}"):
            outname = f"{draw_dir_path}/{img_file.stem}_draw.tif"
            self.logger.info(f"DRAW IMAGE FILE: {img_file.name} to {Path(outname).name}")

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
                if self.noise_rm:
                    # image noise removal
                    im_draw = (im_draw * 256).astype('uint16')
                    imageIP = self.__opencv_to_IPSDK(im_draw)
                    imageIP = morpho.unionLinearOpening2dImg(imageIP, 3.0, PyIPSDK.eBEP_Disable)
                    PyIPSDK.saveTiffImageFile(outname, imageIP)
                else:
                    cv2.imwrite(outname, im_draw)
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
                 model_name_classification=None):
        """
        Args:
            scan_folder (:obj:`str`): Path to scan images
            model_name (:obj:`int`): The IPSDK PixelClassification model name build with Explorer
            model_name_classification (:obj:`int`): The IPSDK Classification model name build with Explorer
            csv_file (:obj:`str`): The file use to rename images
            rename (:obj:`list`): List of columns header used to rename crop image (default first 2 columns)
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

        self.full_leaves_ipsdk_img = None
        self.full_files = []
        self.mask_overlay_files = []
        self.files_to_run = []
        self.csv_dict_list = {}
        self.table_leaves = []
        self.draw_ML_image = draw_ML_image
        self.model_name_classification = model_name_classification

        # TODO add csv_path_merge to user argument ?
        self.csv_path_merge = self.basedir.joinpath("global-merge-ALL.csv").as_posix()
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
            return filter(re.compile(pattern).match, strings)

        # glob scan file to analysis
        full_files_filter = glob_re(r'^(.(?!(_mask_overlay)))*.tif$', os.listdir(self.basedir.as_posix()))

        self.full_files = sorted([self.basedir.joinpath(path) for path in full_files_filter if "overlay" not in path])
        full_files_set = set(sorted(f"{path.stem}" for path in self.full_files))

        # if force_rerun load already file run
        mask_overlay_files_filter = glob_re(r'.*_mask_overlay\.tif$', os.listdir(self.basedir.as_posix()))
        self.mask_overlay_files = sorted([self.basedir.joinpath(path) for path in mask_overlay_files_filter])
        mask_overlay_files_filter_set = set(
            sorted(f"{path.stem.replace('_mask_overlay', '')}" for path in self.mask_overlay_files))

        for label in self.all_ml_labels:
            self.csv_dict_list[label] = [self.basedir.joinpath(path).as_posix() for path in
                                         glob_re(fr'.*_merge-{label}\.csv$', os.listdir(self.basedir.as_posix()))]
        basename_files_to_run = list(full_files_set - mask_overlay_files_filter_set)
        if self.force_rerun:
            self.files_to_run = self.full_files
        else:
            self.files_to_run = sorted([file for file in self.full_files if f"{file.stem}" in basename_files_to_run])
        self.meta_info.check_correspondingML(files_list=self.files_to_run)

    def run_ML(self):
        """loop to apply ML on all images"""
        self.logger.info("~~~~~~~~~ START STEP MACHINE LEARNING ~~~~~~~~~")
        if not self.files_to_run and self.full_files:
            self.logger.info(f"All files already run")
        elif not self.files_to_run and not self.full_files:
            raise FileNotFoundError(f"Not found file extension '.tif' on folder: {self.basedir.as_posix()}")
        nb_scan = len(self.files_to_run)
        for indice, img_file_path in enumerate(self.files_to_run, 1):
            self.logger.info(f"Analyse scan file {indice}/{nb_scan}: {img_file_path.name}")
            self.analyse_leaves(image_path=img_file_path.as_posix())
        self.logger.info("~~~~~~~~~ END STEP MACHINE LEARNING ~~~~~~~~~")
        # if self.files_to_run:
        self.logger.info("~~~~~~~~~ START MERGE CSV ~~~~~~~~~")
        self.__merge_CSV()
        self.logger.info("~~~~~~~~~ END MERGE CSV ~~~~~~~~~")


    def __merge_CSV(self, sep="\t", rm_merge=False):
        """merge all CSV file include on final folder

        Args:
            sep: CSV output separator
            rm_merge: if True, remove intermediate csv files Default: False
        """
        all_merge = []

        for label in self.all_ml_labels:

            if len(self.csv_dict_list[label]) > 1:
                df_list = (pd.read_csv(f, sep=sep) for f in self.csv_dict_list[label])
                df_merge = pd.concat(df_list, ignore_index=True, axis=0, ).fillna(0)
                df_merge.sort_values([self.meta_info.header[0], self.meta_info.header[1]], ascending=(True, True),
                                     inplace=True)
                all_merge.append(df_merge)
                csv_path_file = self.basedir.joinpath(f"global-merge-{label}.csv").as_posix()
                self.logger.info(f"Merge all files for class: {label} to {csv_path_file}")
                with open(csv_path_file, "w") as libsizeFile:
                    df_merge.to_csv(libsizeFile, index=False, sep=sep, float_format='%.6f')
                if rm_merge:
                    for file in self.csv_dict_list[label]:
                        Path(file).unlink(missing_ok=True)
        self.logger.info(f"Merge all csv files to {self.csv_path_merge}")
        all_merge_df = all_merge[0]
        for df_ in all_merge[1:]:
            all_merge_df = all_merge_df.merge(df_, on=self.meta_info.header + ["crop_name", "leaf_ID",
                                                                               f"leaf_Area_{self.calibration_obj.dico_info['unit']}"])

        # all_merge_df = pd.concat(all_merge, ignore_index=True, axis=1, join="inner")#.fillna(0)
        all_merge_df.sort_values([self.meta_info.header[0], self.meta_info.header[1]], ascending=(True, True),
                                 inplace=True)

        def selector(x):
            if x == "leaf_ID":
                return "count"
            elif all_merge_df.head(1)[x].dtype == "object":
                return lambda x: ''.join(x.unique())
            elif "min-size" in x:
                return "min"
            elif "max-size" in x:
                return "max"
            elif "mean-size" in x:
                return "mean"
            elif "median-size" in x:
                return "median"
            elif "SD-size" in x:
                return "std"
            else:
                return "sum"

        agg_dict = {f: selector(f) for f in all_merge_df.columns[2:]}
        all_merge_df_agg = all_merge_df.groupby([self.meta_info.header[0], self.meta_info.header[1]]).agg(agg_dict).reset_index()

        with open(self.csv_path_merge, "w") as libsizeFile:
            all_merge_df.to_csv(libsizeFile, index=False, sep=sep, float_format='%.6f')
        with open(self.csv_path_merge.replace(".csv","_aggragated_leaves.csv"), "w") as libsizeFile:
            all_merge_df_agg.columns = ['number_of_leaves' if x == 'leaf_ID' else x for x in all_merge_df_agg.columns]
            all_merge_df_agg.to_csv(libsizeFile, index=False, sep=sep, float_format='%.6f')

    def analyse_leaves(self, image_path):
        # extract path/name from image path
        path_img = Path(image_path)
        basename = path_img.stem

        # load full image (ie with all leaves)
        self.full_leaves_ipsdk_img = PyIPSDK.loadTiffImageFile(image_path)

        # if self.full_leaves_ipsdk_img.getBufferType() != PyIPSDK.eIBT_UInt16:
        #     range_UInt16 = PyIPSDK.createRange(0, 65535)
        #     convert_UInt16 = util.convertImg(self.full_leaves_ipsdk_img, PyIPSDK.eIBT_UInt16)
        #     self.full_leaves_ipsdk_img = itrans.normalizeImg(convert_UInt16, range_UInt16)
        #     PyIPSDK.saveTiffImageFile(self.basedir.joinpath(f"2021-08-19_convert_fields_magna.tif").as_posix(),
        #     self.full_leaves_ipsdk_img)

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
            list_leaves_overlay = []
            list_leaves_overlay_IPSDK = []
            for leaf in self.table_leaves:
                self.logger.info(f" - Read and extract lesion on leaf {leaf.leaf_id}/{len(self.table_leaves)}")
                leaf.analysis(model_load=self.model_load,
                              model_to_label_dict=self.model_to_label_dict,
                              model_classification_to_label_dict=self.model_classification_to_label_dict,
                              small_object=self.small_object,
                              calibration_obj=self.calibration_obj,
                              model_name_classification=self.model_name_classification)
                dict_frames_separated_leaves.update(leaf.dico_frames_separated)
                list_leaves_overlay.append(leaf)
                list_leaves_overlay_IPSDK.append(leaf)

            # build final image with filter overlay
            geometryRgb2 = PyIPSDK.geometry2d(PyIPSDK.eImageBufferType.eIBT_Int32, x_size, y_size)
            overlayImagefilter = PyIPSDK.createImage(geometryRgb2)
            util.eraseImg(overlayImagefilter, 0)
            # add leaf mask to full leaves overlay
            for leaf in list_leaves_overlay:
                # ui.displayImg(overlayImagefilter, pause=True)
                # ui.displayImg(leaf.image_label_blend, pause=True)
                util.putROI2dImg(overlayImagefilter, leaf.image_label_blend, leaf.x_position, leaf.y_position,
                                 overlayImagefilter)

            # build IPSDK overlay
            geometryRgb2_label = PyIPSDK.geometry2d(PyIPSDK.eImageBufferType.eIBT_Label16, x_size, y_size)
            overlayImagefilterIPSDK = PyIPSDK.createImage(geometryRgb2_label)
            util.eraseImg(overlayImagefilterIPSDK, 0)

            # loop for all label to extract IPSDK label image:
            # print(leaf.image_ipsdk_blend_dict_class.items())
            dico_label_overlay_IPSDK = {}
            for leaf in list_leaves_overlay_IPSDK:
                for label, img in leaf.image_ipsdk_blend_dict_class.items():
                    if label not in dico_label_overlay_IPSDK:
                        dico_label_overlay_IPSDK[label] = PyIPSDK.createImage(geometryRgb2_label)
                        util.eraseImg(dico_label_overlay_IPSDK[label], 0)
                    # ui.displayImg(img, pause=True, title=f"{label} {len(dico_label_overlay_IPSDK)}")
                    util.putROI2dImg(dico_label_overlay_IPSDK[label], img, leaf.x_position, leaf.y_position,
                                     dico_label_overlay_IPSDK[label])
                    # ui.displayImg(overlayImagefilterIPSDK, pause=True)
            for label, img_overlay in dico_label_overlay_IPSDK.items():
                # ui.displayImg(img_overlay, pause=True, title=f" {basename}_{label}_overlay_ipsdk.tif   {label}")
                PyIPSDK.saveTiffImageFile(self.basedir.joinpath(f"{basename}_{label}_overlay_ipsdk.tif").as_posix(),
                                              img_overlay)

            # # build all csv tables
            result_separated = pd.concat(dict_frames_separated_leaves.values(),
                                         keys=dict_frames_separated_leaves.keys(), ignore_index=True)

            self.__build_df_split(basename, result_separated)
            self.__build_df_merge(basename, result_separated)
            # call blend to build mask overlay
            self.__blend_overlay(basename, overlayImagefilter)

    @staticmethod
    def __bary_sort(list_to_order):
        x_min, y_min, x_size, y_size, x_bary, y_bary = list_to_order
        if x_size < 1000:
            return (x_bary * 3 + y_bary * 5) / 2
        else:
            return (x_bary * 6 + y_bary * 3) / (2.5 * x_size)

    def __blend_overlay(self, basename, ov):
        ov = util.convertImg(ov, PyIPSDK.eIBT_UInt16)
        # print(self.model_to_label_dict)
        color_label_to_RGB_Uint16blend = {}
        for label in self.model_to_label_dict.keys():
            i = self.model_to_label_dict[label]["value"]
            colors_label = self.model_to_label_dict[label]["color"]
            colors_UINT16 = [int(i * 256) for i in colors_label]
            color_label_to_RGB_Uint16blend[i - 1] = colors_UINT16  # -1 car pas de leaf
        color_label_to_RGB_Uint16blend[0] = [0, 0, 0]
        # Count the number of labels
        nbLabels = glbmsr.statsMsr2d(ov).max

        # Create 3 random LUTs (one per channel)
        randValues = np.random.rand(3, int(nbLabels + 1)) * 65535
        for i in color_label_to_RGB_Uint16blend:
            if i <= nbLabels:
                for c in range(3):
                    randValues[c][i] = color_label_to_RGB_Uint16blend[i][c]

        lutR = PyIPSDK.createIntensityLUT(0, 1, randValues[0, :])
        lutG = PyIPSDK.createIntensityLUT(0, 1, randValues[1, :])
        lutB = PyIPSDK.createIntensityLUT(0, 1, randValues[2, :])
        colorLut = [lutR, lutG, lutB]

        # Convert the label image to a color image
        overlayImage = PyIPSDK.createImage(self.full_leaves_ipsdk_img.getGeometry())

        for c in range(0, 3):
            plan = PyIPSDK.extractPlan(0, c, 0, overlayImage)
            itrans.lutTransform2dImg(ov, colorLut[c], plan)

        # Blending
        blend = arithm.blendImgImg(self.full_leaves_ipsdk_img, overlayImage, 1 - self.alpha)
        # change alpha blending see https://fr.wikipedia.org/wiki/Alpha_blending
        # blend = itrans.normalizeImg(blend, PyIPSDK.createRange(0, 65535))
        blend = util.convertImg(blend, PyIPSDK.eIBT_UInt16)

        mask = bin.lightThresholdImg(ov, 1)
        binaryGeometry = PyIPSDK.geometryRgb2d(PyIPSDK.eIBT_Binary, self.full_leaves_ipsdk_img.getSizeX(),
                                               self.full_leaves_ipsdk_img.getSizeY())
        maskImage = PyIPSDK.createImage(binaryGeometry)

        for c in range(0, 3):
            plan = PyIPSDK.extractPlan(0, c, 0, maskImage)
            util.copyImg(mask, plan)

        # ui.displayImg(self.full_leaves_ipsdk_img, pause=True)
        logic.maskImgImg(blend, self.full_leaves_ipsdk_img, maskImage, blend)
        # loop to add leaf if draw True
        if self.draw_ML_image:
            leaf_color = [int(i * 256) for i in self.model_to_label_dict["leaf"]["color"]]
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
        df.insert(0, "crop_name", basename)
        df_merge = pd.merge(self.meta_info.dataframe_with_crop_name, df, on="crop_name")  # ,how="outer")
        return df_merge

    def __build_df_split(self, basename, result_separated=None):
        # all leaf and all lesions
        result_separated = self.__append_col_df(basename, result_separated)
        # save results to csv format
        csv_path_file = self.basedir.joinpath(f"{basename}_split-info.csv").as_posix()
        # print(f"CSV SAVE AT {csv_path_file}")
        result_separated.to_csv(csv_path_file, index=False, sep="\t", float_format='%.6f')

    def __build_df_merge(self, basename, result_separated):
        # merge all lesion by leaves
        leaves = result_separated[result_separated['Class'] == 'leaf']['leaf_ID']
        dico_df_by_leaves_label = {}
        for leaf_id in leaves:
            cond1 = result_separated['Class'] == 'leaf'
            cond2 = result_separated['leaf_ID'] == int(leaf_id)
            # area_leaf_px2 = result_separated[cond1 & cond2]['Number of pixels'].values[0]
            area_leaf = result_separated[cond1 & cond2].filter(regex='Area').values[0][0]
            for label in self.all_ml_labels:
                dftmp = pd.DataFrame()

                if label in result_separated['Class'].unique() and label.lower() not in ["leaf"]:
                    cond3 = result_separated['Class'] == label
                    # area_lesion_px2 = result_separated[cond2 & cond3].filter(regex='Number of pixels')
                    area_lesion_cm2 = result_separated[cond2 & cond3].filter(regex='Area')
                    nb_lesion = len(area_lesion_cm2)
                    area_lesion_sum = area_lesion_cm2.sum().values[0]
                    area_lesion_median = area_lesion_cm2.median().values[0]
                    area_lesion_mean = area_lesion_cm2.mean().values[0]
                    area_lesion_std = area_lesion_cm2.std().values[0]
                    area_lesion_min = area_lesion_cm2.min().values[0]
                    area_lesion_max = area_lesion_cm2.max().values[0]
                    percent_lesion = (area_lesion_sum / area_leaf) * 100

                    # build dataframe with resume infos
                    dftmp = pd.DataFrame(data=[{f"leaf_ID": leaf_id,

                                                f"leaf_Area_{self.calibration_obj.dico_info['unit']}": area_leaf,
                                                f"{label}_Area_{self.calibration_obj.dico_info['unit']}": area_lesion_sum,

                                                f"{label}_nb": nb_lesion,
                                                f"{label}_percent": percent_lesion,

                                                f"{label}_median-size_{self.calibration_obj.dico_info['unit']}": area_lesion_median,
                                                f"{label}_mean-size_{self.calibration_obj.dico_info['unit']}": area_lesion_mean,
                                                f"{label}_SD-size_{self.calibration_obj.dico_info['unit']}": area_lesion_std,
                                                f"{label}_min-size_{self.calibration_obj.dico_info['unit']}": area_lesion_min,
                                                f"{label}_max-size_{self.calibration_obj.dico_info['unit']}": area_lesion_max
                                                }])
                elif label.lower() not in ["leaf", "background", "lesion"]:
                    dftmp = pd.DataFrame(data=[{f"leaf_ID": leaf_id,

                                                f"leaf_Area_{self.calibration_obj.dico_info['unit']}": area_leaf,
                                                f"{label}_Area_{self.calibration_obj.dico_info['unit']}": 0,

                                                f"{label}_nb": 0,
                                                f"{label}_percent": 0,

                                                f"{label}_median-size_{self.calibration_obj.dico_info['unit']}": 0,
                                                f"{label}_mean-size_{self.calibration_obj.dico_info['unit']}": 0,
                                                f"{label}_SD-size_{self.calibration_obj.dico_info['unit']}": 0,
                                                f"{label}_min-size_{self.calibration_obj.dico_info['unit']}": 0,
                                                f"{label}_max-size_{self.calibration_obj.dico_info['unit']}": 0
                                                }])
                if label not in dico_df_by_leaves_label:
                    dico_df_by_leaves_label[label] = [dftmp]
                else:
                    dico_df_by_leaves_label[label].append(dftmp)

        for label, list_df in dico_df_by_leaves_label.items():
            dftmp = pd.concat(list_df, axis=0, ignore_index=True)
            if not dftmp.empty:
                dftmp = self.__append_col_df(basename, dftmp)
                # save results to csv format
                csv_path_file = self.basedir.joinpath(f"{basename}_merge-{label}.csv").as_posix()
                if csv_path_file not in self.csv_dict_list[label]:
                    self.csv_dict_list[label].append(csv_path_file)
                    dftmp.to_csv(csv_path_file, index=False, sep="\t", float_format='%.6f')
        # concat

    def __split_leaves(self, image_path, loaded_image):
        # extract path/name from image path
        path_img = Path(image_path)
        basename = path_img.stem
        if self.plant_model == "banana":
            small_size = 105000
        elif self.plant_model == "rice":
            small_size = 37000
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
            if self.plant_model == "banana":
                img1 = util.copyImg(PyIPSDK.extractPlan(0, 0, 1, imagePCA))
            elif self.plant_model == "rice":
                img1 = util.copyImg(PyIPSDK.extractPlan(0, 0, 1, imagePCA))
            # ui.displayImg(img1, pause=True)

            # clusterisation and binarisation
            value5 = bin.otsuThreshold(img1)
            all_mask = bin.darkThresholdImg(img1, value5)

        # ui.displayImg(all_mask, pause=True)
        # suppression des artefacts pour obtenir le mask des feuilles
        # ui.displayImg(all_mask, pause=True, title="all_maskLeaf")
        if self.plant_model == "banana":
            all_mask_filter = advmorpho.removeBorder2dImg(all_mask)
            all_mask_filter = advmorpho.removeSmallShape2dImg(all_mask_filter, small_size)
            structuringElement = PyIPSDK.circularSEXYInfo(3)
            all_mask_filter = morpho.closing2dImg(all_mask_filter, structuringElement, PyIPSDK.eBEP_Disable)
            # ui.displayImg(all_mask_filter, pause=True, title="all_mask_filter")
        else:
            all_mask_filter = advmorpho.removeSmallShape2dImg(all_mask, small_size)
        # all_mask_filter_bin = advmorpho.binaryReconstruction2dImg(all_mask, all_mask_filter)
        # ui.displayImg(all_mask_filter_bin, pause=True, title="all_mask_filter_bin")
        all_mask_filter_bin = advmorpho.fillHole2dImg(all_mask_filter)
        # ui.displayImg(all_mask_filter_bin, pause=True, title="all_mask_filter_bin")
        # trim the edge of the leaf to remove the plastic cover
        structuringElement = PyIPSDK.circularSEXYInfo(int(self.border_leaf))
        all_mask_filter_bin = morpho.erode2dImg(all_mask_filter_bin, structuringElement)
        # ui.displayImg(all_mask_filter_bin, pause=True, title="all_mask_filter_bin erode")

        # split to separated labels (creation mask)
        # TODO: use adaptativeWatershed
        all_mask_label = advmorpho.connectedComponent2dImg(all_mask_filter_bin,
                                                           PyIPSDK.eNeighborhood2dType.eN2T_8Connexity)
        # remove small objects (bad leaves)
        # TODO: get median leaf size as small size
        label_img = advmorpho.removeSmallShape2dImg(all_mask_label, small_size)

        # check if mask is empty after remove small elements
        nbLabels = glbmsr.statsMsr2d(label_img).max
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
            outMeasureSet1 = shapeanalysis.labelAnalysis2d(loaded_image, label_img, inMeasureInfoSet2d)
            x_min_array = outMeasureSet1.getMeasure("BoundingBoxMinXMsr").getMeasureResult().getColl(0)[1:]
            y_min_array = outMeasureSet1.getMeasure("BoundingBoxMinYMsr").getMeasureResult().getColl(0)[1:]
            x_size_array = outMeasureSet1.getMeasure("BoundingBoxSizeXMsr").getMeasureResult().getColl(0)[1:]
            y_size_array = outMeasureSet1.getMeasure("BoundingBoxSizeYMsr").getMeasureResult().getColl(0)[1:]
            x_barycentre = outMeasureSet1.getMeasure("BoundingBoxCenterXMsr").getMeasureResult().getColl(0)[1:]
            y_barycentre = outMeasureSet1.getMeasure("BoundingBoxCenterYMsr").getMeasureResult().getColl(0)[1:]

            order_list_pos = sorted(
                [(x_min, y_min, x_size, y_size, x_bary, y_bary) for x_min, y_min, x_size, y_size, x_bary, y_bary in
                 zip(x_min_array, y_min_array, x_size_array, y_size_array, x_barycentre, y_barycentre)],
                key=lambda x: self.__bary_sort(x))

            table_leaves = []
            for c, tuple_values in enumerate(order_list_pos, 1):
                xmin, ymin, xsize, ysize, xbary, ybary = tuple_values
                # print(xmin, ymin, xsize, ysize)
                table_leaves.append(Leaf(basename=basename,
                                         basedir=self.basedir,
                                         leaf_id=c,
                                         x_pos=xmin,
                                         y_pos=ymin,
                                         x_size=xsize,
                                         y_size=ysize,
                                         full_leaves_ipsdk_img=loaded_image
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
    def __init__(self, basename, basedir, leaf_id, x_pos, y_pos, x_size, y_size, full_leaves_ipsdk_img):
        self.logger = logging.getLogger('Leaf')
        self.basename = basename
        self.basedir = basedir
        self.leaf_id = leaf_id
        self.x_position = int(x_pos)
        self.y_position = int(y_pos)
        self.x_size = int(x_size)
        self.y_size = int(y_size)

        self.full_leaves_ipsdk_img = full_leaves_ipsdk_img

        self.image_label_blend = None
        self.image_ipsdk_blend = None
        self.image_ipsdk_blend_dict_class = {}

        self.dico_frames_separated = {}
        self.model_name_classification = None

    def label_image_to_df(self, split_mask_separated_filter, calibration_obj, label):
        # check if mask is empty after remove small elements
        nbLabels = glbmsr.statsMsr2d(split_mask_separated_filter).max
        if nbLabels > 0:
            if split_mask_separated_filter.hasGeometricCalibration():
                calibration = split_mask_separated_filter.getGeometricCalibration()
            else:
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
            df.insert(0, "Class", label)
            df.insert(0, "leaf_ID", self.leaf_id)
            self.dico_frames_separated[f"{label}-{self.leaf_id}"] = df

    def label_to_overlay_blend(self, split_mask_filter, i):
        # use to generate on overlay image with overlay function
        split_mask_filter_UInt16 = util.convertImg(split_mask_filter, PyIPSDK.eImageBufferType.eIBT_UInt16)
        label_img = arithm.multiplyScalarImg(split_mask_filter_UInt16, i - 1)
        self.image_label_blend = arithm.addImgImg(self.image_label_blend, label_img)
        # ui.displayImg(self.image_label_blend, pause=True, title="self.image_label_blend")

    def analysis(self, model_load, model_to_label_dict, model_classification_to_label_dict, small_object,
                 calibration_obj, save_cut=False,
                 model_name_classification=None):
        self.model_name_classification = model_name_classification
        ipsdk_img = util.getROI2dImg(self.full_leaves_ipsdk_img, self.x_position, self.y_position, self.x_size,
                                     self.y_size)
        if save_cut:
            Path(self.basedir.joinpath("leaf_cut_only")).mkdir(exist_ok=True)
            outimgname = self.basedir.joinpath("leaf_cut_only", f"{self.basename}_{self.leaf_id}.tif")
            self.logger.info(f'Save file leaf: {outimgname.as_posix()}')
            PyIPSDK.saveTiffImageFile(outimgname.as_posix(), ipsdk_img)

        # apply smart segmentation machine learning
        # TODO: SAVE PROBABILITY IMAGE?
        all_masks, imageProbabilities = ml.pixelClassificationRFWithProbabilitiesImg(ipsdk_img, model_load)

        # create empty overlay for label color with original size
        self.image_label_blend = PyIPSDK.createImage(all_masks, PyIPSDK.eImageBufferType.eIBT_UInt16)
        util.eraseImg(self.image_label_blend, 0)

        # create empty overlay for split lesions image with original size
        self.image_ipsdk_blend = PyIPSDK.createImage(all_masks, PyIPSDK.eImageBufferType.eIBT_Label16)
        util.eraseImg(self.image_ipsdk_blend, 0)
        # ui.displayImg(all_masks, pause=True, title="all_masks on LOOP")
        # loop for label found on ML
        for label in model_to_label_dict.keys():
            i = int(model_to_label_dict[label]["value"])
            if i != 0:  # remove first class ie background
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
                    self.label_image_to_df(split_mask_separated_filter, calibration_obj, label)
                else:
                    split_mask = bin.thresholdImg(all_masks, i, i)
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
                        img3 = ui.applySmartClassification(split_mask_separated_filter, ipsdk_img,
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
                            if label_classification.lower() in ["lesion"]:
                                self.label_to_overlay_blend(split_mask_separated_filter, i)
                    if label.lower() in ["lesion"] and not self.model_name_classification:
                        self.label_to_overlay_blend(split_mask_separated_filter, i)
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

        self.AVAIL_TOOLS = ["draw", "crop", "ML", "merge"]
        self.__allow_ext = ["jpg", "JPG", "PNG", "png", "BMP", "bmp", "tif", "tiff", "TIF", "TIFF", "Tif", "Tiff"]
        self.crop_obj = None
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
        cmd_line = f"python3 {__file__} -c {Path(config_file).resolve()}"
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
                    'format': '%(asctime)s | %(name)-16s | %(funcName)-15s | %(levelname)-8s | %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M',
                },
                'colored': {
                    '()': 'colorlog.ColoredFormatter',
                    'format': "%(log_color)s %(asctime)s | %(name)-16s | %(funcName)-15s | %(levelname)-8s | %("
                              "message)s",
                    'datefmt': '%Y-%m-%d %H:%M',
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

        # if Draw or Crop
        if "draw" in self.tools or "crop" in self.tools:
            self.__check_dir(section="DRAWCROP", key="images_path")
            self.__check_dir(section="DRAWCROP", key="out_draw_dir", makedir=True)
            self.__get_allow_extension(section="DRAWCROP", key="extension")
            self.crop_obj = CropAndCutImages(scan_folder=self.get_config_value(section="DRAWCROP", key="images_path"),
                                             rename=self.get_config_value(section="rename"),
                                             extension=self.get_config_value(section="DRAWCROP", key="extension"),
                                             x_pieces=self.get_config_value(section="DRAWCROP", key="x_pieces"),
                                             y_pieces=self.get_config_value(section="DRAWCROP", key="y_pieces"),
                                             top=self.get_config_value(section="DRAWCROP", key="top"),
                                             left=self.get_config_value(section="DRAWCROP", key="left"),
                                             bottom=self.get_config_value(section="DRAWCROP", key="bottom"),
                                             right=self.get_config_value(section="DRAWCROP", key="right"),
                                             noise_remove=self.__var_2_bool("DRAWCROP", "noise_remove",
                                                                            to_convert=self.get_config_value(
                                                                                section="DRAWCROP",
                                                                                key="noise_remove")),
                                             numbering=self.get_config_value(section="DRAWCROP", key="numbering"),
                                             plant_model=self.plant_model,
                                             force_rerun=self.__var_2_bool("DRAWCROP", "force_rerun",
                                                                           to_convert=self.get_config_value(
                                                                               section="DRAWCROP", key="force_rerun")),
                                             )
            if self.config["RUNSTEP"]["draw"]:
                self.crop_obj.loop_draw(draw_dir_name=self.get_config_value(section="DRAWCROP", key="out_draw_dir"))
            if self.config["RUNSTEP"]["crop"]:
                self.crop_obj.loop_crop(cutdir_name=self.get_config_value(section="DRAWCROP", key="out_cut_dir"),
                                        csv_file=self.get_config_value(section="csv_file")
                                        )
        if ("ML" in self.tools or "merge" in self.tools) and (
                not self.crop_obj or (self.crop_obj and self.crop_obj.exit_status)):
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
                                                                                           key="model_name_classification")
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
# CODE RUN
#####################################################
version = "0.0.1"


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
    import re
    if not _nsre:
        _nsre = re.compile('([0-9]+)')
    try:
        return [int(text) if text.isdigit() else f"{text}".lower() for text in re.split(_nsre, in_list)]
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

    Examples:
        >>> l1 = [1, 2, 3, 4, 5, 6]
        >>> l2 = [6, 7, 8, 9]
        >>> com, u1, u2 = compare_list(l1, l2)
        >>> print(com)
        [6]
        >>> print(u1)
        [1, 2, 3, 4, 5]
        >>> print(u2)
        [7, 8, 9]

    """
    list1 = [Path(elm).name for elm in list1]
    list2 = [Path(elm).name.replace('_mask_overlay', "") for elm in list2]
    ens1 = set(list1)
    ens2 = set(list2)
    common = list(ens1 & ens2)
    uniq1 = list(ens1 - ens2)
    uniq2 = list(ens2 - ens1)
    return sorted(common, key=sort_human), sorted(uniq1, key=sort_human), sorted(uniq2, key=sort_human)


def existent_file(path):
    """
    'Type' for argparse - checks that file exists and return the absolute path as PosixPath() with pathlib

    Notes:
        function need modules:

        - pathlib
        - argparse


    Arguments:
        path (str): a path to existent file

    Returns:
        :class:`PosixPath`: ``Path(path).resolve()``

    Raises:
         ArgumentTypeError: If file `path` does not exist.
         ArgumentTypeError: If `path` is not a valid file.

    Examples:
        import argparse
        parser = argparse.ArgumentParser(prog='test.py', description='''This is demo''')
        parser.add_argument('-f', '--file', metavar="<path/to/file>",type=existent_file, required=True,
                            dest='path_file', help='path to file')

    """
    from argparse import ArgumentTypeError
    from pathlib import Path

    if not Path(path).exists():
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise ArgumentTypeError(f'ERROR: file "{path}" does not exist')
    elif not Path(path).is_file():
        raise ArgumentTypeError(f'ERROR: "{path}" is not a valid file')

    return Path(path).resolve()


def welcome_args(version_arg, parser_arg):
    """
    use this Decorator to add information to scripts with arguments

    Args:
        version_arg: the program version
        parser_arg: the function which return :class:`argparse.ArgumentParser`

    Returns:
        None:

    Notes:
        use at main() decorator for script with :class:`argparse.ArgumentParser`

    Examples:
        @welcome_args(version, build_parser())
        def main():
            # some code
        main()
        ################################################################################
        #                             prog_name and version                            #
        ################################################################################
        Start time: 16-09-2020 at 14:39:02
        Commande line run: ./filter_mummer.py -l mummer/GUY0011.pp1.fasta.PH0014.pp1.fasta.mum
        - Intput Info:
                - debug: False
                - plot: False
                - scaff_min: 1000000
                - fragments_min: 5000
                - csv_file: blabla
        PROGRAMME CODE HERE
        Stop time: 16-09-2020 at 14:39:02       Run time: 0:00:00.139732
        ################################################################################
        #                               End of execution                               #
        ################################################################################

    """
    from datetime import datetime

    def welcome(func):
        def wrapper():
            start_time = datetime.now()
            parser = parser_arg
            version = version_arg
            parse_args = parser.parse_args()
            # Welcome message
            print(
                f"""{"#" * 80}\n#{Path(parser.prog).stem + " " + version:^78}#\n{"#" * 80}\nStart time: 
{start_time:%d-%m-%Y at %H:%M:%S}\nCommande line run: {" ".join(sys.argv)}\n""")
            # resume to user
            print(" - Intput Info:")
            for k, v in vars(parse_args).items():
                print(f"\t - {k}: {v}")
            print("\n")
            func()
            print(
                f"""\nStop time: {datetime.now():%d-%m-%Y at %H:%M:%S}\tRun time: {datetime.now() - start_time}\n
{"#" * 80}\n#{'End of execution':^78}#\n{"#" * 80}""")

        return wrapper

    return welcome


def build_parser():
    epilog_tools = """Documentation avail at: \n\n"""
    description_tools = f"""
    More information:
        Script version: {version}
    """
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

    optional = parser_other.add_argument_group('Input infos not mandatory')
    optional.add_argument('-v', '--version', action='version', version=version,
                          help=f'Use if you want to know which version of {Path(__file__).name} you are using')
    optional.add_argument('-h', '--help', action='help', help=f'show this help message and exit')
    optional.add_argument('-d', '--debug', action='store_true', help='enter verbose/debug mode')
    return parser_other


@welcome_args(version, build_parser())
def main():
    from datetime import timedelta
    prog_args = build_parser().parse_args()
    #####################################################
    # EDIT ONLY THIS LINE TO CHANGE CONFIG FILE
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/test-henri/config-henri.yaml"
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/Marie/config-marie.yaml"
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/bug_francoise/config.yaml"
    # config_file_name = "/home/sebastien/Documents/IPSDK/IMAGE/bug_marie/config.yaml"
    #####################################################
    with Timer() as timer:
        instance = LeAFtool(config_file=prog_args.config_file, debug=prog_args.debug)
    instance.logger.info(f'Total time in seconds:{timedelta(seconds=timer.interval)}', extra={'className': 'LeAFtool'})


if __name__ == '__main__':
    main()
