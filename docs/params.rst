Parameters
==========

RUN STEP
--------

- **draw** *(boolean)*: Active draw tool to see the lines of the border and cutting
- **cut** *(boolean)*: Active cut tool to see the lines of the border and cutting
- **ml** *(boolean)*: Active machine learning tool to see apply model build on Explorer
- **merge** *(boolean)*: Active merge tool to stuck original file to image with classes overlay



Global
------

- **log_path** *(str)*: Directory path where log files will be created
- **debug** *(boolean)*: More verbose logging for errors debug
- **PLANT_MODEL** *(banana/rice)*: Select plante model
- **csv_file** *(str)*: CSV path file with Meta-info, scan name and position mandatory at first and second position. Separator is autodetect
- **rename** *(list)*: ordered list of csv header used to rename cut files


DRAW-CUT
--------

- **images_path** *(str)*: Input path directory with raw scan images
- **out_cut_dir** *(str)*: Output path directory for cut images
- **out_draw_dir** *(str)*: Output path directory for draw images
- **extension** *(jpg/JPG/PNG/png/BMP/bmp/tif/tiff/TIF/TIFF/Tif/Tiff)*: The raw scan images extension, must be the same for all scan.
- **x_pieces** *(int)*: The number of output fragments to split vertically *Default: 1*
- **y_pieces** *(int)*: The number of output fragments to split horizontally *Default: 1*
- **top** *(int)*: The top margin to remove before cut *Default: 0*
- **left** *(int)*: The left margin to remove before cut *Default: 0*
- **bottom** *(int)*: The bottom margin to remove before cut *Default: 0*
- **right** *(int)*: The right margin to remove before cut *Default: 0*
- **noise_remove** *(boolean)*: Use IPSDK unionLinearOpening2dImg function to remove small white objet noise *Default: False*
- **force_rerun** *(boolean)*: Force running again even files existed, rerun draw and/or cut. *Default: False*
- **numbering** *(right/bottom)*: if right: the output order cut is left to right, if bottom: the output order is top to bottom then left *Default: right*
.. image:: ./docs/images/splitExemple.png

ML
--

- **images_path_ml** *(str)*: Input path directory with cutted scan images
- **model_name** *(int)*: The IPSDK PixelClassification model name build with Explorer
- **model_name_classification** *(int)*: The IPSDK Classification model name build with Explorer
- **split_ML** *(boolean)*: Use machine learning to split leaves instead of RGB *Default: False*
- **calibration_name** *(str)*: Name of Explorer calibration, no calibration if empty
- **small_object** *(int)*: The minimum area of class, to remove small noise detect object *Default: 100*
- **alpha** *(float)*: The degree of transparency to apply for blend overlay color labels. Must float 0 <= alpha <= 1 *Default: 0.5*
- **color_lesion_individual** *(boolean)*: If `True` apply random color for each separated lesions else use all lesions will colored with color of model *Default: True*
- **leaf_border** *(int)*: The diameter of the brush (in pixels) used to erode the leaf *Default: 0*
- **noise_remove** *(boolean)*: Use IPSDK unionLinearOpening2dImg function to remove small white objet noise *Default: False*
- **force_rerun** *(boolean)*: Force running again even files existed. *Default: False*
- **draw_ML_image** *(boolean)*: If `True`, add overlay rectangle corresponding to image used for apply Machine learning (generally one leaf) *Default: False*

Merge
-----

- **rm_original** *(boolean)*: remove individual files `*_mask_overlay` *Default: False*
- **extension** *(jpg/JPG/PNG/png/BMP/bmp/tif/tiff/TIF/TIFF/Tif/Tiff)*: Merge file extension *Default: jpg*


INTERFACE
---------

- **show_meta** *(boolean)*: Show/Hide meta infos section to reduce windows size
- **cut_part** *(int)*: the number of output fragments to split vertically and/or horizontally.
Example with:
- *X parts = 2*
- *Y parts = 3*
.. image:: ./docs/images/splitExemple2.png
- **margin** *(int)*: the number of pixel margin to remove before cut
Example with:
- *top = 200*
- *left = 250*
- *bottom = 280*
- *right = 150*
.. image:: ./docs/images/draw.png
- **upload** *(file)*: Load previous configuration of LeAFtool, must be yaml file
- **save** *(file)*: Save configuration file of LeAFtool un yaml format
- **run** *(file)*: Run processing with the configuration (force to save before)
- **preview** *(file)*: Show/hide the yaml file build with program.
- **preview_edit** *(file)*: The yaml file build with program.
