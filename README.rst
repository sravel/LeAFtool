
.. image:: Images/LeAFtool-long.png?raw=true
   :target: Images/LeAFtool-long.png?raw=true
   :alt: Alt text
   :width: 50%


.. contents:: Table of Contents
    :depth: 2

About LeAFtool
==============

Research on plant leaf diseases requires the acquisition of quantitative data to characterize the symptoms caused by different pathogens. These symptoms are frequently lesions that are differentiated from the leaf blade by their color and texture. Among the variables used to characterize the impact of a disease, the most relevant are the number of lesions per unit of leaf area, the area and the shape of the lesions. Since visual measurements are only possible on small numbers of images, it is necessary to use computerized image analysis procedures.
Existing procedures can partially meet the needs but are not always adapted to the particularities of the images obtained in the laboratory. From a scanned image under laboratory conditions containing several leaves of plants showing symptoms of a known disease, the algorithm developed makes it possible to obtain for each sheet of the image the number and the characteristics of surface and shape. lesions.

The LeAFtool (Lesion Area Finding tool) is a python script used IPSDK library and also implemented on macro on `Explorer` tool develop by IPSDK.

https://www.reactivip.com/fr/traitement-dimages/#graphic

Install
=======

.. code-block:: bash

    cd .local/ReactivIP/Explorer/Macro_Interface/
    git clone https://github.com/sravel/LeAFtool.git


------------------------------------------------------------------------

USAGE
=====

LeAFtool can be used in command line mode, or GUI macro on Explorer


GUI
---

 +---------------------------------------+---------------------------------------+
 |  |build|                              +  |explore|                            +
 +---------------------------------------+---------------------------------------+
 | Build and Run scripts with interface  +  Explore results                      +
 +---------------------------------------+---------------------------------------+
 | |link|                                                                        +
 +-------------------------------------------------------------------------------+
 |  Live Link to images on Explorer                                              +
 +-------------------------------------------------------------------------------+

.. |build| image:: Images/windows.png?raw=true
   :target: Images/windows.png?raw=true
   :alt: Alt text
   :width: 100%



.. |explore| image:: Images/csv.png?raw=true
   :target: Images/csv.png?raw=true
   :alt: Alt text
   :width: 100%
  
.. |link| image:: Images/explorer.png?raw=true
   :target: Images/explorer.png?raw=true
   :alt: Alt text
   :width: 50%
   :align: middle


------------------------------------------------------------------------

CMD
---

Build config.yaml file like:

.. literalinclude:: ./config.yaml
   :language: YAML


Some exemples
=============


 +------------------------------------------------------+------------------------------------------------------+
 |  |exemple1|                                          +  |exemple2|                                          +
 +------------------------------------------------------+------------------------------------------------------+
 | 1 class:  lesion, with leaf border                   +  2 class: lesion and chloroses, without leaf border  +
 +------------------------------------------------------+------------------------------------------------------+


.. |exemple1| image:: Images/banana.jpg?raw=true
   :target: Images/banana.jpg?raw=true
   :alt: Alt text
   :width: 100%
   :align: middle


.. |exemple2| image:: Images/2class.jpg?raw=true
   :target: Images/2class.jpg?raw=true
   :alt: Alt text
   :width: 72%   
   :align: middle

License
=======

Licencied under `CeCill-C <http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html>`_ and GPLv3.
Intellectual property belongs to `CIRAD <https://www.cirad.fr/>`_ and author.