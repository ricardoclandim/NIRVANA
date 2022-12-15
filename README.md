## NIRVANA:  **N**onaxisymmetric **I**rregular **R**otational **V**elocity **ANA**lysis

NIRVANA forward models galaxy velocity fields using a second-order nonaxisymmetric model. This enables it to reproduce bisymmetric features like bars. It also accounts for beam smearing and velocity dispersion in the resolved spectra. 

MaNGA-NIRVANA will be a Value Added Catalog included in SDSS Data Release 17, the final data release for the SDSS-IV project MaNGA. It will contain rotation curves for several thousand rotating galaxies in the local Universe with quantified maps of bisymmetric irregularities.



Installation
============

Clone the repo
--------------

To download the DAP software and associated data, clone the `mangadap
GitHub repo`_ by executing:


        git clone --single-branch --branch asymdrift  https://github.com/kbwestfall/NIRVANA.git

This will create a ``NIRVANA`` directory in the current directory.

Install Python 3
----------------

NIRVANA is supported for Python 3 only. To install Python, you can do
so along with a full package manager, like `Anaconda`_, or you can
install python 3 directly from `python.org`_.


Install the NIRVANA from source
-------------------------------

The preferred method to install NIRVANA and ensure its dependencies are
met is to, from the top-level, ``NIRVANA`` directory, run:


    pip install -e .

This approach is preferred because it eases uninstalling the code:

    
    pip uninstall nirvana

Installation in this way should also mean that changes made to the code
should take immediate effect when you restart the calling python
session.

----

To install NIRVANA dependencies, run:


    pip install -r requirements.txt
    
    
---

You need first to download DAPall and DRPall files. This is done through


      nirvana_manga_catalogs --redux ./redux/ --analysis ./analysis/ --dr DR17
      
The files dapall-v3_1_1-3.1.0.fits and drpall-v3_1_1.fits will be downloaded in the corresponding directories.
      
---

To download specific data you want to analyze, run


     nirvana_manga_download 8485 1901 --redux ./redux/ --analysis ./analysis/ --dr DR17
     
The necessary data from MaNGA plate identifier (e.g., 8485) and ifu identifier (e.g., 1901) is downloaded to the corresponding directories.

---

To analyze the data simply run


    nirvana_manga_axisym 8485 1901 --redux ./redux/ --analysis ./analysis/ --dr DR17
    
 The outputs are created in the main directory NIRVANA. To change the output directory, create a new directory (e.g. outputs)  and use the option --odir ./outputs/ when running nirvana_manga_axisym
