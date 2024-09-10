=====
SEALS
=====

The SEALS (Source Emission Accounting & Localization System) project is a joint
project between the National Science Foundation (NSF) National Center for
Atmospheric Research (NCAR), managed by the University Corporation for
Atmospheric Research (UCAR), Colorado State University (CSU) and the Planetary
Science Institute (PSI) to create an open-source software tool that uses a
machine learning (ML) approach to locate and quantify methane emissions at
upstream and midstream oil and gas facilities outfitted with low-cost
meteorology and continuous CH4 monitoring instrument platforms.

The NCAR/seals-ml repository contains the ML software developed by the joint project described above.

The NCAR/SEALS repository contains an python base open-source framework designed to facilitate test runs of the ML models in seals-ml.  It
utilizes a Model-View-Controller (MVC) design pattern, along with a parameters file, to enable simple command-line, archive-mode execution of any of the
following 3 models of seals-ml:
 * sealsml.keras.models.BackTrackerDNN
 * sealsml.keras.models.LocalizedLeakRateBlockTransformer
 * sealsml.keras.models.BlockTransformer
It may be extended in the future to support additional models, a real-time mode (currenty stubbed out), additional output options, or other features.  
The framework code runs in the same environment as the seals-ml code itself, defined by seals-ml/enviroment_gpu.yml.

.. toctree::
   :hidden:

   Users_Guide/index

