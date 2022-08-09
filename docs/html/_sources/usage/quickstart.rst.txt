==========
QuickStart
==========


About The Project
=================

| This is a LIneA project to compare detections from Gawa code to the
| input simulations. 

| People involved (alphabetic order):

* Adriano Pieres
* Amanda Fassarela
* Ana Clara de Paula Moreira
* Christophe Benoist
* Cristiano Singulani
* Luiz Nicolaci da Costa


Getting Started
===============

| Whether you use a jupyter notebook or the python script here, an environment
| must be created using conda. The code here is intended to run on cluster of
| LInea, so a few changes should be implemented if you want to run on other
| environment.

| If this is the first time you are running the code in a new environment,
| please create a conda environment and install the dependencies, whether
| using `pip` or `conda`.
| Be sure that these libraries are installed without errors. See the
| information on 'Installation' to a complete list of steps.


Running
=======

| After activate the environment, run the code in terminal:

::

	cd ~/qa_gawa
	python qa_gawa.py


| Be sure that you are in the same folder as the code cloned.

| In case you want to run jupyter notebook:

::

	jupyter-lab qa_gawa.ipynb


| Restart the jn kernel to load the libraries installed.
| Run the jn cell by cell.



Usage
=====

The actions below are executed:

* Read and show config file from Gawa code;
* Plot all the isochronal masks to check if there are areas not covered by masks in Color-Magnitude Diagram (magnitude x color) space;
* Match candidate clusters from detections to simulations, in order to confirm whether a detection is expected or not. Write also a file with undetected clusters (simulated clusters that are not detected);
* Calculate completeness and purity of simulations and shows a few plots where completeness and purity are functions of SMR, deistance, angular sizes, etc;
* Show simulated clusters and detections into the sky color-coded by SNR using different levels of minimum SNR;
* Show CMDs of undetected clusters, to be evaluated by user and show a clue about why cluster were not detected.
