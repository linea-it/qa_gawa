============
Installation
============

| To install the code, you have to clone the repository and
| follow instructions on README to install dependencies.

| The code is intended to run on Linux OS.


Code and dependencies
=====================

| The code is written and runs in Python 3.10.4, but it is compatible
| to python 3.X. The following libraries are mandatory to run the code:

* `numpy <https:/numpy.org/>`_
* `astropy <https:/www.astropy.org/>`_
* `healpy <https:/healpy.readthedocs.io/en/latest>`_
* `json <https:/docs.python.org/3/library/json.html>`_
* `os <https:/docs.python.org/3/library/os.html>`_
* `sys <https:/docs.python.org/3/library/sys.html>`_
* `matplotlib <https:/matplotlib.org/>`_
* `warnings <https:/docs.python.org/3/library/warnings.html>`_
* `tabulate <https:/pypi.org/project/tabulate/>`_


Installation
============

| Clone the repository and create an environment with Conda:

::

	git clone https://github.com/linea-it/qa_gawa && cd qa_gawa
	conda create -p $HOME/.conda/envs/qa_gawa python=3.8
	conda activate qa_gawa
	conda install jupyterlab
	conda install ipykernel
	pip install numpy
	pip install tabulate
	pip install astropy
	pip install healpy
	ipython kernel install --user --name=qa_gawa

| Once you created this env, in the second time (and after)
| you run the code, you can only access the env activating it:

::

	conda activate qa_gawa


| If you have error messages from missing libraries,
| install it in a similar manner as packages installed above.


