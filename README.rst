===============
Image Drug VAEs
===============


Pytorch models to create molecular VAEs using images as a latent space.


How To
--------
Currently, this will just train GridLSTM on perfect RDKIT images.

Set Up Environment

``
git clone #used for data but not needed if you are going to use your own data
``

* Smiles to Image Training.
    Using the config file, one can simply run ``python train_im_smi.py``


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
