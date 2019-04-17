===============
Image Drug VAEs
===============


.. image:: https://img.shields.io/pypi/v/image_drug_vaes.svg
        :target: https://pypi.python.org/pypi/image_drug_vaes

.. image:: https://img.shields.io/travis/aclyde11/image_drug_vaes.svg
        :target: https://travis-ci.org/aclyde11/image_drug_vaes

.. image:: https://readthedocs.org/projects/image-drug-vaes/badge/?version=latest
        :target: https://image-drug-vaes.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/aclyde11/image_drug_vaes/shield.svg
     :target: https://pyup.io/repos/github/aclyde11/image_drug_vaes/
     :alt: Updates



Pytorch models to create molecular VAEs using images as a latent space.


How To
--------
Currently, this will just train GridLSTM on perfect RDKIT images.

* Smiles to Image Training.
    Using the config file, one can simply run ``python train_im_smi.py``

#todo Support training Smiles to Image using a model that generates images

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
