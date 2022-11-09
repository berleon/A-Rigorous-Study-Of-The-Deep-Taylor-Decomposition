# A Rigorous Study Of The Deep Taylor Decomposition


[![Build Status](https://github.com/berleon/sanity_checks_for_relation_networks/actions/workflows/dev.yml/badge.svg)](https://github.com/berleon/sanity_checks_for_relation_networks/actions/workflows/dev.yml)

This is the code for the publications ["A Rigorous Study Of The Deep Taylor
Decomposition"](https://openreview.net/forum?id=Y4mgmw9OgV)

## Pointers to the Code:

* Source code of the sanity checks can be found in the following files: `lrp_relations/sanity_checks.py`. For preprocessing, see
`lrp_relations/preprocessing_clevr.py` and `lrp_relations/preprocessing_clevr_xai.py`.
* Source code of the experiment with the simple neural network can be found in `lrp_relations/notebooks/dtd_local_linear_roots.ipynb`.


## Install

Make sure you installed the packages in the subfolders: `relation_network`, `clevr-xai`. And then run `poetry install`.
