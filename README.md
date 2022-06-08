# c-origami

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ninashenker/c-origami/blob/main/LICENSE)
[![Build and Test]()]()

[Website](https://www.corigami.org/) |
[Dataset](link) |
[GitHub](https://github.com/ninashenker/c-origami) |
[Publications](#list-of-papers)

C-Origami is a deep neural network model for predicting de novo cell type-specific chromatin architecture. By incorporating DNA sequence, CTCF binding, and chromatin accessibility profiles, C-Origami achieves accurate cell type-specific prediction.

Datasets can be downloaded from (link). Publications
associated with the C. Origami project can be found
[at the end of this README](#list-of-papers).


## Documentation

### CTCF/ATAC/DNA data 

### Hi-C ground truth data

### Code Repository

For code documentation, most functions and classes have accompanying docstrings
that you can access via the `help` function in IPython. For example:

```python
from c-origami import Prediction

help(Prediction)
```

## Dependencies and Installation

**Note:** Contributions to the code are continuously tested via GitHub actions.
If you encounter an issue, the best first thing to do is to try to match the
test environments in `requirements.txt` and `dev-requirements.txt`.

First install PyTorch according to the directions at the
[PyTorch Website](https://pytorch.org/get-started/) for your operating system
and CUDA setup. Then, run

```bash
pip install c-origami
```

### Installing Directly from the Github

If you want to install directly from the GitHub source, clone the repository,
navigate to the `c-origami` root directory and run

```bash
pip install -e .
```

## Training a model

## Prediction

## Editing/Perturbation

## Screening


## License

C-Origami is MIT licensed, as found in the [LICENSE file](https://github.com/ninashenker/c-origami/blob/main/LICENSE).

## Cite

If you use the C-Origami code in your project, please cite the bioRxiv 
paper:

```BibTeX
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jimin Tan and Javier Rodriguez-Hernaez and Theodore Sakellaropoulos and Francesco Boccalatte and Iannis Aifantis and Jane Skok and David Fenyö and Bo Xia and Aristotelis Tsirigos},
    journal = {bioRxiv e-prints},
    archivePrefix = "bioRxiv",
    doi = {10.1101/2022.03.05.483136},
    year={2022}
}
```


## List of Papers

The following lists titles of papers from the C-Origami project. 

Cell type-specific prediction of 3D chromatin architecture
Jimin Tan, Javier Rodriguez-Hernaez, Theodore Sakellaropoulos, Francesco Boccalatte, Iannis Aifantis, Jane Skok, David Fenyö, Bo Xia, Aristotelis Tsirigos
bioRxiv 2022.03.05.483136; doi: https://doi.org/10.1101/2022.03.05.483136
