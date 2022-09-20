# c-origami

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ninashenker/c-origami/blob/main/LICENSE)

[Models](#Download-model-and-other-relevant-resource-files) |
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
## Download model and other relevant resource files

Next, download the model and other files needed for running C.Origami. Since the files are large and optional, you can download what you need. It is recommended you download the model needed along with the hg38 or mm10 reference genome. 
You may also download the preprocessed CTCF/ATAC data or use your own fastq files. 

For human genome:
```bash
wget -O human_model.pt https://www.dropbox.com/s/jkx1jxjyoumq6e8/hg38_state_dict_43.pt?dl=0

```
For mouse genome:
```bash
wget -O mouse_model.pt https://www.dropbox.com/s/67kopnqxd08gwum/epoch%3D81-step%3D41737.ckpt?dl=0

```

# Training

## Training your own  model

# Inference

For any inference application, download one of our pre-trained models or use your own model.

## Prediction

C.Origami can perform de novo prediction of cell type-specific chromatin architecture using both DNA sequence features and cell type-specific genomic information.


## Editing/Perturbation

## Screening


## License

C-Origami is MIT licensed, as found in the [LICENSE file](https://github.com/ninashenker/c-origami/blob/main/LICENSE).

## Cite

If you use the C-Origami code in your project, please cite the bioRxiv 
paper:

```BibTeX
@inproceedings{tan2020,
    title={Cell type-specific prediction of 3D chromatin architecture},
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
