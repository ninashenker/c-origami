# c-origami

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ninashenker/c-origami/blob/main/LICENSE)

[Models](#Download-model-and-other-relevant-resource-files) |
[GitHub](https://github.com/ninashenker/c-origami) |
[Publications](#list-of-papers)

C-Origami is a deep neural network model for predicting de novo cell type-specific chromatin architecture. By incorporating DNA sequence, CTCF binding, and chromatin accessibility profiles, C-Origami achieves accurate cell type-specific prediction.

Publications associated with the C. Origami project can be found
[at the end of this README](#list-of-papers).


## Documentation

### CTCF/ATAC/DNA data 
In order to use our pipeline we require the sequencing data to be pre-processed. The input for both the CTCF and ATAC data should be in the form of a bigwig (bw) file. The bigwig should be normalized to the total number of reads and should allow visual inspection of the data using an application such as [IGV](https://igv.org).
C.Origami has been trained on the human (hg38) and mouse (mm10) genome. You can download each chromosome along with a json file containing the length of each chromosome by running our python script under `src/preprocessing/dna_sequence/download.sh`.

For human chromosomes:
```bash
download.sh hg38
```
For mouse chromosomes:
```bash
download.sh mm10
```
## Dependencies and Installation

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

Next, download the model and other files needed for running C.Origami. You can also train your own model instead of using ours. See the [training](#Training) section below.

For human genome:
```bash
wget -O human_model.pt https://www.dropbox.com/s/jkx1jxjyoumq6e8/hg38_state_dict_43.pt?dl=0
```
For mouse genome:
```bash
wget -O mouse_model.pt https://www.dropbox.com/s/67kopnqxd08gwum/epoch%3D81-step%3D41737.ckpt?dl=0
```
We use [Hydra](https://github.com/facebookresearch/hydra)  configs for our training and inference pipelines. 

# Training

Under `src/training/config` you will find the config for both the input into the model and train parameters. 

The config `default.yaml` allows you to specify where you would like to save the model and set your hyperparameters. 

Under `dataset` directory you will find the input data config where you can specify `data_root`, `assembly`, and `celltype`. 

Gather the pre-processed ctcf and atac bigwigs under your root directory followed by a subdirectory with the assembly name and inside another directory named by the celltype e.g. `root/mm10/Patski`. Please make sure the directories match what was specified in the config. 


```python
python train.py
```

# Inference

C.Origami can perform de novo prediction of cell type-specific chromatin architecture using both DNA sequence features and cell type-specific genomic information.

For any inference application, download one of our pre-trained models or use your own model. The config.yaml file under `src/inference` allows you to set the task to **predict**, **perturbation**, or **screening**.

```python
python inference.py
```

## Prediction

Set the parameters in config.yaml in order to specify the path of your inputs and the chromosome/start position of your prediction. 

```python
model_path: root/mouse_model.pt 
input_folder: root/genomic_features/
ctcf_name: ctcf_log2fc.bw
atac_name: atac.bw
chr_fa_path: root/chrs/
chr_lengths: chr_length.json
cell_line: Patski
chr_num: int
start_pos: int
task: prediction
```

## Editing/Perturbation

For now the only perturbation implemented is deletion. Fill out the contig.yaml with the same parameters as predict as well as the start and end position. If you want to do multiple deletions, you can specify in the config by creating additional start and end positions. 

```python
# Same parameters as prediction plus these additional perturbation only criteria
task: perturbation
del_pos
  -start: int 
    end:  int
  -start: int
    end:  int
```
## Screening

In silico genetic screening can be used to see what regions of perturbation lead to the greatest impact on the prediction. Running this task will result in a bedgraph file consisting of the chr number, start position, end position, and impact score. The more impact the perturbation had, the higher the impact score.

Screening can be done only for one chromosome at a time. The end position unless otherwise specified will be 2MB from the start position specified above it. The `del_window` is allows you to set the size of the deletion you want to make or in other words how many base pairs to remove. The `step_size` is how far each deletion is from the past deletion (start position) - please note it is fine for the deletions to overlap. 

```python
 # Same parameters as prediction plus these additional screening only criteria
 task: perturbation
 end_pos: int     # optional (default is to screen 2 MB from the start pos)
 del_window: int  # how many base pairs to remove
 step_size: int   # how far each deletion is from the next

```

**Please note that screening can be very computationally intensive especially when screening at a 1 Kb resolution or less. For instance, screening on chromosome 8, a medium-size chromosome which has a length of 146Mb, requires the model to make 146Mb / 1Kb * 2 predictions = 292,000 separate predictions.**

## Multi-run 

If you would like to run multiple predictions at once you can use Hydra's multirun function. For example if you want to predict chromosome 1,3,7 at positions 2MB, 50MB, 75MB we can run the following command:

```bash
python inference.py --multirun chr_num=1,3,7 start_pos=2000000,50000000,75000000
```

## License

C-Origami is MIT licensed, as found in the [LICENSE file](https://github.com/ninashenker/c-origami/blob/main/LICENSE).

## Cite

If you use the C-Origami code in your project, please cite the bioRxiv paper:

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
