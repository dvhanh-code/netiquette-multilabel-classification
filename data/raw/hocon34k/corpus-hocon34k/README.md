# HOCON34k: A Corpus of Hate speech in Online Comments from German Newspapers
In this repository we provide the data set for the corpus of Hate speech in Online Comments from German Newspapers (HOCON34k). Please see the [corpus website](https://ccwi.github.io/corpus-hocon34k/) for more information about the corpus.

## How to get started?
- datasets: Folder that contains the data set
- expert-agreement-annotation.ipynb: Python jupyter notebook that allows to reproduce the calculation and optimization of the interrater realiability
- environment.yml: File for creating an Anaconda environment, which contains the packages required to execute the code.

## Installation of the conda environment
As a prerequisite, please make sure that you have installed a current version of the [Python distribution Anaconda](https://www.anaconda.com/download). To install the environment, execute the following command:

```
conda env create --file=environment.yml

conda activate hatespeech-dataset-hocon34k
```
