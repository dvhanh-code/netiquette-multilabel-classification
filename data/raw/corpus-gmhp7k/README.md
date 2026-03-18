# GMHP7k: A corpus of German misogynistic hatespeech posts
In this repository we provide the data set for the corpus on German misogynistic hatespeech posts (GMHP7k). Please see the [corpus website](https://ccwi.github.io/corpus-gmhp7k/) for more information about the corpus.

## How to get started?
- [datasets](datasets/README.md): Folder that contains the data set with one file for each phase of the annotation process (See Sections 3.1 and 4.1)
- [dataset-analysis-article-mhs.ipynb](dataset-analysis-article-mhs.ipynb): Python jupyter notebook that allows to reproduce the corpus statistics (See Section 4.2)
- [environment.yml](environment.yml): File for creating an Anaconda environment, which contains the packages required to execute the code.

## Installation of the conda environment
As a prerequisite, please make sure that you have installed a current version of the [Python distribution Anaconda](https://www.anaconda.com/download). To install the environment, execute the following command:

```
conda env create --name hatespeech-dataset-gmhp7k --file=environment.yml

conda activate hatespeech-dataset-gmhp7k
```

## License
The corpus is provided under the terms of the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/). By using the corpus you agree to this license.

<img alt="license" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png" width="118" height="41">
