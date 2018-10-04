# DeepTACT
DeepTACT is a bootstrapping deep learning model, which integrates genome sequences and chromatin accessibility data for the prediction of chromatin contacts among regulatory elements.

# Data preprocessing
Promoter capture Hi-C data of total B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8) are derived from https://osf.io/u8tzp/ (Javierre et al. 2016). Labeled training pairs of these cell lines are available in the DeepTACT/ directory, where each cell line has its own subdirectory. For each cell line, we train and test models for promoter-promoter interactions (P-P) and promoter-enhancer interactions (P-E) separately. The sequence information of each interaction pair is extracted from hg19.fa. The openness signals are calculated from DNase-seq data. 

- Data augmentation. In each 'pairs.csv' file, we give all positive pairs and the augmented negative pairs. With DeepTACT/DataPrepare.py, you can augment positive training pairs, obtain balanced training data and imbalanced testing data for each cell line. Simply run
```
python DataPrepare.py Mon P-E
```

- Bootstrapping. With DeepTACT/Bootstrapping.py, we bootstrap each original dataset 20 times for ensemble learning. We give an example of the inputs for bootstrapping in DeepTACT/demo directory. When running 'Bootstrapping.py', you need to specify the directory of inputs, the type of interactions, and the number of DNase-seq experiments that are used to provide openness signals. Using data in demo directory as an example, run 
```
python Bootstrapping.py demo P-E 3
```

# Training and evaluation
We implemented the DeepTACT model using Keras 1.2.0 on a Linux server. All experiments were carried out with 4 Nvidia K80 GPUs which significantly accelerated the training process than CPUs. We provide examples of sequences and DNase inputs in the DeepTACT/demo directory. If you want to train your own model with DeepTACT, you can simply substitute your data to DeepTACT/demo with the same format. To train a model, you can run
```
python DeepTACT.py demo P-E 3
```
We evaluate the ensemble model with a voting strategy. Given the information of a sample as an input, its final prediction score is the average of the prediction scores derived from all classifiers.

# Improve the resolution of PCHi-C data
We apply the trained DeepTACT model to infer contacts between regulatory elements in situations where one or both interaction regions contain multiple regulatory elements. In this way, we predict promoter-level interactions from bin-level interactions. For each cell line, the promoter-level interactions are predicted and saved in 'predictions.csv' files (e.g. DeepTACT/Mon/P-E/predictions.csv).

# Requirements
- hickle
- numpy=1.13.3
- Theano=0.8.0
- keras=1.2.0
- pandas=0.20.1
- biopython=1.70
- Scikit-learn=0.18.2

# Installation
Download DeepTACT by
```shell
git clone https://github.com/liwenran/DeepTACT
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
