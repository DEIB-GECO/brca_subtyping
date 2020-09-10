# Investigating Deep Learning based Breast Cancer Subtyping using Pan-cancer and Multi-omic Data
Breast Cancer comprises multiple molecular subtypes with great implications on disease prognosis and outcome. Existing stratification methods rely on the expression quantification of only relatively small gene sets.
Next Generation Sequencing has established itself as a leading technology for analysing genome-wide features of an individual and promises to produce large amounts of data in the next years.
In this scenario, we explore the potential of machine learning and, in particular, deep learning techniques in breast cancer patient stratification. Due to the paucity of publicly available data, we leverage on pan-cancer data and also on non-cancer data to train our models in a semi-supervised setting. We make use of multi-omic data, including microRNA expressions and somatic copy number alterations, to better characterize individual patients. We provide an in-depth investigation of several supervised and semi-supervised methods and architectures for breast cancer subtyping.
Obtained results show simpler machine learning models to perform at least as well as (but not necessarily better than) the deep semi-supervised approaches on our task over RNA-seq gene expression data (in terms Accuracy). When multi-omic data types are combined together, performance of deep learning models show little (if any) improvement in terms of prediction accuracy, indicating the need for further analysis on larger datasets of multi-omic data as and when they become available.

## Source Code
The source code used for the experiments can be found under the "src" directory (https://github.com/DEIB-GECO/brca_subtype/tree/master/src), with the model classes and scripts to train the models. 
Under "notebooks" one can find several data exploration examples and standalone model experiments for quick prototyping (https://github.com/DEIB-GECO/brca_subtype/tree/master/notebooks).

## Funding
This research is funded by the ERC Advanced Grant project 693174 GeCo (Data-Driven Genomic Computing), 2016-2021.
