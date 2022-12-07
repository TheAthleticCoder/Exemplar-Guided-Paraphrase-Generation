# **Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation**

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[![forthebadge](https://forthebadge.com/images/badges/powered-by-coders-sweat.svg)](https://forthebadge.com)

-----

**Contributors:** 
1. Harshit Gupta 
2. [Pratyaksh Gautam](https://github.com/hi-im-buggy)

----

## *Objective:* 
The goal of Exemplar-Guided Paraphrase Generation (EGPG) is to produce a target sentence
that matches the style of the provided exemplar while preserving the source sentence's content
information. \
In an effort to learn a better representation of the style and the substance, this study makes a
novel approach suggestion. The recent success of contrastive learning, which has proven its
effectiveness in unsupervised feature extraction tasks, is the key driving force behind this
approach. \
Designing two contrastive losses with regard to the content and style while taking into
account two problem features during training is the idea.

**Paper Citation:** https://arxiv.org/pdf/2109.01484.pdf

-----

## *Datasets Used:*

In order to train and assess our models, we use two datsets. As follows:
1. **ParaNMT Dataset:** Using back translation of the original English sentences from a different
challenge, they were created automatically.
2. **QQPos:** Compared to the dataset above, the QQPos Dataset is more formal. \
We use 93k sentences for training, 3k sentences for validation and 3k sentences for testing from
both the datasets each

-----
## *File Structure:*

1. `contrastive_loss.py` Vectorized and Optimized code for style and content loss.
2. `exemplar_gen.py` Code for generating exemplar sentences using the two datasets.
3. `final_model.py` Full detailed implementation of the model and other relevant functions in Pytorch.
4. `model_nll_loss.py` Code of model with NLL Loss only.
5. `paranmt-txt-to-csv.py` Code to convert txt file to csv format (you can handle the datasets as you wish).
6. `Project Report.pdf` Final report and analysis of the results and metrics. Also qualitative analysis of generated sentences on the test set.
7. `Project Presentation.pdf` Slides dispalying the methodology and results. Similar to project report.

-----


