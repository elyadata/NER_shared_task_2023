# ELYADATA Systems for Arabic NER Shared Task 2023

This repository contains the work carried out by [ELYADATA](https://www.elyadata.com/) for the submission to the [NER shared task of 2023](https://dlnlp.ai/st/wojood/) 
in the context of the first Arabic NLP conference, [ArabicNLP 2023](https://arabicnlp2023.sigarab.org/home).

There is also the `Data_preprocessing` folder which contains the re-sampling experiments, the data cleaning scripts,
and also the scripts for data preparation that convert the Wojood text files from an IOB format, to a format compatible 
with each model.

Detailed instructions on how to train the models and perform data-preprocessing are available in the `readme.md` files 
of each folder.

If you use this code please cite our paper :

```bibtex
@inproceedings{DBLP:conf/wanlp/LaouirineEB23,
  author       = {Imen Laouirine and Haroun Elleuch and Fethi Bougares},
  title        = {{ELYADATA} at WojoodNER Shared Task: Data and Model-centric Approaches for Arabic Flat and Nested {NER}},
  booktitle    = {Proceedings of ArabicNLP 2023, Singapore (Hybrid), December 7, 2023},
  pages        = {759--764},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.arabicnlp-1.84},
}
```
