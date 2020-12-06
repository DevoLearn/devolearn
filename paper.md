--  
title: 'Pre-trained Models That Enable Computational Developmental Biology'  
tags:    
- Python
- pre-trained Models
- deep learning
- cell segmentation
date: "December 6, 2020"  
output:
  pdf_document: default
  html_document:
    df_print: paged
  word_document: default
authors:  
- name: Mayukh Deb   
  affiliation: 1, 2 
- name: Ujjwal Singh  
  affiliation: 1, 3
- name: Bradly Alicea  
  orcid: 0000-0003-3869-3175  
  affiliation: 1, 4
bibliography: paper.bib  
affiliations:  
- name: OpenWorm Foundation    
   index: 1  
- name: Amrita Vishwa Vidyapeetham University  
   index: 2 
- name: IIIT Delhi    
   index: 3 
- name: Orthogonal Research and Education Lab  
   index: 4  
---  

# Summary
Extracting metadata from microscopic videos/images have been one of the key steps in the process of finding emerging patterns from various biological processes. There have been many attempts to develop segmentation tools for cell shape and location [@Cao_1:2019; Cao_2:2019; Chen:2013]. In particular, cell tracking methodologies provide quantitative summaries of cell centroid positions within an embryo [@Ulman:2006]. Our pre-trained model (Devolearn) aims to speed up this process of collecting metadata by using robust deep learning models that can be used through a high level API. Devolearn’s primary focus is the _Caenorhabditis elegans_ embryo and specifically on the early embryogenesis process. This builds upon desired functionality that was first proposed by the DevoWorm group in [@Alicea:2019]. Below are some of the capabilities of the DevoLearn model.

* **Segments images/videos of the _C. elegans_ embryo** and extract the centroids of the cells and save them into a CSV file.  

* **Estimating the population of cells of various lineages within the _C. elegans_ embryo** and upon user request generates plots of the data.  

* **Generating images of the _C. elegans_ embryo with a Generative Adversarial Network (GAN) (beta)** and provides support for extracting so-called _meta-features_. The current version of DevoLearn (0.2.0) also supports bulk generation of images.  

DevoLearn has been made available as an open-source module, available on PyPi ([link](https://pypi.org/project/devolearn/)). All the deep-learning models used in devolearn are built and trained on PyTorch, the PyPI package (https://pypi.org/project/devolearn/) itself does not contain the model files, but the models are downloaded automatically once the user imports the model class from the package. 

## Technical Details  
DevoLearn 0.2.0 is optimized to segment and analyze high-resolution microscopy images such as those acquired using light sheet microscopy. Additionally, the possibility of time-series prediction allows us to capture features related to cellular- and organism-level movement. DevoLearn 0.2.0 uses a ResNet18 architecture to build a pre-trained model of the shape features structure inherent in a microscopy image. Data from the WormImage database [@Hall_and_Altun:2008] is used to train the model for _C. elegans_ embryogenesis. The EPIC data set [@Murray:2012] is used to extract metadata needed to construct individual cells labels used to validate input data. 

## Statement of Need
Devolearn (0.2.0) is a Python package that aims to automate the process of collecting metadata from videos/images of the _C. elegans_ embryo with the help of deep learning models \autoref{fig:1}. This would enable researchers/enthusiasts to analyse features from videos/images at scale without having to annotate their data manually. There are a number of pre-trained models which are already in use in different contexts, but options are fewer within the unique feature space of developmental biology, in particular. Devolearn aims not just to fix this issue, but also work on other aspects around developmental biology with species-specific models.  

![Schematic demonstrating the runtime procedure of the DevoLearn standalone program.\label{fig:1}](https://user-images.githubusercontent.com/19001437/101274836-eef29800-3766-11eb-8001-e64f5a4ca66a.png)
  
DevoLearn is also capable of extracting _meta-features_ that identify movement patterns and multicellular physics in the embryogenetic environment. Exampoles of this include embryo networks [@Alicea_and_Gordon:2018] and motion features. The former capability involves extracting potential structural and functional networks using distance metrics and other information extracted from microscopy images. Motion features can also be extracted and can be used for a variety of purposes, including as a means to build generative adversarial network (GAN) models [@Goodfellow:2014]

Devolearn has been built to be very data science friendly and to be highly compatible with libraries like NumPy [@Harris:2020] and Pandas [@Virtanen:2020] As the Devolearn framework \autoref{fig:2} grows bigger with more tools and deep learning models, the combination of beginner friendliness and support for data science functionality will enable exciting scientific explorations both in developmental biology and data science.   

![Schematic of the DevoLearn Umbrella, which includes the DevoLearn standalone program and the DevoLearn framework.\label{fig:2}](https://user-images.githubusercontent.com/19001437/101274845-03cf2b80-3767-11eb-9541-bc549f697dbb.png)

The DevoLearn pre-trained models is also a part of the [DevoLearn Github organization](https://github.com/devolearn), which serves as a comprehensive open-source research and educational resource. DevoLearn not only features the DevoLearn pre-trained models, but also provides users with Data Science tutorials, web-based applications that offer other Deep Learning and Machine Learning tools for cell segmentation, and other educational resources.  We invite new collaborators to join us on a continual basis in maintaining and expanding the capabilities of the DevoLearn organization.  

# Acknowledgements
We would like to thank the OpenWorm Foundation, the International Neuroinformatics Coordinating Facility (INCF), and Google Summer of Code for their financial and institutional support. Gratitude also goes to the DevoWorm group for their expertise and feedback. 

# References
