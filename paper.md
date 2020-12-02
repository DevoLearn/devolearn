---  
title: 'Pre-trained Deep-Learning Models That Enable Computational Developmental Biology Research and Education'  
tags:  
  - Python  

authors:  
  - name: Mayukh Deb  
    orcid: ---  
    affiliation: 1,2 
  - name: Ujjwal Singh  
    orcid: ---  
    affiliation: 1 ,3
  - name: Bradly Alicea  
    orcid: 0000-0003-3869-3175  
    affiliation: 1 ,4
    
affiliations:  
 - name: OpenWorm Foundation  
   index: --  
 - name: Amrita Vishwa Vidyapeetham University 
   index: -- 
 - name: IIIT Delhi  
   index: -- 
  - name: Orthogonal Research and Education Lab
   index: -- 
 date: November 2020 
bibliography: devolearn-joss.bib  
---  

# Summary
Extracting metadata from microscopic videos/images have been one of the key steps in the process of finding emerging patterns from various biological processes. There have been many attempts to develop segmentation tools for cell shape and location [1-3]. In particular, cell tracking methodologies provide quantitative summaries of cell centroid positions within an embryo [4]. Our pre-trained model (Devolearn) aims to speed up this process of collecting metadata by using robust deep learning models that can be used through a high level API. Devolearn’s primary focus is the _Caenorhabditis elegans_ embryo and specifically on the early embryogenesis process. This builds upon desired functionality that was first proposed by the DevoWorm group in [5]. Below are some of the capabilities of the DevoLearn model.

* **Segments images/videos of the _C. elegans_ embryo** and extract the centroids of the cells and save them into a CSV file.  

* **Estimating the population of cells of various lineages within the _C. elegans_ embryo** and upon user request generates plots of the data.  

* **Generating images of the _C. elegans_ embryo with a Generative Adversarial Network (GAN) (beta)** and provides support for extracting so-called _meta-features_. The current version of DevoLearn (0.2.0) also supports bulk generation of images.  

DevoLearn has been made available as an open-source module, available on PyPi ([link](https://pypi.org/project/devolearn/)). All the deep-learning models used in devolearn are built and trained on PyTorch, the PyPI package (https://pypi.org/project/devolearn/) itself does not contain the model files, but the models are downloaded automatically once the user imports the model class from the package. 

## Technical Details  
DevoLearn 0.2.0 is optimized to segment and analyze high-resolution microscopy images such as those acquired using light sheet microscopy. Additionally, the possibility of time-series prediction allows us to capture features related to cellular- and organism-level movement. DevoLearn 0.2.0 uses a ResNet18 architecture to build a pre-trained model of the shape features structure inherent in a microscopy image. Data from the WormImage database [6] is used to train the model for _C. elegans_ embryogenesis. The EPIC data set [7] is used to extract metadata needed to construct labels for individual cells and validate the input data. 

## Statement of Need
Devolearn (0.2.0) is a python package that aims to automate the process of collecting metadata from videos/images of the _C. elegans_ embryo with the help of deep learning models (Figure 1). This would enable researchers/enthusiasts to analyse features from videos/images at scale without having to annotate their data manually. There are a number of pre-trained models which are already in use in different contexts, but options are fewer within the unique feature space of developmental biology, in particular. Devolearn aims not just to fix this issue, but also work on other aspects around developmental biology with species-specific models.  

<P>
<CENTER>
  <IMG SRC="https://github.com/DevoLearn/Education/blob/master/DevoLearn%20Schematic.png">
</CENTER>
</P>

**Figure 1.** Schematic demonstrating the runtime procedure of the DevoLearn standalone program.  
  
DevoLearn is also capable of extracting _meta-features_ that identify movement patterns and multicellular physics in the embryogenetic environment. Exampoles of this include embryo networks [8] and motion features. The former capability involves extracting potential structural and functional networks using distance metrics and other information extracted from microscopy images. Motion features can also be extracted and can be used for a variety of purposes, including as a means to build generative adversarial network (GAN) models [9].

Devolearn has been built to be very data science friendly and to be highly compatible with libraries like NumPy [10] and Pandas [11]. As the Devolearn framework (Figure 2) grows bigger with more tools and deep learning models, the combination of beginner friendliness and support for data science functionality will enable exciting scientific explorations both in developmental biology and data science.   

<P>
<CENTER>
  <IMG SRC="https://github.com/DevoLearn/Education/blob/master/DevoLearn%20Umbrella.png">
</CENTER>
</P>

**Figure 2.** Schematic of the DevoLearn Umbrella, which includes the DevoLearn standalone program and the DevoLearn framework.  

The DevoLearn pre-trained models is also a part of the [DevoLearn Github organization](https://github.com/devolearn), which serves as a comprehensive open-source research and educational resource. DevoLearn not only features the DevoLearn pre-trained models, but also provides users with Data Science tutorials, web-based applications that offer other Deep Learning and Machine Learning tools for cell segmentation, and other educational resources.  We invite new collaborators to join us on a continual basis in maintaining and expanding the capabilities of the DevoLearn organization.  

# Acknowledgements
We would like to thank the OpenWorm Foundation, the International Neuroinformatics Coordinating Facility (INCF), and Google Summer of Code for their financial and institutional support. Gratitude also goes to the DevoWorm group for their expertise and feedback.

# References
[1] Cao, J., Guan, G., Wong, M-K., Chan, L-Y., Tang, C., Zhao, Z., and Yan, H. (2019). Establishment of morphological atlas of _Caenorhabditis elegans_ embryo with cellular resolution using deep-learning-based 4D segmentation. _bioRxiv_, doi:10.1101/797688.  

[2] Cao, J., Wong, M-K., Zhao, Z., and Yan, H. (2019). 3DMMS: robust 3D Membrane Morphological Segmentation of _C. elegans_ embryo. _BMC Bioinformatics_, 20, 176.  

[3] Chen, L., Hang Chan, L.L., Zhao, Z., and Yan, H. (2013). A novel cell nuclei segmentation method for 3D _C. elegans_ embryonic time-lapse images. _BMC Bioinformatics_, 14, 328.  

[4] Ulman, V. et.al (2017). An objective comparison of cell-tracking algorithms. _Nature Methods_, 14(12), 1141–1152.  

[5] Alicea, B. et.al (2019). Pre-trained Machine Learning Models for Developmental Biology. _The Node blog_, October 29. https://thenode.biologists.com/pre-trained-machine-learning-models-for-developmental-biology/uncategorized/  

[6] Murray, J.I., Boyle, T.J., Preston, E., Vafeados, D., Mericle, B., Weisdepp, P., Zhao, Z., Bao, Z., Boeck, M., and Waterston, R.H. (2012). Multidimensional regulation of gene expression in the _C. elegans embryo_. _Genome Research_, 22(7), 1282–1294.

[7] Hall, D.H. and Altun, Z.F. (2008). _C. elegans_ Atlas. Cold Spring Harbor Labratory Press, Woodbury, NY.

[8] Alicea, B. and Gordon R. (2018). Cell Differentiation Processes as Spatial Networks: identifying four-dimensional structure in embryogenesis. _BioSystems_, 173, 235-246.   

[9] Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative Adversarial Networks. _arXiv_, 1406.2661.  

[10] Harris, C.R. (2020). Array programming with NumPy. _Nature_, 585, 357-362.  

[11] Virtanen, P. et.al (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. _Nature Methods_, 17, 261-272.    
