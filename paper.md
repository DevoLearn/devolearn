# DevoLearn: Machine Learning Models and Education that Enable Computational Developmental Biology
Mayukh Deb <SUP>1,2</SUP>, Ujjwal Singh <SUP>1,3</SUP>, Mainak Deb <SUP>1,2</SUP>, Bradly Alicea <SUP>1 4</SUP><BR>   

<SUP>1</SUP>OpenWorm Foundation, <SUP>2</SUP>Amrita Vishwa Vidyapeetham University, <SUP>3</SUP>IIIT Delhi, <SUP>4</SUP>Orthogonal Research and Education Lab

## Abstract
We introduce the DevoLearn open-source software, a pre-trained model customized for the analysis of _C. elegans_ developmental processes. The model description, training data, and applications of the model are discussed. We also introduce the DevoLearn platform, an educational resource and software suite for analyzing a wider variety of model organisms. Future directions for the  DevoLearn software and platform conclude this paper. 

## Summary
Extracting metadata from microscopic videos/images have been one of the key steps in the process of finding emerging patterns from various biological processes. There have been many attempts to develop segmentation tools for cell shape and location (Cao, 2019a; Cao, 2019b; Chen, 2013). In particular, cell tracking methodologies provide quantitative summaries of cell centroid positions within an embryo (Ulman, 2006). One goal of the DevoWorm group (https://devoworm.weebly.com/) is to foster open science-friendly interdisciplinary research at the intersection of Developmental Biology, Biological Physics, and Biological Computing. Building upon our previous work on using deep learning to segment microscopy images (OpenDevoCell - https://open-devo-cell.herokuapp.com/), our pre-trained model (Devolearn) aims to speed up the analytical process via software that is easy to use and a pipeline that makes the output easy to interpret. Devolearn’s primary focus is the _Caenorhabditis elegans_ embryo and specifically on the early embryogenesis process. This builds upon desired functionality that was first proposed by the DevoWorm group in (Alicea, 2019). Below are some of the capabilities of the DevoLearn model (Figure 1).

* Segments images/videos of the _C. elegans_ embryo and extract the centroids of the cells and save them into a CSV file.  

* Estimating the population of cells of various lineages within the _C. elegans_ embryo and upon user request generates plots of the data.  

* Generating images of the _C. elegans_ embryo with either a Generative Adversarial Network (GAN) or Feature Pyramid Network (FPN) using a ResNet-18 backbone.  

![](https://user-images.githubusercontent.com/19001437/144554648-2b6f4788-fd72-44d9-81f0-228f2d08bc78.gif)  
__Figure 1.__ Example of Cell Membrane Segmentation using DevoLearn web interface.   

## Accessibility
DevoLearn has been made available as an open-source module, available on PyPI ([https://pypi.org/project/devolearn/](https://pypi.org/project/devolearn/)) and as a Web app (https://devolearn.herokuapp.com). All the deep-learning models implemented in DevoLearn are built and trained on PyTorch. The PyPI package itself does not contain the model weights, but the models are downloaded automatically once the user imports a certain model from the package. 
  
## Statement of Need
Devolearn (0.3.0) is a Python package that aims to automate the process of collecting metadata from videos/images of the _C. elegans_ embryo with the help of deep learning models (Figure 2). This would enable researchers/enthusiasts to analyse features from videos/images at scale without having to annotate their data manually. There are a number of pre-trained models which are already in use in different contexts, but options are fewer within the unique feature space of developmental biology, in particular. Devolearn aims not just to fix this issue, but also work on other aspects around developmental biology with species-specific models.  

![](https://user-images.githubusercontent.com/19001437/144554727-c465079a-2812-4637-8e0a-a339c0453791.jpg)
__Figure 2.__ Schematic demonstrating the runtime procedure of the DevoLearn standalone program.

## Technical Details  
DevoLearn 0.3.0 is optimized to segment and analyze high-resolution microscopy images such as those acquired using light sheet microscopy. The deep learning models used for embryo segmentation and cell lineage population prediction were both based on the ResNet18 architecture. Data from the EPIC dataset (Murray, 2012) was used to train the GAN (beta) and the lineage wise cell population prediction model. The embryo segmentation model was trained on a dataset sourced from Cao (2019b). Data for the hyperparameter tuning training set was acquired from the Cell Tracking Challenge (http://celltrackingchallenge.net/). All datasets used in developing the pre-trained model are shown in Table 1.
  
### Segmenting a _C. elegans_ embryo 
These code examples for importing image data, running the model in DevoLearn, and viewing the results are written in Python. Data can be extracted from video or from standalone microscopy images. Devolearn works best on florescence images or augmented/pre-masked high-resolution microscopy images. The output consists of segmented cell nuclei (Figure 3) with information about the non-normalized _x,y_ position of each identified cell centroid.

__Table 1.__ Links to Datasets
| **Model**                                       | **Data source**                |
|-------------------------------------------------|--------------------------------|
| Segmenting the cell membrane in _C. elegans_ embryo    | [3DMMS: robust 3D Membrane Morphological Segmentation of _C. elegans_ embryo](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2720) |
| Segmenting the nucleus in _C. elegans_ embryo    | [C. elegans Cell-Tracking-Challenge dataset](http://celltrackingchallenge.net/3d-datasets/)   |       
| Cell lineage population prediction + embryo GAN | [EPIC dataset](https://epic.gs.washington.edu/)    |  

In some cases, the nucleus is caught in the act of cell division, or is corrupted by an ambiguous boundary. This can happen for images taken at various focal planes near the dorsal and ventral surfaces of the embryo. In such cases, the user might adjust the threshold to compensate.
  
![](https://user-images.githubusercontent.com/19001437/144554772-7c60baad-8f34-4e5e-9a87-610386c79a57.gif)   
__Figure 3.__ An example of nucleus segmentation in DevoLearn.
  
#### Importing the model
```python
from devolearn import cell_nucleus_segmentor
segmentor = cell_nucleus_segmentor()
```

### Generating Synthetic Images Using GANs   
We also use raw image data as input to a pre-trained Generative Adversarial Network (GAN). To generate such images, you must first import the GAN model, and then generate a picture that can be viewed in Matplotlib.   
  
#### Importing the Model      
from devolearn import Generator, embryo_generator_model
generator = embryo_generator_model()
  
#### Generating a picture, view in Matplotlib
gen_image = generator.generate()  
plt.imshow(gen_image)
plt.show()
generator.generate_n_images(n = 5, foldername= "generated_images", image_size= (700,500))

#### Model prediction and viewing the results
You can view the results of the pre-trained model by executing the following code. Alternately, you can use the web-based GUI to download model results.  
  
```python
seg_pred = segmentor.predict(image_path = "sample_data/images/nucleus_seg_sample.jpg")
plt.imshow(seg_pred)
plt.show()   
```
  
### Hyperparameter Optimization
A training pipeline was built using data from the in order to enable Optuna trials. Training involved image augmentation, which were define using Albumentations. Gaussian noise was also added to the in training images in the augmentation step. During the Optuna trials, optimize the learning rate and batch size to maximize the IOU score. Ran 100 Optuna trials, a single epoch each, on 10% of available data.
  
Optuna is a hyperparameter optimization framework capable of automating the process of hyperparameter tuning. The range of our sampled hyperparameters were as follows: Learning rate: 0.5e-3 to 20e-3, Batch Size: 8 to 64. Each trial trained the model on 10% of available data for three epochs, and returned the resulting IOU score. The hyperparams from the best optuna trial is shown in Figure 4.
  
![](https://user-images.githubusercontent.com/19001437/144554864-abd5c1ac-8ab6-4fcf-b4b3-a677ffaee608.png)    
__Figure 4.__ Training metrics for hyperparameter tuning, from left: IOU scores, Validation (val_dice) Loss, and Learning Rate.

### Lineage Prediction
One advantage of DevoLearn is that _C. elegans_ developmental lineages can be predicting using only a few lines of code. Some results of these predictions are demonstrated in Figure 3. We utilize our own lineage population model, which is based on well-established cell identity annotations. This model makes a prediction from an image and saves the predictions in a CSV file. Additionally you can plot the model's predictions to check their integrity.

```from devolearn import lineage_population_model
model = lineage_population_model(device = "cpu")
print(model.predict(image_path = "sample_data/images/embryo_sample.png"))
results = model.predict_from_video(video_path = "sample_data/videos/embryo_timelapse.mov", save_csv = True, csv_name = "video_preds.csv", ignore_first_n_frames= 10, ignore_last_n_frames= 10, postprocess = False)
plot = model.create_population_plot_from_video(video_path = "sample_data/videos/embryo_timelapse.mov", save_plot= True, plot_name= "plot.png", ignore_last_n_frames= 0, postprocess = False)
plot.show()   
```
  
### Meta-feature Detection
DevoLearn is also capable of extracting _meta-features_ that identify movement patterns and multicellular physics in the embryogenetic environment. Examples of this include embryo networks (Alicea and Gordon, 2018) and motion features. The former capability involves extracting potential structural and functional networks using distance metrics and other information extracted from microscopy images. Motion features can also be extracted and can be used for a variety of purposes, including as a means to build generative adversarial network (GAN) models (Goodfellow, 2014). Feature Pyramid Networks (FPNs) enable semantic feature maps (Lin et.al, 2016), which can also be used to approach the identity of anatomical and other biological features in new ways.

Devolearn has been built to be very flexible with respect to exploiting the latest in contemporary deep learning algorithms, but also provides graphical and numerical outpyt that is anemable to novel data science techniques. and to be DevoLearn is highly compatible with libraries such as NumPy (Harris, 2020) and Pandas (Virtanen, 2020) As the Devolearn framework (Figure 5) grows bigger with more tools and deep learning models, the combination of beginner friendliness and support for data science functionality will enable exciting scientific explorations both in developmental biology and data science.   

## DevoLearn platform
The DevoLearn PyPI package is a part of the DevoLearn Github organization (https://github.com/devolearn), which serves as a comprehensive open-source research and educational resource. This platform consists of a model library that has more general Machine Learning models for a wider range of model organisms, theory-building activities, and data science tutorials submitted by different contributors. It aims to provide users with Data Science tutorials, web-based applications that offer other Deep Learning and Machine Learning tools for cell segmentation, and other educational resources.  We invite new collaborators to join us on a continual basis in maintaining and expanding the capabilities of the DevoLearn organization.  
  
![](https://user-images.githubusercontent.com/19001437/144554923-f24ddf9e-4ce8-4927-a8e8-b2035c1a609f.png)     
__Figure 5.__ Schematic of the DevoLearn Umbrella, which includes the DevoLearn standalone program and the DevoLearn framework.  
  
## Future Directions  
DevoLearn has been developed and improved upon over several model improvements and benchmarking exercises. This has included improvements to the main image segmentation algorithm, which utilizes various deep learning approaches to identify cell boundaries, asymmetries and complex geometries, and the emergence of dividing cells. We plan on building upon these techniques with a focus on defined semantic segmentation techniques and training on a wider variety of datasets to xharacterize this single mode of development in a single species. 
  
One drawback of using _C. elegans_ as a model organism is that its developmental dynamics are restricted to a deterministic set of cell division events. Using other species to train the main Devolearn model in addition to integration with other models is critical. As the DevoLearn platform already includes some of these alternative models, integration of the user interface is key. We also look to integrate the auxilliary resource of DevoLearn into this effort, including the DevoZoo resource, which provides learners with sample data for a number of model organisms from throughout the Eukaryotic domain of life.  

## Acknowledgements
We would like to thank the OpenWorm Foundation, the International Neuroinformatics Coordinating Facility (INCF), and Google Summer of Code (GSoC) for their financial and institutional support. Gratitude also goes to the DevoWorm group for their expertise and feedback. 

## References
Alicea, B., Gordon, R., Kohrmann, A., Parent, J., and Varma, V. (2019). Pre-trained Machine Learning Models for Developmental Biology. _The Node blog_, October 29. doi:10.6084/m9.figshare.13347095.

Alicea, B. and Gordon R. (2018). Cell Differentiation Processes as Spatial Networks: identifying four-dimensional structure in embryogenesis. _BioSystems, 173_, 235-246.  

Cao, J., Guan, G., Wong, M-K., Chan, L-Y., Tang, C., Zhao, Z., and Yan, H. (2019a). Establishment of morphological atlas of _Caenorhabditis elegans_ embryo with cellular resolution using deep-learning-based 4D segmentation. _bioRxiv_, doi:10.1101/797688.

Cao, J., Wong, M-K., Zhao, Z., and Yan, H. (2019b). 3DMMS: robust 3D Membrane Morphological Segmentation of _C. elegans_ embryo. _BMC Bioinformatics_, 20, 176.

Chen, L., Hang Chan, L.L., Zhao, Z., and Yan, H. (2013). A novel cell nuclei segmentation method for 3D _C. elegans_ embryonic time-lapse images. _BMC Bioinformatics_, 14, 328.

Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative Adversarial Networks. _arXiv_, 1406.2661.

Hall, D.H. and Altun, Z.F. (2008). _C. elegans_ Atlas. Cold Spring Harbor Labratory Press, Woodbury, NY.

Harris, C.R. (2020). Array programming with NumPy. _Nature_, 585, 357-362.

Lin, T-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. (2016). Feature Pyramid Networks for Object Detection. _arXiv_, 1612.03144.
  
Murray, J.I., Boyle, T.J., Preston, E., Vafeados, D., Mericle, B., Weisdepp, P., Zhao, Z., Bao, Z., Boeck, M., and Waterston, R.H. (2012). Multidimensional regulation of gene expression in the _C. elegans_ embryo. _Genome Research_, 22(7), 1282–1294.

Ulman, V. et.al (2017). An objective comparison of cell-tracking algorithms. _Nature Methods_, 14(12), 1141–1152.

Virtanen, P. et.al (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. _Nature Methods, 17_, 261-272.
