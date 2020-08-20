<p align="center">
<img src = "https://raw.githubusercontent.com/DevoLearn/devolearn/master/images/banner_1.jpg">
</p>

[![Build Status](https://travis-ci.org/DevoLearn/devolearn.svg?branch=master)](https://travis-ci.org/DevoLearn/devolearn)
[![](https://img.shields.io/github/issues/DevoLearn/devolearn)](https://github.com/DevoLearn/devolearn/issues)
[![](https://img.shields.io/github/contributors/DevoLearn/devolearn)](https://github.com/DevoLearn/devolearn/graphs/contributors)
[![](https://img.shields.io/github/last-commit/DevoLearn/devolearn)](https://github.com/DevoLearn/devolearn/commits/master)
[![](https://img.shields.io/twitter/url?color=green&label=Slack&logo=slack&logoColor=blue&style=social&url=https%3A%2F%2Fopenworm.slack.com%2Farchives%2FCMVFU7Q4W)](https://openworm.slack.com/archives/CMVFU7Q4W)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N0jKNYdhDUfCYdx1bA4UI7v8NQ1qPAo0?usp=sharing)


## Contents

* [Example notebooks](https://github.com/DevoLearn/devolearn#example-notebooks)
* [Segmenting the C. elegans embryo](https://github.com/DevoLearn/devolearn#segmenting-the-c-elegans-embryo)
* [Generating synthetic images of embryos with a GAN](https://github.com/DevoLearn/devolearn#generating-synthetic-images-of-embryos-with-a-pre-trained-gan)
* [Predicting populations of cells within the C. elegans embryo](https://github.com/DevoLearn/devolearn#predicting-populations-of-cells-within-the-c-elegans-embryo)
* [Contributing to DevoLearn](https://github.com/DevoLearn/devolearn/blob/master/.github/contributing.md#contributing-to-devolearn)
* [Contact us](https://github.com/DevoLearn/devolearn#contact-us)


### Installation
```python
pip install devolearn
```
### Example notebooks
<code>
<img src = "https://raw.githubusercontent.com/DevoLearn/data-science-demos/master/Networks/nodes_matrix_long_smooth.gif" width = "40%">
<img src = "https://raw.githubusercontent.com/DevoLearn/data-science-demos/master/Networks/3d_node_map.gif" width = "30%">  
</code>

* [Extracting centroid maps and making 3d centroid models](https://nbviewer.jupyter.org/github/DevoLearn/data-science-demos/blob/master/Networks/experiments_with_devolearn_node_maps.ipynb)

### Segmenting the C. elegans embryo 

<img src = "https://raw.githubusercontent.com/DevoLearn/devolearn/master/images/pred_centroids.gif" width = "90%">

* Importing the model
```python
from devolearn import embryo_segmentor
segmentor = embryo_segmentor()

```

* Running the model on an image and viewing the prediction
```python
seg_pred = segmentor.predict(image_path = "sample_data/images/seg_sample.jpg")
plt.imshow(seg_pred)
plt.show()
```

* Running the model on a video and saving the predictions into a folder 
```python
filenames = segmentor.predict_from_video(video_path = "sample_data/videos/seg_sample.mov", centroid_mode = False, save_folder = "preds")
```

* Finding the centroids of the segmented features
```python
seg_pred, centroids = segmentor.predict(image_path = "sample_data/images/seg_sample.jpg", centroid_mode = True)
plt.imshow(seg_pred)
plt.show()
```

* Saving the centroids from each frame into a CSV

```python
df = segmentor.predict_from_video(video_path = "sample_data/videos/seg_sample.mov", centroid_mode = True, save_folder = "preds")
df.to_csv("centroids.csv")
```

### Generating synthetic images of embryos with a Pre-trained GAN
<img src = "https://raw.githubusercontent.com/devoworm/GSoC-2020/master/Pre-trained%20Models%20(DevLearning)/images/generated_embryos_3.gif" width = "30%">

* Importing the model
```python
from devolearn import Generator, embryo_generator_model
generator = embryo_generator_model()

```

* Generating a picture and viewing it with [matplotlib](https://matplotlib.org/)
```python
gen_image = generator.generate()  
plt.imshow(gen_image)
plt.show()

```

* Generating n images and saving them into `foldername` with a custom size

```python
generator.generate_n_images(n = 5, foldername= "generated_images", image_size= (700,500))
```

---

### Predicting populations of cells within the C. elegans embryo


<img src = "https://raw.githubusercontent.com/devoworm/GSoC-2020/master/Pre-trained%20Models%20(DevLearning)/images/resnet_preds_with_input.gif" width = "70%">

*  Importing the population model for inferences 
```python
from devolearn import lineage_population_model
```

* Loading a model instance to be used to estimate lineage populations of embryos from videos/photos.
```python
model = lineage_population_model(mode = "cpu")
```

* Making a prediction from an image
```python
print(model.predict(image_path = "sample_data/images/embryo_sample.png"))
```

* Making predictions from a video and saving the predictions into a CSV file
```python
results = model.predict_from_video(video_path = "sample_data/videos/embryo_timelapse.mov", save_csv = True, csv_name = "video_preds.csv", ignore_first_n_frames= 10, ignore_last_n_frames= 10 )
```

* Plotting the model's predictions from a video
```python
plot = model.create_population_plot_from_video(video_path = "sample_data/videos/embryo_timelapse.mov", save_plot= True, plot_name= "plot.png", ignore_last_n_frames= 0 )
plot.show()
```
## Contact us
### Authors/maintainers:
* [Mayukh Deb](https://twitter.com/mayukh091)
* [Ujjwal Singh](https://twitter.com/ujjjwalll)
* [Dr. Bradly Alicea](https://twitter.com/balicea1)

Feel free to join our [Slack workspace](https://openworm.slack.com/archives/CMVFU7Q4W)!
