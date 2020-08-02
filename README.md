# pyelegans 

<img src = "images/py_elegans_vis.png">

> Still under construction

# Generating synthetic images of embryos with a Pre-trained GAN

* Importing the model
```python
from pyelegans import Generator, embryo_generator_model
```

* Generating a picture
```python
generator = embryo_generator_model()
gen_image = generator.generate()  

```
* Viewing the generated image
```
plt.imshow(gen_image)
plt.show()
```

## Bulk generation using the GAN
* generating n images and saving them into `foldername` with a user set size 

```python
generator.generate_n_images(n = 5, foldername= "generated_images", image_size= (700,500))
```

---

# Predicting populations of cells within the C. elegans embryo

*  Importing the population model for inferences 
```python
from pyelegans import lineage_population_model
```

* Loading a model instance to be used to estimate lineage populations of embryos from videos/photos.
```python
model = lineage_population_model(mode = "cpu")
```

* Making a prediction from an image
```python
pred = model.predict(image_path = "sample.png")
```

* Making predictions from a video and saving the predictions into a CSV file
```python
results = model.predict_from_video(video_path = "sample_videos/20071217_ceh-432x3.mov", save_csv = True, csv_name = "foo.csv")
```

* Plotting the model's predictions
```python
plot = model.create_population_plot_from_video(video_path = "sample_data/sample_videos/20071217_ceh-432x3.mov", save_plot= True, plot_name= "images/plot.png", ignore_last_n_frames= 30 )
plot.show()
```
This would show a plot like:

<img src= "sample_preds/plot.png" width = "40%">
