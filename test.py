from devolearn import lineage_population_model
from devolearn import Generator, embryo_generator_model
from devolearn import embryo_segmentor

import matplotlib.pyplot as plt
import os 

model = lineage_population_model(mode = "cpu")

print(model.predict(image_path = "sample_data/images/embryo_sample.png"))

if os.path.isdir("sample_preds") is not True:
    os.mkdir("sample_preds")

results = model.predict_from_video(video_path = "sample_data/videos/embryo_timelapse.mov", save_csv = True, csv_name = "sample_preds/video_preds.csv", ignore_first_n_frames= 10, ignore_last_n_frames= 10 )
                                    
plot = model.create_population_plot_from_video(video_path = "sample_data/videos/embryo_timelapse.mov", save_plot= True, plot_name= "sample_preds/plot.png", ignore_last_n_frames= 0 )
plot.grid()
plot.show()


generator = embryo_generator_model()
gen_image = generator.generate()  ## 2d numpy array 
plt.imshow(gen_image)
plt.show()

generator.generate_n_images(n = 5, foldername= "generated_images", image_size= (700,500))

segmentor = embryo_segmentor()

seg_pred = segmentor.predict(image_path = "sample_data/images/seg_sample.jpg", centroid_mode =False )
seg_pred, centroids = segmentor.predict(image_path = "sample_data/images/seg_sample.jpg", centroid_mode = True)

plt.imshow(seg_pred)
plt.show()

filenames = segmentor.predict_from_video(video_path = "sample_data/videos/seg_sample.mov", centroid_mode = False, save_folder = "preds")

df = segmentor.predict_from_video(video_path = "sample_data/videos/seg_sample.mov", centroid_mode = True, save_folder = "preds")

df.to_csv("centroids.csv")