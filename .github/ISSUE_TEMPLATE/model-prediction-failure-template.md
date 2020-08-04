---
name: Model prediction failure template
about: inform us about cases where our deep learning models failed to predict correctly
  in your images/videos
title: ''
labels: ''
assignees: ''

---

**Specify the model that failed**
For example: `lineage_population_model()`

**Details about the bad prediction**

1. URL to the input data that failed (you can also email it to the authors/maintainers)
2. A screenshot of the bad prediction (it its a csv file or an image folder, then email it to us or give us a link to it )
3. [optional] Possible reason of failure  (for example: the model did not train on images with random white patches, but the input images has random white patches )
4. [optional] The error might just be because of some error in your code, so a snippet of your  code might help as well.


** Environment :**
 - OS/hosted runtime : [e.g. macOS/Windows/Linux or Google Colab, etc]
 - Version [e.g. 0.0.1]

**Additional context**
Add any other context about the problem here.
