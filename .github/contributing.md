# Contributing to devolearn

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to devolearn and its packages, which are hosted in the [DevoLearn](https://github.com/DevoLearn) organization on GitHub. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[I don't want to read this whole thing, I just have a question!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [What do the devolearn models do ?](#what-do-the-devolearn-models-do-)
  * [How can I add my own model to devolearn ?](#how-can-i-add-my-own-model-to-devolearn-)
  * [Making a code contribution](#making-a-code-contribution)

[Suggesting a model](#suggesting-a-model)

[Git Commit Messages](#git-commit-messages)
  
  ## I don't want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below:

* Join the [devolearn slack](https://openworm.slack.com/archives/CMVFU7Q4W) 
* Try contacting the [contributors/authors](https://github.com/DevoLearn/devolearn/graphs/contributors) we're pretty active on github !

## What should I know before I get started?

### What do the devolearn models do ? 

The models at devolearn can be used to collect various types of useful data from videos/images of embryos which can be used for statistical analysis. Or if you're feeling crafty, you can use that metadata for another deep learning model.

At devolearn, we want to create a collection of robust deep learning models which can be used very easily for research and analysis of various biological processes. 

### How can I add my own model to devolearn ?

The models on devolearn are all based on [PyTorch](https://pytorch.org/), and we'd highly prefer it if your model is PyTorch based as well. 

If you already have a PyTorch model ready, it's time to have a good look at the [devolearn folder tree](https://github.com/DevoLearn/devolearn/tree/master/devolearn)

After adding your new model, it might look like (this is just an example on how to distribute your `.py` file and your `.pt` model):

<pre>
.
├── embryo_generator_model
│   ├── embryo_generator_model.py
|   ├── embryo_generator.pt
│   ├── __init__.py
│        
├── <b>your_model_name</b>
│   ├── <b>your_model_name</b>.py   ## should contain the model class and all of it's functions 
|   ├── <b> your_model_name</b>.pt  ## the trained model 
│   ├── __init__.py 
|
├── __init__.py

</pre>

### Making a code contribution
For this, keep an eye out on the [issues](https://github.com/DevoLearn/devolearn/issues) page, and if you're a beginner, feel free to contact us on [slack](https://openworm.slack.com/archives/CMVFU7Q4W) 

## Suggesting a model

We're always open to new ideas and suggestions that might help us build new and better models for research. If you have an idea, you're always welcome to talk to us on [slack](https://openworm.slack.com/archives/CMVFU7Q4W). Here are the three four points that we'd like to know the most if it's a model that doesn't exist on devolearn already:
* Purpose of the model (classification, estimation, Image segmentation, time series prediction, etc)
* Possible data sources 
* Input and output formats (e.g input = image of an embryo, output = list of populations of various cells within the embryo)
* A brief description on how the model would help in research

##  Git commit messages 

* Use the present tense ("Add feature" not "Added feature")
* If commit messages are new to you, [this would be helpful](https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/)

If you don't want to read all of that, you can just write something like `filename: add function xyz()`

A real example would be like: `readme: add images` 

