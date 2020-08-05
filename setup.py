import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="develearn",
    version="0.0.1",
    author="Mayukh Deb, Ujjwal Singh, Bradley Alicea", #To be decided, also there can be name of "DevoWorm | OpenWorm Foundation"
    author_email="mayukhmainak2000@gmail.com, ujjwal18113@iiitd.ac.in, balicea@openworm.org", #Subject to change, we can also use Devolearn official Email address.
    description="Accelerate data driven research on embryos with Pre-Trained deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DevoLearn/devolearn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],




    #Mayukh, please run pip freeze command to get all the dependencies required and any if missing. Also delete this comment once done.

    install_requires=[
          "Pillow",
          "joblib",
          "matplotlib",
          "numpy",
          "opencv-python",
          "pandas",
          "torch",
          "torchvision",
          "tqdm"
          "segmentation-models-pytorch"
      ],
    python_requires='>=3.6',      
)