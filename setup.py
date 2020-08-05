import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="devolearn",
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

    install_requires=[
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "scikit-learn",
        "scipy",
        "sklearn",
        "matplotlib",
        "numpy",
        "pretrainedmodels",
        "opencv-python",
        "pandas",
        "tqdm",
        "cycler",
        "efficientnet-pytorch",
        "future",
        "imutils",
        "joblib",
        "kiwisolver",
        "munch",
        "Pillow",
        "pyparsing",
        "python-dateutil",
        "pytz",
        "six",
        "threadpoolctl"
      ],
    python_requires='>=3.6',      
)
