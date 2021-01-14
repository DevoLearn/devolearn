import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="devolearn",
    version="0.2.1",
    author="Mayukh Deb, Ujjwal Singh, Bradly Alicea", 
    author_email="mayukhmainak2000@gmail.com, ujjwal18113@iiitd.ac.in, balicea@openworm.org", #Subject to change, we can also use Devolearn official Email address.
    description="Accelerate data driven research in developmental biology with deep learning models",
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
        "cycler==0.10.0",
        "efficientnet-pytorch==0.6.3",
        "imutils==0.5.3",
        "joblib==1.0.0",
        "kiwisolver==1.3.1",
        "matplotlib==3.3.3",
        "munch==2.5.0",
        "numpy==1.19.5",
        "opencv-python==4.5.1.48",
        "pandas==1.2.0",
        "Pillow==8.1.0",
        "pretrainedmodels==0.7.4",
        "pyparsing==2.4.7",
        "python-dateutil==2.8.1",
        "pytz==2020.5",
        "scikit-learn==0.24.0",
        "scipy==1.6.0",
        "segmentation-models-pytorch==0.1.3",
        "six==1.15.0",
        "sklearn==0.0",
        "threadpoolctl==2.1.0",
        "timm==0.3.2",
        "torch==1.7.1",
        "torchvision==0.8.2",
        "tqdm==4.56.0",
        "typing-extensions==3.7.4.3",
        "wget==3.2"
      ],
    python_requires='>=3.6',   
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose']   
)