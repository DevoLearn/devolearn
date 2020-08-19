import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="devolearn",
    version="0.2.0",
    author="Mayukh Deb, Ujjwal Singh, Bradly Alicea", 
    author_email="mayukhmainak2000@gmail.com, ujjwal18113@iiitd.ac.in, balicea@openworm.org", #Subject to change, we can also use Devolearn official Email address.
    description="Accelerate data driven research on embryos with deep learning models",
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
        "torch>=1.6.0",
        "torchvision",
        "pretrainedmodels>=0.7.4",
        "segmentation-models-pytorch",
        "efficientnet-pytorch>=0.6.3",
        "opencv-python",
        "imutils>=0.5.3",
        "scikit-learn",
        "scipy",
        "numpy",
        "matplotlib",
        "pandas",
        "sklearn>=0.0",
        "cycler>=0.10.0",
        "joblib>=0.16.0",
        "kiwisolver>=1.2.0",
        "munch>=2.5.0",
        "pyparsing>=2.4.7",
        "python-dateutil>=2.8.1",
        "six>=1.15.0",
        "wget"
      ],
    python_requires='>=3.6',   
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose']   
)