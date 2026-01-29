from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="tbsim",
    packages=[package for package in find_packages() if package.startswith("tbsim")],
    install_requires=[
        "l5kit==1.5.0",
        "tianshou",
        "pytorch-lightning==2.0",
        "wandb",
        "pyemd",
        "h5py",
        "imageio-ffmpeg",
        "casadi",
        "protobuf==3.20.1", # new version might cause error
        "einops==0.6.0",
        "torchtext", 
        "imageio",
        "pymap3d",
        "transforms3d",
        "prettytable",
        "tqdm",
        "matplotlib",
        "dill",
        "pandas",
        "seaborn",
        "pyarrow",
        "zarr",
        "kornia",
        "nuscenes-devkit==1.1.9",
        "black",
        "isort",
        "pytest",
        "pytest-xdist",
        "twine",
        "build",

    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Traffic Behavior Simulation",
    author="NVIDIA AV Research",
    author_email="danfeix@nvidia.com",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
