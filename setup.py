from distutils.core import setup

_VERSION = "0.3.2"

with open("README", "r") as readme:
    long_desc = readme.read()


setup(
    version=_VERSION,
    name="stan-learn",
    packages=["base", "test", "examples", "linear_regression"],
    author="Ryan J. Kinnear",
    author_email="Ryan@Kinnear.ca",
    description=("Implementation of some Bayesian ML algorithms "
                 "in Stan with an sklearn-like interface."),
    long_description=long_desc,
    url="https://github.com/RJTK/Levinson-Durbin-Recursion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"],
    license="LICENSE"
)
