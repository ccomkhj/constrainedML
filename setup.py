import os
import setuptools

description = "Constrained linear regression in scikit-learn style"
long_description = description
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setuptools.setup(
    name="constrainedML",
    version="0.0.1",
    author="Huijo",
    author_email="huijo@hexafarms.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ccomkhj/constrainedML",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scikit-learn<=1.1",
        "numpy",
    ],
)
