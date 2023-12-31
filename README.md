# Multi-Constrained Regression and Neural Network Repository

## Overview

This repository is dedicated to hosting and sharing advanced techniques in machine learning algorithms, particularly focusing on constraining the weights of certain inputs in regression and multi-layer perceptron. I have ventured into reverse engineering and extending its capabilities to fit custom requirements for specific types of learning problems.

## Purpose

The purpose of this repository is to provide a resource for machine learning practitioners looking to impose constraints on the input features' weights. The reverse-engineered solutions herein allow for greater control over the machine learning model's behavior, ensuring that the influence of some features remains within desired boundaries.

## Tutorials

I provide detailed tutorials for the following topics:

- **Multi-Constrained Linear Regression**: This tutorial takes you through the steps of creating a linear regression model that allows constraints to be placed on the weights of multiple input features.
  - [Multi-Constrained Linear Regression Tutorial](tutorial/MultiConstrainedLinearRegression.md)

- **Multi-Constrained Multi-Layer Perceptron**: Explore the implementation of a multi-layer perceptron (neural network) that incorporates constraints on the weights corresponding to specific input features.
  - [Multi-Constrained Multi-Layer Perceptron Tutorial](tutorial/MultiConstrainedMultiLayerPerceptron.md)

## Features

- Reverse engineering techniques applied to scikit-learn's Linear Regression and MLP models
- Custom weight constraint functionalities
- Step-by-step tutorials for implementing the above models

## Getting Started

To get started with these tutorials and code, you should clone the repository and navigate to the `tutorial` directory where you can find the markdown files with detailed explanations and code samples.

```bash
git clone https://github.com/ccomkhj/constrainedML.git
cd constrainedML/tutorial
```
### Contributing
Welcome contributions from the community! Whether it's improving the tutorials, extending the features of the models, or fixing bugs, please feel free to fork the repo, make your changes, and submit a pull request.

### Acknowledgments
Thanks to the scikit-learn developers for their work on creating a comprehensive machine learning library.
This project was inspired by the need for industry-specific machine learning models that require tailored constraints.

### Contact
If you have any questions or feedback, please open an issue in the repository.