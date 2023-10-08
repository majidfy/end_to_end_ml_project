# End to End Machine Learning Project

This repository contains the code and documentation for an end-to-end machine learning project. The goal of this project is to develop and deploy a machine learning model that can solve a specific problem or task. This README provides an overview of the project, its structure, and instructions for running and replicating the results. The problem solved in this project is to make a regressive model to predict student's overall score based on their performance/demographics data.

## Installation
To start off, create a new virtual environment and install the required dependancies which are all mentioned in the requirements.txt file
## Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

## Usage
The project is segregated into different components which include data ingestion, data transformation, model training, hyperparameter tunning and model deployment.

The project can be directly run using the train_pipeline.py which loads the data, performs the data split, data transformation and model training. The functions for all the steps are mentioned in the components folder.

Finally, app.py can be used to deploy the prediction pipeline locally using Flask. 

