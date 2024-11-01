# Signature_Detection

## Requirements
Recommended: Use a virtual environment like conda or python's in-build virtual environment
Run pip install -r requirements.txt to install required dependencies
## Data Creation
To create a dataset to train, upload signatures to the dataset folder(individual's sigantures in different folders)
### Naming the folders
Suppose the name is Chris Evans the folder name should be 'Chris_Evans'

## Labeling Signatures
In both app.py(train) and test.py(test) label_map is provided. Change the values accordingly in both files

## Train Model
After uploading dataset and labeling the signatures, train model by running: 
```bash
pip install -r requirements.txt
