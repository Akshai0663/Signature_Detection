# Signature_Detection

## Requirements
Recommended: Use a virtual environment like conda or python's in-build virtual environment.
Run this code to install required dependencies:
```bash
pip install -r requirements.txt
```
## Data Creation
To create a dataset to train, upload signatures as images in jpeg/jpg or png to the dataset folder(individual's sigantures in different folders)
### Naming the folders
Suppose the name is Chris Evans the folder name should be 'Chris_Evans'

## Labeling Signatures
In both app.py(train) and test.py(test) label_map is provided. Change the values accordingly in both files

## Train Model
After uploading dataset and labeling the signatures, train model by running: 
```bash
python app.py
```
For greater accuracy increase the no of images provided
## Test Model
After model creation, run this:
```bash
python test.py
```
Run the test server:
```bash
http://127.0.0.1:5000
```
