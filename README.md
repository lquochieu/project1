# Cats, Dogs, Pandas image classification

## Introduction
Cats, Dogs, Pandas image classification problem using Convolution Neural Network (CNN)  to classify images Cats, Dogs, Pandas

## Contribute
+ Trần Trọng Hiệp

## Table of contents
1. [Introduction](#Introduction)
2. [Contribute](#Contribute)
3. [Dependencies](#Dependencies)
4. [INSTALLATION](#INSTALLATION)
5. [TRAINING MODEL](#TRAINING-MODEL)
    + [Change the default model](#Change-the-default-model)
    + [Change the training datasets](#Change-the-training-datasets)
    + [Change the training epochs](#Change-the-training-epochs)
    + [Change the name of the saved model file](#Change-the-name-of-the-saved-model-file)
    + [Change the name of the resulting image file](#Change-the-name-of-the-resulting-image-file)
6. [LOAD MODEL](#LOAD MODEL)
7. [DEVELOPMENT](#DEVELOPMENT)
## Dependencies
Python version 3.8 or greater

## INSTALLATION
Import Open-cv library:
```
pip install opencv-python
```

Import tensorflow library:
```
pip install tensorflow
```

Import Keras libraby:
```
pip install keras
```

## QUICK USE
Classify 10 random images in datasets
```
python load_model.py
```

## TRAINING MODEL
 `Quick train` with default the model is `MyProcessModel`
```
python app.py
```
Build model without defaults expect the default model is `MyProcessModel`
```
python app.py --datasets 'file_name_dataset' --epochs number_of_epochs --model 'file_name_model' --output 'file_name_resulting_image'
```
The model after training is saved in 'file_name_model', default is `'./Model/MyProcessModel_100.hdf5'`. 

The resulting image after training is saved in 'file_name_resulting_image', defaults is `file_name_resulting_image`
### Change the default model
In file app.py, import `your_model`. I have 3 models available in this app.py file which are `MyProcessModel`, `ShallowNet`, `Lenet-5`
```
from pyimagesearch.nn.conv import MyProcessModel
from pyimagesearch.nn.conv import ShallowNet
from pyimagesearch.nn.conv import Lenet5
```
After that at line 56 in this file. Change `MyProcessModel` to your model
```
model = MyProcessModel.build(width=32, height=32, depth=3, classes=3)
```
Example
```
model = Shallownet.build(width=32, height=32, depth=3, classes=3)
```
or
```
model = Lenet5.build(width=32, height=32, depth=3, classes=3)
```
### Change the training datasets
Note when changing datasets, the folder structure is similar to the original folder, including 3 folders Dogs, Cats, Pandas containing datasets of each type. 

```
python app.py --datasets 'file_name_dataset'
```
Default file name datasets is `'./datasets/animals'`
### Change the training epochs
```
python app.py --epochs number_of_epochs
```
Default epochs are `100 epochs`
### Change the name of the saved model file
```
python app.py --model 'file_name_model'
```
Default saved model file name is `'./Model/MyProcessModel_100.hdf5'`
### Change the name of the resulting image file
```
python app.py --output 'file_name_resulting_image'
```
## LOAD MODEL
 `Quick load` with default the saved model is `MyProcessModel` and load random 10 images in datasets.
```
python load_model.py
```
Or 
```
python load_model.py --datasets 'file_your_datasets' --model 'file_saved_model' --output 'file_resulting_output'
```
File_your_datasets defaults is `"./datasets/animals"`. File_saved_model defaults is `"./Model/MyProcessModel.hdf5"`. File_sesulting_ouput default is `"./Outputs/predicts/"`
## DEVELOPMENT
You can add your own CNN network architecture, see the files in the folder `'./pyimagesearch/nn/conv/'` to see how to set up the model and create your network architecture.

To better understand what is written in this, you can see the article [Cats, Dogs, Pandas image classification](https://drive.google.com/file/d/1BiXNQtOKSu1ZkayCZKYBlxukVEfNeapI/view?usp=sharing)
