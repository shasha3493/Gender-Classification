# Gender Classification 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

1. Unzip Shashank_Saurav.tar.gz file
2. Download datset from [here](https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz.) and unzip.
3. Pretrained weights as vgg_face.mat can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).

 On unzipping the folder contains following subfolders and files:
 - agegender_cleaned: original dataset
 - best_model - folder containing the best performing model on the validation dataset
 - results - contains all the results
 - train.py - py script to train the model
 - test.py - py script to evaluate the model
 - utils - contains dependencies base_model.py, data_prep.py, plot_history.py, preprocess.py
 - requirements.txt - txt file containing all the libraries
 - pretrained_weights - contains pretrained weights as vgg_face.mat file

 
 
4. Execute the following command in this folder to set up the require virtual environment for running these experiments.
    
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt


# Summary of files

1. agegender_cleaned: original dataset
2. best_model - folder containing the best performing model on the validation dataset
3. results - contains train_results.txt, val_results.txt, test.txt and history.jpeg. txt files contain the class wise metrics (precision, recall, f1-score, accuracy and overall accuracy) for train, validation and test data correspondingly. history.jpeg contains the plot for training logs.
4. train.py - py script to train the model
5. test.py - py script to evaluate the model
6. utils - contains dependencies base_model.py, data_prep.py, plot_history.py, preprocess.py. base_model.py creates the base vgg16 model and loads pretrained weights to be used as feature extractor. data_prep.py creates train-val-test split of the original dataset. plot_history.py plots the training logs against epochs. preprocess.py subtracts means from the images.
7. requirements.txt - txt file containing all the libraries
8. pretrained_weights - contains pretrained weights as vgg_face.mat file

## Dataset

Dataset can be directly accessed [here](https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz.). 

agegender_cleaned/combined/aligned and agegender_cleaned/combined/valid contains many subfolders like 01_F, 02_M etc. These subfolders in turn contain many .jpg images. The naming convention of the folder is such that 01 in 01_F for example represents the age of person and M represents gender(male). 

All the images under agegender_cleaned/combined/aligned was placed under ./data/train_data/male or ./data/train_data/female depending on the gender. 

For every subfolder under './agegender_cleaned/combined/valid/', placed approximately half of the files in validation(./data/val_data/male or ./data/val_data/female) and half of the files in test (./data/test_data/male or ./data/test_data/female) depending on gender so that in both validation and test set, we have almost equal number of images for all age ranges. As a result, metrics would be a good indication of the performance of the model. In summary, train data has 29,437 data samples(14312 females and 15125 males), validation dataset has 1814 samples(861 females and 953 males) and test dataset has 1867 samples(976 males and 891 females).

The above split can be prepared by running data_prep.py under utils folder.

## Running the Experiment

Please execute the following commands in order to train and evaluate the model

1. Prepare data split
    
    python data_prep.py
    
    
2. Train the model 

    python train.py --seed 13 --epochs 10 --batch_size 64 --weights_path './pretrained_weights/vgg_face.mat' --       train_dir './data/train_data/' --val_dir './data/val_data/' --checkpoint_path './best_model/' --plot_history       './results/history.jpeg'
    

3. Evaluate the model 

    python test.py --model './best_model/' --test_dir './data/test_data/' --result_path './results/results.txt' --     batch_size 64

# Results

### Training History Plot:

![history.jpg](/results/history.jpeg)

### Test Metrics:

![Screenshot%20from%202020-08-10%2005-06-40.png](/results/test_results.txt)

### Validation Metrics:

![Screenshot%20from%202020-08-10%2005-06-55.png](attachment:Screenshot%20from%202020-08-10%2005-06-55.png)

### Training Metrics:

![Screenshot%20from%202020-08-10%2005-31-21.png](attachment:Screenshot%20from%202020-08-10%2005-31-21.png)

# Appendix:

## Steps taken to build the model

### Dataset Preparation(data_prep.py):

All the images under agegender_cleaned/combined/aligned was placed under ./data/train_data/male or ./data/train_data/female depending on the gender. 

For every subfolder under './agegender_cleaned/combined/valid/', placed approximately half of the files in validation(./data/val_data/male or ./data/val_data/female) and half of the files in test (./data/test_data/male or ./data/test_data/female) depending on gender so that in both validation and test set, we have almost equal number of images for all age ranges. This was done so that test metrics would be a good indication of the performance of the model.

### Base Model(base_model.py)

CNN architecture was built based on [paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf). Subsequently pre-trained weights vgg_face.mat was initilized to correponding layers of the model. Since mat files can't be directly loaded to tensorflow, [link](https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/) was referred to load weights.

### Fine Tuning (train.py)

Import the base model built in base_model.py. Created data generator for train and val data by subtracting the means to individual channels (preprocess.py) and thensubsequently resizing the image to (224,224). 

From the base model, upto 5th convolution layer was taken (used as feature extractor) and then on the top classifier network was attached. Classifier network is a fully connected layer with two hidden layers containih 2048 and 1024 units respectively. Additionally batch normalization was applied to dense layers alongwith dropout to improve generalization. 

In order to fine tune, weights for the first 4 conv layers were kept fixed and only weights for 5th conv and dense layers were trained. Also, learning rate chosen was low (0.00001).

Batch size is 64, Optimizer used is Adam, Loss function is Cross Entropy and validation metric is Accuracy.

The model is finallt trained and val accuracy is observed after every epoch. If there's an improvement in the same, the corresponding model is saved.

After training, training logs are plotted in plot_history.py. Plot was also saved as jpeg file under results folder.

### Test the model (test.py)

Imported the best performing model on the val data saved during training. Created data generator for test data by subtracting the means to individual channels (preprocess.py) and then subsequently resizing the image to (224,224). 

Evaluted the model on the test dataset to get the predictions. Finally, using the predictions and true labels, calcualted the metrics (precision, recall, f1-score, accuracy) per class and overall accuracy. These results were also saved in txt files under results folder.

# Built With

- TensorFLow
