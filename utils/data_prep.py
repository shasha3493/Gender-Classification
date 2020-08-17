# Importing necessary libraries
import os
import shutil

def data_prep():
    # Creating following directory structure as required by tensorflow to automatically infer # classes

    '''
    data
        train_data
            male
            female
        val_data
            male
            female
        test_data
            male
            female

    '''

    if not (os.path.isdir('./data/')):
        os.mkdir('./data')

    if not (os.path.isdir('./data/train_data/')):
        os.mkdir('./data/train_data/')

    if not (os.path.isdir('./data/val_data/')):
        os.mkdir('./data/val_data/')

    if not (os.path.isdir('./data/test_data/')):
        os.mkdir('./data/test_data/')

    if not (os.path.isdir('./data/train_data/male/')):
        os.mkdir('./data/train_data/male')

    if not (os.path.isdir('./data/train_data/female/')):
        os.mkdir('./data/train_data/female/')

    if not (os.path.isdir('./data/val_data/male/')):
        os.mkdir('./data/val_data/male/')

    if not (os.path.isdir('./data/val_data/female/')):
        os.mkdir('./data/val_data/female/')

    if not (os.path.isdir('./data/test_data/male/')):
        os.mkdir('./data/test_data/male')

    if not (os.path.isdir('./data/test_data/female/')):
        os.mkdir('./data/test_data/female/')


    # Copying all the images under all the folders inside ./agegender_cleaned/combined/aligned/ to ../data/train_data/male (female)

    source = './agegender_cleaned/combined/aligned/'
    target_male = './data/train_data/male/'
    target_female = './data/train_data/female/'

    # Lists all the subfolders
    folders = os.listdir(source)

    for folder in folders:
        
        # Lists all the files
        files = os.listdir(source + folder)
        
        # Copying images to the respective directory if file doesn't already exists
        if folder.find('M') != -1:
            for file in files:
                if not os.path.isfile(target_male + file):
                    _ = shutil.copy(source + folder + '/' + file, target_male)
        else:
            for file in files:
                if not os.path.isfile(target_female + file):
                    _ = shutil.copy(source + folder + '/' + file, target_female)


    # For every subfolder under './agegender_cleaned/combined/valid/', placing approximately
    # half of the files in validation and half of the  files in test so that in both validation
    # and test set, we have almost equal number of images for all age ranges. As a result, metrics
    # would be a good indication of the performance of the model

    source = './agegender_cleaned/combined/valid/'
    target_val_male = './data/val_data/male/'
    target_val_female = './data/val_data/female/'
    target_test_male = './data/test_data/male/'
    target_test_female = './data/test_data/female/'

    folders = os.listdir(source)

    for folder in folders:
        files = os.listdir(source + folder)
        # Splitting total images/age into half
        ind = int(len(files)/2)
        val_files = files[:ind]
        test_files = files[ind:]

        # Copying images to the respective directory if file doesn't already exists
        if folder.find('M') != -1:
            for file in val_files:
                if not os.path.isfile(target_val_male + file):
                    _ = shutil.copy(source + folder + '/' + file, target_val_male)
            
            for file in test_files:
                if not os.path.isfile(target_test_male + file):
                    _ = shutil.copy(source + folder + '/' + file, target_test_male)
                
        
        else:
            for file in val_files:
                if not os.path.isfile(target_val_female + file):
                    _ = shutil.copy(source + folder + '/' + file, target_val_female)

            for file in test_files:
                if not os.path.isfile(target_test_female + file):
                    _ = shutil.copy(source + folder + '/' + file, target_test_female)




        