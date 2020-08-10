import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *
from sklearn import metrics
from utils.preprocess import subtract_mean

# Reading input from the terminal
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='./best_model/', help="path to the model checkpoint")
parser.add_argument("--test_dir", type=str, default="./data/test_data/", help="path to test dataset")
parser.add_argument("--result_path", type=str, default="./results/results.txt", help="path of file to write results to")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")

opt = parser.parse_args()

# Building the model from the saved model
model = load_model(opt.model)
model.build((None, 224,224, 3))

# Test dataset directory
test_dir = opt.test_dir

batch_size = opt.batch_size

# Creating generator for test dataset
datagen_test = ImageDataGenerator(preprocessing_function = subtract_mean)
generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=(224,224),
                                                  batch_size=batch_size,
                                                  shuffle=False)

steps = generator_test.n / batch_size

# list of Class name
class_names = list(generator_test.class_indices.keys())

# dic with class names as keys and class ids as values
class_2_id = generator_test.class_indices
# dic with class names as values and class ids as keys
id_2_class = {v: k for k,v in class_2_id.items()}

# Predicted probabilities by the model
predictions = model.predict_generator(generator_test, steps=steps)

# Predicted labels by the model
test_preds = np.argmax(predictions, axis=1)
# True labels of the test dataset
test_trues = generator_test.classes

# Confusion matrix
cm = metrics.confusion_matrix(test_trues, test_preds)

# Accuracy for male class
male_acc = np.round(cm[class_2_id['male'], class_2_id['male']]/(cm[class_2_id['male'], class_2_id['male']] + cm[class_2_id['male'], class_2_id['female']]), 3)
# Accuracy for female class
female_acc = np.round(cm[class_2_id['female'], class_2_id['female']]/(cm[class_2_id['female'], class_2_id['male']] + cm[class_2_id['female'], class_2_id['female']]), 3)

# Classification report
cm = metrics.classification_report(test_trues, test_preds, target_names = [id_2_class[0], id_2_class[1]], output_dict=True)

# Different metrics for male and female class
female_prec = np.round(cm['female']['precision'], 2)
female_rec = np.round(cm['female']['recall'], 2)
female_f1 = np.round(cm['female']['f1-score'], 2)

male_prec = np.round(cm['male']['precision'], 2)
male_rec = np.round(cm['male']['recall'], 2)
male_f1 = np.round(cm['male']['f1-score'], 2)

acc = np.round(cm['accuracy'], 3)

# Witing the metrics to the file
with open(opt.result_path,'w') as f:
    f.write('\t \t Precision \t Recall \t F1-score \t Accuracy\n')
    f.write('female:\t {} \t\t {} \t\t {} \t\t {}\n'.format(female_prec, female_rec, female_f1, female_acc))
    f.write('male:\t\t {} \t\t {} \t\t {} \t\t {}\n'.format(male_prec, male_rec, male_f1, male_acc))
    f.write('Avg Accuracy: \t\t\t\t\t\t\t {}'.format(acc))