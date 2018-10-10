import csv
import cv2
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import sklearn
import random
from pathlib import Path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.utils import plot_model


correction = 0.30
batch_size = 32
####################################PART 1: LOAD AND GENERTE DATA ###########################################################
### Load data from Udacity and show its place.
def path_to_filename(path):
    """ Pulls the filename out of a path by splitting on the slashes
    """
    if 'C:' in path:
        return path.split('\\')[-1] # Grab the Windows file name, cutting off the base path
    else:
        return path.split('/')[-1] # Grab the file name, cutting off the base path

def generator(samples, batch_size=32):
    """Takes the a list of lines from a CSV and generates a list of images
    and steering measurements
    """
    n_samples = len(samples)
    while True: # Loop forever so the generator never exits, as expected by keras.fit_generator
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for sample in batch_samples: # This is a line of the CSV file
                for i in range(3):
                    source_path = sample[i]
                    filename = path_to_filename(source_path)
                    path = './data/IMG/' + filename
                    image = cv2.imread(path)
                    if image is None:
                        # Did you know that imread quietly returns None if it can't find the image? Kind of unhelpful. 
                        print('no image found')
                        pdb.set_trace()
                    images.append(image)
                # Append three measurements to the list
                measurement = float(sample[3])
                measurements.append(measurement) # center
                measurements.append(measurement + correction) # left
                measurements.append(measurement - correction) # right

            # add mirrored data
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = measurement * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)
            images += augmented_images
            measurements += augmented_measurements

            X_train = np.array(images) 
            y_train = np.array(measurements) 
            # yield (X_train, y_train) # inputs, targets
            yield sklearn.utils.shuffle(X_train, y_train) # inputs, targets

# Read data
PATH = Path('data/CSVfile')
CSVfile = list(PATH.iterdir())

lines = []
for CSV in CSVfile:
    with open(str(CSV)) as file: 
        reader = csv.reader(file)
        lines_with_labels = []
        for line in reader:
            lines_with_labels.append(line)
        print( str(CSV),  len(lines_with_labels))
        lines_with_labels = lines_with_labels[1:]
        lines += lines_with_labels

#pdb.set_trace()
random.shuffle(lines) # Shuffle the order
print( len(lines))
### Splitting %80 of data for training, %20 for validation
train, val = train_test_split(lines, test_size=0.2)

train_generator = generator(train, batch_size)
val_generator = generator(val, batch_size)
############################### Part 2: BUILDING MODEL AND TRAINING #######################################################
############################### Modified Comma.ai's model #######################################################
### 3 convolution layers followed by 2 fully connected layers
### Changed input size to use whole  size of the input image, added 3 more convolutional layers with kernel sizes of 3x3
### Added more neurons and used ReLU activation function instead of ELU added dropout to increase accuracy
model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Conv2D(32, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(48, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(64, (5, 5), strides=2, activation='relu'))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train) / batch_size, \
    epochs=5, validation_data=val_generator, validation_steps = len(val) / batch_size) # Shuffle is on by default


print('model saved.')
model.save('model0.h5')
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
