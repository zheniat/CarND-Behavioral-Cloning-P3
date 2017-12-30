import csv
import cv2
import numpy as np
import sklearn
import pydot
import random
import math
import os.path

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Dropout
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Model
import matplotlib.pyplot as plt

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

#model parameters
batch_size = 32
epochs = 30
camera_correction = 0.12



# import sample images and measurements from the car simulator output
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#split the data set into two: training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Image pre-processing

# number of training and validation samples.
augment_size = 40 #40% of images will be augmented

# sample multipler
# *2 - brightness
# *2 - random shadow
sample_multiplier = 2 * 2 * augment_size/100

# *2 - all augmented images will be flipped
train_sample_count = math.ceil(len(train_samples) * (1 + sample_multiplier) * 2)
validation_sample_count = math.ceil(len(validation_samples) * (1 + sample_multiplier) * 2)

#convert image path into an image
#returns an RGB image
def process_image(path):
    filename = path.split('/')[-1]
    img_path = 'IMG/' + filename
    bgr_image = cv2.imread(img_path)
    # convert BGR to RGB to ensure consistency with the drive simulator
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return image

def augment_flip_images(image, measurement):
    flipped = np.fliplr(image)
    return flipped, - measurement

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2] > 255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# generator function
def generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, measurements = [], []

            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                     # create adjusted steering measurements for the side camera images
                steering_left = steering_center + camera_correction
                steering_right = steering_center - camera_correction

                # read in images from center only
                path = "IMG/"
                img_center = process_image(batch_sample[0])
                img_left = process_image(batch_sample[1])
                img_right = process_image(batch_sample[2])

                # add images and angles to data set
                #images.append(img_center)
                #measurements.append(steering_center)
                images.extend([img_center, img_left, img_right])
                measurements.extend([steering_center, steering_left, steering_right])

            image_probs = random.sample(range(1, 100), len(images))

            #augment images
            augmented_images, augmented_measurements = [], []

            #augment brightness, add random shadow
            for image, measurement, image_prob in zip(images, measurements, image_probs):
                if image_prob < augment_size:

                    new_image = augment_brightness_camera_images(image)
                    augmented_images.append(new_image)
                    augmented_measurements.append(measurement)

                    new_image = add_random_shadow(image)
                    augmented_images.append(new_image)
                    augmented_measurements.append(measurement)

            #add all images to the augmented lists
            augmented_images.extend(images)
            augmented_measurements.extend(measurements)

            flipped_images, flipped_measurements = [], []
            for image, measurement in zip(augmented_images, augmented_measurements):
                new_image, new_measurement = augment_flip_images(image, measurement)
                flipped_images.append(new_image)
                flipped_measurements.append(new_measurement)
                flipped_images.append(image)
                flipped_measurements.append(measurement)

            # trim image to only see section with road
            X_train = np.array(flipped_images)
            y_train = np.array(flipped_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)


# create generators: one for training and one for validation
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

best_weights_file = "weights.best.hdf5"

#Keras CNN model
model = Sequential()
# crop image to focus on the road
model.add(Cropping2D(cropping=((70,25), (0,0))))
# normalize each image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# convolutional layers
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
# neural network layers with dropouts
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# use Keras 2.0 functionality
if os.path.isfile(best_weights_file):
    model.load_weights(best_weights_file)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#add checkpoint
checkpoint = ModelCheckpoint(best_weights_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

callbacks_list = [checkpoint, earlystop]


history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = train_sample_count//batch_size,
                              validation_data = validation_generator,
                              validation_steps = validation_sample_count//batch_size,
                              nb_epoch=epochs, verbose=1,
                              callbacks=callbacks_list)

### print the keys contained in the history object
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper left')
plt.show()

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#save the model
model.save('model.h5')
