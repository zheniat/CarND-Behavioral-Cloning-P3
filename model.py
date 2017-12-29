import csv
import cv2
import numpy as np
import sklearn

# requires Keras 2.0
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Dropout
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#model parameters
batch_size = 32
epochs = 50


# import sample images and measurements from the car simulator output
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#split the data set into two: training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#convert image path into an image
def process_image(path):
    filename = path.split('/')[-1]
    img_path = 'IMG/' + filename
    image = cv2.imread(img_path)
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
                # read in images from center only
                steering_center = float(batch_sample[3])
                img_center = process_image(batch_sample[0])

                # add images and angles to data set
                images.append(img_center)
                measurements.append(steering_center)

            #flip images to generalize the model and double the number of images
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped = np.fliplr(image)
                augmented_images.append(flipped)
                augmented_measurements.append(-measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# number of training and validation samples.
# We will double the number of sample in the generator by creating a mirror image of each sample
train_sample_count = len(train_samples) * 2
validation_sample_count = len(validation_samples) * 2

# create generators: one for training and one for validation
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#Keras CNN model
model = Sequential()
# normalize each image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# crop image to focus on the road
model.add(Cropping2D(cropping=((70,25), (0,0))))
# convolutional layers
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
# neural network layers
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# use Keras 2.0 functionality
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = train_sample_count//batch_size,
                              validation_data = validation_generator,
                              validation_steps = validation_sample_count//batch_size,
                              nb_epoch=epochs, verbose=1)

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
