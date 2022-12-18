"""Belajar Machine Learning untuk Pemula - Septriane Putri Akprita.ipynb

Original file is located at
    https://colab.research.google.com/drive/1dBIjUdZOu1zJTfx8m61bZ7fUgelkCvlA
"""

!wget https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip

import tensorflow as tf

import os
os.listdir()

import zipfile
local_zip = 'rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()
base_dir = 'rockpaperscissors'
train_dir = os.path.join(base_dir, 'rps-cv-images')

#rock
train_rock_dir = os.path.join(train_dir, 'rock')
#paper
train_paper_dir = os.path.join(train_dir, 'paper')
#scissors
train_paper_dir = os.path.join(train_dir, 'scissors')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    horizontal_flip=True,
                                    shear_range=0.2,
                                    fill_mode = 'nearest',
                                    validation_split=0.4)

train_gen = train_data_gen.flow_from_directory(train_dir,
                                               target_size=(150, 150),
                                               batch_size=4,
                                               class_mode='categorical',
                                               subset='training')
validation_gen = train_data_gen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=4,
                                                    class_mode='categorical',
                                                    subset='validation')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation='relu'),
                                    tf.keras.layers.Dense(1024, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='softmax')])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['accuracy'])

model.fit(train_gen,
          steps_per_epoch=25,
          epochs=20,
          validation_data=validation_gen,
          validation_steps=5,
          verbose=2)

model.evaluate(train_gen)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

uploaded = files.upload()

for fn in uploaded.keys():
  path = fn
  img=image.load_img(path, target_size=(150,150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  print(fn)
  if classes[0][1] == 1:
    print('rock')
  elif classes[0][0] == 1:
    print('paper')
  else :
    print('scissors')
