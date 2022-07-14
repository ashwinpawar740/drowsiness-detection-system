import matplotlib.pyplot as plt
import tf as tf
from keras import backend as K
from keras.callbacks import History
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image

# Dimension of images
img_height, img_width = 50, 50

train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
nb_train_samples = 98
nb_validation_samples = 133
epochs = 30
batch_size = 10



if K.image_data_format == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale = 1. /255,
    shear_range = 0.2,
    zoom_range= 0.2,
    horizontal_flip= True)

train_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size= (img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size= (img_width, img_height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')

#########################################################################################
# Model Creation

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(loss= 'categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

#This is the augmentation configuration we will use for training
history = History()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,callbacks=[history],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size
)

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
plt.savefig('acc.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
loss = plt.show()
plt.savefig('loss.png')

# serialize model to JSON
model_json = model.to_json()
with open("eye.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/eye.h5")
