from keras import layers
from keras import models

train_dir = '../input/messy-vs-clean-room/images/train'
val_dir = '../input/messy-vs-clean-room/images/val'
test_dir = '../input/messy-vs-clean-room/images/test'

from keras.applications import VGG19

conv_base = VGG19(weights='imagenet',include_top=False,input_shape=(299,299,3))

from keras import optimizers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False
model.summary()


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc',f1_m,recall_m,precision_m])


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  shear_range=0.3,
                                  zoom_range=0.3,
                                  horizontal_flip=True,
                                  fill_mode='nearest'
                                  )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(299,299),batch_size=32,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(val_dir,target_size=(299,299),batch_size=32,class_mode='binary')


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.00001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='f1_m', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, callbacks=[es,mc],validation_data=validation_generator,validation_steps=50)

import matplotlib.pyplot as plt
def smoothed_curve(points,factor=0.8):
    smothed_points = []
    for point in points:
        if smothed_points:
            previous = smothed_points[-1]
            smothed_points.append(previous*factor+point*(1-factor))
        else:
            smothed_points.append(point)
    return smothed_points

history_dict = history.history
loss_values = history_dict['loss']
val_loss_val = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)

plt.plot(epochs,smoothed_curve(loss_values),'bo',label='Training Loss')
plt.plot(epochs,smoothed_curve(val_loss_val),'b',label='Validation Loss')
plt.title('Train and valid loss')
plt.xlabel('Epochs')

plt.ylabel('Losses')
plt.legend()
plt.show()

acc_values = history_dict['acc']
val_acc_val = history_dict['val_acc']


plt.plot(epochs,smoothed_curve(acc_values),'bo',label='Training acc')
plt.plot(epochs,smoothed_curve(val_acc_val),'b',label='Validation acc')
plt.title('Train and valid acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()