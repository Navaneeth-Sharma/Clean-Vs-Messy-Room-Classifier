from django.shortcuts import render
from django.shortcuts import HttpResponse
from .models import Image
from .forms import ImageForm
from django.contrib import messages

# Create your views here.

def Resize(file_path,saving_path='static/output/out.jpg',scale_percent=50):

    try:
        import cv2
     
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
         
        # print('Original Dimensions : ',img.shape)
         
        scale_percent = scale_percent # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(saving_path,resized)

    except Exception as e:
        print(e)

def create_model():
    from keras import layers
    from keras import models

    from keras.applications import VGG19

    

    conv_base = VGG19(weights='imagenet',include_top=False,input_shape=(299,299,3))

    from keras import optimizers
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    conv_base.trainable = False
    return model


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



def load_trained_model(weights_path,image_):
    model = create_model()
    model.load_weights(weights_path)
    import tensorflow as tf
    from keras import optimizers
    import keras
    import numpy as np
    image = tf.keras.preprocessing.image.load_img(image_)
    input_arr =tf. keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc',f1_m,recall_m,precision_m])
    val = model.predict(input_arr)
    return val


def index(request):
    if request.method == 'POST':
        a = 0
        form = ImageForm(request.POST, request.FILES)
        form.save()
        from cmr.models import Image
        import cv2
        print(Image.objects.all().last().img)
        test = 'media/'+str(Image.objects.all().last().img)
        print(test)
        tes = 'static/'+'img/j.jpg'
        im = cv2.imread(test)
        print(im.shape)
        if im.shape[0]<299 or im.shape[1]<299:
            print('"""""""Please try another the image""""""')
            val_ = 2

        else:
            if im.shape==(299,299,3):
                create_model()
                print(test)
                try:
                    val_ = load_trained_model('static/messy_and_clean_room.hdf5',test)
                except Exception as e:
                    print(e)
                # print(val_)
                
                
            else:
                for i in range(9):
                    try:
                        Resize(test,tes,(i+1)*10)
                        import cv2
                        img = cv2.imread(tes)
                        img = img[int(img.shape[0]/2-299/2):int(img.shape[0]/2+299/2),int(img.shape[1]/2-299/2):int(img.shape[1]/2+299/2)]
                        cv2.imwrite('static/'+'img/j.jpg',img)
                        val_ = load_trained_model('static/messy_and_clean_room.hdf5',tes)
                        
                        break
                    except:
                        pass
                
                
                

        # cv2.imshow('hey',im)
        # print(im)
        # print(image)
        # print(image.objects.all())
        print(val_)

        if 1-val_<0.5:
            try:
                val = f'\"My model Predics---Messy Room!! You can clean it now and make it look better  \"'
            except:
                val = f'\"My model Predics---Messy Room!! You can clean it now and make it look better \"'
            messages.warning(request, val)
        elif 1-val_>0.5 and 1-val_<=1:
            try:
                val = f'My model Predics---Clean!! Keep it up. Your Room is about {int((1-val_.ravel()[0])*100)}% Clean'
            except:
                val = f'My model Predics---Clean!! Keep it up. Your Room is about {int(val_*100)}% Clean'
            messages.success(request, val)
        else:
            val = 'Please upload different picture. !!!!The size is small!!!! '
            messages.warning(request,val)
    
    return render(request, 'index.html')


# from django.http import HttpResponse 
# from django.shortcuts import render, redirect 
# from .forms import *
  
# # Create your views here. 
# def index(request): 
  
#     if request.method == 'POST': 
#         name = request.POST.get('name')
#         img = request.POST.get('img') 
  
#         # if img.is_valid(): 
#         print('ji')
#         img.save() 
#         return redirect('success') 
#     else: 
#         img = ImageForm() 
#     return render(request, 'index.html') 
  
  
# def success(request): 
#     return HttpResponse('successfully uploaded') 


# # from PIL import Image as Im
# Im.open(Image.objects.all().last().img)