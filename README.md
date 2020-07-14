# Clean-Vs-Messy-Room-Classification

## Introduction

   A simple Website to identify the Clean and Messy Room. **Amazing right!!** This Project is the Combination of Deep Learning and Django.I have used VGG19 CNN 
architecture and Keras framework to classify the Clean and Messy Room.The Model is trained with input shape (299,299,3).

## The DataSet :
     
   This Dataset of Rooms is from Kaggle Datasets. https://www.kaggle.com/cdawn1/messy-vs-clean-room is the link to download the dataset. I have created some functions and some operations in cmr/views.py which is also a part of django as a frontend. ( https://github.com/Navaneeth-Sharma/Clean-and-Messy-Room-Classification/blob/master/cmr/views.py ) This will resize any image greater than (299+ ,299+ ,3 ) shape into the input shape (299,299,3).

## The VGG19 Architecture :


![alt text](https://www.researchgate.net/profile/Michael_Wurm/publication/331258180/figure/fig1/AS:728763826442243@1550762244632/Architecture-of-the-FCN-VGG19-adapted-from-Long-et-al-2015-which-learns-to-combine.png)  



## The Metrics:

### The Accuracy
![alt text](https://github.com/Navaneeth-Sharma/Clean-and-Messy-Room-Classification/blob/master/static/img/acc.png)


### The Loss
![alt text](https://github.com/Navaneeth-Sharma/Clean-and-Messy-Room-Classification/blob/master/static/img/loss.png)

## The Weights :
     
   The Weights of this couldn't be uploaded to the github bcz it exceeds 100Mb. If you want the wieghts you can generate it from main_cmr.py python file it saves the weights as best_model.h5 and you can use it in the website

## Django :

   I have used Django Framework (since it supports  Python programming). It has a basic form which takes the user input image(as a file input) of the room and gives the user an alert (a feature of bootstrap) will display on the screen, based on how clean is the room. So why wait, check out the code above !!
   
## Screenshots of the Website :

![alt text](https://github.com/Navaneeth-Sharma/Clean-Vs-Messy-Room-Classifier/blob/master/static/img/Screenshot%20from%202020-07-14%2000-45-50.png)
![alt text](https://github.com/Navaneeth-Sharma/Clean-Vs-Messy-Room-Classifier/blob/master/static/img/Screenshot%20from%202020-07-14%2000-46-05.png)


###    Thank You ...
