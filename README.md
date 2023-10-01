<h1>Brain Tumor Classification</h1>

<h3>Problem Statement: To generate a Deep-Learning based model which can 
classify MRI images of Brain Tumor into four categories.
The data is taken from kaggle platform.
<a href = "https://www.kaggle.com/datasets/iashiqul/mri-image-based-brain-tumor-classification">Dataset link</a
</h3>
<pre>
The data is in a zipfile, so we have to unzip the file and then check how many images are there in each class.
We will try to visualize some images from each class.
Then we are going with our important step of preparing images to train the model.
We are using the ImageDataGenerator class from Keras to prepare the images, we are also scaling the images with the same.

We will define some callbacks like EarlyStopping , ModelCheckpoint , TensorBoard.
Early Stopping : To check for the overfitting.
ModelCheckpoint : To track weights after every epoch.
TensorBoard : To visualize the graphs of losses and accuracy.

After defining some callbacks, we will train the model. Either we can build our model or we can use some pre-trained model.
Here we are using the pretrained model known as Xception. We are using this model with its weights fixed.
We are fitting our Dense layer with our requirements.
Then we will move towards training the model. But after training for some epoch we are fine tuning the model.
Fine tuning means we are training some layers of our base layer also, and then again training the model.

We are getting the validation accuracy of 98.32 which is pretty good. Finally, we are saving our model and using it for prediction.

UI includes an option for uploading an image, then in the backend, the model will predict the probability of each class and give the answer.
</pre>