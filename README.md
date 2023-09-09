# Fashion-MNIST


`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example of how the data looks (*each class takes three-rows*):

![](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

## Get the Data

You can clone this GitHub repository; the dataset appears under `data/fashion`. This repo also contains some scripts for benchmark and visualization.
   
```bash
git clone git@github.com:zalandoresearch/fashion-mnist.git
```

### Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Usage

### Loading data with Tensorflow
Make sure you have downloaded the data and placed it in `data/fashion`. Otherwise, *Tensorflow will download and use the original MNIST.*

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(BATCH_SIZE)
```

Note, Tensorflow supports passing in a source url to the `read_data_sets`. You may use: 
```python
data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
```


## Model
```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=input_shape)) #should be the first layer 28 represents the pixels
#kernel size for scnning the image is (3,3) pixels 
#input shape is the input of the pixel size and the shader

model.add(MaxPooling2D(pool_size=(2,2)))#pools the image size into a smaller matrix max pooling take the max size in the 
#frame we have set and it will be pooled in a (2,2) matrix

model.add(Flatten()) #flattens the data into one dimentional layer i.e array so it can be connected into many layers

model.add(Dense(128,activation=tf.nn.relu)) #connects the layers together relu = rectified linear unit 128 is the output
#tf = tensorflow . nn = neural networks . relu

model.add(Dropout(0.1)) #removes some connections of neural network because it will remember exactly the same of how a 
#image looks like hence it will be a problem detecting the similar images
#0.1 is the percentage of neural networks you want to cut

model.add(Dense(10,activation = tf.nn.softmax)) #softmax takes care of the max probabilistic outcome
```

```python
model.compile(optimizer='adam', #minimizes the loss
              loss = 'sparse_categorical_crossentropy', #calculates the loss
              metrics=['accuracy'])#calculates accuracy
model.fit(x=x_train,y=y_train,epochs=6) #data is sent into model and epochs = 6 are the iterations performed on the model to
#make it corrent
```




