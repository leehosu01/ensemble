# ensemble
automatic ensemble tuning


## Prepare dataset and trained model before ensemble
```python
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

def build_model():
    def scc_ls(y_true, y_pred):
        y_true = tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), 10))
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = 0.1)
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
              loss=scc_ls,
              metrics=['accuracy'])
    return model

n_model = 16
models = [build_model() for _ in range(n_model)]
for model in models:
    model.fit(train_images, train_labels, epochs=5, batch_size = 256)
for model in models:
    model.evaluate(test_images,  test_labels)
```

## define metrics
```python
def loss_evalueate(y_true):
    def _sub(y_pred):
        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true, y_pred)).numpy()
    return _sub
def acc_evalueate(y_true):
    def _sub(y_pred):
        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)).numpy()
    return _sub
```

## ensembing
```python
!pip install git+github.com/leehosu01/ensemble.git
import ensemble
import numpy as np
weights = ensemble.ensemble(
    model_predict_functions     = [model.predict for model in models],
    dataset                     = test_images     ,
    eval_function               = loss_evalueate(test_labels),
    eval_method                 = np.argmin,
    random_sample_count         = 8,
    random_order_length         = 256,
    rate_underbound             = 0.10,
    rate_upperbound             = 2.00,
    search_method               = ternary_search,
    search_precision            = 5,
    verbose                     = 1 )

predict = ensemble.ensemble_predict_function(
    [model.predict for model in models], weights
) # get predict function

loss_evalueate(test_labels)(predict(test_images)), acc_evalueate(test_labels)(predict(test_images))
```
