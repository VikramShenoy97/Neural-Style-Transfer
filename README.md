# Neural Style Transfer

Neural Style Transfer implemented using Keras and TensorFlow.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For using Neural Style Transfer, you need to install Keras, TensorFlow and PIL.

```
pip install keras
pip install tensorflow
pip install Pillow
```

### Images

#### Content Image - A picture taken by me while on my trip to Amsterdam.

![Content_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Input_Images/Amsterdam.jpg)

#### Style Image - Starry Night by Vincent Van Gogh.

![Style_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Input_Images/Starry_Night.jpg)


### Run

Run the script *test.py* in the terminal as follows.

```
Python test.py
```

## Results
The final output is stored in Output Images.

### Intermediate Stages of Style Transfer

Here is the generated image through different intervals of the run.

![Intermediate_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Output_Images/Intermediate_Images.jpg)

### Transition through epochs

![Transition](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Transition/nst.gif)

### Result of Style Transfer

![Final_Image](https://github.com/VikramShenoy97/Neural-Style-Transfer/blob/master/Output_Images/Style_Transfer.jpg)


## Built With

* [Keras](https://keras.io) - Deep Learning Framework
* [TensorFlow](https://www.tensorflow.org) - Deep Learning Framework

## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Project is based on **Leon A. Gaty's** paper, [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576)
* Project is inspired by **Raymond Yuan's** blog, [*Neural Style Transfer*](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
