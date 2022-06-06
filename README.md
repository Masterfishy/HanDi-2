# HanDi-2

A Discord Bot trained to identify images of numbers and correctly classify them.

## About HanDi-2

HanDi-2 is a Discord Bot based on my final group project for the machine learning course I took in my senior year at Virginia Tech.

For this project, our team trained two models using different approaches and compared the results. The original models were a multi-layer perceptron and K-Nearest Neighbors. We trained and tested the two models on data from the [MNIST handwritten digit dataset](https://www.tensorflow.org/datasets/catalog/mnist). On the dataset our models both had 97% accuracy.

In an attempt to make the results of our model more interesting, we programmed a Discord Bot using [Python Discord API](https://github.com/Rapptz/discord.py) to take user digit image input and classify the digit. In this real environment, HanDi did not perform as well.

While researching and training our model, we attempted to see if we could improve performance with Principal Component Analysis. The hope of this decomposition was to see if our models would generalize better. We found that while our models trained faster, our models' performance did not improve.

In another groups presentation of the same problem, they made use of a [Histogram of Oriented Gradients (HOG)](https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f) algorithm. This algorithm assisted with edge detection and toted higher accuracies than our model. After seeing their success with implementing HOG, I thought I'd like to see what this feature decomposition might do for our HanDi.

So that's what this is, this is HOG wild HanDi.
