import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys


class DataHandler(object):

    def __init__(self):

        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

        # Load Lena image to memory
        self.lena = Image.open('lena.jpg')

    def load_dataset(self):
        # Credit for this function: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_batch(self, subset, batch_size, use_target_distribution=False):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, 0, :].reshape((batch_size, 28, 28))

        # Resize from 28x28 to 64x64
        batch_resized = []
        for i in range(batch.shape[0]):
            # resize to 64x64 pixels
            batch_resized.append(scipy.ndimage.zoom(batch[i, :, :], 2.3, order=1))
        batch = np.stack(batch_resized)
        batch = batch.reshape((batch_size, 1, 64, 64))

        # Convert to RGB
        batch = np.concatenate([batch, batch, batch], axis=1)

        # Modify images if target distribution requested
        if use_target_distribution:

            # Binarize images
            batch[batch >= 0.5] = 1
            batch[batch < 0.5] = 0

            # For each image in the mini batch
            for i in range(batch_size):

                # Take a random crop of the Lena image (background)
                x_c = np.random.randint(0, self.lena.size[0] - 64)
                y_c = np.random.randint(0, self.lena.size[1] - 64)
                image = self.lena.crop((x_c, y_c, x_c + 64, y_c + 64))
                image = np.asarray(image).transpose((2, 0, 1)) / 255.0

                # Randomly alter the color distribution of the crop
                for j in range(3):
                    image[j, :, :] = (image[j, :, :] + np.random.uniform(0, 1)) / 2.0

                # Invert the color of pixels where there is a number
                image[batch[i, :, :, :] == 1] = 1 - image[batch[i, :, :, :] == 1]
                batch[i, :, :, :] = image

        # Rescale to range [-1, +1]
        batch = batch * 2 - 1

        # Image label
        labels = y[idx]

        return batch.astype('float32'), labels.astype('int32')

