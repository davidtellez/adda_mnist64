import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, get_output
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
import numpy as np


class ADDANet(object):

    def __init__(self):

        # Define inputs
        input_var = T.ftensor4('input_var')  # input images (batchx3x64x64)
        labels_classifier_var = T.ivector('labels_classifier_var')  # labels for images
        labels_domain_var = T.ivector('labels_domain_var')  # labels for domain (1 for source, 0 for target)
        learning_rate = T.fscalar('learning_rate')

        # Define classifier networks
        network_classifier = self.network_classifier(input_var)
        network_discriminator = self.network_discriminator(network_classifier['classifier/pool1'])

        # Define outputs
        prediction_classifier = get_output(network_classifier['classifier/output'])  # prob image classification
        prediction_discriminator = get_output(network_discriminator['discriminator/output'])  # prob image domain (should be 1 for source)

        # Define losses (objectives)
        loss_classifier_only = T.mean(categorical_crossentropy(prediction_classifier, labels_classifier_var) * labels_domain_var)
        loss_discriminator = T.mean(categorical_crossentropy(prediction_discriminator, labels_domain_var))
        loss_classifier = loss_classifier_only - loss_discriminator

        # Define performance
        perf_classifier_only = categorical_accuracy(prediction_classifier, labels_classifier_var).mean()
        perf_discriminator = categorical_accuracy(prediction_discriminator, labels_domain_var).mean()

        # Define params
        params_classifier = lasagne.layers.get_all_params(network_classifier['classifier/output'], trainable=True)
        params_discriminator = lasagne.layers.get_all_params(network_discriminator['discriminator/output'], trainable=True)
        params_discriminator = [param for param in params_discriminator if 'discriminator' in param.name]

        # Define updates
        updates_classifier = lasagne.updates.adam(loss_classifier, params_classifier, learning_rate=learning_rate)
        updates_classifier_only = lasagne.updates.adam(loss_classifier_only, params_classifier, learning_rate=learning_rate)
        updates_discriminator = lasagne.updates.adam(loss_discriminator, params_discriminator, learning_rate=learning_rate)

        # Define training functions
        self.train_fn_classifier = theano.function(
            [input_var, labels_classifier_var, labels_domain_var, learning_rate],
            [loss_classifier, loss_classifier_only, prediction_classifier],
            updates=updates_classifier)
        self.train_fn_classifier_only = theano.function(
            [input_var, labels_classifier_var, labels_domain_var, learning_rate],
            [loss_classifier, loss_classifier_only, prediction_classifier],
            updates=updates_classifier_only)
        self.train_fn_discriminator = theano.function(
            [input_var, labels_domain_var, learning_rate],
            [loss_discriminator, prediction_discriminator],
            updates=updates_discriminator)

        # Define validation functions
        self.valid_fn_classifier = theano.function(
            [input_var, labels_classifier_var],
            [perf_classifier_only, prediction_classifier])

        self.valid_fn_discriminator = theano.function(
            [input_var, labels_domain_var],
            [perf_discriminator, prediction_discriminator])

    def network_classifier(self, input_var):

        network = {}
        network['classifier/input'] = InputLayer(shape=(None, 3, 64, 64), input_var=input_var, name='classifier/input')
        network['classifier/conv1'] = Conv2DLayer(network['classifier/input'], num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='classifier/conv1')
        network['classifier/pool1'] = MaxPool2DLayer(network['classifier/conv1'], pool_size=2, stride=2, pad=0, name='classifier/pool1')
        network['classifier/conv2'] = Conv2DLayer(network['classifier/pool1'], num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='classifier/conv2')
        network['classifier/pool2'] = MaxPool2DLayer(network['classifier/conv2'], pool_size=2, stride=2, pad=0, name='classifier/pool2')
        network['classifier/conv3'] = Conv2DLayer(network['classifier/pool2'], num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='classifier/conv3')
        network['classifier/pool3'] = MaxPool2DLayer(network['classifier/conv3'], pool_size=2, stride=2, pad=0, name='classifier/pool3')
        network['classifier/conv4'] = Conv2DLayer(network['classifier/pool3'], num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='classifier/conv4')
        network['classifier/pool4'] = MaxPool2DLayer(network['classifier/conv4'], pool_size=2, stride=2, pad=0, name='classifier/pool4')
        network['classifier/dense1'] = DenseLayer(network['classifier/pool4'], num_units=64, nonlinearity=rectify, name='classifier/dense1')
        network['classifier/output'] = DenseLayer(network['classifier/dense1'], num_units=10, nonlinearity=softmax, name='classifier/output')

        return network

    def network_discriminator(self, features):

        network = {}
        network['discriminator/conv2'] = Conv2DLayer(features, num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='discriminator/conv2')
        network['discriminator/pool2'] = MaxPool2DLayer(network['discriminator/conv2'], pool_size=2, stride=2, pad=0, name='discriminator/pool2')
        network['discriminator/conv3'] = Conv2DLayer(network['discriminator/pool2'], num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='discriminator/conv3')
        network['discriminator/pool3'] = MaxPool2DLayer(network['discriminator/conv3'], pool_size=2, stride=2, pad=0, name='discriminator/pool3')
        network['discriminator/conv4'] = Conv2DLayer(network['discriminator/pool3'], num_filters=32, filter_size=3, stride=1, pad='valid', nonlinearity=rectify, name='discriminator/conv4')
        network['discriminator/pool4'] = MaxPool2DLayer(network['discriminator/conv4'], pool_size=2, stride=2, pad=0, name='discriminator/pool4')
        network['discriminator/dense1'] = DenseLayer(network['discriminator/pool4'], num_units=64, nonlinearity=rectify, name='discriminator/dense1')
        network['discriminator/output'] = DenseLayer(network['discriminator/dense1'], num_units=2, nonlinearity=softmax, name='discriminator/output')

        return network

    def classifier_forward_pass(self, tensor, labels):

        [perf_classifier_only, prediction_classifier] = self.valid_fn_classifier(tensor.astype('float32'), labels.astype('int32'))
        return perf_classifier_only, prediction_classifier

    def discriminator_forward_pass(self, tensor, labels):

        [perf_discriminator, prediction_discriminator] = self.valid_fn_discriminator(tensor.astype('float32'), labels.astype('int32'))
        return perf_discriminator, prediction_discriminator

    def classifier_backward_pass(self, input_var, labels_classifier_var, labels_domain_var, learning_rate, classifier_only=False):

        if classifier_only:
            [loss_classifier, loss_classifier_only, prediction_classifier] = \
                self.train_fn_classifier_only(input_var.astype('float32'), labels_classifier_var.astype('int32'),
                                         labels_domain_var.astype('int32'), np.array(learning_rate).astype('float32'))
        else:
            [loss_classifier, loss_classifier_only, prediction_classifier] = \
                self.train_fn_classifier(input_var.astype('float32'), labels_classifier_var.astype('int32'),
                                         labels_domain_var.astype('int32'), np.array(learning_rate).astype('float32'))

        return loss_classifier, loss_classifier_only, prediction_classifier

    def discriminator_backward_pass(self, input_var, labels_domain_var, learning_rate):

        [loss_discriminator, prediction_discriminator] = self.train_fn_discriminator(input_var.astype('float32'),
             labels_domain_var.astype('int32'), np.array(learning_rate).astype('float32'))

        return loss_discriminator, prediction_discriminator
