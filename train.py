from os.path import exists, join, dirname, basename
import os
import time
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image
import scipy
import sys

from adda_network import ADDANet
from data_handler import DataHandler


class Trainer():

    def __init__(self, ):

        # Initialize data loader
        self.data = DataHandler()

        # Initialize model
        self.ada_network = ADDANet()

    def train(self, n_epochs, n_iterations, mini_batch_size, lr_0, k_classifier, k_discriminator, train_classifier_only):

        # Params
        list_perf_classifier_source = []
        list_perf_classifier_target = []
        list_perf_discriminator = []
        lr = lr_0

        # Epochs
        for epoch_i in range(n_epochs):

            # Iterations
            for iteration_i in range(n_iterations):

                # Default initialization
                loss_classifier = np.nan
                loss_classifier_only = np.nan
                loss_discriminator = np.nan

                # Train classifier
                for _ in range(k_classifier):

                    # Get data
                    inputs_source, labels_source = self.data.get_batch('train', mini_batch_size / 2, use_target_distribution=False)
                    inputs_target, labels_target = self.data.get_batch('train', mini_batch_size / 2, use_target_distribution=True)
                    inputs = np.concatenate([inputs_source, inputs_target], axis=0)
                    labels_classifier_var = np.concatenate([labels_source, np.zeros_like(labels_source)], axis=0)
                    labels_domain_var = np.concatenate([np.ones_like(labels_source), np.zeros_like(labels_source)], axis=0)

                    # Train classifier
                    (loss_classifier, loss_classifier_only, prediction_classifier) = self.ada_network.classifier_backward_pass(
                        inputs, labels_classifier_var, labels_domain_var, lr, train_classifier_only
                    )

                # Train domain discriminator
                if not train_classifier_only:
                    for _ in range(k_discriminator):

                        # Get data
                        inputs_source, labels_source = self.data.get_batch('train', mini_batch_size / 2, use_target_distribution=False)
                        inputs_target, labels_target = self.data.get_batch('train', mini_batch_size / 2, use_target_distribution=True)
                        inputs = np.concatenate([inputs_source, inputs_target], axis=0)
                        labels_domain_var = np.concatenate([np.ones_like(labels_source), np.zeros_like(labels_source)], axis=0)

                        # Train discriminator
                        (loss_discriminator, prediction_discriminator) = self.ada_network.discriminator_backward_pass(
                            inputs, labels_domain_var, lr
                        )

                # Get validation data
                inputs_source, labels_source = self.data.get_batch('valid', mini_batch_size / 2, use_target_distribution=False)
                inputs_target, labels_target = self.data.get_batch('valid', mini_batch_size / 2, use_target_distribution=True)
                inputs = np.concatenate([inputs_source, inputs_target], axis=0)
                labels_domain_var = np.concatenate([np.ones_like(labels_source), np.zeros_like(labels_source)], axis=0)

                # Forward pass for validation data
                perf_classifier_source, pred_classifier_source = self.ada_network.classifier_forward_pass(inputs_source, labels_source)
                perf_classifier_target, pred_classifier_target = self.ada_network.classifier_forward_pass(inputs_target, labels_target)
                perf_discriminator, prediction_discriminator = self.ada_network.discriminator_forward_pass(inputs, labels_domain_var)

                # Accumulate results
                list_perf_classifier_source.append(perf_classifier_source)
                list_perf_classifier_target.append(perf_classifier_target)
                list_perf_discriminator.append(perf_discriminator)

                # Print some results
                print('[E %04d, It %04d, LR %0.6f] lc %0.3f - lco %0.3f - ld %0.3f - pcs %0.3f - pct %0.3f - pd %0.3f' %
                      (epoch_i, iteration_i, lr,
                       loss_classifier, loss_classifier_only, loss_discriminator,
                       np.mean(list_perf_classifier_source[-200:]),
                       np.mean(list_perf_classifier_target[-200:]),
                       np.mean(list_perf_discriminator[-200:]),
                     ))

            # Decrease learning rate at end of epoch
            lr *= 0.1

        # Collect results
        results = {
            'classifier_source': (inputs_source, labels_source, pred_classifier_source),
            'classifier_target': (inputs_target, labels_target, pred_classifier_target)
        }

        return results


def plot_results(results, output_path):

    # Plot each input image with its respective label and predicted prob
    for i in range(results['classifier_target'][0].shape[0]):
        plt.subplot(6, 6, i + 1)
        image = (results['classifier_target'][0][i, ...].transpose(1, 2, 0) + 1) / 2.0
        pred = int(np.argmax(results['classifier_target'][2], axis=1)[i])
        label = int(results['classifier_target'][1][i])
        plt.imshow(image)
        plt.title('%d - %d' % (label, pred))
        plt.axis('off')
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":

    # Train classifier only (for comparison)
    trainer = Trainer()
    results = trainer.train(n_epochs=2, n_iterations=500, mini_batch_size=64, lr_0=0.001,
                            k_classifier=10, k_discriminator=1, train_classifier_only=True)
    plot_results(results, output_path='resources/validation_64_classifier_only.png')

    # Train classifier with domain adaptation
    trainer = Trainer()
    results = trainer.train(n_epochs=2, n_iterations=500, mini_batch_size=64, lr_0=0.001,
                            k_classifier=10, k_discriminator=1, train_classifier_only=False)
    plot_results(results, output_path='resources/validation_64_full.png')
