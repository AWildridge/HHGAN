from __future__ import division, print_function

import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

import keras.backend as K
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import _Merge
from keras.models import Model, Sequential
from keras.utils import plot_model
"""
Based on code by Erik Linder-Norén: https://github.com/eriklindernoren/Keras-GAN
Modified to accept inputs from non-image data and fixed some training issues
resulting from calculating the loss functions incorrectly. Also added more functionality
for monitoring the training of the WGAN-GP
"""


LAMBDA = 15


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((25, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():
    def __init__(self):

        self.feature_dim = 25
        self.img_shape = (self.feature_dim,)
        self.latent_dim = 200

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)
        #optimizer = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # MC input
        mc_event = Input(shape=self.img_shape)

        # Noise input
        latent_input = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        generated_event = self.generator(latent_input)

        # Discriminator determines validity of the real and fake images
        generated_validity = self.critic(generated_event)
        mc_validity = self.critic(mc_event)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([mc_event, generated_event])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[mc_event, latent_input],
                                  outputs=[mc_validity, generated_validity, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, LAMBDA])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(
            loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        self.gradients = K.gradients(y_pred, averaged_samples)[0]

        # compute the euclidean norm by squaring ...
        self.gradients_sqr = K.square(self.gradients)
        #   ... summing over the rows ...
        self.gradients_sqr_sum = K.sum(self.gradients_sqr,
                                  axis=np.arange(1, len(self.gradients_sqr.shape)))
        #   ... and sqrt
        self.gradient_l2_norm = K.sqrt(self.gradients_sqr_sum)

        # compute lambda * (1 - ||grad||)^2 still for each single sample
        self.gradient_penalty = K.square(1 - self.gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(self.gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(250, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(25, activation="tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Dense(250, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(125, activation="tanh"))
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):
        # Load the dataset
        X_train = np.load('NHETraining.npy')

        # Rescale -1 to 1
        #        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                          [valid, fake, dummy])
                '''generated_events = self.generator(noise)
                interpolated_img = RandomWeightedAverage()([imgs, generated_events])
                validity_interpolated = self.critic(interpolated_img)
                gradient_penalty_loss(validity_interpolated, validity_interpolated, interpolated_img)
                print('Using gradients: ' + str(self.gradients.eval()))
                print('l2 norm is: ' + str(self.gradient_l2_norm.eval()))
                print('gradient penalty is: ' + str(self.gradient_penalty.eval()))'''

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print(
                "%d [D loss: %f + %f = %f [G loss: %f]" % (epoch, -(d_loss[0] + d_loss[1]), d_loss[2],
                                                         -(d_loss[0] + d_loss[1]) + d_loss[2], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r = 38333
        noise = np.random.normal(0, 1, (r, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        np.save('../../tests/lambda15/' + str(epoch) + '_20%_out', gen_imgs)
        # SECTION MODIFIED (This samples for the output images.  Produces 60K images in a .npy file at the specified epochs(sample_interval).)


#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig("images/mnist_%d.png" % epoch)
#        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    # Training parameters.  30001 is used to ensure the code can process operations occouring after 30000 epochs.
    wgan.train(epochs=3001, batch_size=250, sample_interval=1000)

print(np.load('Train7/3000_20%_out.npy')[100])
