import sys
import tensorflow as tf
import numpy as np
import time
import keras.backend as K

from inspect import signature

class FGSM:
    def __init__(self, sess, model, use_log=True, targeted=True, batch_size=1, epsilon=0.3):
        """
        The implementation of Ian Goodfellow's FGSM attack.

        Returns adversarial examples for the supplied model.

        targeted: True if we should perform a targetted attack, False otherwise.
        default is targeted.
        """

        if use_log == False:
            print("use_log should be set to True for FGSM attack")
            #throw an exception
            return

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        #self.epsilon = tf.placeholder(tf.float32)
        self.epsilon = epsilon
        self.batch_size = batch_size

        shape = (batch_size,image_size,image_size,num_channels)

        self.x = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.y = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.logits = self.model.predict(self.x)
        #Generate the gradient of the loss function.
        self.adv_loss = K.categorical_crossentropy(self.logits, self.y, from_logits=True)
        if targeted:
            self.adv_loss = -self.adv_loss
        #grad = K.gradients(self.adv_loss, x)
        self.grad = K.gradients(self.adv_loss, [self.x])[0]

        # signed gradient
        self.normed_grad = K.sign(self.grad)

        # Multiply by constant epsilon
        self.scaled_grad = self.epsilon * self.normed_grad

        # Add perturbation to original example to obtain adversarial example
        self.adv_x = K.stop_gradient(self.x + self.scaled_grad)

        self.adv_x = K.clip(self.adv_x, -0.5, 0.5)

        return

    def attack(self, imgs, targets):
        """
        Perform the one-shot L_infinity FGSM attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """

        adv_x_concrete, adv_x_grad = self.sess.run([self.adv_x, self.grad], feed_dict={self.x: imgs,
                                                                                        self.y: targets})

        return adv_x_concrete, adv_x_grad

    """
    A dummy wrapper for the FGSM attack.  Just for invoked by test_FGSM.py (based on test_all.py)
    """
    def attack_batch(self, imgs, targets):

        adv_x_concrete, adv_x_grad = self.attack(imgs, targets)

        return adv_x_concrete, adv_x_grad
