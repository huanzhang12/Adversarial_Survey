import sys
import tensorflow as tf
import numpy as np
import time
#import keras.backend as K
import tensorflow as tf

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
        #preds_max = tf.reduce_max(self.logits, 1, keep_dims=True)
        #self.y = tf.to_float(tf.equal(self.logits, preds_max))
        self.y = self.y / tf.reduce_sum(self.y, 1, keep_dims=True)
        #Generate the gradient of the loss function.
        #decide whether use logits or softmax
        if use_log:#use logits
            op = self.logits.op
            if "softmax" in str(op).lower():
                self.logits, = op.inputs
            else:
                self.logits = self.logits
        #else:
            #use softmax

        self.adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        if targeted:
            self.adv_loss = -self.adv_loss
        
        self.grad, = tf.gradients(self.adv_loss, self.x)
        # signed gradient
        
        self.normalized_grad = tf.sign(self.grad)
        self.normalized_grad = tf.stop_gradient(self.normalized_grad)
        # Multiply by constant epsilon
        self.scaled_grad = self.epsilon * self.normalized_grad
        self.adv_x = self.x + self.scaled_grad
        # Add perturbation to original example to obtain adversarial example
       

        self.adv_x = tf.clip_by_value(self.adv_x, -0.5, 0.5)
        
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
