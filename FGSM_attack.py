import sys
import tensorflow as tf
import numpy as np
import time
from six.moves import xrange
#from inspect import signature

class FGSM:
    def __init__(self, sess, model, eps, use_log=True, targeted=True, batch_size=1, ord=np.inf, clip_min=-0.5, clip_max=0.5):
        """
        The implementation of Ian Goodfellow's FGSM attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        default is targeted.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.eps = eps
        self.batch_size = batch_size

        self.clip_min = clip_min
        self.clip_max = clip_max

        shape = (batch_size,image_size,image_size,num_channels)

        self.x = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.y = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        
        self.logits = self.model.predict(self.x)
        
        
        self.y = self.y / tf.reduce_sum(self.y, 1, keep_dims=True)
        #Generate the gradient of the loss function.
        #decide whether use logits or softmax
        
        # if use_log:
        #     op = self.logits.op
        #     if "softmax" in str(op).lower():
        #         self.logits, = op.inputs

        # self.adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        if use_log:
            self.adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        else:
            self.adv_loss = -tf.reduce_sum(self.logits * self.y, axis = 1)
        
        if targeted:
            self.adv_loss = -self.adv_loss

        self.grad, = tf.gradients(self.adv_loss, self.x)
        
        # signed gradient
        if ord == np.inf:
        # Take sign of gradient
            self.normalized_grad = tf.sign(self.grad)
            self.normalized_grad = tf.stop_gradient(self.normalized_grad)
        elif ord == 1:
            red_ind = list(xrange(1, len(self.x.get_shape())))
            self.normalized_grad = self.grad / tf.reduce_sum(tf.abs(self.grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
        elif ord == 2:
            red_ind = list(xrange(1, len(self.x.get_shape())))
            square = tf.reduce_sum(tf.square(self.grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
            self.normalized_grad = self.grad / tf.sqrt(square)
        else:
            raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")
        
        # Multiply by constant epsilon
        self.scaled_grad = self.eps * self.normalized_grad
        
        # Add perturbation to original example to obtain adversarial example
        self.adv_x = self.x + self.scaled_grad

        self.adv_x = tf.clip_by_value(self.adv_x, self.clip_min, self.clip_max)
        
        return

    def attack(self, imgs, targets):
        """
        Perform the one-shot FGSM attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """

        print("Perofming FGSM attack")

        adv_x_concrete = self.sess.run(self.adv_x, feed_dict={self.x: imgs,
                                                               self.y: targets })
        print("Done on the FGSM attack")

        return adv_x_concrete

    """
    A dummy wrapper for the FGSM attack.  Just for invoked by test_FGSM.py (based on test_all.py)
    """
    def attack_batch(self, imgs, targets):

        adv_x_concrete = self.sess.run(self.adv_x, feed_dict={self.x: imgs, self.y: targets})

        return adv_x_concrete
