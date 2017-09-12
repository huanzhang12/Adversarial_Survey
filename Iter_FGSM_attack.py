import sys
import tensorflow as tf
import numpy as np
import time

class Iter_FGSM:
    def __init__(self, sess, model, eps, eps_iter, iter_num, use_log = True,targeted=True, batch_size=1):
        
        """
        The implementation of Ian Goodfellow's FGSM attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        default is targeted.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.batch_size = batch_size
        self.eps = eps
        self.eps_iter = eps_iter
        self.iter_num = iter_num
        shape = (batch_size,image_size,image_size,num_channels)

        self.x = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.y = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.eta = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.logits = self.model.predict(self.x)
        #Generate the gradient of the loss function.
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
        self.scaled_grad = self.eps_iter * self.normalized_grad
        self.adv_x = self.x + self.scaled_grad

        self.adv_x = tf.clip_by_value(self.adv_x, -0.5, 0.5)
        
        return

    def attack(self, imgs, targets):
        print("performing iter_FGSM attack")
        eta =  0
        for i in range(self.iter_num):
            adv_x = self.sess.run(self.adv_x, feed_dict={self.x: imgs + eta, self.y: targets })
            eta = adv_x - imgs
            eta = np.clip(eta, -self.eps, self.eps)
        adv_x = eta + imgs
        adv_x = np.clip(adv_x, -0.5,0.5)
        print("Done on the iter_FGSM attack")
        return adv_x
