import sys
import tensorflow as tf
import numpy as np
import time
#import keras.backend as K
import tensorflow as tf

class FGSM:
    def __init__(self, sess, model, use_log=True, targeted=True, batch_size=1):
        """
        The implementation of Ian Goodfellow's FGSM attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        default is targeted.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.epsilon = tf.placeholder(tf.float32)
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

    def attack(self, imgs, targets, epsilon):
        """
        Perform the one-shot L_infinity FGSM attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """

        print("Perofming FGSM attack")

        adv_x_concrete, loss = self.sess.run([self.adv_x, self.adv_loss], feed_dict={self.x: imgs,
                                                               self.y: targets,
                                                               self.epsilon: epsilon})
        print("Done on the FGSM attack")

        return adv_x_concrete, loss
# import sys
# import tensorflow as tf
# import numpy as np
# import time
# #import keras.backend as K
# import tensorflow as tf

# class FGSM:
#     def __init__(self, sess, model, use_log=True, targeted=True, batch_size=1):
#         """
#         The implementation of Ian Goodfellow's FGSM attack.
#         Returns adversarial examples for the supplied model.
#         targeted: True if we should perform a targetted attack, False otherwise.
#         default is targeted.
#         """

#         image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
#         self.sess = sess
#         self.model = model
#         self.epsilon = tf.placeholder(tf.float32)
#         self.batch_size = batch_size

#         shape = (batch_size,image_size,image_size,num_channels)

#         self.x = tf.Variable(np.zeros(shape), dtype=tf.float32)
#         self.y = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
#         self.logits = self.model.predict(self.x)
#         #Generate the gradient of the loss function.
#         # self.adv_loss = K.categorical_crossentropy(self.logits, self.y, from_logits=True)
#         self.adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
#         if targeted:
#             self.adv_loss = -self.adv_loss
#         #grad = K.gradients(self.adv_loss, x)
#         # self.grad = K.gradients(self.adv_loss, [self.x])[0]
#         self.grad, = tf.gradients(self.adv_loss, self.x)
#         # signed gradient
#         # self.normed_grad = K.sign(self.grad)
#         self.normalized_grad = tf.sign(self.grad)
#         self.normalized_grad = tf.stop_gradient(self.normalized_grad)
#         # Multiply by constant epsilon
#         self.scaled_grad = self.epsilon * self.normalized_grad
#         self.adv_x = self.x + self.scaled_grad
#         # Add perturbation to original example to obtain adversarial example
#         # self.adv_x = K.stop_gradient(self.x + self.scaled_grad)

#         #self.adv_x = K.clip(self.adv_x, -0.5, 0.5)
#         self.adv_x = tf.clip_by_value(self.adv_x, -0.5, 0.5)
        

#     def attack(self, imgs, targets, epsilon):
#         """
#         Perform the one-shot L_infinity FGSM attack on the given images for the given targets.
#         If self.targeted is true, then the targets represents the target labels.
#         If self.targeted is false, then targets are the original class labels.
#         """

#         print("Perofming FGSM attack")

#         adv_x_concrete = self.sess.run(self.adv_x, feed_dict={self.x: imgs,
#                                                                self.y: targets,
#                                                                self.epsilon: epsilon})
#         print("Done on the FGSM attack")

#         return adv_x_concrete
