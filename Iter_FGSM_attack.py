import sys
import tensorflow as tf
import numpy as np
from six.moves import xrange
import time

class Iter_FGSM:
    def __init__(self, sess, model):
        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.x = tf.placeholder(shape=[None, image_size, image_size, num_channels], dtype=tf.float32)
        self.logits = self.model.predict(self.x)
        self.predicts = self.logits

    def init_attack(self, sess, model, iter_num, use_log=True, targeted=True, batch_size=1, ord=np.inf, clip_min=-0.5, clip_max=0.5):
       
        """
        The implementation of Ian Goodfellow's Iterative-FGSM attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        default is targeted.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.batch_size = batch_size
        # self.eps = eps
        # self.eps_iter = eps_iter
        self.iter_num = iter_num
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        shape = (batch_size,image_size,image_size,num_channels)

        self.eps = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='eps_var')
        self.eps_iter = tf.Variable(np.zeros(batch_size), dtype=tf.float32,name='eps_iter_var')
        self.assign_eps = tf.placeholder(tf.float32,[batch_size],name='feed_eps')
        self.assign_eps_iter = tf.placeholder(tf.float32,[batch_size],name='feed_eps_iter')
        self.setup = []
        self.assign_eps_op = self.eps.assign(self.assign_eps)
        self.setup.append(self.assign_eps_op)
        self.assign_eps_iter_op = self.eps_iter.assign(self.assign_eps_iter)
        self.setup.append(self.assign_eps_iter_op)

        self.imgs = tf.Variable(np.zeros(shape), dtype=tf.float32)
        # self.x = tf.Variable(np.zeros(shape), dtype=tf.float32)
        # self.x = tf.placeholder(shape=[None, image_size, image_size, num_channels], dtype=tf.float32)
        self.y = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        # self.logits = self.model.predict(self.x)
        
        #Generate the gradient of the loss function.
        #preds_max = tf.reduce_max(self.logits, 1, keep_dims=True)
        #self.y = tf.to_float(tf.equal(self.logits, preds_max))
        self.y = self.y / tf.reduce_sum(self.y, 1, keep_dims=True)
        #Generate the gradient of the loss function.
        #decide whether use logits or softmax
        # self.predicts = self.logits
        
        # if use_log:
        #     op = self.logits.op
        #     if "softmax" in str(op).lower():
        #         self.logits, = op.inputs

        if use_log:
            self.adv_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        else:
            self.adv_loss = -tf.reduce_sum(self.logits * self.y, axis = 1)
        
        if targeted:  
            self.adv_loss = -self.adv_loss

        self.grad, = tf.gradients(self.adv_loss, self.x)
        # signed gradient
        if self.ord == np.inf:
        # Take sign of gradient
            self.normalized_grad = tf.sign(self.grad)
            self.normalized_grad = tf.stop_gradient(self.normalized_grad)
        elif self.ord == 1:
            red_ind = list(xrange(1, len(self.x.get_shape())))
            self.normalized_grad = self.grad / tf.reduce_sum(tf.abs(self.grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
        elif self.ord == 2:
            red_ind = list(xrange(1, len(self.x.get_shape())))
            square = tf.reduce_sum(tf.square(self.grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
            self.normalized_grad = self.grad / tf.sqrt(square)
        else:
            raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")
            
        # Multiply by constant epsilon
        
        #self.eps_iter = tf.cast(self.eps_iter, tf.float32)
        self.EPS_ITER = tf.reshape(tf.tile(tf.reshape(self.eps_iter,[-1,1]),[1, image_size*image_size*num_channels]),shape)
        # self.eps_iter = tf.reshape(self.eps_iter,[-1,1])
        # self.eps_iter = tf.tile(self.eps_iter, [1, image_size*image_size*num_channels])
        # self.eps_iter = tf.reshape(self.eps_iter,[batch_size,image_size,image_size,num_channels])


        #self.eps = tf.cast(self.eps, tf.float32)
        self.EPS = tf.reshape(tf.tile(tf.reshape(self.eps,[-1,1]),[1, image_size*image_size*num_channels]),shape)
        # self.eps = tf.reshape(self.eps,[-1,1])
        # self.eps = tf.tile(self.eps, [1, image_size*image_size*num_channels])
        # self.eps = tf.reshape(self.eps,[batch_size,image_size,image_size,num_channels])
        self.scaled_grad = tf.multiply(self.EPS_ITER,self.normalized_grad)
        print(self.scaled_grad.get_shape())
        self.adv_x = self.x + self.scaled_grad

        self.adv_x = tf.clip_by_value(self.adv_x, self.clip_min, self.clip_max)
        self.eta = self.adv_x - self.imgs
        
        #self.eta = tf.clip_by_value(self.eta, -self.eps, self.eps)
        
        # Clipping perturbation eta to self.ord norm ball
        if self.ord == np.inf:
            self.eta = tf.clip_by_value(self.eta, -self.EPS, self.EPS)
        elif self.ord in [1, 2]:
            reduc_ind = list(xrange(1, len(self.eta.get_shape())))
            if self.ord == 1:
                norm = tf.reduce_sum(tf.abs(self.eta), reduction_indices=reduc_ind, keep_dims=True)
            elif self.ord == 2:
                norm = tf.sqrt(tf.reduce_sum(tf.square(self.eta), reduction_indices=reduc_ind, keep_dims=True))
            self.eta = tf.multiply(self.eta, self.EPS) / norm
        return

    def predict(self, imgs):
        # imgs = np.array(imgs, dtype = 'f')
        # self.sess.run(self.setup, {self.assign_eps:np.zeros(self.batch_size), self.assign_eps_iter:np.zeros(self.batch_size)} )
        predicts = self.sess.run([self.predicts], feed_dict = {self.x : imgs})
        return predicts

    def one_attack(self, imgs, targets, eps, eps_iter, verbose = False):
        #eta =  np.dtype('Float64')
        #adv_x = np.dtype('Float64')
        eta = 0
        imgs = np.array(imgs, dtype = 'f')
        self.sess.run(self.setup, {self.assign_eps:eps, self.assign_eps_iter:eps_iter} )
        for i in range(self.iter_num):
            #eta = self.sess.run(self.adv_x, feed_dict = {self.x : imgs + eta, self.y : targets}) - imgs
            eta, loss, predicts = self.sess.run([self.eta, self.adv_loss, self.predicts], feed_dict = {self.imgs : imgs, self.x : imgs + eta, self.y : targets})
            if verbose:
                print('iteration:', i, 'loss is: ', loss[:10])
                for idx, predict in enumerate(predicts[:10]):
                    print("Classification {}: {}".format(idx, np.argsort(predict)[-1:-11:-1]))
                    print("Probabilities/Logits {}: {}".format(idx, np.sort(predict)[-1:-11:-1]))
                print()
        adv_x = eta + imgs
        adv_x = np.clip(adv_x, self.clip_min, self.clip_max)


        return adv_x


    def attack(self, inputs, targets, targeted, initial_eps = 0.3, max_attempts = 10, verbose = False):
        batch_size = len(inputs)
        success_log = [[] for _ in range(batch_size)]
        eps_val = [[initial_eps] for _ in range(batch_size)] 
        best_adv = np.full(inputs.shape,np.nan)
        for try_index in range(max_attempts):
            eps = np.array([item[try_index] for item in eps_val])
            print("***********eps:***********:",eps)
            eps_iter = eps / self.iter_num
            adv = self.one_attack(inputs, targets,  eps=eps, eps_iter=eps_iter)
            for i in range(len(adv)):
                print('----------------------------------------')
                # show(inputs[i], "original_{}.png".format(i))
                original_predict = np.squeeze(self.predict(inputs[i:i+1]))
                print("Original Classification:", np.argsort(original_predict)[-1:-11:-1])
                print("Original Probabilities/Logits:", np.sort(original_predict)[-1:-11:-1])
                print("Original Classification minimum:", np.argsort(original_predict)[0:10])
                print("Original Probabilities/Logits minimum:", np.sort(original_predict)[0:10])

                target_label = np.argmax(targets[i])
                attack_label = None
                success = False
                print("Target:", target_label)
                # if the array contains NaN, the solver did not return a solution
                if (np.any(np.isnan(adv[i:i+1]))):
                    print('Attack failed. (solver returned NaN)')
                    l0 = l1 = l2 = linf = np.nan
                    continue
                else:
                    # print("Adversarial:")
                    # show(adv[i], "adversarial_{}.png".format(i))
                    # print("Noise:")
                    # show(adv[i] - inputs[i], "attack_diff.png")

                    adv_predict = np.squeeze(self.predict(adv[i:i+1]))
                    print("Adversarial Classification:", np.argsort(adv_predict)[-1:-11:-1])
                    print("Adversarial Probabilities/Logits:", np.sort(adv_predict)[-1:-11:-1])
                    print("Adversarial Classification minimum:", np.argsort(adv_predict)[0:10])
                    print("Adversarial Probabilities/Logits minimum:", np.sort(adv_predict)[0:10])
                    attack_label = np.argmax(adv_predict)

                    if targeted:
                        success = np.argsort(adv_predict)[-1] == target_label
                    else:
                        success = np.argsort(adv_predict)[-1] != target_label
                    success_log[i] += [success]
                    if success:
                        best_adv[i] = adv[i]
                        print("Attack succeeded.")
                        if try_index + 1 < max_attempts:
                            if any(not _ for _ in success_log[i]):
                                last_false = len(success_log[i]) - success_log[i][::-1].index(False) - 1
                                eps_val[i] += [0.5 * (eps_val[i][try_index] + eps_val[i][last_false])]
                            else:
                                eps_val[i] += [eps_val[i][try_index] * 0.5] 
                    else:
                        print("Attack failed.")
                        if try_index + 1 < max_attempts:
                            if any(_ for _ in success_log[i]):
                                last_true = len(success_log[i]) - success_log[i][::-1].index(True) - 1
                                eps_val[i] += [0.5 * (eps_val[i][try_index] + eps_val[i][last_true])]
                            else:
                                eps_val[i] += [eps_val[i][try_index] * 2.0]
                        continue
        print("success log:", success_log)
        print("eps values:", eps_val)
        return best_adv, eps_val


    def attack_batch(self, imgs, targets, targeted):
        return self.attack(imgs, targets, targeted)

    
