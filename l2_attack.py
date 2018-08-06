## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np
import time

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 100   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 0.1     # the initial constant c to pick as a first guess

class CarliniL2:
    def __init__(self, sess, model,max_batch_size=3000):
        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.model = model
        self.modifier = tf.Variable(tf.zeros(shape=(max_batch_size, image_size, image_size, num_channels),dtype=np.float32), validate_shape=False)
        #self.modifier = tf.Variable(np.zeros(shape=(1, image_size, image_size, num_channels),dtype=np.float32), validate_shape=True)
        print("**************self.modifier shape:",self.modifier.get_shape())
        self.assign_modifier = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels])
        self.assign_modifier_op = tf.assign(self.modifier, self.assign_modifier, validate_shape=False)
        # self.assign_modifier_op = tf.assign(self.modifier, self.assign_modifier, validate_shape=True)
        self.timg = tf.placeholder(shape=[None, image_size, image_size, num_channels], dtype=tf.float32)
        self.newimg = tf.tanh(self.modifier + self.timg)/2
        self.predicts = self.model.predict(self.newimg)

    def init_attack(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS, print_every = 1, early_stop_iters = 0,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 use_log = False, adam_beta1 = 0.9, adam_beta2 = 0.999):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        print("early stop:", self.early_stop_iters)
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)
        self.shape = shape
        
        # the variable we're going to optimize over
        # self.modifier = tf.Variable(np.load('black_iter_350.npy').astype(np.float32).reshape(shape)?)

        # these are variables to be more efficient in sending data to tf
        # self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        
        # prediction BEFORE-SOFTMAX of the model
        # self.output = self.predict(self.newimg)
        self.output = self.predicts
        # self.output = self.predict(self.newimg)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.timg)/2),[1,2,3])
        
        # compute the probability of the label class versus the maximum other
        self.real = tf.reduce_sum((self.tlab)*self.output,1)
        self.other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        if self.TARGETED:
            if use_log:
                # loss1 = tf.maximum(- tf.log(self.other), - tf.log(self.real))
                # loss1 = - tf.log(self.real)
                loss1 = tf.maximum(0.0, tf.log(self.other + 1e-30) - tf.log(self.real + 1e-30))
            else:
                # if targetted, optimize for making the other class most likely
                loss1 = tf.maximum(0.0, self.other-self.real+self.CONFIDENCE)
        else:
            if use_log:
                # loss1 = tf.log(self.real)
                loss1 = tf.maximum(0.0, tf.log(self.real + 1e-30) - tf.log(self.other + 1e-30))
            else:
            # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0.0, self.real-self.other+self.CONFIDENCE)

        # sum up the losses
        # self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss2 = self.l2dist
        # self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss1 = self.const * loss1
        self.loss = self.loss1+self.loss2
        # self.loss = self.loss1
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        # optimizer = tf.train.MomentumOptimizer(self.LEARNING_RATE, 0.99)
        # optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE)
        # optimizer = tf.train.AdadeltaOptimizer(self.LEARNING_RATE)
        # optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE, adam_beta1, adam_beta2)
        # self.train = optimizer.minimize(self.loss, var_list=[self.modifier])
        self.grad = tf.gradients(self.output,self.newimg)[0]
        self.train = self.adam_optimizer_tf(self.loss, self.modifier)
        # self.train = self.IFGSM_optimizer_tf(self.loss, self.modifier)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        # these are the variables to initialize when we run
        self.setup = []
        # self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        # self.grad_op = tf.gradients(self.loss, self.modifier)
        
        # self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)
        self.init = tf.variables_initializer(var_list=new_vars)

    def adam_optimizer_tf(self, loss, var):
        with tf.name_scope("adam_optimier"):
            self.grad = tf.gradients(loss, var)[0]
            # self.noise = tf.random_normal(self.shape, 0.0, 1.0)
            # self.noise = 0
            self.beta1 = tf.constant(0.9)
            self.beta2 = tf.constant(0.999)
            self.lr = tf.constant(self.LEARNING_RATE)
            self.epsilon = 1e-8
            self.epoch = tf.Variable(1, dtype = tf.float32)
            self.mt = tf.Variable(np.zeros(self.shape), dtype = tf.float32)
            self.vt = tf.Variable(np.zeros(self.shape), dtype = tf.float32)
            self.new_mt = self.beta1 * self.mt + (1 - self.beta1) * self.grad
            self.new_vt = self.beta2 * self.vt + (1 - self.beta2) * tf.square(self.grad)
            self.corr = tf.sqrt(1 - tf.pow(self.beta2, self.epoch)) / (1 - tf.pow(self.beta1, self.epoch))
            self.delta = self.lr * self.corr * (self.new_mt / tf.sqrt(self.new_vt+self.epsilon)) 
            # self.delta = self.lr * self.corr * ((self.new_mt / tf.sqrt(self.new_vt+self.epsilon)) + self.noise / tf.pow(self.epoch + 1, 0.2))
            # delta = self.lr * corr * ((new_mt / tf.sqrt(new_vt + self.epsilon)) + self.noise)
            # delta = self.lr * (self.grad + self.noise)

            # assign_var = tf.assign_sub(var, delta)
            assign_var = tf.assign(var, var - self.delta, validate_shape=False)
            # assign_var = tf.assign(var, var - self.lr * self.grad, validate_shape=False)
            # assign_var = tf.assign(var, var - delta, validate_shape=True)
            assign_mt = tf.assign(self.mt, self.new_mt)
            assign_vt = tf.assign(self.vt, self.new_vt)
            assign_epoch = tf.assign_add(self.epoch, 1)
            return tf.group(assign_var, assign_mt, assign_vt, assign_epoch)
        '''
        with tf.name_scope("adam_optimier"):
            self.grad = tf.gradients(loss, var)[0]
            # self.noise = tf.random_normal(self.shape, 0.0, 1.0)
            self.noise = 0
            self.beta1 = tf.constant(0.9)
            self.beta2 = tf.constant(0.999)
            self.lr = tf.constant(self.LEARNING_RATE)
            self.epsilon = 1e-8
            self.epoch = tf.Variable(1, dtype = tf.float32)
            self.mt = tf.Variable(np.zeros(self.shape), dtype = tf.float32)
            self.vt = tf.Variable(np.zeros(self.shape), dtype = tf.float32)

            self.new_mt = self.beta1 * self.mt + (1 - self.beta1) * self.grad
            new_mt = self.new_mt
            self.new_vt = self.beta2 * self.vt + (1 - self.beta2) * tf.square(self.grad)
            new_vt = self.new_vt
            self.pow1 = tf.pow(self.beta2, self.epoch)
            self.pow2 = tf.pow(self.beta1, self.epoch)
            self.corr_num = tf.sqrt(1 - self.pow1) 
            self.corr_deno = 1 - self.pow2
            # self.corr_num = tf.sqrt(1 - tf.pow(self.beta2, self.epoch)) 
            # self.corr_deno = 1 - tf.pow(self.beta1, self.epoch)
            corr = self.corr_num/self.corr_deno 
            # delta = self.lr * corr * (new_mt / (tf.sqrt(new_vt) + self.epsilon))
            self.sq  = tf.sqrt(new_vt+self.epsilon)
            self.lr_corr = self.lr * corr
            self.delta = self.lr_corr * ((new_mt / self.sq))# + self.noise / tf.pow(self.epoch + 1, 0.2))
            delta = self.delta
            # delta = self.lr * corr * ((new_mt / tf.sqrt(new_vt + self.epsilon)) + self.noise)
            # delta = self.lr * (self.grad + self.noise)

            # assign_var = tf.assign_sub(var, delta)
            assign_var = tf.assign(var, var - delta, validate_shape=False)
            # assign_var = tf.assign(var, var - self.lr * self.grad, validate_shape=False)
            # assign_var = tf.assign(var, var - delta, validate_shape=True)
            assign_mt = tf.assign(self.mt, new_mt)
            assign_vt = tf.assign(self.vt, new_vt)
            assign_epoch = tf.assign_add(self.epoch, 1)
            return tf.group(assign_var, assign_mt, assign_vt, assign_epoch)
        '''
        
    def IFGSM_optimizer_tf(self, loss, var):
        with tf.name_scope("IFGSM_optimizer"):
            self.grad = tf.gradients(loss, var)[0]
            print("self.grad.shape:",self.grad.shape)
            y,x = np.ogrid[-149: 149+1, -149: 149+1]
            mask_slice = np.maximum(np.abs(x),np.abs(y))<=111
            mask_slice = mask_slice.astype(int)
            mask = np.expand_dims(np.stack((mask_slice,mask_slice,mask_slice),axis=2),axis=0)
            self.VGG_mask = tf.constant(mask,tf.float32)
            self.new_grad = tf.multiply(tf.sign(self.grad),self.VGG_mask)
            self.lr = tf.constant(self.LEARNING_RATE)
            delta = self.lr * self.new_grad
            assign_var = tf.assign_sub(var, delta)
            self.grad_norm = tf.norm(self.grad)
            return assign_var
 
    def predict(self, imgs):
        imgs = np.arctanh(imgs*1.999999)
        self.sess.run([self.assign_modifier_op], feed_dict = {self.assign_modifier : np.zeros_like(imgs)})
        predicts = self.sess.run([self.predicts], feed_dict = {self.timg : imgs})
        return predicts

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        np.set_printoptions(precision=5)
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            print("imgs size batch_size:",imgs[i:i+self.batch_size].shape)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size])[0])
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh(imgs*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10
        n_success = 0

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        # fill the array as nan to indicate attack failure
        for b in o_bestattack:
            b.fill(np.nan)
        o_best_const = [self.initial_const]*batch_size
        modifier = self.sess.run((self.modifier),{self.timg:np.zeros_like(imgs[:batch_size])})
        print("modifier before attack:",np.sum(modifier))
        print("modifier shape:",modifier.shape)
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print("current best l2", o_bestl2)
            # completely reset adam's internal state. 
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) rep:eat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            # self.sess.run(self.setup, {self.assign_timg: batch,
            self.sess.run(self.setup, {self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            '''
            print("batch shape:",batch.shape)
            self.sess.run([self.assign_modifier_op], feed_dict = {self.assign_modifier : np.zeros_like(batch)})
            timg,modifier,newimg,loss,grad,l2dist = self.sess.run((self.timg,self.modifier,self.newimg,self.loss,self.grad,self.l2dist),feed_dict={self.timg:np.zeros_like(batch)})
            # print("**************self.grad shape:",self.sess.run(tf.shape(self.grad),feed_dict={self.timg:np.zeros_like(batch)}))
            print("newimg:",np.sum(newimg))
            print("timg",np.sum(timg))
            print("self.grad",np.sum(grad))
            '''
            prev = 1e6
            train_timer = 0.0
            for iteration in range(self.MAX_ITERATIONS):
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//self.print_every) == 0:
                    # print(iteration,self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2)))
                    # grad = self.sess.run(self.grad_op)
                    # old_modifier = self.sess.run(self.modifier)
                    # np.save('white_iter_{}'.format(iteration), modifier)
                    loss, real, other, loss1, loss2 = self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2), {self.timg: batch})
                    if self.batch_size == 1:
                        print("[STATS][L2] iter = {}, time = {:.3f}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, train_timer, loss[0], real[0], other[0], loss1[0], loss2[0]))
                    elif self.batch_size > 10:
                        print("[STATS][L2][SUM of {}] iter = {}, time = {:.3f}, batch_size = {}, n_success = {:.5g}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(self.batch_size, iteration, train_timer, batch_size, n_success, sum(loss), sum(real), sum(other), sum(loss1), sum(loss2)))
                    else:
                        print("[STATS][L2] iter = {}, time = {:.3f}".format(iteration, train_timer))
                        print("[STATS][L2] real =", real)
                        print("[STATS][L2] other =", other)
                        print("[STATS][L2] loss1 =", loss1)
                        print("[STATS][L2] loss2 =", loss2)
                        print("[STATS][L2] loss =", loss)
                    sys.stdout.flush()

                attack_begin_time = time.time()
                # perform the attack
                # print("batch shape:",batch.shape)
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss, 
                                                         self.l2dist, self.output, 
                                                         self.newimg],{self.timg: batch})
                # new_modifier = self.sess.run(self.modifier)
                
                # print(grad[0].reshape(-1))
                # print((old_modifier - new_modifier).reshape(-1))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                    if np.all(l > prev*.9999):
                        print("Early stopping because there is no improvement")
                        break
                    prev = l

                # adjust the best result found so far
                read_last_loss = False
                # test_prediction = self.predict(nimg)
                # print("test_predction sum:",np.sum(test_prediction))
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    rank = np.argsort(sc)
                    # test_prediction = self.predict(nimg)
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        # print a message if it is the first attack found
                        if o_bestl2[e] == 1e10:
                            if not read_last_loss: 
                                loss, real, other, loss1, loss2 = self.sess.run((self.loss,self.real,self.other,self.loss1,self.loss2),{self.timg: batch})
                                read_last_loss = True
                            print("[STATS][L3][First valid attack found!] iter = {}, time = {:.3f}, img = {}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(iteration, train_timer, e, loss[e], real[e], other[e], loss1[e], loss2[e]))
                            n_success += 1
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        o_best_const[e] = CONST[e]

                train_timer += time.time() - attack_begin_time

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # modifier = self.sess.run(self.modifier)
                    # np.save("best.model", modifier)
                    print('old constant: ', CONST[e])
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    print('new constant: ', CONST[e])
                else:
                    print('old constant: ', CONST[e])
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10
                    print('new constant: ', CONST[e])

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return np.array(o_bestattack), o_best_const
