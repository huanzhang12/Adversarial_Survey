from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import time
import numpy as np


import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_tf import model_train, model_eval, batch_eval, tf_model_load

from cleverhans.utils_keras import KerasModelWrapper
from setup_mnist import MNIST, MNISTModel
#from setup_mnist import MNISTModel
from FGSM_attack import FGSM
FLAGS = flags.FLAGS
from Iter_FGSM_attack import Iter_FGSM
from cleverhans.attacks import BasicIterativeMethod

from PIL import Image


def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = (img + 0.5)*255
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            print ('image label:', np.argmax(data.test_labels[start+i]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    train_data = data.train_data
    train_labels = data.train_labels
    return train_data, train_labels, inputs, targets

def mnist_compare(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1, attack="fgsm", targeted=False):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    #report = AccuracyReport()
    use_log = True
    targeted = True
    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    #set_log_level(logging.DEBUG)

    # Get MNIST data
    data =  MNIST()

    print('Generate data')
    X_train, Y_train, inputs, targets = generate_data(data, samples=10, targeted=targeted)
	# Redefine test set as remaining samples unavailable to adversaries
    print('training data shape: ', len(X_train), len(Y_train))
    print('inputs shape: ', len(inputs), len(targets))
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

	 # Define TF model graph 
    modelTop = MNISTModel(use_log = use_log)
    model = modelTop.model
    preds = model(x)
    model_path = 'modelsLog/mnist'
    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        'filename': os.path.split(model_path)[-1]
    }
    print('training or loading the model')
    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path+".meta"):
        tf_model_load(sess, model_path)
    else:
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    save=os.path.exists("modelsLog"))
    print('done')
            
	# Instantiate a fgsm attack object

    print("Running cleverhans iter_FGSM attack...")
    timestart = time.time()
    if targeted:
        attacker_params = {'eps': 0.3, 'y_target':targets,'eps_iter': 0.3, 'nb_iter':20, 'clip_min': -0.5, 'clip_max': 0.5}
    else:
        attacker_params= {'eps': 0.3, 'eps_iter': 0.3, 'nb_iter':20, 'clip_min': -0.5, 'clip_max': 0.5}
    wrap = KerasModelWrapper(model) 
    cleverhans_iter_fgsm = BasicIterativeMethod(wrap, back='tf',sess=sess)
    x_adv = cleverhans_iter_fgsm.generate(x, **attacker_params)
    x_adv1 = sess.run(x_adv, feed_dict = {x : inputs})
    timeend = time.time()
    print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
	

    print("Running self iter_FGSM attack...")
    timestart = time.time()
    eps_iter = 0.3
    steps = 20
    eps = 0.3
    self_iter_fgsm = Iter_FGSM(sess, modelTop, eps = eps, eps_iter = eps_iter, iter_num = steps, targeted=targeted, batch_size = 90)
    x_adv2 = self_iter_fgsm.attack(inputs, targets)
    timeend = time.time()
    print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
    num = 0
    num1 = 0
    for i in range(len(x_adv2)):
        #print("Valid:")
        #show(inputs[i], "original_{}.png".format(i))
        #original_predict = np.squeeze(model.predict(inputs[i:i+1]))
        original_predict = sess.run(preds, feed_dict = {x : inputs[i:i+1]})
        #print(original_predict)
        original_predict = np.reshape(original_predict,(10,))
        if not targeted:
            if np.argsort(original_predict)[-1] != np.argmax(targets[i]):
            #print('true label id different with predict label. Skip this')
                continue
        #print(np.argsort(original_predict)[-1])
        #print(np.argmax(targets[i]))
        #if original_predict !=        
        #print("Original Classification:", np.argsort(original_predict)[-1:-11:-1])
        #print("Original Probabilities/Logits:", np.sort(original_predict)[-1:-11:-1])
        #print("Target:", np.argmax(targets[i]))
        #print("Adversarial:")
        #show(x_adv1[i], "adversarial_{}.png".format(i))
        #print("Noise:")
        #show(x_adv1[i] - inputs[i], "attack_diff.png")
        #adv_predict = np.squeeze(model.predict(x_adv1[i:i+1]))
        adv_predict_cleverhans = sess.run(preds, feed_dict = {x : x_adv1[i:i+1]})
        adv_predict_me = sess.run(preds, feed_dict = {x : x_adv2[i:i+1]})
        adv_predict_cleverhans = np.reshape(adv_predict_cleverhans,(10,))
        adv_predict_me = np.reshape(adv_predict_me,(10,))
        #print("Adversarial Classification in cleverhans:", np.argsort(adv_predict_cleverhans)[-1:-11:-1])
        #print("Adversarial Probabilities/Logits in cleverhans:", np.sort(adv_predict_cleverhans)[-1:-11:-1])
        #print("Adversarial Classification in me:", np.argsort(adv_predict_me)[-1:-11:-1])
        #print("Adversarial Probabilities/Logits in me:", np.sort(adv_predict_me)[-1:-11:-1])
        print(x_adv1[i][np.abs(x_adv1[i] - x_adv2[i]) > 1e-10])
        print(x_adv2[i][np.abs(x_adv1[i] - x_adv2[i]) > 1e-10])
        print(inputs[i][np.abs(x_adv1[i] - x_adv2[i]) > 1e-10])
        if targeted:
            success_cleverhans = np.argsort(adv_predict_cleverhans)[-1] == np.argmax(targets[i])
            success_me = np.argsort(adv_predict_me)[-1] == np.argmax(targets[i])
        else:
            success_cleverhans = np.argsort(adv_predict_cleverhans)[-1] != np.argmax(targets[i])
            success_me = np.argsort(adv_predict_me)[-1] != np.argmax(targets[i])
        if success_cleverhans:
            #print("Attack succeeded.")
            num = num + 1
        #else:
        #    print("Attack failed.")
        if success_me:
            num1 = num1 + 1
    print('total number of success in cleverhans: ',num)
    print('total number of success in me: ',num1)

def main(argv=None):
    mnist_compare()


if __name__ == '__main__':
    # General flags
    #flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    #flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    #flags.DEFINE_integer('n_attack', -1, 'No. of images used for attack')
    #flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to oracle
    #flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')

    # Flags related to substitute
    #flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    #flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    #flags.DEFINE_integer('nb_epochs_s', 30, 'Training epochs for substitute')
    #flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    # Flags related to attack
    #flags.DEFINE_string('attack', 'cwl2', 'cwl2 = Carlini & Wagner\'s L2 attack, fgsm = Fast Gradient Sign Method')
    #flags.DEFINE_bool('targeted', False, 'use targeted attack')

    # Flags related to saving/loading
    #flags.DEFINE_bool('load_pretrain', False, 'load pretrained model from sub_saved/mnist-model')
    #flags.DEFINE_bool('cached_aug', False, 'use cached augmentation in sub_saved')
    #flags.DEFINE_string('train_dir', 'sub_saved', 'model saving path')
    #flags.DEFINE_string('filename', 'mnist-model', 'cifar model name')

    #os.system("mkdir -p sub_saved")

    app.run()
