import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
import scipy as sp
import json

from setup_imagenet import ImageNet, ImageNetModel

from PIL import Image

def dump(img, path):
    #save as npy
    np.save(path + ".npy", img)

    #save as png file
    fig = (img + 0.5)*255
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(path + ".png")

    return

#This function assumes the input image has width == height
def adapt_image_size(image_fn, src_image_size, target_image_size):


    img = Image.open(image_fn)

    if src_image_size < target_image_size:

        print("src_image_size < target_image_size")

        size_diff = target_image_size - src_image_size

        padding_size_1 = size_diff // 2

        if size_diff % 2 > 0:
            padding_size_2 = padding_size_1 + 1
        else:
            padding_size_2 = padding_size_1

        cropped_image = img.crop(
            (
                -padding_size_1,
                -padding_size_1,
                img.size[0] + padding_size_2,
                img.size[1] + padding_size_2
            )
        )

        print("cropped_image.size = ", cropped_image.size)

        cropped_image_np = np.array(cropped_image)

        return cropped_image_np

    elif src_image_size > target_image_size:

        print("src_image_size > target_image_size")

        half_the_size = src_image_size / 2

        crop_size_1 = target_image_size // 2

        if target_image_size % 2 > 0:
            crop_size_2 = crop_size_1 + 1
        else:
            crop_size_2 = crop_size_1

        cropped_image = img.crop(
            (
                half_the_size - crop_size_1,
                half_the_size - crop_size_1,
                half_the_size + crop_size_2,
                half_the_size + crop_size_2
            )
        )

        cropped_image_np = np.array(cropped_image)

        print("cropped_image.size = ", cropped_image.size)

        return cropped_image_np



def adapt_image_size_on_ndarray(img, src_image_size, target_image_size):

    print("img.shape = ", img.shape)

    if src_image_size < target_image_size:

        print("src_image_size < target_image_size")

        inflated_img = np.zeros((target_image_size, target_image_size, 3), dtype=img.dtype)

        size_diff = target_image_size - src_image_size

        padding_size_1 = size_diff // 2

        inflated_img[padding_size_1: padding_size_1 + img.shape[0],
                    padding_size_1: padding_size_1 + img.shape[1],
                    :] = img

        print("inflated_img.shape = ", inflated_img.shape)

        return inflated_img

    elif src_image_size > target_image_size:

        print("src_image_size > target_image_size")

        mid = src_image_size // 2

        crop_size_1 = target_image_size // 2

        if target_image_size % 2 > 0:
            crop_size_2 = crop_size_1 + 1
        else:
            crop_size_2 = crop_size_1

        print("mid = ", mid)
        print("crop_size_1 = ", crop_size_1)
        print("crop_size_2 = ", crop_size_2)

        cropped_image = img[mid - crop_size_1:mid + crop_size_2,
                            mid - crop_size_1:mid + crop_size_2,
                            :]

        print("cropped_image.shape = ", cropped_image.shape)

        return cropped_image



def adv_sample_fn_parser(fn):
    # para_list = fn.split('.')[0].split('_')
    para_list = fn[:-4].split('_')
    para_dict = {}
    for para in para_list:
        if '=' in para:
            para_name, para_value = para.split('=')
            para_dict[para_name] = para_value
        else:
            para_dict[para] = para

    return para_dict


def adv_sample_fn_assembler_and_loader(adv_sample_path, para_dict, src_input_size, target_input_size):

    # content_list = ['original', 'adversarial', 'noise', 'labels', 'targets']
    content_list = ['original', 'adversarial', 'labels', 'targets']

    adv_sample_dict = {}

    print("adv_sample_fn_assembler_and_loader")
    print("para_dict = ", para_dict)

    if 'target' in para_dict:
        para_format = "{0}/imgno={1}_content={2}_id={3}_seq={4}_prev={5}_target={6}_adv={7}_res={8}.npy"

        for content in content_list:
            fn = para_format.format(adv_sample_path, para_dict['imgno'], content, para_dict['id'], para_dict['seq'],
                                    para_dict['prev'], para_dict['target'], para_dict['adv'], para_dict['res'])

            obj = np.load(fn)
            adv_sample_dict[content] = obj

        if src_input_size != target_input_size:

            print("adv_sample_dict['original'].shape = ", adv_sample_dict['original'].shape)
            print("adv_sample_dict['adversarial'].shape = ", adv_sample_dict['adversarial'].shape)

            if len(adv_sample_dict['original'].shape) > 3:
                original = adv_sample_dict['original'][0]
            else:
                original = adv_sample_dict['original']

            if len(adv_sample_dict['adversarial'].shape) > 3:
                adversarial = adv_sample_dict['adversarial'][0]
            else:
                adversarial = adv_sample_dict['adversarial']


            adapted_orig_image_obj = adapt_image_size_on_ndarray(original, src_input_size, target_input_size)
            adapted_adv_image_obj = adapt_image_size_on_ndarray(adversarial, src_input_size, target_input_size)
            adv_sample_dict['original'] = adapted_orig_image_obj
            adv_sample_dict['adversarial'] = adapted_adv_image_obj

    else:
        para_format = "{0}/imgno={1}_content={2}_id={3}_seq={4}_prev={5}_adv={6}_res={7}.npy"

        for content in content_list:
            fn = para_format.format(adv_sample_path, para_dict['imgno'], content, para_dict['id'], para_dict['seq'],
                                    para_dict['prev'], para_dict['adv'], para_dict['res'])

            obj = np.load(fn)
            adv_sample_dict[content] = obj

        if src_input_size != target_input_size:

            if len(adv_sample_dict['original'].shape) > 3:
                original = adv_sample_dict['original'][0]
            else:
                original = adv_sample_dict['original']

            if len(adv_sample_dict['adversarial'].shape) > 3:
                adversarial = adv_sample_dict['adversarial'][0]
            else:
                adversarial = adv_sample_dict['adversarial']

            adapted_orig_image_obj = adapt_image_size_on_ndarray(original, src_input_size, target_input_size)
            adapted_adv_image_obj = adapt_image_size_on_ndarray(adversarial, src_input_size, target_input_size)
            adv_sample_dict['original'] = adapted_orig_image_obj
            adv_sample_dict['adversarial'] = adapted_adv_image_obj

    adv_sample_dict['config'] = para_dict

    return adv_sample_dict



def source_model_adversarial_sample_group_generator(adv_sample_path, src_model_name, target_model_name):
    with open("model_input_size_info.json") as f:
        input_size_info = json.load(f)

    src_input_size = input_size_info[src_model_name]
    target_input_size = input_size_info[target_model_name]

    for entry in os.scandir(adv_sample_path):
        print("entry = ", entry)
        if 'adversarial' in entry.name and 'npy' in entry.name:
            print(entry.name)
            para_dict = adv_sample_fn_parser(entry.name)
            adv_sample_group = adv_sample_fn_assembler_and_loader(adv_sample_path, para_dict, src_input_size, target_input_size)

            yield adv_sample_group

def main(args):
    with tf.Session() as sess:

        use_log = not args['use_zvalue']

        print('Loading target model', args['target_model_name'])
        target_model = ImageNetModel(sess, use_log, args['target_model_name'])

        print('args = ', args)

        targeted_flag = not args['untargeted']

        print("targeted_flag = ", targeted_flag)

        if args['attack'] not in ["FGSM", "IterFGSM", "EADL1", "CW"]:
            print("Unknown attack methods, exit 1")
            return

        random.seed(args['seed'])
        np.random.seed(args['seed'])
        tf.set_random_seed(args['seed'])

        #saved_path = "{}/{}/{}/targeted_{}".format(args['save'], args['dataset'], args['attack'], targeted_flag)
        adv_sample_path = args['src_adv_sample_path']

        img_no = 0
        total_success = 0
        total_top_5_success = 0
        time_total = 0.0

        os.system("mkdir -p " + args['save'])

        if args['attack'] == "FGSM" or args['attack'] == "IterFGSM":
            verbose_f = open(args['save'] + "/" + "_".join(
                [args['dataset'], args['src_model_name'], args['target_model_name'], args['attack'], str(targeted_flag), str(args['epsilon']), "verbose.txt"]), "w")
            aggre_f = open(args['save'] + "/" + "_".join(
                [args['dataset'], args['src_model_name'], args['target_model_name'], args['attack'], str(targeted_flag), str(args['epsilon']), "aggre.txt"]), "w")
        elif args['attack'] == "AdaptiveFGSM":
            verbose_f = open(
                args['save'] + "/" + "_".join([args['dataset'], args['src_model_name'], args['target_model_name'], args['attack'], str(targeted_flag), "verbose.txt"]),
                "w")
            aggre_f = open(
                args['save'] + "/" + "_".join([args['dataset'], args['src_model_name'], args['target_model_name'], args['attack'], str(targeted_flag), "aggre.txt"]), "w")
        elif args['attack'] == 'CW' or args['attack'] == 'EADL1':
            verbose_f = open(args['save'] + "/" + "_".join(
                [args['dataset'], args['src_model_name'], args['target_model_name'], args['attack'], str(targeted_flag), str(args['kappa']), "verbose.txt"]), "w")
            aggre_f = open(args['save'] + "/" + "_".join(
                [args['dataset'], args['src_model_name'], args['target_model_name'], args['attack'], str(targeted_flag), str(args['kappa']), "aggre.txt"]), "w")

        if targeted_flag == True:
            verbose_head_str = '\t'.join(
            ['total', 'seq', 'id', 'time', 'success', 'success_top_5', 'src_orig_class', 'src_pred_class', 'attack_target_class', 'target_orig_class', 'target_pred_class'])
        else:
            verbose_head_str = '\t'.join(
            ['total', 'seq', 'id', 'time', 'success', 'src_orig_class', 'src_pred_class', 'target_orig_class', 'target_pred_class'])

        #top-5 success rate column is only for targeted attack
        #for untargeted attack, top-5 == top_1
        aggre_head_str = '\t'.join(
            ['total_count', 'success_rate', 'top_5_success_rate', 'time_avg'])

        verbose_f.write(verbose_head_str + '\n')
        aggre_f.write(aggre_head_str + '\n')

        if "vgg" in args['src_model_name'] or 'densenet' in args['src_model_name'] or 'alexnet' in args['src_model_name']:
            src_model_remove_background_class = True
        else:
            src_model_remove_background_class = False

        if "vgg" in args['target_model_name']  or 'densenet' in args['target_model_name'] or 'alexnet' in args['target_model_name']:
            target_model_remove_background_class = True
        else:
            target_model_remove_background_class = False

        print("src_model_remove_background_class = ", src_model_remove_background_class, ", target_model_remove_background_class = ", target_model_remove_background_class)

        print("adv_sample_path = ", adv_sample_path)

        for adv_sample_group in source_model_adversarial_sample_group_generator(adv_sample_path, args['src_model_name'], args['target_model_name']):

            print("true labels.shape:", np.argmax(adv_sample_group['labels'].shape))
            print("true labels, argmax:", np.argmax(adv_sample_group['labels']))
            print("target.shape:", np.argmax(adv_sample_group['targets'].shape))
            print("target, argmax:", np.argmax(adv_sample_group['targets']))

            # test if the image is correctly classified
            target_original_predict = target_model.model.predict(adv_sample_group['original'])
            target_original_predict = np.squeeze(target_original_predict)

            print("target_original_predict.shape", target_original_predict.shape)

            if target_model_remove_background_class == True:
                target_original_predict = np.insert(target_original_predict, 0, 0)
                print("vgg case, insert a leading 0 to prediction vector, shape = ", target_original_predict.shape)

            print("target model predict, argmax", np.argmax(target_original_predict))

            target_original_prob = np.sort(target_original_predict)
            target_original_class = np.argsort(target_original_predict)
            target_original_class_label = np.argmax(target_original_predict)

            print("target original probabilities:", target_original_prob[-1:-6:-1])
            print("target original classification:", target_original_class[-1:-6:-1])
            print("target original probabilities (most unlikely):", target_original_prob[:6])
            print("target original classification (most unlikely):", target_original_class[:6])
            print("target_original_class_label = ", target_original_class_label)


            true_label = adv_sample_group['labels']

            # true_label = np.insert(true_label, 0, 0)

            print("true_label.shape = ", true_label.shape)

            #The 2nd condition is for dealing with the case that using the original label in attacking vgg models
            if src_model_remove_background_class == True and true_label.shape == tuple([1000,]):
                true_label = np.insert(true_label, 0, 0)
                print("after expanding, true_label.shape", true_label.shape)
                print("true_label[:5]", true_label[:5])

            print("Image no. {}, original class {}, classified as {}".format(
                adv_sample_group['config']['seq'], np.argmax(true_label), target_original_class_label))
            if target_original_class_label != np.argmax(true_label):
                print("skip wrongly classified image no. {}, original class {}, classified as {}".format(
                    adv_sample_group['config']['seq'], np.argmax(true_label), target_original_class_label))
                continue

            img_no += 1
            timestart = time.time()

            target_adversarial_predict = target_model.model.predict(adv_sample_group['adversarial'])
            target_adversarial_predict = np.squeeze(target_adversarial_predict)

            if target_model_remove_background_class == True:
                target_adversarial_predict = np.insert(target_adversarial_predict, 0, 0)
                print("after expanding, target_adversarial_predict.shape", target_adversarial_predict.shape)
                print("target_adversarial_predict[:5]", target_adversarial_predict[:5])

                # attack_target_class = int(adv_sample_group['config']['target']) + 1
                attack_target_class = np.argmax(adv_sample_group['targets']) + 1

            else:
                #attack_target_class = int(adv_sample_group['config']['target'])
                attack_target_class = np.argmax(adv_sample_group['targets'])


            target_adversarial_prob = np.sort(target_adversarial_predict)
            target_adversarial_class = np.argsort(target_adversarial_predict)
            target_adversarial_class_label = np.argmax(target_adversarial_predict)

            top_5_target_adversarial_class = target_adversarial_class[-1:-6:-1]



            print("top_5_target_adversarial_class = ", top_5_target_adversarial_class)
            print("set(top_5_target_adversarial_class) = ", set(top_5_target_adversarial_class))

            timeend = time.time()

            time_used = timeend - timestart

            print("target adversarial probabilities:", target_adversarial_prob[-1:-6:-1])
            print("target adversarial classification:", target_adversarial_class[-1:-6:-1])
            print('target_adversarial_predict.shape = ', target_adversarial_predict)
            print("target_adversarial_class_label = ", target_adversarial_class_label)

            src_model_target = adv_sample_group['targets']
            if src_model_remove_background_class == True:
                src_model_target = np.insert(src_model_target, 0, 0)
                src_orig_class = int(adv_sample_group['config']['prev']) + 1
                # src_pred_class = int(adv_sample_group['config']['adv']) + 1
                src_pred_class = int(float(adv_sample_group['config']['adv'])) + 1
            else:
                src_orig_class = int(adv_sample_group['config']['prev'])
                # src_pred_class = int(adv_sample_group['config']['adv'])
                src_pred_class = int(float(adv_sample_group['config']['adv']))

            print('src_model_target.shape = ', src_model_target.shape)

            print("src_model_target label = ", np.argmax(src_model_target))

            print('target_adversarial_class_label = ', target_adversarial_class_label)
            print('attack_target_class = ', attack_target_class)

            success = False
            top_5_success = False
            if targeted_flag:
                if target_adversarial_class_label == np.argmax(src_model_target):
                    success = True

                # if attack_target_class in set(top_5_target_adversarial_class):
                if np.argmax(src_model_target) in set(top_5_target_adversarial_class):
                    total_top_5_success += 1
                    top_5_success = True

            else:
                if target_adversarial_class_label != target_original_class_label:
                    success = True

            if success:
                total_success += 1

            if targeted_flag == True:
                verbose_str = '\t'.join(
                    [str(img_no), str(adv_sample_group['config']['seq']), str(adv_sample_group['config']['id']),
                     str(time_used), str(success), str(top_5_success), str(src_orig_class), str(src_pred_class),
                     str(attack_target_class), str(target_original_class_label), str(target_adversarial_class_label)])

            else:
                verbose_str = '\t'.join(
                    [str(img_no), str(adv_sample_group['config']['seq']), str(adv_sample_group['config']['id']),
                     str(time_used), str(success), str(src_orig_class), str(src_pred_class),
                     str(target_original_class_label), str(target_adversarial_class_label)])

            verbose_f.write(verbose_str + "\n")
            print(verbose_str)
            sys.stdout.flush()

        verbose_f.close()

        if img_no == 0:
            success_rate = 0.0
            top_5_success_rate = 0.0
        else:
            success_rate = total_success / float(img_no)
            if targeted_flag:
                top_5_success_rate = total_top_5_success / float(img_no)
            else:
                top_5_success_rate = success_rate

        if total_success == 0:
            time_avg = 0.0
        else:
            time_avg = time_total / total_success

        aggre_str = "\t".join(
            [str(img_no), str(success_rate), str(top_5_success_rate), str(time_avg)])
        aggre_f.write(aggre_str + "\n")
        print(aggre_str)
        aggre_f.close()

        print("ALL DONE!!!")
        return

if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["imagenet"], default="imagenet")
    parser.add_argument("-s", "--save", default="./saved_results")

    parser.add_argument("--src_adv_sample_path")

    parser.add_argument("-a", "--attack", choices=["FGSM", "IterFGSM", "CW", 'EADL1'], default="FGSM")
    parser.add_argument("-u", "--untargeted", action='store_true')
    parser.add_argument("-e", "--epsilon", type=float, default=0.3)
    parser.add_argument("--kappa", type=int, default=0, help = "initial_constance")


    parser.add_argument("--eps_iter", type=float, default=0.3)
    parser.add_argument("--iter_num", type=int, default=10)

    parser.add_argument("--src_model_name", default="resnet_v2_50")
    parser.add_argument("--target_model_name", default="resnet_v2_101")

    parser.add_argument("-z", "--use_zvalue", action='store_true')
    parser.add_argument("--seed", type=int, default=1216)
    args = vars(parser.parse_args())

    # add some additional parameters
    # learning rate
    args['lr'] = 1e-2
    args['inception'] = False
    args['use_tanh'] = True

    # set up some parameters based on datasets
    if args['dataset'] == "imagenet":
        args['inception'] = True
        args['lr'] = 2e-3

    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    tf.set_random_seed(args['seed'])
    print(args)

    main(args)

    print("Experiment Done!!!")
