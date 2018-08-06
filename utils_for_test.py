import numpy as np
from PIL import Image
import random



def linf_loss(x, y):
#    return np.max(np.abs(x-y))
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_loss(x, y):
#    return (np.sum((x - y) ** 2) ** .5)
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def l1_loss(x, y):
#    return np.sum(np.abs(x-y))
    return np.linalg.norm(x.flatten() - y.flatten(), ord=1)

def l0_loss(x, y):
#    return np.sum(np.abs(x - y) >= 1e-10)
    return np.linalg.norm(x.flatten() - y.flatten(), ord=0)

def show(img, name = "output.png"):
    np.save('img', img)
    fig = (img + 0.5)*255
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
    return


def dump(img, path, save_png=False):
    #save as npy
    np.save(path + ".npy", img)

    if save_png == True:
        #save as png file
        fig = (img + 0.5)*255
        fig = fig.astype(np.uint8).squeeze()
        pic = Image.fromarray(fig)
        pic.save(path + ".png")

    return


def generate_data(data, samples, targeted=True, random_and_least_likely = False, skip_wrong_label = True, start=0, ids = None,
        target_classes = None, target_type=0b1111, predictor = None, imagenet=False, remove_background_class=False,
        total_num_valid_samples=1, num_random_targets=1):

    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    target_candidate_pool = np.eye(data.test_labels.shape[1])
    target_candidate_pool_remove_background_class = np.eye(data.test_labels.shape[1] - 1)
    print('generating labels...')

    print('target_type = ', target_type)

    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        if target_classes:
            target_classes = target_classes[start:start+samples]
        start = 0
    total = 0
    num_valid_samples = 0
    for i in ids:
        total += 1

        if num_valid_samples >= total_num_valid_samples:
            print("reaching the total_num_valid_samples, ", total_num_valid_samples)
            break

        if remove_background_class == True:
            true_label = np.argmax(data.test_labels[start + i][1:])
        else:
            true_label = np.argmax(data.test_labels[start + i])

        print("true_label = ", true_label)

        if targeted:
            predicted_label = -1 # unknown

            if random_and_least_likely:
                # if there is no user specified target classes
                if target_classes is None:
                    original_predict = np.squeeze(predictor(np.array([data.test_data[start+i]])))

                    print("original_predict.shape = ", original_predict.shape)

                    num_classes = len(original_predict)
                    predicted_label = np.argmax(original_predict)
                    least_likely_label = np.argmin(original_predict)
                    top2_label = np.argsort(original_predict)[-2]
                    start_class = 1 if (imagenet and not remove_background_class) else 0

                    new_seq = [least_likely_label, top2_label, predicted_label]

                    if imagenet:
                        if remove_background_class:
                            sample_pool = [x for x in range(0, 1000) if x != true_label]
                        else:
                            sample_pool = [x for x in range(1, 1001) if x != true_label]
                    else:
                        sample_pool = [x for x in range(data.test_labels.shape[1]) if x != true_label]

                    random_seq = random.sample(sample_pool, num_random_targets)
                    new_seq[2] = random_seq[0]

                    seq = []
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    else:
                        num_valid_samples += 1

                        if target_type & 0b0100:
                            # least
                            seq.append(new_seq[0])
                            information.append('least')
                        if target_type & 0b0001:
                            # top-2
                            seq.append(new_seq[1])
                            information.append('top2')
                        if target_type & 0b0010:
                            # random
                            seq.append(new_seq[2])
                            information.append('random')
                else:
                    # use user specified target classes
                    seq = target_classes[total - 1]
                    information.extend(len(seq) * ['user'])


            else:
                if imagenet:
                    if remove_background_class:
                        seq = random.sample(range(0,1000), 10)
                    else:
                        seq = random.sample(range(1,1001), 10)
                    information.extend(data.test_labels.shape[1] * ['random'])
                else:
                    seq = range(data.test_labels.shape[1])
                    information.extend(data.test_labels.shape[1] * ['seq'])

            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i,
                true_label, predicted_label, true_label == predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))


            for j in seq:
                if(j == true_label):
                    print("=======skip the original image label========")
                    continue
                inputs.append(data.test_data[start+i])
                if remove_background_class:
                    targets.append(target_candidate_pool_remove_background_class[j])
                else:
                    targets.append(target_candidate_pool[j])
                true_labels.append(data.test_labels[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
        else:

            original_predict = np.squeeze(predictor(np.array([data.test_data[start + i]])))
            predicted_label = np.argmax(original_predict)

            print("i = %d, true_label = %d, predicted_label = %d" % (i, true_label, predicted_label))

            if true_label != predicted_label and skip_wrong_label:
                print("untargeted setting: skipping wrongly classified samples")
                continue
            else:
                num_valid_samples += 1

            inputs.append(data.test_data[start+i])
            if remove_background_class:
                # shift target class by 1
                print(np.argmax(data.test_labels[start+i]))
                print(np.argmax(data.test_labels[start+i][1:1001]))
                targets.append(data.test_labels[start+i][1:1001])
            else:
                targets.append(data.test_labels[start+i])
            true_labels.append(data.test_labels[start+i])
            if remove_background_class:
                true_labels[-1] = true_labels[-1][1:]
            true_ids.append(start+i)
            information.extend(['original'])

    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)


    print("total = ", total)
    print("len(inputs) = ", len(inputs))
    print("num_valid_samples = ", num_valid_samples)

    print('labels generated')

    return inputs, targets, true_labels, true_ids, information

