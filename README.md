Is Robustness the Cost of Accuracy? – A Comprehensive Study on the Robustness of 18 Deep Image Classification Models
=====================================

The prediction accuracy has been the long-lasting and sole standard for comparing the performance of different image classification models, including the ImageNet competition.  However, recent studies have highlighted the lack of robustness in well-trained deep neural networks to adversarial examples. Visually imperceptible perturbations to natural images can easily be crafted and mislead the image classifiers towards misclassification. To demystify the trade-offs between robustness and accuracy, in this paper we thoroughly benchmark 18 ImageNet models using multiple robustness metrics, including the distortion, success rate and transferability of adversarial examples between 306 pairs of models. Our extensive experimental results reveal several new insights: (1) linear scaling law - the empirical $\ell_2$ and $\ell_\infty$ distortion metrics scale linearly with the logarithm of classification error; (2) model architecture is a more critical factor to robustness than model size, and the disclosed accuracy-robustness Pareto frontier can be used as an evaluation criterion for ImageNet model designers; (3) for a similar network architecture, increasing network depth slightly improves robustness in $\ell_\infty$ distortion;  (4) there exist models (in VGG family) that exhibit high adversarial transferability, while most adversarial examples crafted from one model can only be transferred within the same family.

For more details, please see our paper:

[Is Robustness the Cost of Accuracy? – A Comprehensive Study on the Robustness of 18 Deep Image Classification Models]
by Dong Su\*, Huan Zhang\*, Hongge Chen, Jinfeng Yi, Pin-Yu Chen, Yupeng Gao.  To appear in ECCV 2018. 
(https://arxiv.org/abs/1808.01688) by Dong Su\*, Huan Zhang\*, Hongge Chen, Jinfeng Yi, Pin-Yu Chen, Yupeng Gao, ECCV 2018. 

\* Equal contribution


Experiment Setup
-------------------------------------

The code is tested with python 3.6.5 and TensorFlow v1.8. We suggest to use Conda to manage your Python environments. The following Conda packages are required:

```
conda install pillow numpy scipy pandas tensorflow-gpu h5py
grep 'AMD' /proc/cpuinfo >/dev/null && conda install nomkl
```
Note: the second command is only needed in the linux environment.  


Then clone this repository:
```
git clone git@github.com:huanzhang12/Adversarial_Survey.git
cd Adversarial_Survey
```

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)

create the `./imagenetdata` directory, and put the `imgs` folder under the `./imagenetdata` directory, relative to the Adversarial_Survey repository. This path can be changed in `setup_imagenet.py`.

To prepare the ImageNet models:
Create the `./tmp/` directory.  In the Adversarial_Survey directory, run
```
python setup_imagenet.py
```
All pretrained model will be saved to `./tmp/imagenet` directory.


Run the experiment
--------------------------------------
The following are some examples of attacks:

To run the FGSM untargeted attack on the densenet169_k32 model with epsilon=0.3, 
```
python test_FGSM.py --dataset=imagenet --attack=FGSM --num_valid_test_imgs=10 --attack_batch_size=1 --model_name=densenet169_k32 --numimg=0 --firstimg=0 --save=./saved_results/FGSM/epsilon_0.3_imagenet_FGSM_targeted_densenet169_k32 --epsilon=0.3 --target_type=7 --use_zvalue --seed=1215  --untargeted
```

To run the IFGSM targeted attack on the densenet169_k32 model with optimal attack budget, 
```
python test_IterFGSM.py --dataset=imagenet --attack=IterFGSM --num_valid_test_imgs=10 --attack_batch_size=1 --model_name=densenet169_k32 --numimg=0 --firstimg=0 --save=./saved_results/IterFGSM/epsilon_0.02_iterations_50_imagenet_IterFGSM_targeted_densenet169_k32  --initial_eps=0.02 --max_attempts=9 --iter_num=50 --target_type=7 --use_zvalue --seed=1215
```

One can also run IFGSM targeted attack with fixed attack budget , 
```
python test_IterFGSM.py --dataset=imagenet --attack=IterFGSM --num_valid_test_imgs=10 --attack_batch_size=1 --model_name=densenet169_k32 --numimg=0 --firstimg=0 --save=./saved_results/IterFGSM/epsilon_0.2_iterations_50_imagenet_IterFGSM_targeted_densenet169_k32 --initial_eps=0.2 --max_attempts=1 --iter_num=50 --target_type=7 --use_zvalue --seed=1215
```

To run the CW targeted attack on the densenet169_k32 model
```
python test_CW_EADL1.py --dataset=imagenet --attack=CW --numimg=0 --firstimg=0 --num_valid_test_imgs=10 --save=./saved_results/CW/kappa_0_imagenet_CW_targeted_densenet161_k48 --maxiter=1000 --lr=0.001 --binary_steps=9 --init_const=0.01 --use_zvalue --target_type=7 --kappa=0 --model_name=densenet169_k32 --attack_batch_size=1 --seed=1215 
```

To run the EADL1 targeted attack on the densenet169_k32 model
```
python test_CW_EADL1.py --dataset=imagenet --attack=EADL1 --numimg=0 --firstimg=0 --num_valid_test_imgs=10 --save=./saved_results/EADL1/kappa_0_imagenet_EADL1_targeted_densenet161_k48 --maxiter=1000 --lr=0.001 --binary_steps=9 --init_const=0.01 --use_zvalue --target_type=7 --kappa=0 --model_name=densenet169_k32 --attack_batch_size=1 --seed=1215 
```

To run the untargeted transferability attack to the given target model, e.g. densenet161_k48, from the generated untargeted adversarial examples from the model densenet169_k32
```
python test_transferability.py --dataset=imagenet --attack=FGSM --epsilon=0.3 --src_adv_sample_path=./saved_results/FGSM/densenet169_k32_eps_0.3_untargeted/imagenet/FGSM/targeted_False --save=./saved_results/transferability/FGSM/epsilon_0.3_imagenet_FGSM_untargeted_densenet169_k32_densenet161_k48/ --use_zvalue --src_model_name=densenet169_k32 --target_model_name=densenet161_k48 --untargeted
```

Several global parameters:

`--model_name`: it can be one from the supported 18 ImageNet model list: 
```
['resnet_v2_50','resnet_v2_101','resnet_v2_152','inception_v1','inception_v2','inception_v3','inception_v4', 'inception_resnet_v2','vgg_16','vgg_19','mobilenet_v1_025','mobilenet_v1_050','mobilenet_v1_100', 'densenet121_k32', 'densenet169_k32', 'densenet161_k48', 'nasnet_large', 'alexnet']
```

`--untargeted`: For running untargeted attacks.  For running targeted attacks, do not add this option in the command. 

`--target_type`: For running targeted attacks, you can specify the target type: least likely target (type: `0b0100`), top2 likely (type: `0b0001`) or random target (type: `0b0010`).  In the above examples, we run targeted attack over all of these three target types and use `0b0111` which is 7 as the `target_type`. 

`--seed`: For setting the random seed. 

`--save`: For setting the location for storing the evaluation results

`--firstimg`: For setting the starting image id in the pool.  The default value is 0. 

`--numimg`: For setting the number of images to be examed in the attack.  The default value of it is 0 which means using all loaded images. 

`--num_valid_test_imgs`: the number of valid images (correctly classified by the target model) to be used in the attack. 

`--attack_batch_size`: For setting the number of images to attack in a batch way. 






