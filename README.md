# MaybeShewill-CV.github.io

Here are some of my deep-learning projects which can be showed online. 
You may go to sub folder to test the model

## Nsfw-Classify-Tensorflow
NSFW classify model implemented with tensorflow. Use nsfw dataset provided here
https://github.com/alexkimxyz/nsfw_data_scraper Thanks for sharing the dataset
with us. You can find all the model details here. Don not hesitate to raise an
issue if you're confused with the model.

This software has only been tested on ubuntu 16.04(x64). Here is the test environment
info

**OS**: Ubuntu 16.04 LTS

**GPU**: Two GTX 1070TI 

**CUDA**: cuda 9.0

**Tensorflow**: tensorflow 1.12.0

**OPENCV**: opencv 3.4.1

**NUMPY**: numpy 1.15.1

The main model's hyperparameter are as follows:

**iterations nums**: 160010

**learning rate**: 0.1

**batch size**: 32

**origin image size**: 256

**cropped image size**: 224

**training example nums**: 159477

**testing example nums**: 31895

**validation example nums**: 21266

The rest of the hyperparameter can be found [here](https://github.com/MaybeShewill-CV/nsfw-classification-tensorflow/blob/master/config/global_config.py).

You may monitor the training process using tensorboard tools

During my experiment the `train loss` drops as follows:  
![train_loss](/nsfw_classification/data/images/avg_train_loss.png)

The `train_top_1_error` rises as follows:  
![train_top_1_error](/nsfw_classification/data/images/avg_train_top1_error.png)

The `validation loss` drops as follows:  
![validation_loss](/nsfw_classification/data/images/avg_val_loss.png)

The `validation_top_1_error` rises as follows:  
![validation_top_1_error](/nsfw_classification/data/images/avg_val_top1_error.png)

#### The Model Evaluation 

Some of the evaluation results atr like this 
![evaluation_result](/nsfw_classification/data/images/evaluation_nsfw.png)

The model's main evaluation index are as follows:

**Precision**: 0.92406 with average weighted on each class

**Recall**: 0.92364 with average weighted on each class

**F1 score**: 0.92344 with average weighted on each class

The `Confusion_Matrix` is as follows:  
![confusion_matrix](/nsfw_classification/data/images/confusion_matrix.png)

The `Precison_Recall` is as follows:  
![precision_recall](/nsfw_classification/data/images/precision_recall.png)


#### Online demo

##### URL: https://maybeshewill-cv.github.io/nsfw_classification

Since tensorflo-js is well supported the online deep learning is easy to deploy.
Here I have make a online demo to do local nsfw classification work. The whole js work
can be found here https://github.com/MaybeShewill-CV/MaybeShewill-CV.github.io/tree/master/nsfw_classification
I have supplied a tool to convert the trained tensorflow saved model file into 
tensorflow js model file. In order to generate saved model you can read the 
description about it above. After you generate the tensorflow saved model you 
can simply modify the file path and run the following script

```
cd ROOT_DIR
bash tools/convert_tfjs_model.sh
```
The online demo's example are as follows:
![online_demo](/nsfw_classification/data/images/online_demo.png)

#### TODO
- [ ] Add tensorflow serving script
