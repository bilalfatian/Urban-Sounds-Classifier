# Urban sounds classification with Convolutional Neural Networks

## Objective 🎯

The objective of this project is to implement CNN models to recognize sound events from the **UrbanSound9K** dataset. The work has been divided into the following notebooks:

1. Data analysis (and related papers brief)
2. Pre-processing and feature evaluation
3. CNN model with MFCC 
4. CNN model with Log-MEL Spectrograms
5. Data augmentation
6. Data augmentation pre-processing
7. CNN model with augmented data (Log-MEL Spectrograms)

## Notebooks 📒

1. [Data analysis](https://github.com/GorillaBus/urban-audio-classifier/blob/master/1-data-analysis.ipynb): a brief about previous works with the URbanSound8K dataset (scientific papers), dataset exploration, distribution analysis, listening.

2. [Pre-processing](https://github.com/GorillaBus/urban-audio-classifier/blob/master/2-pre-processing.ipynb): an introduction to different audible features we can use to work with digital audio, the pre-processing pipeline, STFT, MFCC and Log-MEL Spectrograms, feature extraction and data normalization.

3. [CNN model with MFCC features](https://github.com/GorillaBus/urban-audio-classifier/blob/master/3-cnn-model-mfcc.ipynb): data preparation, CNN model definition (with detailed explanation) using Keras and TensorFlow back-end. Solution of a multi-class classification problem, model evaluation and testing, Recall, Precision and F1 analysis.

4. [CNN Model with Log-MEL Spectrograms](https://github.com/GorillaBus/urban-audio-classifier/blob/master/4-cnn-model-mel_spec.ipynb): a performance comparison using the same CNN model architecture with MEL spectrograms. Same training and evaluation than notebook #3.

5. [Data augmentation](https://github.com/GorillaBus/urban-audio-classifier/blob/master/5-data-augmentation.ipynb): creation of augmented data from UrbanSound8K original sounds, using common audio effects like pitch shifting, time stretching, adding noise, with LibROSA.

6. [Augmented pre-processing](https://github.com/GorillaBus/urban-audio-classifier/blob/master/6-augmented-pre-processing.ipynb): audible features extraction from the new generated data.

7. [CNN model with augmented data](https://github.com/GorillaBus/urban-audio-classifier/blob/master/7-cnn-model-augmented.ipynb): using the same CNN architecture and almost identical training procedures with the generated data. Model evaluation and test to compare with previous achievements.

## Acquired Skills 📊
- Understanding of audio data analysis and processing techniques.
- Implementation of CNN models for audio classification.
- Evaluation of different audio features for model performance.
- Experience with data augmentation techniques for improving model generalization.
- Interpretation of results and analysis of model performance.

## Technology 📦
The project utilizes the following technologies:
- Programming Language: Python
- Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Librosa
- Tools: Jupyter Notebooks

## Getting The Dataset 📂

Download a copy of the UrbanSounds8K dataset from the [UrbanSound8K home page](https://urbansounddataset.weebly.com/urbansound8k.html).

Make sure to uncompress the dataset root directory into the project root, you should end up with a directory like "UrbanSounds8K" (or a symbolic link to it) in the project root.


## Install Required Libraries ⬇️

Make sure that Tensorflow, Keras, LibROSA, IPython, NumPy, Pandas, Matplotlib and SciKit Learn are already installed in your environment.

Note that we are using Tensorflow as Keras back-end, you must set this in your ~/.keras/keras.json file, this is an example:

```
{
    "image_dim_ordering": "tf",
    "image_data_format": "channels_first",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

## The UrbanSound8K Dataset 🎧

The **UrbanSound8K** dataset is a compilation of urban sound recordings, classified in 10 categories according to the paper ["A Dataset and Taxonomy for Urban Sound Research"](https://urbansounddataset.weebly.com/taxonomy.html), which proposes a taxonomical categorization to describe different environmental sound types.

The UrbanSound8K dataset contains 8732 labeled sound slices of varying duration up to 4 seconds. The categorization labels being:

1. Air Conditioner
1. Car Horn
1. Children Playing
1. Dog bark
1. Drilling
1. Engine Idling
1. Gun Shot
1. Jackhammer
1. Siren
1. Street Music

Note that the dataset comes already organized in 10 validation folds. In the case we want to compare our results with other we should stick with this schema.


### Dataset Metadata 📋

The included metadata file ("UrbanSound8K/metadata/metadata.csv") provides all the required information about each audio file:

* slice_file_name: The name of the audio file.
* fsID: The Freesound ID of the recording from which this excerpt (slice) is taken
* start: The start time of the slice in the original Freesound recording
* end: The end time of slice in the original Freesound recording
* salience: A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
* fold: The fold number (1-10) to which this file has been allocated.
* classID: A numeric identifier of the sound class.
* class: The class label name.

## Related Papers 🔍 

* [Environmental sound classification with convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/7324337), Karol J. Piczak

* [Dilated convolution neural network with LeakyReLU for environmental sound classification](https://ieeexplore.ieee.org/abstract/document/8096153),  Xiaohu Zhang ; Yuexian Zou ; Wei Shi.

* [Deep Convolutional Neural Network with Mixup for Environmental Sound Classification](https://arxiv.org/abs/1808.08405), Zhichao Zhang, Shugong Xu, Shan Cao, Shunqing Zhang

* [End-to-End Environmental Sound Classification using a 1DConvolutional Neural Network](https://arxiv.org/abs/1904.08990)Sajjad Abdoli, Patrick Cardinal, Alessandro Lameiras Koerich

* [An Ensemble Stacked Convolutional Neural Network Model for Environmental Event Sound Recognition](https://www.mdpi.com/2076-3417/8/7/1152), Shaobo Li, Yong Yao, Jie Hu, Guokai Liu, Xuemei Yao 3, Jianjun Hu

* [Classifying environmental sounds using image recognition networks](https://www.sciencedirect.com/science/article/pii/S1877050917316599), Venkatesh Boddapati, Andrej Petef, Jim Rasmusson, Lars Lundberg

* [Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/), Yu Su, Ke Zhang, Jingyu Wang, Kurosh Madani

## Conclusion 📝

This project demonstrates the effectiveness of CNN models for urban sound classification using the UrbanSound9K dataset. By experimenting with different audio features and data augmentation methods, valuable insights are gained into improving model performance and robustness. The project serves as a valuable resource for understanding and implementing sound event classification systems in urban environments.

## Contact Me 📨

- **LinkedIn:** [Bilal Fatian](https://www.linkedin.com/in/bilal-fatian-806813254/)
- **Gmail:** [fatian.bilal@gmail.com](mailto:fatian.bilal@gmail.com)
