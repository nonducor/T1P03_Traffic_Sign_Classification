# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class_histogram]: ./imgs_writeup/class_histogram.png "Class histogram"
[intensity_histogram]: ./imgs_writeup/intensity_histogram.png "Intensity histogram"
[class_examples]: ./imgs_writeup/class_examples.png "Class examples"
[images_transformed]: ./imgs_writeup/images_transformed.png "Images transformed"
[il_std_orig]: ./imgs_writeup/il_std_orig.png "Input layer standardization - original image"
[il_std_gray]: ./imgs_writeup/il_std_gray.png "Input layer standardization - grayscale image"
[il_std_rgb]: ./imgs_writeup/il_std_rgb.png "Input layer standardization - color image"
[network_arch]: ./imgs_writeup/network_arch.svg "Network architecture"
[training_convergence]: ./imgs_writeup/training_convergence.png "Training convergence"
[additional_test_images]: ./imgs_writeup/additional_test_images.png "Additional test images"
[unknown_signs]: ./imgs_writeup/unknown_signs.png "Unknown signs classification confidence"
[led_speed_limit_example]: ./imgs_writeup/led_speed_limit_example.png "LED Speed Limit Analysis"
[LED_color_std]: ./imgs_writeup/LED_color_std.png "Color standardized LED input"
[LED_first_layer_activations]: ./imgs_writeup/LED_first_layer_activations.png "First layer activation for the LED sign"
[Non_LED_color_std]: ./imgs_writeup/Non_LED_color_std.png "Color standardized non-LED input"
[Non_LED_first_layer_activations]: ./imgs_writeup/Non_LED_first_layer_activations.png "First layer activation for the Non-LED sign"

## Rubric Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb). Most figures shown in this writeup can be found on the notebook as well

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the basic numpy API on the training, validation and test arrays to get some statistics of the training set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

(**Note:** the number of unique classes was cross-checked with the file `signnames.csv` to make sure the training set covered all classes.

#### 2. Include an exploratory visualization of the dataset.

Three basic analysis were done on the dataset (always only looking at the training set): 

1. A histogram of samples per class 

![alt text][class_histogram]

2. A histogram of lightness/darkness of the training examples.

![alt text][intensity_histogram]

3. Plotting a random image of each class

![alt text][class_examples]



### Design and Test a Model Architecture

#### 1. Preprocessing and additional training data

I used the keras `ImageDataGenerator` class to generate more training data. During experimentation, I found that this improved the generalization of the network. To include the generated data on the training process, I duplicated every batch, using the basic training data as the first part of the batch and a second random batch using the image generator. The reason for this will be discussed on the section about training of the network.

The `ImageDataGenerator` was configured with the following parameters:

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.1],
    fill_mode='reflect',
    data_format='channels_last'

To find out a good set of parameters, some manual experimentation was done and the results visually inspected.

Some transformation options should obviously not be used for this application, such as mirroring the image. 
Some examples of the transformations are shown on the figures below:

![alt text][images_transformed]

The only pre-processing steps used are conversion to grayscale (as suggested in [LeCun:2011]) and image normalization.

To avoid errors (e.g. forgetting to apply the preprocessing steps), the preprocessing of each image was embedded on the tensorflow network, using the functions `tf.image.rgb_to_grayscale` and `tf.image.per_image_standardization`. This is quite convenient, though slightly slower for training (due to the re-conversion of the images every epoch).

The effect of per-image standardization can be seem on the example below (using the same image #16149, shown on the class examples.

![alt text][il_std_orig]

![alt text][il_std_gray]

![alt text][il_std_rgb]

The first layer of the network uses a separate convolution for a grayscale feature (42 filters) and for the color features (16 filters). The value number of filters was found through a lot of trial and error, but generally the network presents similar performance when up to 100 filters are used on the first layer (beyond that the training starts becoming too slow).




#### 2. Model architecture

After a lot of experimentation (using ideas mostly from [LeCun:2011] and [Szegedy:2015]), the architecture below was found. The most important characteristic of the network is that it uses a multi-scale feed on the first fully connected layer (it receives as input all convolutional layers, with a heavier pooling). An overview of the network structure is on the picture below:

![alt text][network_arch]

The two basic layers of the network are a convolutional layer and a fully connected layer. A convolutional layer has the following operations:

1. A linear convolution kernel (with bias)
2. A **tanh** activation function
3. A local response normalization function

All fully connected layers have this structure:
1. Linear combination of the previous inputs (with bias)
2. A **tanh** activation function (except on the logits layer)
3. During training, a dropout operation on the two mid layers.

To develop the network, I started from the LeNet network presented on the classes and from there, experimented with different activation functions, compositions of each layer and connections (using the directions provided in the aforementioned papers). I found that:

1. Application of multi-scale feed made the accuracy jump from 89% to around 92%.
1. Usage of tanh instead of RELU greatly improved training convergence and accuracy (from around 92% to 96%), without any noticeable change on time to execute each training epoch.
1. The next jump from around 96% accuracy to 97% accuracy was achieved by using local response normalization and the layer with 1x1 convolutional kernels on the 2nd convolutional layer, forcing a compression on the number of filters.
1. Finally, experimenting with dropout rates on the fully connected layers helped improve the accuracy by around 1 percent point also. I tested dropouts from 0 to 60%, in steps of 10%. Dropouts above 30% generally gave better results, so I chose 50% , also due to the theoretical reasons behind it.

During the development, a systematic experiment was done to find out the influence of the number of neurons on the fully connected layer. The experiment is not recorded on the python notebook but, basically, 36 combinations of number of neurons on the two hidden layers were tried out (from 300 to 100 on the first layer and from 200 to 50 on the second layer). It was found that the network is not exceptionally sensitive to these parameters, as long as the second layer has between 100 to 150 neurons.


#### 3. Training

To train the model, I used the suggested Adam optimizer, as it [seems relatively robust](https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2). I experimented with learning rates from 0.001 to 0.000005, including trying to do a step reduction on the training rate after an accuracy of 0.97 was found and also progressively reducing it every epoch. Neither method seemed to cause any perceivable gain on speed or quality of the final result, thus I decided to use a fixed learning rate on the final training.

Most of the experiments executed were done using something between 10 to 20 epoch of training (in my computer this gives something between 5 to 30 min, depending on the experiment). When I settled on a suitable architecture, I run the training through 200 epochs. The decrease in error rate (1 - accuracy) can be seen below:

![alt text][training_convergence]

When I added the generation of variants based on the original training set, I did it in a slightly hackish way. As I wanted to make sure that all images on the training set were shown to the network, I made the main training loop (the one that gets a batch and trains it) actually run twice: one with a batch from the training set and the second run with a same sized batch of variants. Thus, when the training script outputs the training of an "EPOCH", what actually happened is that the network was trained in a training set twice the size of the original training set. No randomization of images was used on the validation set as it is simply used as a reference to understand how well the training is going.

One thing that, in retrospective, I should have done while experimenting with the structure of the network was to avoid training the convolution layers until I got a good structure and only after being happy with the other parameters, start training then. That would have greatly speed up investigation and, as mentioned by [LeCun:2011], random features give a reasonable estimate of the performance of the network during exploratory phases.

#### 4. Results

My final model results were:
* training set accuracy of 99.99 % (without including random generate examples)
* validation set accuracy of 98.87 % 
* test set accuracy of 98.44 %

Most of the analysis of how I got to these results are shown on the previous discussion. A important remark is that I did not consider test set results until all parameters of the network were fixed, so there is no "leakage" of the test results on the network architecture or hyperparameters.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report

As I happen to live in Germany, instead of getting traffic signs from the Internet I took some pictures on my path from work to home. All pictures were taken at dusk, on the same day.

From the pictures, I them manually cut several traffic signs, of different resolutions and separated 52 images. Due to the characteristics of the place were I took the pictures, there are not too many variations (I have examples of only 13 of the 43 classes). But I collected two interesting examples: 
1. LED light speed signs, which are part of a category on the training set but are a configuration not on the training set
2. Two signs that are not on the training set.

All new test images, together with their classification results, can be seem on the image below. My original expectation on the classification results were:

1. Most of the images that are present on the training set should be correctly classified, as they are not particularly hard (from a human perspective, of course. I found some images on the training set quite hard to identify)
2. The LED speed signs should be classified as speed signs, even if the model cannot recognize the text.
3. The two images not on the original categories should be classified as categories of visually similar things (e.g. I expect the left arrow over a blue field to be classified as one of the blue signs). 
 
 Overall, I was generally satisfied with the results. I was surprised by the number of LED signs that were classified as 120 km/h (I was expecting confusions more on the sort of 60 being classified as 80 or 50). After observing the first internal layer of the network and the normalized images it struck me that the network is probably expecting the numbers to be darker than the surroundings. This explains the noted bias towards 120 km/h (because it has the darkest content inside the red circle).
 
 The classification of the new signs (the two last ones on the picture) was exactly as I expected, them being assigned to similar categories.
 
 The only unexpected error was a distorted (due to perspective) "Turn left" sign that was classified as a "Straight ahead". Probably the ratio of the sign influenced on the classification.

![alt text][additional_test_images]

#### 2. Accuracy of the predictions

Since the set had some cases that were never seen by the network, I'll analyse them separately.

| Set | Accuracy | Comments |
|:---:|:--------:|:--------:|
| Already seen signs | 1 - 1/43 (97.6%) | Accuracy in line with the expected |
| LED signs | 1 - 6/7 (14%) | Most were classified with a different speed, but all were classified as traffic signs |
| Unknown cases | - | Both were classified as similar signs, which was as expected |
 
Without considering the unknown cases (but including the LED speed signs), the overall model accuracy was 86%.

#### 3. Model certainty

For each of the 52 example images, the top 5 predictions were calculated. As most of them were correct, they will not be shown in this writeup. The reader should refer to the python notebook (the table is on the end of the notebook). Nevertheless, the two unknown signs and one example of the LED speed sign will be analysed here, as they are a good example of how the network was able to generalize some features.

![alt text][unknown_signs]

In both unknown signs, it can be seen that the top 5 hypotheses by the network were consistent with the visual appearance of the signs. I understand this as a sign of good generalization, in the sense that the network ended up creating features that somehow map to human intuition.

On the LED speed signs, the network typically failed to "read" the speed, being usually able to detect that it was a speed sign (and most hypotheses were consistent with that).

![alt text][led_speed_limit_example]


### Visualizing the NN

I'll use one of the 60 kp/h misclassification cases as an example for visualizing the network features. I'll concentrate on the first convolutional layer as I found it easier to interpret (to be honest, I couldn't make sense of the meaning of the 3rd convolutional layer. Nevertheless, they are quite different for both inputs.

On the other hand, on the first layer, it becomes clear that the number information is less distorted on the correctly classified sign than in the wrongly classified sign. Analysing the color only filters and comparing them to the normalized image is quite informative (the color filters start from Feature Map 42 until the end). 

All plots can be found on the python notebook. On this writeup I added only the color channel with standardized colors and the first layer

#### Color standardization

For the LED sign:

![alt text][LED_color_std]

For the normal sign:

![alt text][Non_LED_color_std]


#### First layer activations

For the LED sign:

![alt text][LED_first_layer_activations]

For the normal sign:

![alt text][Non_LED_first_layer_activations]



# References

[LeCun:2011]: Pierre Sermanet and Yann LeCun, "Traffic sign recognition with multi-scale Convolutional Networks"

[Szegedy:2015]: Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jonathon and Wojna, Zbigniew, "Rethinking the Inception Architecture for Computer Vision"