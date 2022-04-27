# Deep Neural Networks are Easily Fooled

In recent times, deep neural networks (DNNs) have seen ubiquitous use in a wide number of different fields. Their popularity is in part due to their applications in image processing and recognition. However, what happens if a malicious actor were to be able to fool a highly-tuned DNN? How can this be accomplished? And more importantly, how fundamentally different is human vision compared to computer vision?

This article explores and disseminates a paper presented to the Computer Vision & Pattern Recognition COnference in 2015 titled _"Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images"_<sup>[1]</sup>, by a team of researchers from the University of Wyoming and Cornell University.

## Aside

The work presented in this post is a report in partial requirement for the EEL 6812 - Intro to Neural Networks course project, overseen by Dr. Ying Ma.

To the best of the authors knowledge, citations have been made where appropriate.

### Authors

Gustavo Diaz Galeas, MS CpE
Carlos Stanton, MS EE

Department of Electrical & Computer Engineering
University of Central Florida, Orlando FL

## Human Vision vs Computer Vision

Many layers of complexity exist with our understanding of human vision.  What we do know today is that human vision does not involve repetitions or patterns. Some would say that we do not need to learn to see, however the reality is more complex than this statement.  In regards to how humans process images, it involves the light from the image recoiling back into the eyes through the cornea and then passed on to the retina which involves the processing of colors from the image being viewed.  With Computer Vision, we seek to further explore the process of image classification by attempting to involve patterns and repetitions.  Computer Vision is interpreted via models such as neural networks (NNs) that employ different algorithms in our attempts to simulate the process of human vision.  

## Background

Convolutional neural networks (CNNs) are most commonly applied to analyze visual imagery. AlexNet and LeNet are both two networks that employ a CNN architecture. Due to their popularity, both are available pretrained from the CAFFE (Convolutional Architecture For Feature Extraction) software package. It supplies definitions for different models, optimization settings, and pretrained weights.

### ImageNet Dataset

The ImageNet dataset consists of which consists of 1.3 million images and 1000 classes. The ImageNet dataset has visually discernable features that allows Deep Neural Networks such as AlexNet DNN to determine the classification of images. Below is a sample of the images contained in the ImageNet dataset:

![A subset of the IMAGENET dataset](https://i.imgur.com/hwMoZQc.jpg)


ImageNet classes may sometimes contain multiple labels associated with it <sup>[7]</sup>. It can be of a common name, species, etc.. For example: ImageNet class 14 (i.e. classID of 13) contains two class labels: junco and snowbird. In ornithological terms, a snowbird are any of several birds seen primarily during the winter time. On the other hand, the junco is a species of small North American finches that some ornithogolists state are _the_ snowbird as they can often be seen foraging in flocks. This means that networks tuned for ImageNet classification will look for visual features corresponding to the junco, and the output label will have both labels of "junco" and "snowbird" together. 

![A dark-eyed junco perched on a branch.](https://i.imgur.com/Ufrn2Qu.jpg)

Below we have an example of 14 out of the 1000 classes of images from the ImageNet dataset.

![](https://i.imgur.com/zXyYMZa.png)

### MNIST Dataset

The MNIST dataset contains 60,000 images and 10 classes of handwritten digits, ranging from 0 to 9. Below is an example of the MNIST dataset, where each row corresponds to a specific digit, and each column corresponds to a unique image representing the respective digit.

![](https://i.imgur.com/r5LtWQh.png)

### AlexNet DNN

In 2012, AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). This challenge strives to obtain networks that achieve top-1 and top-5 error rates, using a subset of ImageNet's images. The challenge provides about 1.2 million training images, 50 thousand validation images, and 150 thousand testing images, along with enforcing a fixed image resolution of 256x256 pixels.

Its famed recognition comes from having achieved a top-5 error of 15.3%, and the primary designer, Alex Krizhevsky, states that the depth of the model was the key in achieving such high performance. <sup>[4]</sup>

![Typical architecture of the AlexNet DNN](https://i.imgur.com/0kN9BqG.png)

AlexNet is composed of a total of 8 layers with learnable parameters. Although the current specifications of the model are different than from when it first competed, it still achieves high performance. This is due to its underlying architecture, consisting of three sets of convolutional layers in a 1:1:3 configuration, each feeding into their own max-pooling layers (which accounts for 3) to reduce overall dimensionality of the data. The output of the final convolutional layer is then fed into two sets of fully connected layers, before being fed into a final fully-connected layer that has the softmax activation function for label output.

The input expected by the AlexNet DNN are three images of size 256x256, with the output being the categorical label that the network predicts the image belongs to. The total number of parameters present in this architecture is 62.3 million.

### LeNet DNN

The most common configuration of the LeNet DNN is LeNet-5, due to consisting of 5 different layers, which will be explained in more detail later.

LeNet-5 is one of the earliest pre-trained models, proposed in 1998 as part of a paper in applying gradient-based learning to document recognition <sup>[3]</sup>. Part of the reason for its enduring popularing in image recongition is due to its simple and straightforward architecture.

![Typical architecture of the LeNet-5 DNN](https://i.imgur.com/Z9cTLoH.png)

As previously mentioned, the network is composed of 5 layers, each with learnable parameters. The first three layers are sets of convolution layers that are supported by average pooling layers that exist in-between. After the data has been processed by the convolutional layers, it is then fed into two fully connected layers before leading towards a Softmax activation function acting as the classifier.

The input expected by LeNet-5 is a single 32x32 grayscale image, and the output is 10, one for each handwritten digit. Out of these 10 values, one will have the highest value and this is the classification that the image belongs to. In addition, the number of learnable parameters is 60 thousand.


### Evolutionary Algorithms

Evolutionary algorithms are a class of optimization algorithms that have drawn inspiration from Darwinian evolution. It encapsulates the idea of "survival of the most adaptable" in obtaining a high-performing result. In the case of fooling DNNs, evolutionary algorithms are utilized in the generation of images that seem undiscernable to humans, but are classified with high confidence by networks.

At a high level, the typical flow of execution of an evolutionary algorithm is as follows. First, there is an initialization of an initial population of solutions. Upon the creation of this initial population, the algorithm then proceeds into the selection phase, where the members of the population are evaluted according to a fitness function. The members that perform the best are kept, and the algorithm moves towards the crossover phase. This phase makes use of the top performing members identified in the selection phase, and are used to generate new members (i.e. children) that contain similar features as the parents. Finally, there is a mutation phase that is done to ensure the algorithm does not get stuck in a local extrema. This phase simply involves randomly mutating the new generated population at random.

Each pass of this evolutionary cycle is referred as a generation. The algorithm can be terminated upon reaching a specified number of generations, or until the performance of each member population in the selection phase converges.

![Diagram highlighting the high-level execution flow of an evolutionary algorithm.](https://i.imgur.com/M04Y2EC.png)

In the case of image generation, the initialization step involves in the random selection of images from either the ImageNet or MNIST datasets. The fitness function is the DNN to be fooled (i.e. AlexNet for ImageNet evolution or LeNet for MNIST evolution). In crossover, the best performing images are then used to generate additional images that contain features similar to the images used in generating them. Finally, pixels are mutated at random, dependent on how the evolutionary algorithm has been constructed.


## Methods of Image Generation

Two methods of generating images were presented, both used in conjunction with an evolutionary algorithm. The first makes use of direct encoding, which often produces irregular images, so called due to the lack of any underlying regularity that humans are able to recognize. The second makes use of indirect encoding, which produces regular images that are highly geometrical in nature. Both of these methods are discussed in further detail below.

### Direct Encoding

Direct encoding is a method where information is encoded in a 1-to-1 fashion without transformation. This is perhaps the most straightforward method of representing information as nothing else needs to be done. For example, in the case of MNIST images, which are 28x28 in size, are represented by one grayscale integer for each pixel. Conversely, for ImageNet images, a three-tuple value of (H, S, V) are used to represent each 256x256 pixel in the image.

In the context of evolution, the initial population is created by initiating each pixel value with uniform random noise, i.e. p ∈ [0, 255]. For image mutation, each pixel value is independently mutated at an initial rate of 0.1, dropping by half every 1000 generations. This mutation rate is analogous to the learning rate in training NNs.

The direct encoded images are evolved repeatedly into highly distorted images of each digit (0-9), making use of the LeNet DNN as the fitness function due to it performing well on images from the MNIST dataset. After multiple generations, the evolved images were classified by LeNet into corresponding digits with a 99.99% confidence for it to be digits, despite the high irregularity present that make it unrecognizable to the human sight.

Shown below are images generated from the MNIST dataset. Each column represents a different digit, whereas each row is the result after 200 generations of evolution. Despite many of the images appearing to lack any distinct and discernable features, on some images there does appear to be some form of regularity. For example, in the images identified as "1" by the DNN appear to have a repeating pattern of vertical lines. In this case, it appears that the LeNet DNN is recognizing these repeating vertical bars as being a handwritten "1" digit. This observation leads to the suggestion that DNNs seem to attempt and identify certain discriminative featues that aid it in its classification of the image with such a high confidence. The remaining images, on the other hand, appear to be more random and seem nothing more than noise.

![](https://i.imgur.com/DN26Eih.png)

Below are images evolved from the ImageNet dataset. As previously stated, there are 1000 classes that identify different labeled objects.  In this case the images you can see discern certain features of these objects identified in the Imagenet database and declare with greater than 99% confidence that for example, the image labeled “robin” represents a particular feature from the dataset.  The feature could be the color, it could be the beak, or several other features.  The color does appear indicative of a real image of a robin, but this could also be something else entirely.  The same can be said for the image that was identified as a “peacock”.  We know that the colors of a peacock typically are blue and green in nature, so we could say that the AlexNet DNN trained on the Imagenet framework discerned this particular feature to identify with 99% certainty or more that this is a peacock. 

![](https://i.imgur.com/Q9wbTua.jpg)

### Indirect Encoding

Indirect encoding is a method of representing information through changing it from one form to another. To better explain how this can be accomplished, a compositional pattern-producing network (CPPN) was utilized to aid in generation of images. The way how this network works is by accepting a (x, y) position of a pixel as its input, with the output being a grayscale value for MNIST images, or a HSV tuple for ImageNet images. The output activation function is what makes these images appear to be highly regular in nature. The available activation functions that the CPPN can make use of are either sine, sigmoid, Gaussian, or linear. In some shape or form, these activation functions are related in the realm of analytical geometry. At the beginning of evolution, the CPPN begins simply with an input and output layer. That is, it does not have any hidden layers. The process of evolution helps determine the topology, weights, and activation function of the CPPN, and there exists one CPPN for each pixel value in each image.

The indirectly encoded images as shown below are evolved repeatedly to produce images that seem more recognizable as each digit class 0-9 from the MNIST dataset.  This resulted in more distinguishable patterns of images however, the LeNet DNN also labeled with 99.99% confidence as with the directly encoded case previously.  As stated previously, the pattern form to identify as digit class “1” shows more visually comprehensible vertical bars, however now we can see smaller vertical bars that can be understood as matching another segment of the number 1.  For the digit class “0” in the first image at the top of this class, it appears to repeat the pattern of the top portion of the segment of “0”, and also in the fourth image down from digit class “0”, it also appears to repeat the pattern of the top portion of the segment of “0”. 

![](https://i.imgur.com/Ql12PUh.png)

In the next image below, we can see evolved images from indirect encoding that appear to have more identifiable features.  For example, if we look at the picture labeled as a starfish, we can see the colors blue and an orange-brown combination.  These visually discernable features allow the Alex-Net DNN to identify this as a starfish due to the original images from the Imagenet Dataset. The same can be said for the electric guitar.  The image shown appears to be a morphed picture of the guitar strings on a guitar.  In the example for the remote control, we all know remote controls have buttons.  Although this pattern of buttons is repeated several times in this image, we can see how the AlexNet DNN would pick up this evolved image and identify it as a remote control.  Some images are more difficult to understand as to why the DNN would label it as such an object.  Take the “African Grey” for example, one can surmise from this picture that the Alex-Net DNN focused more on the colors of the African grey bird which is a darker grey in some parts of the body and a lighter grey in other parts.  The background stitching colored in red and green could possibly be the background that the particular original image was taken in.

It is clear to see that the AlexNet DNN used can correctly identify some images based on visual features such as color, and patterns as discussed above, however the confidence score of 99% or greater remains questionable which further demonstrates how DNN’s can be fooled.

![](https://i.imgur.com/cwgKnhu.jpg)

## Resulting Images

Below, we can see a image of Evolved Images before selection and evolution and then we can see images after selection and evolution. Compositional Pattern Producing Networks can be used to evolve images.  In this paper, originally the website used was picbreeder.org which involved CPPN-Encoded Evolutionary algorithms which is a indirect method that produces images that both humans and Deep Neural Networks can recognize.  This original website has since been retired due to the outdated code.  However, a team of programmers back in 2020 from southwestern university came together and created an updated version of this website.

In this instance, the user is considered as the fitness function since we can select any image that we choose, and then this image will become the parents of the next generation via the CPPN network architecture as previously discussed.  
 If you look closely at the first image presented below, you will see a red box that contains the image that we selected to be mutated. 



### Original Images before selection and evolution
![](https://i.imgur.com/mQLUvqz.png)





 For the image below, you can see the results after this image goes into the CPPN as input and generates a tuple of HSV color values.  HSV(Hue Saturation Value) is an alternate representation of the RGB color model.

### Evolved Images after selection and evolution
![](https://i.imgur.com/DvckGmy.png)


## Discussion

The results of the paper are some of the more interesting that have been produced in this field, as it leads towards several questions, and above all, the introduction of different techniques in the generation of unique imagery that contains information embedded in them that is otherwise undiscernable to the human vision.

### Video Games

The methods of direct and indirect encoding images used in neural networks have helped pave ways in their application towards video games. One such example is where researchers made use of both methods in the construction of two different networks, NEAT and HyperNEAT, to gauge their performance as both play the popular video game _Tetrs_ <sup>[6]</sup>. The results indicate that a hybrid approach in both direct and indirect encoding should be pursued, as each have differing strengths that affect overall performance.

### Algorithms

As a follow-up from the above, another team of researchers have combined both methods of direct and indirect encoding in the context of evolutionary algorithms <sup>[5]</sup>. This work was done as the researchers observed that many of the challenging problems that plague engineers tend to be regular. In other words, the solutions to one part of a problem can often be reused to solve other parts, almost in a case of a divide-and-conquer strategy. Due to this observation, work was done in developing an algorithm called HybrID, which combines the best of both direct and indirect encodings through evolutionary algorithms.

### Security

Perhaps the largest possible implication regarding the results produced in the paper is the security aspect to it. The LeNet DNN was able to confidently classify evolved images as digits, despite it not appearing as such to the human sight. Likewise, the AlexNet DNN also confidently classified evolved images to be of specific objects. Suppose that a malicious actor were to create a method of generating evolved images to fool networks tuned to identifying whether the image is of an personal identification document, such as a security badge, but also contains encoded information that is identifiable to the system to be valid. Thus, this allows the malicious actor to be able to obtain access to a secured system due to the network identifying it as valid, but when a security team inspects footage, all that is seen are images that lack any discernable features, as in the case of direct encoding, or one that contains geometrical features that don't hold significance, as in the case of indirect encoding.

This can also be generalized in a scenario where a malicious actor would be able to produce a trail of image evidence that contains encoded information only a network could identify but is still utter nonsense, while simultaneously being of no use to law enforcement due to the lack of any features that can be quickly distinguishable to the human sight.


## Conclusion

The work presented has helped drive much research in different fields, namely in the methodologies that were used. It also helped shed intriguing insight into the major differences that still exist between how humans see and how computers see. Ultimately, it also raises additional questions in the implications of making use of these results towards malicious use in attempting to obtain unauthorized access to a secure facility, or perhaps leave behind an unidentifiable trail of evidence that can not be linked back to a malicious actor.

## Sources

[1] Nguyen A, Yosinski J, Clune J. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images. In Computer Vision and Pattern Recognition (CVPR ’15), IEEE, 2015.

[2] J. Secretan, N. Beato, D. B. D Ambrosio, A. Rodriguez, A. Campbell, and K. O. Stanley. Picbreeder: evolving pictures collaboratively online. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 1759–1768. ACM, 2008.

[3] Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.

[4] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” in Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1, 2012, pp. 1097–1105.

[5] Helms L, Clune J (2017) Improving HybrID: How to best combine indirect and direct encoding in evolutionary algorithms. PLoS ONE 12(3): e0174635. https://doi.org/10.1371/journal.pone.0174635

[6] Lauren E. Gillespie, Gabriela R. Gonzalez, and Jacob Schrum. 2017. Comparing Direct and Indirect Encodings Using Both Raw and Hand-Designed Features in Tetris. In Proceedings of GECCO ’17, Berlin, Germany, July 15-19, 2017, 8 pages. DOI: http://dx.doi.org/10.1145/3071178.3071195

[7] https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/