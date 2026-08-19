---
documentclass: extarticle
papersize: letter
geometry: "margin=0.75in"
fontsize: 14pt
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Stanford CS231N | Spring 2025 | Lecture 2: Image Classification with Linear Classifiers


<p align="center"><img src="./lecture_02_slides/slide_4_00-00-00.133.jpg" width="75%" alt="Lecture Video at 00:00:00.133" /></p>

We will be talking today about image classification, continuing our discussion on the topic of image classification from last lecture.


<p align="center"><img src="./lecture_02_slides/slide_302_00-00-10.077.jpg" width="75%" alt="Lecture Video at 00:00:10.077" /></p>

We'll get into some topics that get us closer to neural networks and, ultimately, convolutional neural networks and so on. We will start with linear classifiers.


<p align="center"><img src="./lecture_02_slides/slide_1034_00-00-34.501.jpg" width="75%" alt="Lecture Video at 00:00:34.501" /></p>

Moving to the next slide, this was the syllabus that we talked about last lecture. We also covered some subtopics that we will be covering in the class, including discussions around human-centered AI aspects at the end.


<p align="center"><img src="./lecture_02_slides/slide_3058_00-01-42.035.jpg" width="75%" alt="Lecture Video at 00:01:42.035" /></p>

Just like in the previous lecture, let's start with our core task of image classification. Again, it is a core task in computer vision. We actually come back to this task quite often throughout the quarter because it is a very good benchmark. This is one of the items that we come back to quite often.

We want to define the image classification task today and then introduce two of the data-driven approaches: nearest neighbor, and the other one, linear classifier.


<p align="center"><img src="./lecture_02_slides/slide_5058_00-02-48.768.jpg" width="75%" alt="Lecture Video at 00:02:48.768" /></p>

So what is image classification?


<p align="center"><img src="./lecture_02_slides/slide_5748_00-03-11.792.jpg" width="75%" alt="Lecture Video at 00:03:11.792" /></p>

Given an image and a number of predefined labels—a set of possible labels such as dog, cat, truck, plane, and so on—the job of the system is to assign one of those labels to this image. To us, this is actually a very easy task because our brains—our cognitive system—are wired to get a holistic understanding of this image and assign a label to it. But when it comes to coding this and looking at how a computer can make sense of this image, that's completely a different story.


<p align="center"><img src="./lecture_02_slides/slide_6854_00-03-48.695.jpg" width="75%" alt="Lecture Video at 00:03:48.695" /></p>

We want to see how machines can make sense of such data. Images are often defined by matrices of data, more broadly, generally, tensors of data. Often, the numbers—each of the pixel values—are between 0 and 255, which is an 8-bit data structure. Since this is a colored image assuming that it has a resolution of $800$ by $600$, since it's an RGB image, it has three channels: red, green, and blue (RGB).

Therefore, it is a tensor of $800 \times 600 \times 3$, as you can see on this slide. So as you can probably guess, this is the semantic gap between our perception of this image and how the machine perceives and sees the image. To understand how challenging this could be, let's look at some challenges and variations.


<p align="center"><img src="./lecture_02_slides/slide_9526_00-05-17.851.jpg" width="75%" alt="Lecture Video at 00:05:17.851" /></p>

In this type of imaging data, let's assume that we move the camera.


<p align="center"><img src="./lecture_02_slides/slide_10230_00-05-41.341.jpg" width="75%" alt="Lecture Video at 00:05:41.341" /></p>

So all these pixels will have a new value. Again, for us humans, it's the same object; there is no difference. But from a computer's perspective, it's completely a new data point. This is one of the challenges, but there are quite a few others as well.


<p align="center"><img src="./lecture_02_slides/slide_10994_00-06-06.833.jpg" width="75%" alt="Lecture Video at 00:06:06.833" /></p>

For example, illumination is another challenge. That's why a same cat, same object, may look differently in terms of numbers when it comes to being pictured in different illumination conditions. With that in mind, whether the cat is in a dark room or under the sun, still, it's the cat; it's one cat. But this is creating challenges for the machine.

Background clutter, background objects—which is actually our next slide.


<p align="center"><img src="./lecture_02_slides/slide_13584_00-07-33.253.jpg" width="75%" alt="Lecture Video at 00:07:33.253" /></p>

Background clutter is another challenge. What else? Zooming in and out, so scale basically of the object in the image.


<p align="center"><img src="./lecture_02_slides/slide_18140_00-10-05.271.jpg" width="75%" alt="Lecture Video at 00:10:05.271" /></p>

The resolution of the image could be considered as definitely a challenge. Occlusion is one of the major problems. As humans, it's very easy to say this is a cat; these are cats. Even the last one, which is actually very challenging, the one on the right—you can only see a tail and a little bit of probably a paw on the right side.

One could say that it could be a tiger, or it could be a raccoon with a tiny tale. But because this is because of the context, because we know this is inside a living room on a couch, most probably, it's a cat. So again, for us humans, it's not that hard. Beyond that, there are many other problems: deformation.

Deformation is one of the other major challenges. And beyond that, intraclass variation is one more important challenge. We know that cats can come in different sizes, colors, patterns, or even they have different breeds. All of those are still cats, but for machines, it's not that easy to recognize the intraclass variations.

One other interesting challenge is context, because if you only look at that part—that image on the right—or if an algorithm looks at this without considering the context, It's very easy to classify this as a tiger or some other animal. But because of the context and because we know that there is the effect of shadows and so on, this could probably be classified correctly.


<p align="center"><img src="./lecture_02_slides/slide_19450_00-10-48.982.jpg" width="75%" alt="Lecture Video at 00:10:48.982" /></p>

In this class, what we want to do is to get to a place that we build models that can recognize activity—recognize objects, and also other aspects within the image. Before doing so, we have to look at the most basic building block of classifying an image, and that is building, implementing a function like this.


<p align="center"><img src="./lecture_02_slides/slide_21564_00-11-59.518.jpg" width="75%" alt="Lecture Video at 00:11:59.518" /></p>

There is a clear flowchart of tasks and steps—if-then-else steps that creates an algorithm for sorting. But when it comes to images and understanding the visual world, that is not happening; that is a challenge. There is no way to hard code the steps for classifying images. Although there has been some effort in this space.


<p align="center"><img src="./lecture_02_slides/slide_23298_00-12-57.377.jpg" width="75%" alt="Lecture Video at 00:12:57.377" /></p>

There are papers that have tried to come up with algorithms and steps to recognize objects.


<p align="center"><img src="./lecture_02_slides/slide_24082_00-13-23.537.jpg" width="75%" alt="Lecture Video at 00:13:23.537" /></p>

While this has been an interesting effort and it had some success on very limited variability type of images, it is very hard to scale these types of algorithms. Also, finding the logic for each of those requires a lot of effort by itself as well. Because of these challenges, I think these types of algorithms, which are based on creating logics and procedures for detecting objects or classifying images, have not been quite successful.

Machine learning comes with this data-driven approach.


<p align="center"><img src="./lecture_02_slides/slide_26890_00-14-57.230.jpg" width="75%" alt="Lecture Video at 00:14:57.230" /></p>

With this new paradigm of looking at this problem from a data-driven perspective, we define a procedure of a three-step process.


<p align="center"><img src="./lecture_02_slides/slide_26994_00-15-00.700.jpg" width="75%" alt="Lecture Video at 00:15:00.700" /></p>

The first one is to collect data sets of images and their labels. We used to be doing this 10, 20 years ago, using search engines and image search engines over the internet to create these types of data sets.


<p align="center"><img src="./lecture_02_slides/slide_28386_00-15-47.146.jpg" width="75%" alt="Lecture Video at 00:15:47.146" /></p>

Now we have all of the data sets. The second step is using machine learning algorithms to train a classifier.


<p align="center"><img src="./lecture_02_slides/slide_29146_00-16-12.505.jpg" width="75%" alt="Lecture Video at 00:16:12.505" /></p>

So, it is a very simple procedure. But instead of building a logic, we are building a data-driven approach for it. I said we want to talk about two popular methods and classifiers.


<p align="center"><img src="./lecture_02_slides/slide_30456_00-16-56.215.jpg" width="75%" alt="Lecture Video at 00:16:56.215" /></p>

One of them is the nearest neighbor classifier. Then we'll move to the topic of linear classification.


<p align="center"><img src="./lecture_02_slides/slide_31430_00-17-28.715.jpg" width="75%" alt="Lecture Video at 00:17:28.715" /></p>

To build the nearest neighbor classifier, we need to build the `train` and `predict` functions.


<p align="center"><img src="./lecture_02_slides/slide_31680_00-17-37.056.jpg" width="75%" alt="Lecture Video at 00:17:37.056" /></p>

The `train` function needs to just memorize all of the data and labels.


<p align="center"><img src="./lecture_02_slides/slide_32022_00-17-48.468.jpg" width="75%" alt="Lecture Video at 00:17:48.468" /></p>

So, the training function basically doesn't do anything other than keeping everything in memory. The prediction function—the `predict` function looks for the most similar training image. It creates a lookup table of all of the images and all of their labels. During the prediction or testing time, what it does is tries to find the closest one, the most similar image, and outputs the label for that image.


<p align="center"><img src="./lecture_02_slides/slide_32904_00-18-17.897.jpg" width="75%" alt="Lecture Video at 00:18:17.897" /></p>

Let's look at an example. Assuming that we have these five as in our training data, then this is the query image, the input image for prediction. What we want to do is to see which of these training data and training images is the most similar to this one. For that, we need the distance function.


<p align="center"><img src="./lecture_02_slides/slide_34456_00-19-09.682.jpg" width="75%" alt="Lecture Video at 00:19:09.682" /></p>

There are many different ways of doing that. One of the most popular ones is $L_1$ distance, which is defined as the sum over all absolute values of pixel differences between the two images, $I_1$ and $I_2$. This defines the value as the distance between these two images. So, this is the most basic distance function, but it's actually very useful in many applications.

We'll be coming back to this $L_1$ and other variations of distances in the class quite often.


<p align="center"><img src="./lecture_02_slides/slide_36366_00-20-13.412.jpg" width="75%" alt="Lecture Video at 00:20:13.412" /></p>

With this very simple definition, we want to see how we can get it implemented.


<p align="center"><img src="./lecture_02_slides/slide_36514_00-20-18.350.jpg" width="75%" alt="Lecture Video at 00:20:18.350" /></p>

The first step is to just memorize the training data. So, the `train` function just keeps the data in memory.


<p align="center"><img src="./lecture_02_slides/slide_36860_00-20-29.895.jpg" width="75%" alt="Lecture Video at 00:20:29.895" /></p>

The `predict` function uses Python libraries like NumPy, and so on, We can implement this in just four lines. So this is going to be the implementation for the `predict` function.


<p align="center"><img src="./lecture_02_slides/slide_38356_00-21-19.812.jpg" width="75%" alt="Lecture Video at 00:21:19.812" /></p>

The pixel values, as I explained, in the simplest form, this is a tensor of $800$ by $600$ by $3$ and three channels. These are RGB values for each of the pixel locations. I should actually repeat the questions for online students too. The question was what do the pixel values represent?

The next question is why it's between $0$ and $255$. There are many different standards for storing images. The most popular one that we use in almost all images that you see online and here they are RGB. RGB is a $24$-bit format, sometimes $32$ because there is another channel (alpha).

That's the standard that is defined; there are some other frameworks too, but this is the most popular one. With that, let me go back to the code and ask you a question.


<p align="center"><img src="./lecture_02_slides/slide_41550_00-23-06.385.jpg" width="75%" alt="Lecture Video at 00:23:06.385" /></p>

I'm hoping that you are familiar with big-$O$ notation that often represents computational and sometimes space complexities with. If you look at the algorithms, let's go with the training data.


<p align="center"><img src="./lecture_02_slides/slide_42464_00-23-36.882.jpg" width="75%" alt="Lecture Video at 00:23:36.882" /></p>

For the training step, it is of $O(1)$ because we are not actually doing anything; we are not even moving any data. We are just keeping a copy of the data in memory. It means that without operations—with an operation order of 1—we can complete the training step. What about the prediction step for each single example of the testing data?

How many operations should we take?


<p align="center"><img src="./lecture_02_slides/slide_44532_00-24-45.885.jpg" width="75%" alt="Lecture Video at 00:24:45.885" /></p>

It wouldn't work when it needs to scale for very simple problems. We used to be using these types of approaches. What we often want is to build classifiers that are fast during prediction. They do it much faster, but it's okay if they take a lot of time to do during the training because that...

could be done offline.


<p align="center"><img src="./lecture_02_slides/slide_46926_00-26-05.764.jpg" width="75%" alt="Lecture Video at 00:26:05.764" /></p>

But with that, I want to look at some of the visualizations and how this algorithm in general works. If you have a testing sample that is in that specific region, the color of that region shows what the nearest neighbor for that sample will be. This is going to be the nearest neighbor algorithm; one nearest neighbor algorithm partitions the space in this setting.

But do you see a problem here in this example? The yellow one is exactly in the middle of all of the greens. And this means that probably that's an outlier. That's probably noise.

This is the case for many, many problems that we have to solve. The reason there is this big yellow region in the middle is just this single point, and because we are only using one nearest neighbor, this happens.


<p align="center"><img src="./lecture_02_slides/slide_49864_00-27-43.795.jpg" width="75%" alt="Lecture Video at 00:27:43.795" /></p>

To make it a little bit more robust, we can increase the number of nearest neighbors that we take, which turns the nearest neighbor algorithm into a $k$-nearest neighbor. But the problem that you can see here is now we have some white regions. There is no way to identify what the label of that example in the white region is. It's a good way of finding regions that are important for data—more data collection.

We can go larger on the value of $k$, but one of the factors that plays an important role is the value of $k$.


<p align="center"><img src="./lecture_02_slides/slide_52826_00-29-22.627.jpg" width="75%" alt="Lecture Video at 00:29:22.627" /></p>

If you remember, we had another decision to make, which was the distance function. We talked about the $L_1$ distance, again the sum of all of the absolute values between pairwise differences of the pixels. If I visualize the $L_1$ distance, or sometimes in some context, we call it Manhattan distance, the distance function is kind of visualized in this way.

This is a good way of visualizing and seeing how this $L_1$ distance function works. Another popular distance function that we use is $L_2$, which instead of the absolute value, calculates the square of the differences, sums it up. But because of the square, we also do a square root. And visualizing that, we'll get the circle visualization, where each of the points on the circle are they

have the same distance from the center, from the origin. So this visualization actually helps us understand the differences between these distances too. And these are the most basic and easiest distance functions that we can use. So there are, again, a lot more.

If you have two pixel values, two features, then we have this 2D space. And this $x$ and $y$ are often those features. So if I rotate these features, meaning, if I use other types of features, this $L_1$ will have a different framework, different value, while it's not any different for $L_2$. So that's why this is a big difference between $L_1$ and $L_2$.

But if those features are more arbitrary, then $L_2$ distance makes more sense. If I want to calculate the distance—so the distance of all of the points on this shape from the origin are exactly the same, if I use the $L_1$ distance. But for $L_2$ distance, the points on this circle have the same distance from the center or the origin of this space.

So that's basically the main what these two images are showing. Any point on this shape, when using an $L_1$ distance, has the same distance from the origin. And for the circle, any point on the circle, if you're using $L_2$ distance, you'll have the same distance from the origin. Yeah, it's important to—it's better to use $L_1$ if we want to preserve the features.

So to answer that question, if I rotate the feature axis, the distances and this distance function changes completely. While if I do the same here, nothing changes. It's the exact same value of the features distance. Sorry.

In this case, $L_1$ is very sensitive on the feature values, while $L_2$ is not. If you select another feature in the same space that is having a different creates a different shape, then your $L$ function, the distance function changes as well. So if I draw the lines here, again, the question for online students is why it changes if you rotate. If I select another feature that goes from this side, then the lines will look different.

So if you rotate this thing, but for that shape it's not—it's agnostic.


<p align="center"><img src="./lecture_02_slides/slide_61912_00-34-25.797.jpg" width="75%" alt="Lecture Video at 00:34:25.797" /></p>

With these two distance functions that we talked about, if I visualize the space, you can see with $k=1$ with one nearest neighbor with $L_1$ and $L_2$, these are the space partitionings. While there, we have a little bit of more smooth boundary separation.


<p align="center"><img src="./lecture_02_slides/slide_63246_00-35-10.308.jpg" width="75%" alt="Lecture Video at 00:35:10.308" /></p>

So there is a tool online on the lab website that you can play around with this, with different distance functions and different number of $k$. You can see, you can create a different setup.


<p align="center"><img src="./lecture_02_slides/slide_63684_00-35-24.923.jpg" width="75%" alt="Lecture Video at 00:35:24.923" /></p>

So you can play around with it. But why did we talk about nearest neighbor to begin with? First, yes, it's the easiest problem to solve, easiest solution, easiest data-driven approach, and great to start with. But one of the main reasons that we want to iterate and discuss nearest neighbor is the fact that we can look into the topic of hyperparameters.

Hyperparameters are often some of the variables that you have to make a decision on to be able to run your algorithm. In this case, the value of $k$, the number of nearest neighbors, is defined as a hyperparameter. Depending on how many nearest neighbors you take, your outputs will be different.

Another choice that you have here is the distance function. The choices of hyperparameters are often very much dataset-dependent and sometimes problem-dependent. We have to have a way to identify those parameters to optimize for them for each single problem. This process is what is often referred to as hyperparameter tuning in machine learning algorithms, deep learning algorithms, and so on.


<p align="center"><img src="./lecture_02_slides/slide_66556_00-37-00.752.jpg" width="75%" alt="Lecture Video at 00:37:00.752" /></p>

How do we do that? There are different approaches. One approach is choosing the hyperparameters that work best for the training data. You have a set of images or data; you look for the best set of hyperparameters that generates the best training result or minimum training loss.


<p align="center"><img src="./lecture_02_slides/slide_67444_00-37-30.382.jpg" width="75%" alt="Lecture Video at 00:37:30.382" /></p>

While this works for the training data, it is not a good idea at all because, especially with nearest neighbor, $k=1$ is always the best value. Since you are memorizing the training data, $k=1$ will give you 100% accuracy.


<p align="center"><img src="./lecture_02_slides/slide_68060_00-37-50.936.jpg" width="75%" alt="Lecture Video at 00:37:50.936" /></p>

We know that this is not a great idea. The second approach is choosing hyperparameters that work best for held-out testing sets. While this is a little bit better than the first one, there is also a big problem here.


<p align="center"><img src="./lecture_02_slides/slide_69330_00-38-33.311.jpg" width="75%" alt="Lecture Video at 00:38:33.311" /></p>

This is not a good idea because we don't know how the model will generalize.


<p align="center"><img src="./lecture_02_slides/slide_69382_00-38-35.047.jpg" width="75%" alt="Lecture Video at 00:38:35.047" /></p>

Never do this; it is kind of cheating.


<p align="center"><img src="./lecture_02_slides/slide_69572_00-38-41.385.jpg" width="75%" alt="Lecture Video at 00:38:41.385" /></p>

A better idea is to always separate some part of the training data as a validation set. You train your model on the original training data, and then you try to find or optimize your hyperparameter on the validation set. After you've found the best set of hyperparameters, then use those parameters to replicate the results for the testing set and do the predictions for the testing set.


<p align="center"><img src="./lecture_02_slides/slide_71404_00-39-42.514.jpg" width="75%" alt="Lecture Video at 00:39:42.514" /></p>

For this reason, one better approach is to use cross-validation for setting hyperparameters. Basically, you split your training data into a number of folds (partitions), here five. Each fold plays as the validation set once. You iteratively run this process five times for five-fold cross-validation.

You do this five times and average the accuracies. For a given hyperparameter value, you calculate the accuracy on the validation set across all these five runs and then average it. You repeat this multiple times to find the best setting for the hyperparameter. After finding the optimal hyperparameter setting, you apply it to the testing set.

This is a little bit more reliable and generates much better results. However, in larger scale deep learning, it is less practiced because repeating this process multiple times across five folds with huge datasets is very hard. Therefore, we often use intuitions for setting hyperparameters, and using a single validation set is sometimes the approach we go with.

But this is pretty much advised. There are different approaches.


<p align="center"><img src="./lecture_02_slides/slide_79412_00-44-09.713.jpg" width="75%" alt="Lecture Video at 00:44:09.713" /></p>

Let's finalize the topic, wrap up the topic of nearest neighbor, and look at some examples and some results. So let me introduce you to the CIFAR10 data set. It's one of the data sets that you're going to be using in your assignments quite often. It has 10 classes with a number of training images and testing images.

The 10 classes, some of the examples are shown here with nearest neighbor for each of the testing images. If we run nearest neighbor and select the top 10 nearest neighbors, they are all visualized there. As you can imagine and guess, one of the first questions to answer is how many should be the value for $k$? How many nearest neighbors should we take?

We want to study one of the quick experiments with five-fold; each of those points is one of the folds in five-fold for each of the values of $k$, which shows different values here. And with a 10-class classification problem, often the random guess gets you a 10% accuracy. So this is much better than random guess. It's working.

It's doing something, but there is a lot of room to improve. If we go back and look at the examples, we can actually see there are so many mistakes, especially with the one that is closest. For example, the fourth row, if you look at that, it's a frog, but the first example seems to be a cat—sorry, a dog. You can guess why this is happening because the distance is being applied on pixels.

And pixel wise, they look like each other; they have the same type of colors in most pixels, so they are much closer. This example and many other examples show that distances that work on pixels and pixel values are not the best choices. There are much better approaches that we'll be discussing at the end of—more in the future lectures. And just to wrap up the topic, this is another example.

Although from a human eyes perspective, there is absolutely no difference. But the distance between that and the original image is the same as the other two examples that you see here.


<p align="center"><img src="./lecture_02_slides/slide_81330_00-45-13.711.jpg" width="75%" alt="Lecture Video at 00:45:13.711" /></p>

This is the summary of what we've discussed. So the question is, how do we make a decision in those cases? You often go with randomly selected one of the top ones.


<p align="center"><img src="./lecture_02_slides/slide_82676_00-45-58.623.jpg" width="75%" alt="Lecture Video at 00:45:58.623" /></p>

So summarizing what we've talked about with k-nearest neighbor, it was mostly about understanding the easiest


<p align="center"><img src="./lecture_02_slides/slide_83442_00-46-24.181.jpg" width="75%" alt="Lecture Video at 00:46:24.181" /></p>

Moving on to the next topic, which is linear classifiers. I want to spend the remaining time of this lecture to talk about this very important topic. This is one of the most important building blocks for deep learning. We need to see how this approach is different.


<p align="center"><img src="./lecture_02_slides/slide_84714_00-47-06.623.jpg" width="75%" alt="Lecture Video at 00:47:06.623" /></p>

First, we want to see how it's different from nearest neighbor.


<p align="center"><img src="./lecture_02_slides/slide_85918_00-47-46.797.jpg" width="75%" alt="Lecture Video at 00:47:46.797" /></p>

With this setup that we build, a linear classifier first uses $\mathbf{w}$, these parameters to map each of the inputs $\mathbf{x}$ into a value, which is the output $y$.


<p align="center"><img src="./lecture_02_slides/slide_86382_00-48-02.279.jpg" width="75%" alt="Lecture Video at 00:48:02.279" /></p>

And how this is done is very simple. This image is basically an area of say, $32$ by $32$ by $3$, so $3072$ numbers. This defines our $\mathbf{x}$, which is a $3072 \times 1$ vector. We know that we have $10$ output classes, so we need $10$ different scores.

The output will be kind of a vector of $10 \times 1$. This means that we have to find a weight matrix $\mathbf{W}$ that is $10$ by $3072$ that maps $\mathbf{x}$ into the output scores.


<p align="center"><img src="./lecture_02_slides/slide_87788_00-48-49.193.jpg" width="75%" alt="Lecture Video at 00:48:49.193" /></p>

To complete this linear function, we often use this bias term as well. It's an input-independent value which actually has a lot of different use cases. I can talk about it when I do some geometric visualizations, but it sometimes creates a shift for different class scores and helps with much better separation of each class.


<p align="center"><img src="./lecture_02_slides/slide_88782_00-49-22.359.jpg" width="75%" alt="Lecture Video at 00:49:22.359" /></p>

As I said, these linear functions are actually building blocks for building neural networks. Each of these linear classifiers, linear functions when put together one after the other, create large neural networks. Although there are a lot of other things that need to be added here, this is one of the most important components.


<p align="center"><img src="./lecture_02_slides/slide_89524_00-49-47.118.jpg" width="75%" alt="Lecture Video at 00:49:47.118" /></p>

If we look at some of the popular neural networks, we can see that linear functions are everywhere in the architectures.


<p align="center"><img src="./lecture_02_slides/slide_90040_00-50-04.334.jpg" width="75%" alt="Lecture Video at 00:50:04.334" /></p>

To better understand what this mapping and this function is doing, let's go back to our example of CIFAR10 and our training and testing samples, and make it a little bit simpler.


<p align="center"><img src="./lecture_02_slides/slide_90420_00-50-17.014.jpg" width="75%" alt="Lecture Video at 00:50:17.014" /></p>

Instead of looking at large images of $32$ by $32$, let's look at images of $2$ by $2$, an input image that has four pixels. This means that the input image is turned into a vector.


<p align="center"><img src="./lecture_02_slides/slide_91014_00-50-36.834.jpg" width="75%" alt="Lecture Video at 00:50:36.834" /></p>

As you can see here, we have to find a $\mathbf{w}$ and the values of $b$. So the input image is mapped into some scores as the output. This is how the linear function from an algebraic viewpoint looks like. The output scores here, we are considering three classes of cat, dog, and ship.

And as you can see, this function maps the image—the vector representing the image—into those scores. So, algebraic viewpoint of linear classification.


<p align="center"><img src="./lecture_02_slides/slide_92346_00-51-21.278.jpg" width="75%" alt="Lecture Video at 00:51:21.278" /></p>

Now let's look at some visual perspectives of this linear classifier. As you can see, we often create each of these images, as we talked about for this image.


<p align="center"><img src="./lecture_02_slides/slide_92774_00-51-35.559.jpg" width="75%" alt="Lecture Video at 00:51:35.559" /></p>

For each of the classes, we define some sort of—we have a row of this—row in the matrix $W$. So this row is kind of a template for that specific class.


<p align="center"><img src="./lecture_02_slides/slide_94988_00-52-49.433.jpg" width="75%" alt="Lecture Video at 00:52:49.433" /></p>

So the visual aspect viewpoint of the linear classifier, and there is another aspect of geometric viewpoint. What this linear classifier often does is finding those lines if it's in 2D space, finding those lines that separates each class from the others. And as you can see here, red, blue, and green are defining different classes. In higher dimensional space, instead of those lines, it's these hyperplanes, as you can see in this example on the left.

But with the bias, we can actually create more reliable functions and decision boundaries. A linear function is very useful. A linear classifier is very useful for many applications, as we talked about, and it's a building block of more complex neural networks.


<p align="center"><img src="./lecture_02_slides/slide_97316_00-54-07.111.jpg" width="75%" alt="Lecture Video at 00:54:07.111" /></p>

However, it does have its own challenges because it can't classify many instances of separate data. For example, in this case, if class 1 is the first and third quadrant and the second class is second and the fourth, there is no way to linearly separate these.


<p align="center"><img src="./lecture_02_slides/slide_98434_00-54-44.414.jpg" width="75%" alt="Lecture Video at 00:54:44.414" /></p>

Similarly, if there are three modes, three areas in the space that are one class, and then the second class is everything else.


<p align="center"><img src="./lecture_02_slides/slide_98784_00-54-56.093.jpg" width="75%" alt="Lecture Video at 00:54:56.093" /></p>

In all of these cases, it's actually very hard to do the separation.


<p align="center"><img src="./lecture_02_slides/slide_99028_00-55-04.234.jpg" width="75%" alt="Lecture Video at 00:55:04.234" /></p>

So what we should do—we talked about linear classifiers and how they can actually map the input images into any form of labels in the output.


<p align="center"><img src="./lecture_02_slides/slide_99320_00-55-13.977.jpg" width="75%" alt="Lecture Video at 00:55:13.977" /></p>

But now what remains is how to choose the value $W$, that for each of these images maps the image into a score for each single class as the output.


<p align="center"><img src="./lecture_02_slides/slide_99708_00-55-26.923.jpg" width="75%" alt="Lecture Video at 00:55:26.923" /></p>

In order to do that, we need to define a loss function, sometimes referred to as objective function, that quantifies how bad the classifier, how bad the model is working. This is the level of unhappiness with respect to the score on the training data.


<p align="center"><img src="./lecture_02_slides/slide_100398_00-55-49.946.jpg" width="75%" alt="Lecture Video at 00:55:49.946" /></p>

After defining those, we need to find a way to efficiently change the values of $W$ to be able to minimize that unhappiness—basically, minimize the loss function. And this is the optimization process. So, the topic of next class, next lecture.


<p align="center"><img src="./lecture_02_slides/slide_101242_00-56-18.108.jpg" width="75%" alt="Lecture Video at 00:56:18.108" /></p>

In order to do that, again for simplicity let's look at an easier and easier example, having These three classes—a linear function, as you can see here—and the three classes of cat, car, and frog.


<p align="center"><img src="./lecture_02_slides/slide_101876_00-56-39.262.jpg" width="75%" alt="Lecture Video at 00:56:39.262" /></p>

We need a loss function that tells how good our current classifier is. In order to do that, we need to parameterize the problem: $x_i$ and $y_i$, defining the input image, label images, and the corresponding labels.


<p align="center"><img src="./lecture_02_slides/slide_102324_00-56-54.210.jpg" width="75%" alt="Lecture Video at 00:56:54.210" /></p>

We often normalize them based on the number of samples as well, but it's not that important.


<p align="center"><img src="./lecture_02_slides/slide_105804_00-58-50.327.jpg" width="75%" alt="Lecture Video at 00:58:50.327" /></p>

This defines the loss function, the objective function for how we can do the optimization and how we can really find the $W$'s. There are different ways of defining this $\mathcal{L}$. I want to talk about softmax classifier right now. This defines the probability of the class being this class $k$ for each input image $x_i$.

To do that, we first use the softmax function. We exponentiate the values of the scores to create these numbers. When we use $\exp$ on these numbers, the outputs will always be positive, and we need to make sure that the probabilities are always positive.


<p align="center"><img src="./lecture_02_slides/slide_106692_00-59-19.957.jpg" width="75%" alt="Lecture Video at 00:59:19.957" /></p>

After creating these numbers, what we can do is just normalize them. So, exponentiate and then normalize based on the sum of all samples. This creates a very good set of values that define a probability function; this is a distribution function—they sum to 1. It's simple to say it’s this set of parameters thinks that this image is a cat with a probability of $0.13$.

Obviously, this is making a mistake in this example because the $W$ is not good.


<p align="center"><img src="./lecture_02_slides/slide_108390_01-00-16.614.jpg" width="75%" alt="Lecture Video at 01:00:16.614" /></p>

We should optimize it and change it. These probabilities are counterparts of unnormalized log probabilities, which are often referred to as logits. If you've taken other machine-learning courses, or if you've used logistic regression in other fields, this is a similar type of framework. This is the exact same framework as logistic regression.

And since we have multiple classes here, it's a multinomial logistic regression. How do we define the function $\mathcal{L}$? There are different ways of defining the function $\mathcal{L}$. We want to define a loss function.

What is the objective here? We want to maximize the probability of the sample belonging to the correct class. So, we want to maximize the value of $0.13$. Now we have other larger values in that set.

If you want to maximize this, this is a maximization problem. In order to turn it—because all of the objectives that we define, we try to build a minimization objective function, the first step is just to negate the values. We negate it so the maximization problem turns into a minimization problem.


<p align="center"><img src="./lecture_02_slides/slide_111320_01-01-54.377.jpg" width="75%" alt="Lecture Video at 01:01:54.377" /></p>

And then we also take the $\log$ of the value just to make the numbers a little bit more manageable. So negative log of that value will define the objective function, the loss function for solving this problem. That's the objective, or the loss function for softmax and for this logistic regression function.


<p align="center"><img src="./lecture_02_slides/slide_111986_01-02-16.599.jpg" width="75%" alt="Lecture Video at 01:02:16.599" /></p>

And if you've taken, as I said, other classes like CS-229, it's often referred to as maximum likelihood estimation as well. It's the same algorithm. And that's basically that simple. But there are other types of interpreting this framework as well.


<p align="center"><img src="./lecture_02_slides/slide_113308_01-03-00.710.jpg" width="75%" alt="Lecture Video at 01:03:00.710" /></p>

What we want to do is to match these two probability functions.


<p align="center"><img src="./lecture_02_slides/slide_114076_01-03-26.336.jpg" width="75%" alt="Lecture Video at 01:03:26.336" /></p>

And in order to do that, we want to minimize the $KL$ divergence, Kullback-Leibler divergence. This is an information theoretic perspective of looking at this loss function. And again, those are exactly the same.


<p align="center"><img src="./lecture_02_slides/slide_114908_01-03-54.097.jpg" width="75%" alt="Lecture Video at 01:03:54.097" /></p>

This $KL$ divergence in this setting simplifies into the same negative $\log$ function that we defined. And that's because when we use one-hot encoding setting for the classes, the entropy is 0. So that's one of the reasons that we call this function cross entropy or binary cross entropy function. So this is the same framework.

We start very simple, but we got to the similarities and differences between each of those.


<p align="center"><img src="./lecture_02_slides/slide_116962_01-05-02.632.jpg" width="75%" alt="Lecture Video at 01:05:02.632" /></p>

So the objective—sorry, the loss function was defined as negative $\log$ of this probability, and the probability was defined by the softmax, which we talked about. And then optimizing for this, which is the topic of next session, will give us the right $\mathbf{w}$'s.


<p align="center"><img src="./lecture_02_slides/slide_117474_01-05-19.716.jpg" width="75%" alt="Lecture Video at 01:05:19.716" /></p>

But before I end, I want to ask a couple of questions with this definition that you see here. What is the mean and maximum value that you can see for the loss function $l_i$? Yes, it's 0, which turns into minus minus infinity. But we have a negative negation there, so it would be infinity.

That is correct. But then we also have to yes, that's definitely that's right. And let me actually look at a second question.


<p align="center"><img src="./lecture_02_slides/slide_118586_01-05-56.820.jpg" width="75%" alt="Lecture Video at 01:05:56.820" /></p>

Yes, this one. So when we initialize all of the $\mathbf{w}$'s, so basically, the $w$'s, in the beginning, it's almost random. So the probabilities of each of the classes becomes mostly equal. What is the softmax $l_i$, assuming we have $c$ classes?


<p align="center"><img src="./lecture_02_slides/slide_119598_01-06-30.587.jpg" width="75%" alt="Lecture Video at 01:06:30.587" /></p>

And especially if it's $c_{10}$. So because the probabilities are equal, it means that all of the probabilities are around 1 And then that will be defined as $\log(C)$. If we have 10 classes, then the $\log$ or $\ln$ of 10 is 2.3, which is the exp.


<p align="center"><img src="./lecture_02_slides/slide_120384_01-06-56.813.jpg" width="75%" alt="Lecture Video at 01:06:56.813" /></p>

We know about it.
