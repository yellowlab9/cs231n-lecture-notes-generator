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

# Stanford CS231N | Spring 2025 | Lecture 9: Object Detection, Image Segmentation, Visualizing


<p align="center"><img src="./lecture_09_slides/slide_4_00-00-00.133.jpg" width="75%" alt="Lecture Video at 00:00:00.133" /></p>

Today, we'll be talking about different tasks of core Computer Vision: algorithms and tasks like detection and segmentation. We will also be covering topics around visualization and understanding. I will cover the most important ones.



<p align="center"><img src="./lecture_09_slides/slide_918_00-00-30.630.jpg" width="75%" alt="Lecture Video at 00:00:30.630" /></p>

Like the previous lecture, last time what we discussed was around the topic of transitioning from sequence-to-sequence models—RNNs to transformers. This was ultimately called something that we now refer to as encoder encoding the sequence.



<p align="center"><img src="./lecture_09_slides/slide_2146_00-01-11.604.jpg" width="75%" alt="Lecture Video at 00:01:11.604" /></p>

Then if we need to decode an image or a language—a sequence as the output—then similar type of architecture is used for decoder getting the encoder tokens as input. Taking those as input and then generating what is the desired output.



<p align="center"><img src="./lecture_09_slides/slide_3266_00-01-48.975.jpg" width="75%" alt="Lecture Video at 00:01:48.975" /></p>

But we talked about ultimately that self-attention is what we work with in many of the applications these days. They work much better than the other two. They do add computation and memory requirements, but that comes with much better modeling of the sequence and better results in terms of any of the tasks. So until here, it was mostly talking about self-attention.

We also talked a little bit about cross-attention and related topics.



<p align="center"><img src="./lecture_09_slides/slide_5362_00-02-58.912.jpg" width="75%" alt="Lecture Video at 00:02:58.912" /></p>

And then we got to the topic of Vision Transformers, which is one of the core models that is being used in modern applications, Computer Vision applications. We did go through this in the last minutes of the previous lecture, and I want to revisit the topic. After that, I'll stop and hear any questions or comments you may have regarding the assignments and everything that I talked about so far.



<p align="center"><img src="./lecture_09_slides/slide_6288_00-03-29.809.jpg" width="75%" alt="Lecture Video at 00:03:29.809" /></p>

We talked about the fact that what we do with transformers when we want to process images is that we split the image into patches, basically creating a sequence.



<p align="center"><img src="./lecture_09_slides/slide_6760_00-03-45.558.jpg" width="75%" alt="Lecture Video at 00:03:45.558" /></p>

The image was split to $S$ by $S$, or in this case, maybe $3 \times 3$ patches.



<p align="center"><img src="./lecture_09_slides/slide_7260_00-04-02.242.jpg" width="75%" alt="Lecture Video at 00:04:02.242" /></p>

Each of those patches are then represented by what we call tokens. Tokens are often a linear projection of the vector, the reshaped version of the image into a vector. It's basically a $D$-dimensional vector, as you can see in this slide. But because we have turned the image into patches, what becomes important—what are we losing here?

It's basically we're losing the position, the 2D position of the image.



<p align="center"><img src="./lecture_09_slides/slide_8392_00-04-40.013.jpg" width="75%" alt="Lecture Video at 00:04:40.013" /></p>

So that's why we often create or add something that we call positional embedding. There are many different ways of doing this. You can create a sequence and just put numbers of sequence as $1, 2, 3$, and so on. Or you can do a 2D version of $x$ and $y$-coordinates, and adding these two together creates the new token that goes to the transformer layer the same way all



<p align="center"><img src="./lecture_09_slides/slide_9254_00-05-08.775.jpg" width="75%" alt="Lecture Video at 00:05:08.775" /></p>

of the self-attention, layer norm, and everything we talked about—the MLP, everything we talked about last week. The output layer will generate the output vectors for us. It could be used for any application. One of the major applications in Computer Vision has been classification.



<p align="center"><img src="./lecture_09_slides/slide_10206_00-05-40.540.jpg" width="75%" alt="Lecture Video at 00:05:40.540" /></p>

With image classification, what becomes important is to somehow be able to encode or generate something as the output that is representative of the class. Often, we add one token, a special extra input to the transformer, which is of the same dimensionality but it's a learnable parameter.



<p align="center"><img src="./lecture_09_slides/slide_11232_00-06-14.774.jpg" width="75%" alt="Lecture Video at 00:06:14.774" /></p>

In the output space, whatever that represents is going to be turned into the class probability vector—a $C$-dimensional vector, which are the class probabilities. But transformers are not only used for classification; they could be used for many other tasks that we will be covering some of today as well.



<p align="center"><img src="./lecture_09_slides/slide_12336_00-06-51.611.jpg" width="75%" alt="Lecture Video at 00:06:51.611" /></p>

<p align="center"><img src="./lecture_09_slides/slide_12396_00-06-53.613.jpg" width="75%" alt="Lecture Video at 00:06:53.613" /></p>

Last week, we also talked about another variant of transformers, again, using tokens that go through transformer layers.



<p align="center"><img src="./lecture_09_slides/slide_12448_00-06-55.348.jpg" width="75%" alt="Lecture Video at 00:06:55.348" /></p>

<p align="center"><img src="./lecture_09_slides/slide_12834_00-07-08.227.jpg" width="75%" alt="Lecture Video at 00:07:08.227" /></p>

If you remember last time, we talked about multiple layers of transformers where positional embeddings are added.



<p align="center"><img src="./lecture_09_slides/slide_12978_00-07-13.032.jpg" width="75%" alt="Lecture Video at 00:07:13.032" /></p>

<p align="center"><img src="./lecture_09_slides/slide_13432_00-07-28.181.jpg" width="75%" alt="Lecture Video at 00:07:28.181" /></p>

Ultimately, transformers give an output of a vector patch for each of the inputs.



<p align="center"><img src="./lecture_09_slides/slide_13810_00-07-40.793.jpg" width="75%" alt="Lecture Video at 00:07:40.793" /></p>

This was ViTs; this is ViTs in a nutshell.



<p align="center"><img src="./lecture_09_slides/slide_16134_00-08-58.337.jpg" width="75%" alt="Lecture Video at 00:08:58.337" /></p>

There are some optimizations that were in the slides last week, but I will quickly spend a couple of minutes on them. I want you to understand that there are many different tweaks and optimizations for better performance and also making the transformers training a little bit more stable.



<p align="center"><img src="./lecture_09_slides/slide_16876_00-09-23.095.jpg" width="75%" alt="Lecture Video at 00:09:23.095" /></p>

One of them is actually residual connections. This layer norm is basically outside the residual. This means that whatever we get here, we normalize it. That doesn't really mean that we can't replicate any form of identity function anymore, which is what we wanted to do.

The solution for that is to bring in the layer normalization.



<p align="center"><img src="./lecture_09_slides/slide_17756_00-09-52.458.jpg" width="75%" alt="Lecture Video at 00:09:52.458" /></p>

We often put it before self-attention and the second one before the MLP layer.



<p align="center"><img src="./lecture_09_slides/slide_18098_00-10-03.869.jpg" width="75%" alt="Lecture Video at 00:10:03.869" /></p>

Normalization is there, but we also preserve our identity function. There are also other ways of normalizing, like RMSNorm (root mean square normalization), which is actually a very basic type of normalization. For each of the features, it doesn't use the mean value of the feature for normalization. But this makes the training more stable.

Again, these are all empirically shown to be better options, although there are some justifications why they work well. But mostly, the reason for adopting these is just the fact that they make the trainings more stable.



<p align="center"><img src="./lecture_09_slides/slide_19554_00-10-52.451.jpg" width="75%" alt="Lecture Video at 00:10:52.451" /></p>

The other option is to instead of using a simple MLP, we use a SwiGLU MLP, where we actually do what we call gated nonlinearity.



<p align="center"><img src="./lecture_09_slides/slide_19718_00-10-57.923.jpg" width="75%" alt="Lecture Video at 00:10:57.923" /></p>

Instead of having two vectors of weights, matrices of $W_1$ and $W_2$, we add a third one, $W_3$. Here, we create some gated nonlinearity. Basically, what it does is getting more trainable parameters, not just necessarily trainable parameters, but creating a better nonlinearity for a small architecture.



<p align="center"><img src="./lecture_09_slides/slide_21666_00-12-02.922.jpg" width="75%" alt="Lecture Video at 00:12:02.922" /></p>

The last piece is mixture of experts, which is often used in even the very modern architectures these days. Instead of having one set of MLP layers, we can have multiple sets of MLP layers; each of those will be an expert.



<p align="center"><img src="./lecture_09_slides/slide_22260_00-12-22.742.jpg" width="75%" alt="Lecture Video at 00:12:22.742" /></p>

What we do is through a router, the tokens will be routed to $A$ of those $E$ experts. In this way, we actually have $A$ active experts. But then again, what it does is it increases the number of parameters and it helps learning more robust models without increasing too much on the compute. These are all parallel MLPs, so we can have multiple experts in parallel.



<p align="center"><img src="./lecture_09_slides/slide_23566_00-13-06.318.jpg" width="75%" alt="Lecture Video at 00:13:06.318" /></p>

<p align="center"><img src="./lecture_09_slides/slide_23936_00-13-18.664.jpg" width="75%" alt="Lecture Video at 00:13:18.664" /></p>

And this is the summary of all of the tweaks that I just mentioned.



<p align="center"><img src="./lecture_09_slides/slide_24148_00-13-25.738.jpg" width="75%" alt="Lecture Video at 00:13:25.738" /></p>

This is similar to a bias, no. This is completely a trainable parameter by itself that you train—either a feedforward network or just a linear projection—to turn that into the probability vector. So it's not just bias. And then again, remember that you have so many self-attention networks and layers here.

Those self-attention layers are basically fusing the information, creating attention between all of the tokens and this class token. So when you supervise it from here, the loss function comes in. This will represent the class probabilities vector. So the question is, what different experts are doing?

That's a great question. Because they are trained in parallel and they are initialized differently, they often try to learn one aspect or a related—maybe also sometimes very much related aspect. It's just adding more compute and more parameters, giving the network to learn different things if it does have to learn multiple concepts. For example, if you have to cover multiple probability distributions, then with these MLPs, you often have the power to separate those modes of data.

So the question is, if the number of experts is a hyperparameter or not? Yes, definitely, it's a hyperparameter. From what I know, it's often predefined; don't necessarily like over-fine tune them. But yes, they are all hyperparameters; they are also learned.

Yes. And they are learned. So why moving the layer norm helps us learn identity transformation?



<p align="center"><img src="./lecture_09_slides/slide_27970_00-15-33.265.jpg" width="75%" alt="Lecture Video at 00:15:33.265" /></p>

So look at this architecture. Will you be able to create any form of identity? Because right after that residual connection, the feature values are changed because you have a normalization. You will never have the identity in the features.

Because right after that, you see the layer normalization, and that's why what we do is we bring it in.



<p align="center"><img src="./lecture_09_slides/slide_28632_00-15-55.354.jpg" width="75%" alt="Lecture Video at 00:15:55.354" /></p>

<p align="center"><img src="./lecture_09_slides/slide_28694_00-15-57.423.jpg" width="75%" alt="Lecture Video at 00:15:57.423" /></p>

We have quite a few different tasks in Computer Vision, and these were the core and the most important tasks over the years for Computer Vision applications. Although these days, we are solving much harder tasks, and nobody cares about object detection anymore because now, we can just do it with one line of code. But over the past 10, 15 years, there has been a lot of advances, and we want to cover—I want to really cover some of those today.

Just so if you have to design something new yourself, you know where to look and how to design your models. Ultimately, there is the topic of visualization and understanding, which is very important in many applications.



<p align="center"><img src="./lecture_09_slides/slide_30694_00-17-04.156.jpg" width="75%" alt="Lecture Video at 00:17:04.156" /></p>

The way we started the class, and this slide is probably very much familiar to everybody, we talked about different tasks.



<p align="center"><img src="./lecture_09_slides/slide_31154_00-17-19.505.jpg" width="75%" alt="Lecture Video at 00:17:19.505" /></p>

For object classification, for the task of classification, we talked about this. We spent quite a lot of time over the first few lectures seeing how we can create a classifier that classifies images from pixels into labels.



<p align="center"><img src="./lecture_09_slides/slide_31704_00-17-37.856.jpg" width="75%" alt="Lecture Video at 00:17:37.856" /></p>

But then one of the other tasks important similarly is semantic segmentation.



<p align="center"><img src="./lecture_09_slides/slide_32012_00-17-48.133.jpg" width="75%" alt="Lecture Video at 00:17:48.133" /></p>

<p align="center"><img src="./lecture_09_slides/slide_32732_00-18-12.157.jpg" width="75%" alt="Lecture Video at 00:18:12.157" /></p>

So basically, when we train a model that does this at test time, we want to take an image and generate the same map as the output.



<p align="center"><img src="./lecture_09_slides/slide_33114_00-18-24.903.jpg" width="75%" alt="Lecture Video at 00:18:24.903" /></p>

How do we do that? There are many different options.



<p align="center"><img src="./lecture_09_slides/slide_33262_00-18-29.842.jpg" width="75%" alt="Lecture Video at 00:18:29.842" /></p>

<p align="center"><img src="./lecture_09_slides/slide_33398_00-18-34.379.jpg" width="75%" alt="Lecture Video at 00:18:34.379" /></p>

Let's say what I can do is just look at each pixel, every single pixel, and say what the value or what the label for that pixel should be. In the very basic form, as you can see here, it's actually very much impossible. It's hard to say what pixel that represents—what object that specific pixel represents—because there is no context if you only look at the pixel itself.



<p align="center"><img src="./lecture_09_slides/slide_34164_00-18-59.938.jpg" width="75%" alt="Lecture Video at 00:18:59.938" /></p>

So that's why context is important. We look at the surrounding areas.



<p align="center"><img src="./lecture_09_slides/slide_34664_00-19-16.622.jpg" width="75%" alt="Lecture Video at 00:19:16.622" /></p>

Because now, you're classifying the entire image. It could be a CNN, it could be a ResNet, it could be a ViT, or whatever.



<p align="center"><img src="./lecture_09_slides/slide_35348_00-19-39.444.jpg" width="75%" alt="Lecture Video at 00:19:39.444" /></p>

This is really time-consuming because if you want to run one full network for every single pixel in an image, it will take forever to turn this into a segmentation map.



<p align="center"><img src="./lecture_09_slides/slide_35792_00-19-54.259.jpg" width="75%" alt="Lecture Video at 00:19:54.259" /></p>

And in that case, we will have our segmentation task solved.



<p align="center"><img src="./lecture_09_slides/slide_36748_00-20-26.158.jpg" width="75%" alt="Lecture Video at 00:20:26.158" /></p>

And in order to do that, we need to have a layer in the input that is the same size as the image. And also in the output, we also need some sort of an inflated layer. You can't go to fully connected layers, and so on, because now, we are generating an image.

And because of that, we need to keep the network inflated. That's what we call, often, fully convolutional neural networks or FCNs. With fully convolutional neural networks, this is definitely a great idea. But there is a caveat; there is a problem: these images are large.



<p align="center"><img src="./lecture_09_slides/slide_38214_00-21-15.073.jpg" width="75%" alt="Lecture Video at 00:21:15.073" /></p>

And these networks, these layers will become very large.



<p align="center"><img src="./lecture_09_slides/slide_38934_00-21-39.097.jpg" width="75%" alt="Lecture Video at 00:21:39.097" /></p>

Somewhere in the middle, we will have a low resolution, but somehow thick in terms of the number of channels. From there, what we do is we go back up to the same size of the image to create the output pixel.



<p align="center"><img src="./lecture_09_slides/slide_40108_00-22-18.270.jpg" width="75%" alt="Lecture Video at 00:22:18.270" /></p>

We know how to do the downsampling; it was easy. We've talked about pooling operation, strided convolution, and several other steps or operations that could be used here. But on the upsampling side, we don't really know how to do the upsampling because we don't have pooling or reverse of uppooling or reverse strided convolutions. Because of that, we had to invent some new operations that reverses downsampling by itself.

Before I go to upsampling, defining what upsampling is, I just briefly want to tell you that: How do you think this network is trained? We now have a network that starts from an image and ends with an image. The tools that we have for training this network was a loss function. How do you think it is best to train or define a loss function for this network?

We talked about softmax loss, and also a little bit about some regression losses and SVM loss. But assuming that we want to use the softmax loss function, how could we define or train this network? What would the objective be? You said mean classification loss for each of the pixels, and that's correct.

You can add the loss function for every single pixel because every single pixel is doing a classification. So you will have $\sum$ over all pixels of the image, and the loss function is just a simple softmax. Then you can backprop; that's the entire loss function that you need. The question is: Do we need what we call ground truths for training?

That's actually the ground truths of segmentation. Yes, for these types of algorithms, because they are fully supervised, we do need the ground truth label maps. In early years, there has been a lot of work doing and sitting down and manually labeling the pixels to be able to train these algorithms. These days, we don't need that because we have tools, but early on, in order to train these algorithms, we needed the ground truth.

Very briefly, let me tell you what we do with upsampling.



<p align="center"><img src="./lecture_09_slides/slide_45500_00-25-18.183.jpg" width="75%" alt="Lecture Video at 00:25:18.183" /></p>

Upsampling is actually not that hard. We can use an unpooling operation. There are different ways of doing it. One is nearest neighbor.

If I want to go from a $2 \times 2$ as an example here, a matrix $2 \times 2$, $4 \times 4$. I just need to copy the data for each of these, taking the nearest neighbor in the lower resolution one. Or bed of nails, you just select one of those in the upsampled version—you only select one of those, the one in the corner. To copy the data, replace everything else with zero.

Through multiple layers of convolution, these values will start appearing.



<p align="center"><img src="./lecture_09_slides/slide_46898_00-26-04.829.jpg" width="75%" alt="Lecture Video at 00:26:04.829" /></p>

If we use max pooling in our network on the encoding side, what we can do is we can save the locations of the $\text{max}$, the ones that were selected. We can then copy the data during the unpooling $\text{max}$ stage right over where the $\text{max}$ was defined. Basically, we save the locations in the encoding part and, during the decoding part's upsampling step, we reuse those saved coordinates.



<p align="center"><img src="./lecture_09_slides/slide_48160_00-26-46.938.jpg" width="75%" alt="Lecture Video at 00:26:46.938" /></p>

The other option is to do a learned upsampling.



<p align="center"><img src="./lecture_09_slides/slide_48318_00-26-52.210.jpg" width="75%" alt="Lecture Video at 00:26:52.210" /></p>

<p align="center"><img src="./lecture_09_slides/slide_48366_00-26-53.812.jpg" width="75%" alt="Lecture Video at 00:26:53.812" /></p>

All of these methods that I showed have no parameter to be learned; it's just an operation.



<p align="center"><img src="./lecture_09_slides/slide_48472_00-26-57.349.jpg" width="75%" alt="Lecture Video at 00:26:57.349" /></p>

<p align="center"><img src="./lecture_09_slides/slide_48484_00-26-57.749.jpg" width="75%" alt="Lecture Video at 00:26:57.749" /></p>

But learned upsamplings are also possible. Very simply, let's revisit convolution.



<p align="center"><img src="./lecture_09_slides/slide_48720_00-27-05.624.jpg" width="75%" alt="Lecture Video at 00:27:05.624" /></p>

<p align="center"><img src="./lecture_09_slides/slide_48934_00-27-12.764.jpg" width="75%" alt="Lecture Video at 00:27:12.764" /></p>

<p align="center"><img src="./lecture_09_slides/slide_49056_00-27-16.835.jpg" width="75%" alt="Lecture Video at 00:27:16.835" /></p>

<p align="center"><img src="./lecture_09_slides/slide_49284_00-27-24.442.jpg" width="75%" alt="Lecture Video at 00:27:24.442" /></p>

<p align="center"><img src="./lecture_09_slides/slide_49384_00-27-27.779.jpg" width="75%" alt="Lecture Video at 00:27:27.779" /></p>

When we wanted to downsample, what we did was strided convolution, where, instead of taking steps of $1$, we take steps of $2$ and generate the outputs step by step. If you don't remember this part, go back to the lecture; we talked about it in the third lecture, I think.



<p align="center"><img src="./lecture_09_slides/slide_49672_00-27-37.389.jpg" width="75%" alt="Lecture Video at 00:27:37.389" /></p>

<p align="center"><img src="./lecture_09_slides/slide_49772_00-27-40.725.jpg" width="75%" alt="Lecture Video at 00:27:40.725" /></p>

<p align="center"><img src="./lecture_09_slides/slide_49796_00-27-41.526.jpg" width="75%" alt="Lecture Video at 00:27:41.526" /></p>

We can then replicate the same process for upsampling.



<p align="center"><img src="./lecture_09_slides/slide_49896_00-27-44.863.jpg" width="75%" alt="Lecture Video at 00:27:44.863" /></p>

This one will represent this area in the upsampled image. Then we define some weights here to map that to the output map.



<p align="center"><img src="./lecture_09_slides/slide_50292_00-27-58.076.jpg" width="75%" alt="Lecture Video at 00:27:58.076" /></p>

For the next one, it's the same story, but there will be overlaps.



<p align="center"><img src="./lecture_09_slides/slide_50498_00-28-04.949.jpg" width="75%" alt="Lecture Video at 00:28:04.949" /></p>

For the overlaps, we often sum over the values. Let me give you an example: summing over the outputs.



<p align="center"><img src="./lecture_09_slides/slide_50636_00-28-09.554.jpg" width="75%" alt="Lecture Video at 00:28:09.554" /></p>

Using a simple $1\text{D}$ function, if the input is just two values of $A$ and $B$, we learn a filter that maps it to the higher resolution output. To do this, we just apply the filter to each of the values and write the outputs here. For the parts where there is an overlap, it's the summation—addition of what is coming from each of the two locations.



<p align="center"><img src="./lecture_09_slides/slide_51822_00-28-49.127.jpg" width="75%" alt="Lecture Video at 00:28:49.127" /></p>

We did talk about fully convolutional neural networks and how they are being used; these are actually some of the most basic and mostly widely used algorithms for segmentation. I want to also very quickly highlight one of the widely used network units.



<p align="center"><img src="./lecture_09_slides/slide_52798_00-29-21.693.jpg" width="75%" alt="Lecture Video at 00:29:21.693" /></p>

<p align="center"><img src="./lecture_09_slides/slide_52908_00-29-25.363.jpg" width="75%" alt="Lecture Video at 00:29:25.363" /></p>

<p align="center"><img src="./lecture_09_slides/slide_52952_00-29-26.831.jpg" width="75%" alt="Lecture Video at 00:29:26.831" /></p>

As you can see, the shape U—it's actually the same architecture as what I showed here. We can just draw it similar to a $\text{U}$ shape.



<p align="center"><img src="./lecture_09_slides/slide_53930_00-29-59.464.jpg" width="75%" alt="Lecture Video at 00:29:59.464" /></p>

<p align="center"><img src="./lecture_09_slides/slide_54172_00-30-07.539.jpg" width="75%" alt="Lecture Video at 00:30:07.539" /></p>

<p align="center"><img src="./lecture_09_slides/slide_54404_00-30-15.280.jpg" width="75%" alt="Lecture Video at 00:30:15.280" /></p>

And then during upsampling, if you don't have the information... It's going to be a little bit hard, and we often get into sometimes boundaries are faded. This was the idea behind U-Net, and as I said, it's actually being used quite often.



<p align="center"><img src="./lecture_09_slides/slide_56272_00-31-17.609.jpg" width="75%" alt="Lecture Video at 00:31:17.609" /></p>

Summary of semantic segmentation—what we talked about today, the fully convolutional neural networks.



<p align="center"><img src="./lecture_09_slides/slide_56538_00-31-26.484.jpg" width="75%" alt="Lecture Video at 00:31:26.484" /></p>

You have same filter as before that we had for downsampling here. To save time, I actually removed some of the slides from this part; I have it in the back of the slides. You should check it out. This is a reverse; this is transformed transposed convolution.

We do have a $3 \times 3$ matrix here. Instead of convolving the input image or data, we apply convolution on the transposed version of the input, and it actually generates a larger output. It's the transposed convolution; it's the reverse of the regular convolution. But why transposed?

I would refer you to take a look at the additional slides. The question is: Is the filter trained? Yes. It's very much similar to other convolution layers.

All of the filters are trained. This was the topic of semantic segmentation.



<p align="center"><img src="./lecture_09_slides/slide_58968_00-32-47.565.jpg" width="75%" alt="Lecture Video at 00:32:47.565" /></p>

As we talked about this, we only get labels for the pixels. But if there are two instances of the same object, we have no idea which one is which because this is just generating or outputting the pixel labels.



<p align="center"><img src="./lecture_09_slides/slide_59546_00-33-06.851.jpg" width="75%" alt="Lecture Video at 00:33:06.851" /></p>

<p align="center"><img src="./lecture_09_slides/slide_60338_00-33-33.277.jpg" width="75%" alt="Lecture Video at 00:33:33.277" /></p>

For doing that, what we need is understanding multiple objects in the image, which brings us to the topic of object detection. Object detection has been, besides image classification, one of the core Computer Vision problems and tasks. For many years, many different algorithms were proposed just for doing the task of object detection. We are going to fly over some of them and highlight a couple of important ones.



<p align="center"><img src="./lecture_09_slides/slide_62118_00-34-32.670.jpg" width="75%" alt="Lecture Video at 00:34:32.670" /></p>

If it's just a single object, it means that we need to do the classification, generate label class scores, as well as getting a bounding box or coordinates of a box. So we need the coordinates of the box $(x, y)$ and $(h, w)$ as the output, as well as what class it is. This is exactly the task of object detection.



<p align="center"><img src="./lecture_09_slides/slide_63034_00-35-03.234.jpg" width="75%" alt="Lecture Video at 00:35:03.234" /></p>

How can we solve this? It's very simple. We can define a softmax loss function for the class scores and we can define an $L_2$ loss function, which is a simple distance metric, a regression loss for the box coordinates.



<p align="center"><img src="./lecture_09_slides/slide_63696_00-35-25.323.jpg" width="75%" alt="Lecture Video at 00:35:25.323" /></p>

Having these two defined, we have a multi-task loss; we are solving two tasks at the same time. For doing that, we again add the loss values and generate a compound loss function, as you can see here.



<p align="center"><img src="./lecture_09_slides/slide_64348_00-35-47.078.jpg" width="75%" alt="Lecture Video at 00:35:47.078" /></p>

So this is simple. It's doable. If you have one single object, you can for sure solve this problem using this architecture that I talked about.



<p align="center"><img src="./lecture_09_slides/slide_64666_00-35-57.688.jpg" width="75%" alt="Lecture Video at 00:35:57.688" /></p>

But this is not that easy if you have multiple objects in the scene. For three objects, you have to generate 12 output numbers, and if there are more, then it's going to be too many numbers to generate. So this algorithm is not really scalable.



<p align="center"><img src="./lecture_09_slides/slide_65316_00-36-19.377.jpg" width="75%" alt="Lecture Video at 00:36:19.377" /></p>

<p align="center"><img src="./lecture_09_slides/slide_65700_00-36-32.190.jpg" width="75%" alt="Lecture Video at 00:36:32.190" /></p>

One solution is instead of going or getting the entire image as the input, why not to look at bounding boxes?



<p align="center"><img src="./lecture_09_slides/slide_65798_00-36-35.459.jpg" width="75%" alt="Lecture Video at 00:36:35.459" /></p>

For each bounding box, we can say we only have one label: whether it's a cat or a dog or the background.



<p align="center"><img src="./lecture_09_slides/slide_66244_00-36-50.341.jpg" width="75%" alt="Lecture Video at 00:36:50.341" /></p>

<p align="center"><img src="./lecture_09_slides/slide_66332_00-36-53.277.jpg" width="75%" alt="Lecture Video at 00:36:53.277" /></p>

<p align="center"><img src="./lecture_09_slides/slide_66352_00-36-53.945.jpg" width="75%" alt="Lecture Video at 00:36:53.945" /></p>

And if I have this way of classifying each of the bounding boxes, I can do a sliding window.



<p align="center"><img src="./lecture_09_slides/slide_66884_00-37-11.696.jpg" width="75%" alt="Lecture Video at 00:37:11.696" /></p>

<p align="center"><img src="./lecture_09_slides/slide_66930_00-37-13.231.jpg" width="75%" alt="Lecture Video at 00:37:13.231" /></p>

Step by step, I can create—I can find the bounding boxes that I have the maximum probability of each of the objects.



<p align="center"><img src="./lecture_09_slides/slide_67324_00-37-26.377.jpg" width="75%" alt="Lecture Video at 00:37:26.377" /></p>

But there is a huge problem here. Again, there are so many different combinations of bounding boxes that you can use.



<p align="center"><img src="./lecture_09_slides/slide_67596_00-37-35.453.jpg" width="75%" alt="Lecture Video at 00:37:35.453" /></p>

And again, this algorithm is not scalable. These are region proposals. If I have a way to find region proposals, that's actually going to be an easy-ish problem.



<p align="center"><img src="./lecture_09_slides/slide_68738_00-38-13.557.jpg" width="75%" alt="Lecture Video at 00:38:13.557" /></p>

<p align="center"><img src="./lecture_09_slides/slide_68852_00-38-17.361.jpg" width="75%" alt="Lecture Video at 00:38:17.361" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69022_00-38-23.034.jpg" width="75%" alt="Lecture Video at 00:38:23.034" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69108_00-38-25.903.jpg" width="75%" alt="Lecture Video at 00:38:25.903" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69248_00-38-30.574.jpg" width="75%" alt="Lecture Video at 00:38:30.574" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69348_00-38-33.911.jpg" width="75%" alt="Lecture Video at 00:38:33.911" /></p>

In order to even I can refine the bounding boxes.



<p align="center"><img src="./lecture_09_slides/slide_69496_00-38-38.849.jpg" width="75%" alt="Lecture Video at 00:38:38.849" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69518_00-38-39.583.jpg" width="75%" alt="Lecture Video at 00:38:39.583" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69532_00-38-40.051.jpg" width="75%" alt="Lecture Video at 00:38:40.051" /></p>

<p align="center"><img src="./lecture_09_slides/slide_69574_00-38-41.452.jpg" width="75%" alt="Lecture Video at 00:38:41.452" /></p>

So, classify and then refine the bounding boxes to have the objects detected. We can classify the boxes and also refine the bounding boxes if I have to change the coordinates a little bit.



<p align="center"><img src="./lecture_09_slides/slide_69652_00-38-44.055.jpg" width="75%" alt="Lecture Video at 00:38:44.055" /></p>

This is what is called R-CNN algorithm.



<p align="center"><img src="./lecture_09_slides/slide_71030_00-39-30.034.jpg" width="75%" alt="Lecture Video at 00:39:30.034" /></p>

We always have a way to track where in the pixel space they are. So, in that case, what we do is instead of running the convolutional neural network on the patches, let's say, we run one big convolution on the entire image.



<p align="center"><img src="./lecture_09_slides/slide_71990_00-40-02.066.jpg" width="75%" alt="Lecture Video at 00:40:02.066" /></p>

<p align="center"><img src="./lecture_09_slides/slide_72022_00-40-03.134.jpg" width="75%" alt="Lecture Video at 00:40:03.134" /></p>

<p align="center"><img src="./lecture_09_slides/slide_72162_00-40-07.805.jpg" width="75%" alt="Lecture Video at 00:40:07.805" /></p>

Now we have the regions in that feature map corresponding to the entire image.



<p align="center"><img src="./lecture_09_slides/slide_72354_00-40-14.211.jpg" width="75%" alt="Lecture Video at 00:40:14.211" /></p>

<p align="center"><img src="./lecture_09_slides/slide_72382_00-40-15.146.jpg" width="75%" alt="Lecture Video at 00:40:15.146" /></p>

<p align="center"><img src="./lecture_09_slides/slide_72510_00-40-19.417.jpg" width="75%" alt="Lecture Video at 00:40:19.417" /></p>

Or what's the object category is? This is the fast version of R-CNN. These are some basic algorithms that we can use convolutional neural networks for detecting objects, their bounding boxes, and so on.



<p align="center"><img src="./lecture_09_slides/slide_73312_00-40-46.177.jpg" width="75%" alt="Lecture Video at 00:40:46.177" /></p>

The question is, if the number of proposed regions are predefined? The short answer to that is yes. I will talk very briefly about what region proposal networks do.



<p align="center"><img src="./lecture_09_slides/slide_73742_00-41-00.524.jpg" width="75%" alt="Lecture Video at 00:41:00.524" /></p>

<p align="center"><img src="./lecture_09_slides/slide_73760_00-41-01.125.jpg" width="75%" alt="Lecture Video at 00:41:01.125" /></p>

<p align="center"><img src="./lecture_09_slides/slide_73778_00-41-01.725.jpg" width="75%" alt="Lecture Video at 00:41:01.725" /></p>

<p align="center"><img src="./lecture_09_slides/slide_73824_00-41-03.260.jpg" width="75%" alt="Lecture Video at 00:41:03.260" /></p>

Easy algorithms put the bounding boxes of the regions—the proposed regions—under an image.



<p align="center"><img src="./lecture_09_slides/slide_74116_00-41-13.003.jpg" width="75%" alt="Lecture Video at 00:41:13.003" /></p>

<p align="center"><img src="./lecture_09_slides/slide_74370_00-41-21.479.jpg" width="75%" alt="Lecture Video at 00:41:21.479" /></p>

<p align="center"><img src="./lecture_09_slides/slide_74426_00-41-23.347.jpg" width="75%" alt="Lecture Video at 00:41:23.347" /></p>

There has been research on building region proposal networks, RPNs.



<p align="center"><img src="./lecture_09_slides/slide_75174_00-41-48.305.jpg" width="75%" alt="Lecture Video at 00:41:48.305" /></p>

What we do is we just randomly start with a CNN; we try to randomly start in different locations in the image.



<p align="center"><img src="./lecture_09_slides/slide_75532_00-42-00.251.jpg" width="75%" alt="Lecture Video at 00:42:00.251" /></p>

<p align="center"><img src="./lecture_09_slides/slide_75640_00-42-03.854.jpg" width="75%" alt="Lecture Video at 00:42:03.854" /></p>

Through layers of convolution, we refine those regions where they have the higher probability of having an object in them because we have the object labels and locations. We can optimize and supervise this process, and then each of those also refines the box coordinates.



<p align="center"><img src="./lecture_09_slides/slide_76174_00-42-21.672.jpg" width="75%" alt="Lecture Video at 00:42:21.672" /></p>

<p align="center"><img src="./lecture_09_slides/slide_76222_00-42-23.274.jpg" width="75%" alt="Lecture Video at 00:42:23.274" /></p>

<p align="center"><img src="./lecture_09_slides/slide_77456_00-43-04.448.jpg" width="75%" alt="Lecture Video at 00:43:04.448" /></p>

<p align="center"><img src="./lecture_09_slides/slide_77474_00-43-05.049.jpg" width="75%" alt="Lecture Video at 00:43:05.049" /></p>

This image only has one object, so most regions are centered around that single object, but generally, that's not the case.



<p align="center"><img src="./lecture_09_slides/slide_78040_00-43-23.934.jpg" width="75%" alt="Lecture Video at 00:43:23.934" /></p>

<p align="center"><img src="./lecture_09_slides/slide_78278_00-43-31.875.jpg" width="75%" alt="Lecture Video at 00:43:31.875" /></p>

In many setups, we can use region proposals to get different objects with higher probabilities. However, those types of algorithms are not being used anymore these days because they are computationally very heavy. While understanding how we got to this point is important, there are several reasons. One reason is that we need two separate networks: a region proposal network and then a classification and box refinement network.



<p align="center"><img src="./lecture_09_slides/slide_80158_00-44-34.605.jpg" width="75%" alt="Lecture Video at 00:44:34.605" /></p>

If you work with any Computer Vision problem, you've probably heard about YOLO even to date.



<p align="center"><img src="./lecture_09_slides/slide_81340_00-45-14.044.jpg" width="75%" alt="Lecture Video at 00:45:14.044" /></p>

I want to briefly tell you what YOLO does: it's basically looking only once in a single pass on the image; it generates all of the bounding boxes.



<p align="center"><img src="./lecture_09_slides/slide_81992_00-45-35.799.jpg" width="75%" alt="Lecture Video at 00:45:35.799" /></p>

How it works is it divides the image into a grid of $S \times S$, and in this example, it's $7$ by $7$.



<p align="center"><img src="./lecture_09_slides/slide_82492_00-45-52.483.jpg" width="75%" alt="Lecture Video at 00:45:52.483" /></p>

So it generates $B$ bounding boxes, a new hyperparameters, bounding $B$ boxes, that is the refinement of the object that is present in that box. And also, it generates the class probability, object class probabilities.



<p align="center"><img src="./lecture_09_slides/slide_83554_00-46-27.918.jpg" width="75%" alt="Lecture Video at 00:46:27.918" /></p>

In this case, for example, if it's $B$ equal to 2, it generates just two bounding boxes with different probabilities.



<p align="center"><img src="./lecture_09_slides/slide_83824_00-46-36.927.jpg" width="75%" alt="Lecture Video at 00:46:36.927" /></p>

It does this for all of the boxes at the same time. Basically, it's the same network that is generating something as the output for each of these bounding boxes.



<p align="center"><img src="./lecture_09_slides/slide_84334_00-46-53.944.jpg" width="75%" alt="Lecture Video at 00:46:53.944" /></p>

It does generate a number of different options for the object.



<p align="center"><img src="./lecture_09_slides/slide_85034_00-47-17.301.jpg" width="75%" alt="Lecture Video at 00:47:17.301" /></p>

<p align="center"><img src="./lecture_09_slides/slide_86718_00-48-13.490.jpg" width="75%" alt="Lecture Video at 00:48:13.490" /></p>

<p align="center"><img src="./lecture_09_slides/slide_86740_00-48-14.224.jpg" width="75%" alt="Lecture Video at 00:48:14.224" /></p>

<p align="center"><img src="./lecture_09_slides/slide_86868_00-48-18.495.jpg" width="75%" alt="Lecture Video at 00:48:18.495" /></p>

<p align="center"><img src="./lecture_09_slides/slide_86890_00-48-19.229.jpg" width="75%" alt="Lecture Video at 00:48:19.229" /></p>

<p align="center"><img src="./lecture_09_slides/slide_86902_00-48-19.630.jpg" width="75%" alt="Lecture Video at 00:48:19.630" /></p>

<p align="center"><img src="./lecture_09_slides/slide_87004_00-48-23.033.jpg" width="75%" alt="Lecture Video at 00:48:23.033" /></p>

<p align="center"><img src="./lecture_09_slides/slide_87038_00-48-24.167.jpg" width="75%" alt="Lecture Video at 00:48:24.167" /></p>

<p align="center"><img src="./lecture_09_slides/slide_87058_00-48-24.835.jpg" width="75%" alt="Lecture Video at 00:48:24.835" /></p>

<p align="center"><img src="./lecture_09_slides/slide_87092_00-48-25.969.jpg" width="75%" alt="Lecture Video at 00:48:25.969" /></p>

<p align="center"><img src="./lecture_09_slides/slide_87484_00-48-39.049.jpg" width="75%" alt="Lecture Video at 00:48:39.049" /></p>

As I said, each of those boxes are associated with a probability. In this example, the probability is shown with the weights of the edges in each of those boxes. For these many different bounding boxes and object probabilities, now we can do thresholding. This is a simple implementation or use of object detection.

Again, this is something very useful if you have time to spend with the repositories of YOLO. There are so many different newer versions of YOLO that are being used for many applications in medicine, robotics, and also in many industrial applications. The question is, how do we get this second image? And what's the intuition behind it?

As I said, for each of the grids, we generate bounding boxes. For this one, we generated two; for all others, we also generate two. This $B$ is again a probability vector, and each of these boxes are associated with a probability of existing an object in them. If I put all of them together for all of the patches, I have so many boxes.

Now each of those are associated with a probability.



<p align="center"><img src="./lecture_09_slides/slide_87940_00-48-54.264.jpg" width="75%" alt="Lecture Video at 00:48:54.264" /></p>

<p align="center"><img src="./lecture_09_slides/slide_87966_00-48-55.132.jpg" width="75%" alt="Lecture Video at 00:48:55.132" /></p>

Perfect. Let's move on. One of the more recent approaches for object detection is DETR, a Detection Transformer. This is purely based on transformers and the topic that we discussed last week.

Also, I started today—same type of self-attention and cross-attention modules could also generate some object detections and bounding boxes for us. How this works? This is actually not a very old paper, 2020, almost five years ago. Although it's now deprecated, nobody uses this for real applications.

But it's a very good example of how to use transformers for object detection.



<p align="center"><img src="./lecture_09_slides/slide_89688_00-49-52.589.jpg" width="75%" alt="Lecture Video at 00:49:52.589" /></p>

What we do here is basically similar to what we've explained earlier: we may turn the image into patches.



<p align="center"><img src="./lecture_09_slides/slide_89976_00-50-02.199.jpg" width="75%" alt="Lecture Video at 00:50:02.199" /></p>

Then those patches are passed through CNNs creating a token.



<p align="center"><img src="./lecture_09_slides/slide_90280_00-50-12.342.jpg" width="75%" alt="Lecture Video at 00:50:12.342" /></p>

<p align="center"><img src="./lecture_09_slides/slide_90968_00-50-35.298.jpg" width="75%" alt="Lecture Video at 00:50:35.298" /></p>

They are trainable parameters themselves. Each of those, for example, if I add five queries as input, or 10, or 20 queries as input—if I'm seeking up to 20 objects to be detected in that image. Then again, through a combination of self-attention layers at the beginning of this transformer decoder, as well as cross-attention with the encoder output.



<p align="center"><img src="./lecture_09_slides/slide_93022_00-51-43.834.jpg" width="75%" alt="Lecture Video at 00:51:43.834" /></p>

At the end, we have the bounding boxes and the classes associated with those bounding boxes as the output. The question is, are we inputting every possible box to the transformer? No. The input here are some trainable parameters that are queries, asking the questions that I actually want an object to be outfitted in place of this input query.

So there is no box or anything as the input; it's part of the output that generates the class label and the box coordinates. The question is, if the queries are formed in a way that it actually represents what we want to look for and where in the image? In this case, what we are looking for is defined by class labels, which are predefined, and they are part of the output.

Our supervision is based on the class labels; we have a class probability vector, the same way that we defined it for the other algorithms. That's how the algorithm knows what type of classes to look for. In terms of the outputs, again, these outputs are supervised. If you remember, this is based on the $L_2$ norm, $L_2$ loss of the ground truth boxes.

We are not telling anything in the query part what or where to look for any of the objects. The training process itself is backpropagating; if there are any losses, any errors, it backpropagates the outputs. Basically, we are not determining anything in the beginning or in this part. The question was, the query is give me up to nine objects.

Yes, that's basically what this means. Your question is if the queries are image patches or not? No, they are not image patches. They are just queries for trainable parameters.

You put them in to generate the outputs; for each of the inputs, you get a value as the output, and that value is turned into class and box coordinates. Again, the question is, what are object queries? They are trainable learnable parameters. So you initialize them, and the network finds the best values for them, and that's what you get as the output.

The question is, if there is any intuition of which FNN gets which box? So it's not generating the exact same thing as the output. We also have control over supervising those FNNs as well. This algorithm does not require the pixel-level segmentations.

It is only supervised based on class labels and bounding boxes. The question is, if it's possible to generalize unseen objects. By unseen, you mean a new class label? Yes.

For these types of algorithms that are fully supervised, often there is no way because you are creating a class probability vector. There is no way of adding something at the end for a new class without previously knowing there is some other classes. Fully supervised networks often have issues with new objects. We can have a background object or "no object."

As you can see, we have the label of "no object." But there are many algorithms and mixed extensions of these types of algorithms that are used for zero-shot learning. Zero-shot means understanding finding something new without having an example of those in the training data. But it is beyond this topic.

What happens if you have more objects in the scene than what you put in as the query? That's a great question. It often generates the ones that has the highest confidence on the objects, so bounding boxes with the highest confidence. In those cases, you often want to add more queries just so you can get more objects.

I'll be here to answer questions if you have any after the class. But we have a bunch of other topics to cover, and I want to make sure we go over them.



<p align="center"><img src="./lecture_09_slides/slide_104146_00-57-55.004.jpg" width="75%" alt="Lecture Video at 00:57:55.004" /></p>

At least you get familiar with the topics.



<p align="center"><img src="./lecture_09_slides/slide_104648_00-58-11.754.jpg" width="75%" alt="Lecture Video at 00:58:11.754" /></p>

And that's actually not too hard. We talked about this when we were talking about R-CNN algorithms, where we run a CNN on the image. Then we have a region proposal network that gives us the bounding boxes. Those bounding boxes are turned into either class labels and bounding box refinements.

This is what we've talked so far with R-CNN and so on.



<p align="center"><img src="./lecture_09_slides/slide_105618_00-58-44.120.jpg" width="75%" alt="Lecture Video at 00:58:44.120" /></p>

Now, we can turn this into a Mask R-CNN that also generates the mask. So basically, same architecture that we talked about earlier. We can take one more output, make it more multitask, and generate the mask predictions.



<p align="center"><img src="./lecture_09_slides/slide_106230_00-59-04.541.jpg" width="75%" alt="Lecture Video at 00:59:04.541" /></p>

What we used to be doing before was: image $\rightarrow$ region proposals $\rightarrow$ CNN gives us class label and the box coordinates.



<p align="center"><img src="./lecture_09_slides/slide_106556_00-59-15.418.jpg" width="75%" alt="Lecture Video at 00:59:15.418" /></p>

Now, we add another layer of convolution that generates the masks for that object at the pixel level. And that mask could be the same size as the input image, and basically on the layer itself.



<p align="center"><img src="./lecture_09_slides/slide_107174_00-59-36.039.jpg" width="75%" alt="Lecture Video at 00:59:36.039" /></p>

If we use fully convolutional neural networks, that is what we often get as the output.



<p align="center"><img src="./lecture_09_slides/slide_107518_00-59-47.517.jpg" width="75%" alt="Lecture Video at 00:59:47.517" /></p>

<p align="center"><img src="./lecture_09_slides/slide_107680_00-59-52.922.jpg" width="75%" alt="Lecture Video at 00:59:52.922" /></p>

<p align="center"><img src="./lecture_09_slides/slide_107778_00-59-56.192.jpg" width="75%" alt="Lecture Video at 00:59:56.192" /></p>

And this is an extension of the R-CNN algorithm, which we call Mask R-CNN.



<p align="center"><img src="./lecture_09_slides/slide_108074_01-00-06.069.jpg" width="75%" alt="Lecture Video at 01:00:06.069" /></p>

<p align="center"><img src="./lecture_09_slides/slide_108348_01-00-15.211.jpg" width="75%" alt="Lecture Video at 01:00:15.211" /></p>

With Mask R-CNN, the results have been very good in detecting different objects, different known objects that we could train the algorithms for.



<p align="center"><img src="./lecture_09_slides/slide_108568_01-00-22.552.jpg" width="75%" alt="Lecture Video at 01:00:22.552" /></p>

<p align="center"><img src="./lecture_09_slides/slide_108584_01-00-23.086.jpg" width="75%" alt="Lecture Video at 01:00:23.086" /></p>

<p align="center"><img src="./lecture_09_slides/slide_108618_01-00-24.220.jpg" width="75%" alt="Lecture Video at 01:00:24.220" /></p>

There are so many APIs and open-source versions of object detectors that you can explore.



<p align="center"><img src="./lecture_09_slides/slide_109030_01-00-37.967.jpg" width="75%" alt="Lecture Video at 01:00:37.967" /></p>

There are some links and resources here, but this all basically rounds up and summarizes some of the tasks that we wanted to cover. They are actually very important for you to understand these tasks; they have been core Computer Vision tasks. Although these days, Computer Vision is much more advanced, it is not bound to these tasks. But if you have industrial applications, for example, quality control or separating rotten tomatoes...

...and good tomatoes in an industrial pipeline. Then, with Computer Vision, you need to be able to detect objects and then classify them into good or bad. That's why it's important to still understand and know these steps and pipelines and how to do them in real time. But now, there are larger scale models that you are all familiar with.



<p align="center"><img src="./lecture_09_slides/slide_110648_01-01-31.954.jpg" width="75%" alt="Lecture Video at 01:01:31.954" /></p>

This summarizes the first part, the Computer Vision tasks that I wanted to talk about. And the last piece that I want to spend 10 minutes on is around visualization and understanding.



<p align="center"><img src="./lecture_09_slides/slide_111044_01-01-45.168.jpg" width="75%" alt="Lecture Video at 01:01:45.168" /></p>

Again, this has been a big lecture by itself. I'm going to summarize some of those most important ones here that you may need to use in your applications. But before that, let me go back to the linear classifier that we talked about. We spent quite a lot of time on linear classifiers.

With the linear classifiers, what we did was at the end, we said, if I look at the linear function, what the network is learning, I can have a template for each of those classes. Like, for example, for this car, you can always see a front-facing car as a template.



<p align="center"><img src="./lecture_09_slides/slide_113100_01-02-53.770.jpg" width="75%" alt="Lecture Video at 01:02:53.770" /></p>

We can do the same with neural networks.



<p align="center"><img src="./lecture_09_slides/slide_113252_01-02-58.841.jpg" width="75%" alt="Lecture Video at 01:02:58.841" /></p>

<p align="center"><img src="./lecture_09_slides/slide_113280_01-02-59.776.jpg" width="75%" alt="Lecture Video at 01:02:59.776" /></p>

If I visualize one of the filters—so here, we visualize the weights of the linear function.



<p align="center"><img src="./lecture_09_slides/slide_113360_01-03-02.445.jpg" width="75%" alt="Lecture Video at 01:03:02.445" /></p>

It was the visual viewpoint. I can do the same with visualizing the filters in the neural networks. For each of the filters, the network is learning something that is basically some basic shapes, orientations, or simple shapes, as you can see here. Although this visualization, we can only do it for the layers that have few channels.

For example, if we have three channels, I can put them in an RGB image and just visualize it. But as you remember, in CNNs, that was not the case. In CNNs, we had different sometimes quite a few channels in the middle layer, so it's not easy to visualize those in something that we can see.



<p align="center"><img src="./lecture_09_slides/slide_115118_01-04-01.103.jpg" width="75%" alt="Lecture Video at 01:04:01.103" /></p>

In earlier layers that we have fewer channels, we can visualize them and see the network is actually learning some patterns; so it starts learning patterns.



<p align="center"><img src="./lecture_09_slides/slide_116030_01-04-31.534.jpg" width="75%" alt="Lecture Video at 01:04:31.534" /></p>

I want to highlight a couple of ways of understanding and visualizing neural networks which are actually important. One is the concept of saliency. In many applications, it's very important for you to know which pixel matters. Because if you want to automate this, nobody cares about knowing if there is tumor or not; everybody cares about where in the image the tumor is.

In order to do that, simplest application is we train a network, a feedforward neural network that generates the value or the class label $\text{doc}$.



<p align="center"><img src="./lecture_09_slides/slide_117658_01-05-25.855.jpg" width="75%" alt="Lecture Video at 01:05:25.855" /></p>

<p align="center"><img src="./lecture_09_slides/slide_117850_01-05-32.261.jpg" width="75%" alt="Lecture Video at 01:05:32.261" /></p>

<p align="center"><img src="./lecture_09_slides/slide_118500_01-05-53.950.jpg" width="75%" alt="Lecture Video at 01:05:53.950" /></p>

Now, what I need is for each pixel, I want to see how much changing the pixel value would affect the $\text{doc}$ score. What does this mean? What I explained is the meaning of the variation. So this means that the meaning of basically gradient.

If I take the gradient of the score with respect to now the pixel values—not the network weights anymore, with pixel values—I can visualize those gradients.



<p align="center"><img src="./lecture_09_slides/slide_120546_01-07-02.218.jpg" width="75%" alt="Lecture Video at 01:07:02.218" /></p>

<p align="center"><img src="./lecture_09_slides/slide_120924_01-07-14.830.jpg" width="75%" alt="Lecture Video at 01:07:14.830" /></p>

<p align="center"><img src="./lecture_09_slides/slide_122386_01-08-03.612.jpg" width="75%" alt="Lecture Video at 01:08:03.612" /></p>

<p align="center"><img src="./lecture_09_slides/slide_122520_01-08-08.084.jpg" width="75%" alt="Lecture Video at 01:08:08.084" /></p>

<p align="center"><img src="./lecture_09_slides/slide_122698_01-08-14.023.jpg" width="75%" alt="Lecture Video at 01:08:14.023" /></p>

<p align="center"><img src="./lecture_09_slides/slide_122898_01-08-20.696.jpg" width="75%" alt="Lecture Video at 01:08:20.696" /></p>

<p align="center"><img src="./lecture_09_slides/slide_124244_01-09-05.608.jpg" width="75%" alt="Lecture Video at 01:09:05.608" /></p>

<p align="center"><img src="./lecture_09_slides/slide_124362_01-09-09.545.jpg" width="75%" alt="Lecture Video at 01:09:09.545" /></p>

<p align="center"><img src="./lecture_09_slides/slide_124854_01-09-25.961.jpg" width="75%" alt="Lecture Video at 01:09:25.961" /></p>

And visualizing those means that these are the pixels that matter; in order to classify a dog on this image, those are the pixels that matter. If I change the values of those pixels, the dog score will be changed. Again, this is the basic meaning and definition of gradients that we've talked about. This is one way.

If you run this on different objects that you've trained in the network, then this is what you get. That’s one way of understanding saliency, and it's very effective in many cases. However, sometimes it's not just about the pixel values all the way to the back. You want to see for each of the classes how the activations work.

This brings us to Class Activation Maps or CAM algorithm (Class Activation Mapping), or Grad-CAM, that I'll talk about in two minutes. These are one of the most and widely used algorithms for understanding CNNs, and also could be used for other architectures too. For transformers, we have a much better way of making sense of those, which actually we talked about in the last lecture.

What happens is that for each of the convolution layers, we often do pooling. The pooling generates feature maps, and the feature maps are then turned into scores. Those scores involve values of the weights. If we extend the math, basically, we simply can highlight the class scores in a weighted sum form.

We do convolution as the spatial consistency across all of the operations can help us trace back all the way to the image space. With that, if I do this multiplication of weights versus the weights that we've learned on top of the feature values, we create some class activations. We can have different class activation maps. These are the weights; these are the pixels or areas of the convolution layer that have been driving the scores for these specific classes.

It's the same for others, like class activation maps for one single object in different images.



<p align="center"><img src="./lecture_09_slides/slide_126314_01-10-14.677.jpg" width="75%" alt="Lecture Video at 01:10:14.677" /></p>

But there is a problem with this.



<p align="center"><img src="./lecture_09_slides/slide_126506_01-10-21.083.jpg" width="75%" alt="Lecture Video at 01:10:21.083" /></p>

<p align="center"><img src="./lecture_09_slides/slide_126700_01-10-27.556.jpg" width="75%" alt="Lecture Video at 01:10:27.556" /></p>

<p align="center"><img src="./lecture_09_slides/slide_126718_01-10-28.157.jpg" width="75%" alt="Lecture Video at 01:10:28.157" /></p>

<p align="center"><img src="./lecture_09_slides/slide_126790_01-10-30.559.jpg" width="75%" alt="Lecture Video at 01:10:30.559" /></p>

<p align="center"><img src="./lecture_09_slides/slide_126814_01-10-31.360.jpg" width="75%" alt="Lecture Video at 01:10:31.360" /></p>

<p align="center"><img src="./lecture_09_slides/slide_126836_01-10-32.094.jpg" width="75%" alt="Lecture Video at 01:10:32.094" /></p>

In order to solve that problem, there is one variant of the algorithm called Grad-CAM (Gradient-Weighted Class Activation Maps).



<p align="center"><img src="./lecture_09_slides/slide_127190_01-10-43.906.jpg" width="75%" alt="Lecture Video at 01:10:43.906" /></p>

It's basically the same algorithm. We need to calculate the weights with respect to—we basically take one of the layers that created some activation at the class level.



<p align="center"><img src="./lecture_09_slides/slide_127858_01-11-06.195.jpg" width="75%" alt="Lecture Video at 01:11:06.195" /></p>

<p align="center"><img src="./lecture_09_slides/slide_128030_01-11-11.934.jpg" width="75%" alt="Lecture Video at 01:11:11.934" /></p>

This is then used instead of the weights; it's an aggregate of all of the weights and gradients up to that specific layer. And then we weight that with that. We also use ReLU to pass the positive ones, and then that could also be all the way shown in image space.



<p align="center"><img src="./lecture_09_slides/slide_128654_01-11-32.755.jpg" width="75%" alt="Lecture Video at 01:11:32.755" /></p>

<p align="center"><img src="./lecture_09_slides/slide_128890_01-11-40.629.jpg" width="75%" alt="Lecture Video at 01:11:40.629" /></p>

<p align="center"><img src="./lecture_09_slides/slide_128906_01-11-41.163.jpg" width="75%" alt="Lecture Video at 01:11:41.163" /></p>

<p align="center"><img src="./lecture_09_slides/slide_128916_01-11-41.497.jpg" width="75%" alt="Lecture Video at 01:11:41.497" /></p>

<p align="center"><img src="./lecture_09_slides/slide_128946_01-11-42.498.jpg" width="75%" alt="Lecture Video at 01:11:42.498" /></p>

I talked about CAM, which was only applied to the last convolution layer.



<p align="center"><img src="./lecture_09_slides/slide_128974_01-11-43.432.jpg" width="75%" alt="Lecture Video at 01:11:43.432" /></p>

This is not possible because in most of the CNN algorithms, we don't have just one convolution layer at the end; we always have some operations fully connected and so on.



<p align="center"><img src="./lecture_09_slides/slide_129742_01-12-09.058.jpg" width="75%" alt="Lecture Video at 01:12:09.058" /></p>

<p align="center"><img src="./lecture_09_slides/slide_129752_01-12-09.391.jpg" width="75%" alt="Lecture Video at 01:12:09.391" /></p>

<p align="center"><img src="./lecture_09_slides/slide_129764_01-12-09.792.jpg" width="75%" alt="Lecture Video at 01:12:09.792" /></p>

<p align="center"><img src="./lecture_09_slides/slide_129778_01-12-10.259.jpg" width="75%" alt="Lecture Video at 01:12:10.259" /></p>

<p align="center"><img src="./lecture_09_slides/slide_129792_01-12-10.726.jpg" width="75%" alt="Lecture Video at 01:12:10.726" /></p>

<p align="center"><img src="./lecture_09_slides/slide_129812_01-12-11.393.jpg" width="75%" alt="Lecture Video at 01:12:11.393" /></p>

<p align="center"><img src="./lecture_09_slides/slide_130002_01-12-17.733.jpg" width="75%" alt="Lecture Video at 01:12:17.733" /></p>

<p align="center"><img src="./lecture_09_slides/slide_130154_01-12-22.805.jpg" width="75%" alt="Lecture Video at 01:12:22.805" /></p>

<p align="center"><img src="./lecture_09_slides/slide_130198_01-12-24.273.jpg" width="75%" alt="Lecture Video at 01:12:24.273" /></p>

Then, we can actually do the visualization; they create these heat maps for each of the objects.



<p align="center"><img src="./lecture_09_slides/slide_130260_01-12-26.342.jpg" width="75%" alt="Lecture Video at 01:12:26.342" /></p>

This was about CNNs, but we talked about transformers last week in the last lecture that they inherently come with activation maps. Do you remember that language matrix that Justin showed, where for each of the output words, there is an attention weight for the input? We can do that for pixels—the same thing for the pixels. For each of the outputs, we can create these maps in the pixel space and visualize the features of the ViTs in the pixel space.

But with CNNs, we often use Grad-CAM or these types of algorithms.



<p align="center"><img src="./lecture_09_slides/slide_131924_01-13-21.864.jpg" width="75%" alt="Lecture Video at 01:13:21.864" /></p>

I did this task that I thought I wouldn't be able to, completing the topics I wanted to talk about today.



<p align="center"><img src="./lecture_09_slides/slide_132198_01-13-31.006.jpg" width="75%" alt="Lecture Video at 01:13:31.006" /></p>

Next session, we will have the lecture around video understanding.



<p align="center"><img src="./lecture_09_slides/slide_132418_01-13-38.347.jpg" width="75%" alt="Lecture Video at 01:13:38.347" /></p>

Thank you.



