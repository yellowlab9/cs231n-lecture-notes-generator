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

# Stanford CS231N | Spring 2025 | Lecture 7: Recurrent Neural Networks


<p align="center"><img src="./lecture_07_slides/slide_4_00-00-00.133.jpg" width="75%" alt="Lecture Video at 00:00:00.133" /></p>

<p align="center"><img src="./lecture_07_slides/slide_272_00-00-09.075.jpg" width="75%" alt="Lecture Video at 00:00:09.075" /></p>

Hello everyone. Welcome to Lecture 7. I also wanted to go over some clarifications from last time. When I gave a lecture last time, there were two ed posts that I think were good that you all might want to check out.

In case you haven't seen it, I'll just go through it really quickly.



<p align="center"><img src="./lecture_07_slides/slide_790_00-00-26.359.jpg" width="75%" alt="Lecture Video at 00:00:26.359" /></p>

I think when describing dropout and how to scale probabilities at test time during the lecture, there was a bit of confusion. Basically, what I said in the slide had a mismatch.



<p align="center"><img src="./lecture_07_slides/slide_1510_00-00-50.383.jpg" width="75%" alt="Lecture Video at 00:00:50.383" /></p>

Generally, they do it's the number of the ones you drop out, so in most libraries, that's what $p$ means. But the basic idea is that at test time, you want the expected output to be the same as at training time. This means that if you dropped $25\%$ of your activations during training time, at test time, you would scale by $0.75$, so that the expected output is the same.



<p align="center"><img src="./lecture_07_slides/slide_2092_00-01-09.803.jpg" width="75%" alt="Lecture Video at 00:01:09.803" /></p>

<p align="center"><img src="./lecture_07_slides/slide_2444_00-01-21.548.jpg" width="75%" alt="Lecture Video at 00:01:21.548" /></p>

There was also a question in class from last time about how normalization can be useful and maybe resolve the issues that arise when you have weights that are initialized incorrectly. We have this choice setting where we have 2D inputs to our model and a two-layer neural network with ReLU. It's outputting basically this quadrant function. If the point lies in the top right, it'll output 1, or 2, or 3, or 4, depending on which quadrant the point lies in.

The blue plot here represents bad, and the green represents bad initialization with LayerNorm. You can see it actually does resolve a lot of the issues, but to get the best performance, you still need good weight initialization, which are the two lines afterwards. So you can go dive in. Also, whether or not LayerNorm helps depends on the problem.

In this quadrant I, you can imagine that you don't need to know the exact 2D position of each point, so LayerNorm was actually helping.



<p align="center"><img src="./lecture_07_slides/slide_4930_00-02-44.497.jpg" width="75%" alt="Lecture Video at 00:02:44.497" /></p>

<p align="center"><img src="./lecture_07_slides/slide_5032_00-02-47.901.jpg" width="75%" alt="Lecture Video at 00:02:47.901" /></p>

Just some notes here: Basically, at a high level, it does help with the issue, but a gap remains. You can't get by this weight initialization issue with just normalization.



<p align="center"><img src="./lecture_07_slides/slide_5198_00-02-53.439.jpg" width="75%" alt="Lecture Video at 00:02:53.439" /></p>

<p align="center"><img src="./lecture_07_slides/slide_5396_00-03-00.046.jpg" width="75%" alt="Lecture Video at 00:03:00.046" /></p>

As I mentioned, it may not always make sense depending on what you're trying to model. To recap also from last time, we've been mainly talking about these sorts of vanilla, standard, non-recurrent neural networks so far. This is a fixed size input and a fixed size output. You have your weight initialization and normalization functions that you use, as well as transfer learning.

If you pre-train on one data set, like ImageNet or some other large-scale internet data set, you can get better results if you initialize your weights to those values.



<p align="center"><img src="./lecture_07_slides/slide_7148_00-03-58.504.jpg" width="75%" alt="Lecture Video at 00:03:58.504" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7180_00-03-59.572.jpg" width="75%" alt="Lecture Video at 00:03:59.572" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7192_00-03-59.973.jpg" width="75%" alt="Lecture Video at 00:03:59.973" /></p>

A really good tool for points 2 and 3 here is something I use in basically all my projects called Weights & Biases.



<p align="center"><img src="./lecture_07_slides/slide_7224_00-04-01.040.jpg" width="75%" alt="Lecture Video at 00:04:01.040" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7246_00-04-01.774.jpg" width="75%" alt="Lecture Video at 00:04:01.774" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7272_00-04-02.642.jpg" width="75%" alt="Lecture Video at 00:04:02.642" /></p>

You might find this useful.



<p align="center"><img src="./lecture_07_slides/slide_7334_00-04-04.711.jpg" width="75%" alt="Lecture Video at 00:04:04.711" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7360_00-04-05.578.jpg" width="75%" alt="Lecture Video at 00:04:05.578" /></p>

It's a really neat way that you can essentially look at different runs; you set different runs with different hyperparameters.



<p align="center"><img src="./lecture_07_slides/slide_7374_00-04-06.045.jpg" width="75%" alt="Lecture Video at 00:04:06.045" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7394_00-04-06.713.jpg" width="75%" alt="Lecture Video at 00:04:06.713" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7420_00-04-07.580.jpg" width="75%" alt="Lecture Video at 00:04:07.580" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7444_00-04-08.381.jpg" width="75%" alt="Lecture Video at 00:04:08.381" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7456_00-04-08.781.jpg" width="75%" alt="Lecture Video at 00:04:08.781" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7486_00-04-09.782.jpg" width="75%" alt="Lecture Video at 00:04:09.782" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7558_00-04-12.185.jpg" width="75%" alt="Lecture Video at 00:04:12.185" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7568_00-04-12.518.jpg" width="75%" alt="Lecture Video at 00:04:12.518" /></p>

In this case, they show a dropout column here.



<p align="center"><img src="./lecture_07_slides/slide_7614_00-04-14.053.jpg" width="75%" alt="Lecture Video at 00:04:14.053" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7646_00-04-15.121.jpg" width="75%" alt="Lecture Video at 00:04:15.121" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7674_00-04-16.055.jpg" width="75%" alt="Lecture Video at 00:04:16.055" /></p>

These are all the different values of dropout.



<p align="center"><img src="./lecture_07_slides/slide_7734_00-04-18.057.jpg" width="75%" alt="Lecture Video at 00:04:18.057" /></p>

The color coding is really nice.



<p align="center"><img src="./lecture_07_slides/slide_7758_00-04-18.858.jpg" width="75%" alt="Lecture Video at 00:04:18.858" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7774_00-04-19.392.jpg" width="75%" alt="Lecture Video at 00:04:19.392" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7794_00-04-20.059.jpg" width="75%" alt="Lecture Video at 00:04:20.059" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7820_00-04-20.927.jpg" width="75%" alt="Lecture Video at 00:04:20.927" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7856_00-04-22.128.jpg" width="75%" alt="Lecture Video at 00:04:22.128" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7884_00-04-23.062.jpg" width="75%" alt="Lecture Video at 00:04:23.062" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7956_00-04-25.465.jpg" width="75%" alt="Lecture Video at 00:04:25.465" /></p>

<p align="center"><img src="./lecture_07_slides/slide_7968_00-04-25.865.jpg" width="75%" alt="Lecture Video at 00:04:25.865" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8014_00-04-27.400.jpg" width="75%" alt="Lecture Video at 00:04:27.400" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8044_00-04-28.401.jpg" width="75%" alt="Lecture Video at 00:04:28.401" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8072_00-04-29.335.jpg" width="75%" alt="Lecture Video at 00:04:29.335" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8134_00-04-31.404.jpg" width="75%" alt="Lecture Video at 00:04:31.404" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8158_00-04-32.205.jpg" width="75%" alt="Lecture Video at 00:04:32.205" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8174_00-04-32.739.jpg" width="75%" alt="Lecture Video at 00:04:32.739" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8194_00-04-33.406.jpg" width="75%" alt="Lecture Video at 00:04:33.406" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8220_00-04-34.274.jpg" width="75%" alt="Lecture Video at 00:04:34.274" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8244_00-04-35.074.jpg" width="75%" alt="Lecture Video at 00:04:35.074" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8256_00-04-35.475.jpg" width="75%" alt="Lecture Video at 00:04:35.475" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8286_00-04-36.476.jpg" width="75%" alt="Lecture Video at 00:04:36.476" /></p>

You can see based on many runs get an idea of which hyperparameters work best.



<p align="center"><img src="./lecture_07_slides/slide_8358_00-04-38.878.jpg" width="75%" alt="Lecture Video at 00:04:38.878" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8368_00-04-39.212.jpg" width="75%" alt="Lecture Video at 00:04:39.212" /></p>

I always use this; I think it's great.



<p align="center"><img src="./lecture_07_slides/slide_8414_00-04-40.747.jpg" width="75%" alt="Lecture Video at 00:04:40.747" /></p>

Especially if you have the compute where you can just run something over and over again to improve performance more, this is a really neat way of visualizing it.



<p align="center"><img src="./lecture_07_slides/slide_8446_00-04-41.814.jpg" width="75%" alt="Lecture Video at 00:04:41.814" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8474_00-04-42.749.jpg" width="75%" alt="Lecture Video at 00:04:42.749" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8534_00-04-44.751.jpg" width="75%" alt="Lecture Video at 00:04:44.751" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8550_00-04-45.285.jpg" width="75%" alt="Lecture Video at 00:04:45.285" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8560_00-04-45.618.jpg" width="75%" alt="Lecture Video at 00:04:45.618" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8574_00-04-46.085.jpg" width="75%" alt="Lecture Video at 00:04:46.085" /></p>

I think they do it well.



<p align="center"><img src="./lecture_07_slides/slide_8594_00-04-46.753.jpg" width="75%" alt="Lecture Video at 00:04:46.753" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8610_00-04-47.287.jpg" width="75%" alt="Lecture Video at 00:04:47.287" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8620_00-04-47.620.jpg" width="75%" alt="Lecture Video at 00:04:47.620" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8646_00-04-48.488.jpg" width="75%" alt="Lecture Video at 00:04:48.488" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8656_00-04-48.821.jpg" width="75%" alt="Lecture Video at 00:04:48.821" /></p>

<p align="center"><img src="./lecture_07_slides/slide_8668_00-04-49.222.jpg" width="75%" alt="Lecture Video at 00:04:49.222" /></p>

There are other tools like TensorBoard, but this is personally the one that I like. Okay, so for the rest of lecture today, we'll be discussing sequence modeling. This is, in contrast to a fixed sized input as input to our model, what if we have a sequence of variable length? We'll also be discussing what are the simple neural networks that people used before the era of transformers.

which mainly consists of RNNs and some variants of RNNs. I'll also relate in one slide how RNNs are actually similar to a lot of, and inspire a lot of, the modern type of language models that you see called state space models. So you might have heard of Mamba; there are some other ones too that we'll talk about in the slide, but the basic idea is the key concepts from RNNs are still being used today.

They are not just used in the past, and they have a lot of nice advantages over transformers that will go into.



<p align="center"><img src="./lecture_07_slides/slide_10164_00-05-39.138.jpg" width="75%" alt="Lecture Video at 00:05:39.138" /></p>

<p align="center"><img src="./lecture_07_slides/slide_10544_00-05-51.818.jpg" width="75%" alt="Lecture Video at 00:05:51.818" /></p>

In contrast, you could have a one-to-many sequence modeling task. Here we still have a fixed size input, like say an image, but we want to output a sequence of variable length. One common example is image captioning.



<p align="center"><img src="./lecture_07_slides/slide_11324_00-06-17.844.jpg" width="75%" alt="Lecture Video at 00:06:17.844" /></p>

You could also have a many-to-one sequence modeling task. Here, we could imagine our inputs are, say, a video and we are trying to classify what this video is of.



<p align="center"><img src="./lecture_07_slides/slide_12000_00-06-40.400.jpg" width="75%" alt="Lecture Video at 00:06:40.400" /></p>

Another setting is many-to-many. The number of inputs and outputs in the sequences don't need to match.



<p align="center"><img src="./lecture_07_slides/slide_12550_00-06-58.751.jpg" width="75%" alt="Lecture Video at 00:06:58.751" /></p>

For every single input, you have one output. While we are discussing RNNs, we will mainly be focusing on the many-to-many setting on the far right. There are basically a lot of small changes you can make to reformulate the problem to apply to the other settings, but this is the most straightforward one. Every time there's an input, there's an output, and we'll be using it for the beginning of class to talk about how RNNs work.

A canonical example problem here would be video classification, where you are classifying every single frame.



<p align="center"><img src="./lecture_07_slides/slide_13448_00-07-28.714.jpg" width="75%" alt="Lecture Video at 00:07:28.714" /></p>

What is an $\text{RNN}$? The basic idea is that you have an input sequence $x$, and an output sequence $y$. What makes an $\text{RNN}$ an $\text{RNN}$ is this recurrent nature. Often people will diagram it by this arrow that's feeding back into the block; this is how you know it's a recurrent layer when you are reading different diagrams.



<p align="center"><img src="./lecture_07_slides/slide_14096_00-07-50.336.jpg" width="75%" alt="Lecture Video at 00:07:50.336" /></p>

Every time there is a new input to the model, we process that and we calculate a new hidden state or internal state. There is a hidden state; it updates, and it depends on the new inputs as well as the previous internal or hidden state.



<p align="center"><img src="./lecture_07_slides/slide_15024_00-08-21.300.jpg" width="75%" alt="Lecture Video at 00:08:21.300" /></p>

I think this diagram is sometimes a bit confusing when you are trying to think about how the gradients are actually calculated and what the order of operations are. People will often draw this diagram of an unrolled $\text{RNN}$. We are more explicitly modeling what is exactly needed to calculate each output and each $\text{RNN}$ move backwards in the computational graph.



<p align="center"><img src="./lecture_07_slides/slide_15812_00-08-47.593.jpg" width="75%" alt="Lecture Video at 00:08:47.593" /></p>

So let's formulate this with mathematical equations now. This is the same thing here. The only change is that it is now a recurrence formula. We are using the same set of $W$'s and the same activation function each time we are computing the hidden state.



<p align="center"><img src="./lecture_07_slides/slide_17456_00-09-42.448.jpg" width="75%" alt="Lecture Video at 00:09:42.448" /></p>

As I mentioned, this is a recurrence formula. To get the actual output—how do we calculate this blue block? We have a separate function that depends on a separate set of parameters that convert our hidden dimension state into the dimension of our output. There is also a set of weights to convert the hidden state to the output.

This does two things: it changes the dimension of our vectors from the dimension size of our hidden state, which [The dimension size of our output]. And then also it provides a transformation there. So $W_y$ is a weight matrix that you will multiply by your hidden state to get the—so it does two things.

It converts your hidden state to the dimension of your output. So your hidden state and output could be different dimensions. And then also, it's a weight matrix that you learn. So not only does it do this dimension change, but also it applies a transformation to your hidden state.

It is how you convert your hidden states to your outputs; what $W_y$ is.



<p align="center"><img src="./lecture_07_slides/slide_19330_00-10-44.977.jpg" width="75%" alt="Lecture Video at 00:10:44.977" /></p>

The previous slide was how we calculate the new hidden state. We have another set of parameters and another function for calculating the output depending on what type of task it is, and how we want to model the RNN. They still share the same weights for each time step, but there are two different things here. And then how do you convert that hidden state to the output, which is this slide.



<p align="center"><img src="./lecture_07_slides/slide_20568_00-11-26.285.jpg" width="75%" alt="Lecture Video at 00:11:26.285" /></p>

Looking through this unrolled diagram here, we can see that you need to initialize your hidden state to some value. We usually call this $h_0$, and you can initialize it to whatever you want.



<p align="center"><img src="./lecture_07_slides/slide_21560_00-11-59.385.jpg" width="75%" alt="Lecture Video at 00:11:59.385" /></p>

You really do notice how the same function and the same set of parameters are used at every time step when computing the hidden state. A separate function and a separate set of parameters are always used at each time step when predicting the output from the hidden state. Can old values of $y$ affect the new hidden state? Under some formulations, yes.

We'll actually go through one example of why that's used. So that's generally how people do that explicit formulation of how can $y$ affect the next hidden state. What is the difference between $h_0$ and $x$ at the first time step? They use basically different weights.

The $h_0$ is using all of—it's using the weights that are used to update every hidden state to the next one. We'll go through exactly what the weights look like, but basically they're using different weights; that's the short answer.



<p align="center"><img src="./lecture_07_slides/slide_23716_00-13-11.323.jpg" width="75%" alt="Lecture Video at 00:13:11.323" /></p>

This is nice because it's bounded between $-1$ and $1$. So as you do the operation over and over again, your values will stay within this range. This is a nice property to have. It's also zero centered and you can represent both positive and negative values; this is why people use $\tanh$.

Also we sometimes have an output function $f_y$ here, but in the simplest case your output $y_t$ could just be a matrix multiplied by your hidden state. This is really the most simple formulation of an RNN.



<p align="center"><img src="./lecture_07_slides/slide_25212_00-14-01.240.jpg" width="75%" alt="Lecture Video at 00:14:01.240" /></p>

What we'll specifically go into in our concrete example today in lecture is this idea of just manually creating a recurrent neural network. We're not going to learn this through gradient descent or all these different methods; I'm just going to show you how you could construct one by hand. You're basically detecting repeated 1s, and you'll output a 0 otherwise.

You can see this input sequence coming in: 0, 1, 0, 1. So far, there's been no repeated ones. But now we have a repeated one, and we have another repeated one because there are two in a row here, and so on. This is the type of model we're building; it's trying to do this task.

This is specifically the many-to-many sequence modeling tasks where we have one output for every input. We've been talking high level so far, but if you're trying to create an RNN to do this, what information should be captured in the hidden state? You have this internal state of your model; what information needs to be captured there in order to do this task?

The input to the previous time step. If our output is only dependent on the hidden state, what else do we need to know?



<p align="center"><img src="./lecture_07_slides/slide_27888_00-15-30.529.jpg" width="75%" alt="Lecture Video at 00:15:30.529" /></p>

This is the information that we need to capture in our hidden state: the previous input and the current value for $\mathbf{x}$, which is either 0 or 1.



<p align="center"><img src="./lecture_07_slides/slide_28020_00-15-34.934.jpg" width="75%" alt="Lecture Video at 00:15:34.934" /></p>

We will set the hidden state $\mathbf{h}_t$ to be a three-dimensional vector. The reason why it's 3 is that this one will come in handy when we're trying to do the output stage calculation, although you could probably construct one without the $1$. This is just to make the math easy and simple for the purposes of the lecture today. The other information is the current value, which will either be 0 or 1, along with the previous values of 0 or 1.

We'll initialize it to be $\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$, so that we are basically assuming it is seeing two zeros in a row before at this point.



<p align="center"><img src="./lecture_07_slides/slide_29396_00-16-20.846.jpg" width="75%" alt="Lecture Video at 00:16:20.846" /></p>

<p align="center"><img src="./lecture_07_slides/slide_29700_00-16-30.990.jpg" width="75%" alt="Lecture Video at 00:16:30.990" /></p>

<p align="center"><img src="./lecture_07_slides/slide_30654_00-17-02.821.jpg" width="75%" alt="Lecture Video at 00:17:02.821" /></p>

<p align="center"><img src="./lecture_07_slides/slide_31436_00-17-28.914.jpg" width="75%" alt="Lecture Video at 00:17:28.914" /></p>

This will be the type of variables we are trying to track in our hidden state, and this is how we'll initialize $\mathbf{h}_0$. I talked about how you can initialize it using very different strategies, or you could learn it; this is what we'll initialize it to. Now let's walk through the code step by step. We are also setting our activation functions to be ReLU just to make the math easy.

It will just be $\max(0, \text{value})$. Since we are only dealing essentially with zeros and ones in this case, it makes it pretty simple to think about. You probably could construct it so that it works with $\tanh$, but we're using ReLU because it is an example for how to run it, keeping the math really easy. We have two specific weights here.

The first weight applies a transformation to the previous hidden state to calculate the next one. The second weight converts our input $\mathbf{x}$ to the dimension of our hidden state and also applies a transformation. Our current hidden state is a function of the previous hidden state, along with the current time step $t$. When we're trying to calculate this hidden state at time step $t$, we are looking to calculate the current value first.

We add this to another term; basically, this is calculating what is the current value.



<p align="center"><img src="./lecture_07_slides/slide_33162_00-18-26.505.jpg" width="75%" alt="Lecture Video at 00:18:26.505" /></p>

Now we talk about how we are doing the hidden state transformation. We want to use the current value for the top value here. In our weight matrix, we'll just have zeros in the top row.



<p align="center"><img src="./lecture_07_slides/slide_33920_00-18-51.797.jpg" width="75%" alt="Lecture Video at 00:18:51.797" /></p>

We set the next row to $\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$. Why do this? You can imagine we have the hidden state from the previous time step here. We want to set the "now previous" to be the former current time step, so we use $\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$.

What this will do is it'll multiply by $\mathbf{h}_{t-1}$. We set the current value over to now the previous value for this time step. Basically, this term will be a zero on top, and it will be whatever the previous time step input value was as the second term.



<p align="center"><img src="./lecture_07_slides/slide_34932_00-19-25.564.jpg" width="75%" alt="Lecture Video at 00:19:25.564" /></p>

This final bit just maintains the $1$ so that we are keeping this $1$ across all calculations.



<p align="center"><img src="./lecture_07_slides/slide_35136_00-19-32.371.jpg" width="75%" alt="Lecture Video at 00:19:32.371" /></p>

<p align="center"><img src="./lecture_07_slides/slide_35146_00-19-32.704.jpg" width="75%" alt="Lecture Video at 00:19:32.704" /></p>

To recap, we have zeros here because we want the right hand side term to track this current value.



<p align="center"><img src="./lecture_07_slides/slide_35338_00-19-39.111.jpg" width="75%" alt="Lecture Video at 00:19:39.111" /></p>

We have a one here to copy over the current from the former time step to be the previous of the current time step.



<p align="center"><img src="./lecture_07_slides/slide_35528_00-19-45.450.jpg" width="75%" alt="Lecture Video at 00:19:45.450" /></p>

<p align="center"><img src="./lecture_07_slides/slide_35656_00-19-49.721.jpg" width="75%" alt="Lecture Video at 00:19:49.721" /></p>

<p align="center"><img src="./lecture_07_slides/slide_35998_00-20-01.133.jpg" width="75%" alt="Lecture Video at 00:20:01.133" /></p>

We are doing $\mathbf{h}_{t}^{\text{previous}}$ is equal to $h_t$, and we also want to move the corresponding value down one.



<p align="center"><img src="./lecture_07_slides/slide_36046_00-20-02.734.jpg" width="75%" alt="Lecture Video at 00:20:02.734" /></p>

How do we actually get our output? We talked about how we can track these values given the weight matrices, specifically $W_{hh}$ and $W_{xh}$. If we have a weight matrix to convert our hidden state into the output dimension, we want it to be $1$ by $3$. It's a single value that's being output when we have this hidden dimension as input.

This is a dot product between the values here and the values here. What this will correspond to is the current plus the previous minus 1, because we multiply the minus $1$ here. This is where the one became useful: the current associated here with a one, and then also the previous associated here with the one as well. So that's how we actually do it.

If you think about it, this general formula will work. For example, if we're looking here, we have the current plus the previous is $2 - 1 = 1$ for this left-hand term inside the $\text{ReLU}$. The $\max$ of $1$ and $0$ is $1$. If these are both $0$, you'll have a minus $1$, so we get $0$.

With a $1$ and a $0$, then you'll still get $0$. So these are how you can construct these weight matrices. I just want to pause and see if there's a question about how the matrices and vectors are tracked, multiplied, and updated. The question is: How do you go about constructing the weight matrices?



<p align="center"><img src="./lecture_07_slides/slide_39360_00-21-53.312.jpg" width="75%" alt="Lecture Video at 00:21:53.312" /></p>

That is a really great question, and I thought to put it in the slide here. How would you actually do this? It's the same way we're always finding the weight matrices in this class. That'll be a lot of what we go into right next.

This is just an example so you can see how all of the weight matrices are multiplied. Basically, if you were trying to initialize with this and then train it to do another task, that would be transfer learning where you're initializing the weights with this. But in practice, I don't think it would work very well at all because your hidden state is really small, and people normally use much larger hidden states.

I'll go over the second row again. For the second row that gets calculated, the entry here will be equal to the top of the vector here. This is how we move the current down to the previous in this step here. The end result of this matrix multiply will be such that the second value is the current value from $\mathbf{t}-1$.

The left is doing the previous carryover, and the right is doing the current. That's what these weight matrices track more generally than the specific problem as well. So, how do you actually compute the gradients?



<p align="center"><img src="./lecture_07_slides/slide_43586_00-24-14.319.jpg" width="75%" alt="Lecture Video at 00:24:14.319" /></p>

Let's look at the computational graph. To draw a little bit more explicitly than before, we have these $x_1$ coming in and $x_2$, and we have a sequence of $x$'s.



<p align="center"><img src="./lecture_07_slides/slide_43720_00-24-18.790.jpg" width="75%" alt="Lecture Video at 00:24:18.790" /></p>

<p align="center"><img src="./lecture_07_slides/slide_43758_00-24-20.058.jpg" width="75%" alt="Lecture Video at 00:24:20.058" /></p>

<p align="center"><img src="./lecture_07_slides/slide_43868_00-24-23.728.jpg" width="75%" alt="Lecture Video at 00:24:23.728" /></p>

We're calculating a hidden state at each time step, and we're specifically using the same $\mathbf{W}$'s—same weight matrices—for each of these calculations as well.



<p align="center"><img src="./lecture_07_slides/slide_44214_00-24-35.273.jpg" width="75%" alt="Lecture Video at 00:24:35.273" /></p>

We need to be thinking about this when we're thinking about how we're computing the gradients, and let's start with the many-to-many scenario. We have an output for each input. In this scenario, you can often also calculate a loss for each output, which is how correct is the output at that stage.



<p align="center"><img src="./lecture_07_slides/slide_44598_00-24-48.086.jpg" width="75%" alt="Lecture Video at 00:24:48.086" /></p>

<p align="center"><img src="./lecture_07_slides/slide_44764_00-24-53.625.jpg" width="75%" alt="Lecture Video at 00:24:53.625" /></p>

If we're in this setting, you have a loss at each step, and you can sum them all together to get your total loss. This would be your loss across the entire input sequence. If we're calculating a loss per time step, you can treat them independently. Sometimes you have an overall loss based on the loss per time step too.

We can also get basically the final gradients for each of these $W$'s. You can calculate the gradient for each time step separately, and then you're going to sum them all together. So this is how it works in practice.

You could imagine if it were different $W$'s at each time step. You could probably easily see how the computational graph could be structured such that you're calculating a different gradient for each of these different $W$'s.



<p align="center"><img src="./lecture_07_slides/slide_47354_00-26-20.045.jpg" width="75%" alt="Lecture Video at 00:26:20.045" /></p>

<p align="center"><img src="./lecture_07_slides/slide_47404_00-26-21.713.jpg" width="75%" alt="Lecture Video at 00:26:21.713" /></p>

In the many-to-one scenario, you'll just have a single $\text{loss}$ calculated here. Sometimes you'll only use the final hidden state to calculate the value depending on the problem setting.



<p align="center"><img src="./lecture_07_slides/slide_47754_00-26-33.391.jpg" width="75%" alt="Lecture Video at 00:26:33.391" /></p>

You're going to do some pooling, like average pooling or max pooling or something like that, to compute your $y$ value.



<p align="center"><img src="./lecture_07_slides/slide_48166_00-26-47.138.jpg" width="75%" alt="Lecture Video at 00:26:47.138" /></p>

And then if you have this one-to-many mapping, like in image captioning, there was a question about how you could incorporate the previous $y$'s.



<p align="center"><img src="./lecture_07_slides/slide_49010_00-27-15.300.jpg" width="75%" alt="Lecture Video at 00:27:15.300" /></p>

So you can imagine that you can put a lot of values in here.



<p align="center"><img src="./lecture_07_slides/slide_49162_00-27-20.372.jpg" width="75%" alt="Lecture Video at 00:27:20.372" /></p>

You could just put zeros, or you could put the previous output here. I explained at a high level how you do the backpropagation.



<p align="center"><img src="./lecture_07_slides/slide_49978_00-27-47.599.jpg" width="75%" alt="Lecture Video at 00:27:47.599" /></p>

So when you're computing, say, a $\text{loss}$ at each time step, and you have an extremely long input sequence, it's really easy to understand. You need to be keeping the activations and the gradients at each time step in memory and then summing them all together. This is going to get extremely large as your input sequence increases. What can you do practically to resolve this issue?

So what you can do is, it's called truncated backpropagation through time.



<p align="center"><img src="./lecture_07_slides/slide_51036_00-28-22.901.jpg" width="75%" alt="Lecture Video at 00:28:22.901" /></p>

You basically fix a time window, and you can look at basically pretending that this is all the model was trained on so far. We start with our $h_0$. We calculate based on the input at time step 1, and our previous $h$ value. We can calculate what is the current hidden state at $h_1$, and then we can use that to calculate our output.

We'll have our $\text{loss}$. And we can run this for each of our examples. You can imagine how in this setting, it's relatively easy to see how you just treat the beginning sequence as if this is all we're seeing during training.



<p align="center"><img src="./lecture_07_slides/slide_52182_00-29-01.139.jpg" width="75%" alt="Lecture Video at 00:29:01.139" /></p>

Moving to the next block, now essentially, you're starting your $h_0$ with—now it's the output of your previous step here. So we're initializing the hidden state with whatever the output was in our final step, but the gradients are no longer carrying over. So we're basically batching the computational graph such that we're only looking at the $\text{loss}$ in the neighborhood of these time steps at a time.

This is a fixed window size that you set. So this is how you get around this relatively, I would say, common issue, especially as you have really long input sequences.



<p align="center"><img src="./lecture_07_slides/slide_53256_00-29-36.975.jpg" width="75%" alt="Lecture Video at 00:29:36.975" /></p>

And so yeah, you basically are batching it out, and you can just keep doing this for the entire input sequence.



<p align="center"><img src="./lecture_07_slides/slide_53406_00-29-41.980.jpg" width="75%" alt="Lecture Video at 00:29:41.980" /></p>

One other thing is that you might ask, "How does this work if we have just a single output at the very end?" You can imagine we're looking at the far right of the diagram here, and we have our $\text{loss}$ that we calculate based on the output of the final time step. We can calculate what is the gradient with respect to our current hidden state at the end.

And understanding based on the previous hidden state and the weight matrix $W$. How can we change this transformation matrix $W$ such that we would be changing our loss? And then basically just applying the gradient rule to $W$ over and over again here. You're only looking at how the hidden state changed, the next hidden state, and how that contributed to the loss.

You look at the final example here. So using different $W$'s at each time step would essentially mean that you're no longer modeling it as a recurrence relation. Basically, you can think of it as one layer for each different possible time step. That would make sense for a problem where it's not a sequence modeling problem; you just have a set of things that you want to classify.

You will need to know the amount of the sequence ahead of time. So how does this work with chunking? Do you understand how, at the point right here where the red dot, we can calculate the gradient of the loss with respect to our final hidden state? If we can do that, then we can calculate the gradient of our loss with respect to our second-to-final hidden state.

Because we know our final hidden state is dependent on our previous hidden state times this weight matrix $W$. We can do this and we can go back and forth until here. At this point, all we need to save is this final step here. When we're calculating backwards, we just use that value to calculate all the previous time steps.

That's the overall process. You're only looking at how the hidden state transforms to form the new hidden state, and that's the only value that's getting updated here. Also, how the input changes the hidden state. So you're looking at two values: both how the input affects it, and how the input affects the next hidden state and the previous hidden state.

The learning still occurs for all the batches. You have your loss with respect to each of your parameters in $W$ here. When you're calculating it for the previous time step, you basically keep this one value. If you change the initial hidden state here, how does that change the loss?

But when you're actually moving to the next chunk over, you only need to look at how does this hidden state here affect the hidden state in the next chunk. So you're looking at this division boundary. The one variable you need to track over is what is the gradient of the hidden state that occurs after the chunk. Then you can use that to calculate the gradient of the current hidden state, which is dependent on the input $x$ and the previous.

There are different ways you can formulate it, but you can imagine we just apply the update to all the weights here and zero out the memory. The only thing we're tracking is this gradient right here. You can do a gradient apply step where you apply all the gradients to the weights depending on the learning rate and your optimizer and all this stuff.

Then you move on to calculating the next batch. The reason why this isn't a perfect calculation is because you're calculating these independently rather than all at once. So you have three different updates rather than just one update at a time, but you are still calculating the gradient for each step here. You keep one thing in memory: how does this hidden state—the first one in the batch—how can you update the hidden state here to determine the loss?

We throw out all the other ones. So you have the weights in memory. You can apply the gradient, you do your learning rate multiply, and you apply it to the weights. You'll also see a similar thing if you do distributed learning.

So if you have a gradient calculated on each GPU separately, they will apply them all to the same set of weights even though they're calculated independently. It's a similar thing where you're not tracking it all in the same memory at the same time, and you're applying it to the weights one at a time.



<p align="center"><img src="./lecture_07_slides/slide_65016_00-36-09.367.jpg" width="75%" alt="Lecture Video at 00:36:09.367" /></p>

I mean, this is mainly for this one, it's essentially the same. But in this setting maybe it's more clear how you're explicitly losing information. Here, you're only looking at some of the outputs at a time.



<p align="center"><img src="./lecture_07_slides/slide_65692_00-36-31.923.jpg" width="75%" alt="Lecture Video at 00:36:31.923" /></p>

So you lose information here, but in this case, you wouldn't lose information.



<p align="center"><img src="./lecture_07_slides/slide_65772_00-36-34.592.jpg" width="75%" alt="Lecture Video at 00:36:34.592" /></p>

One more practical example where we can't fit the whole RNN on the slide is this idea of a character level language model. It's really funny because these were shown to be quite effective 10 years ago. You can see how the current wave of language models are a build up of this really simple approach of just predicting characters with RNNs.



<p align="center"><img src="./lecture_07_slides/slide_67092_00-37-18.636.jpg" width="75%" alt="Lecture Video at 00:37:18.636" /></p>

It's the index here. You can encode this as the index, and then we can use these as inputs. We can calculate our hidden layers based on the previous hidden layer as well as the current input.



<p align="center"><img src="./lecture_07_slides/slide_67374_00-37-28.045.jpg" width="75%" alt="Lecture Video at 00:37:28.045" /></p>

So we want the output for example to be 'e'. We map it over here; you can imagine this is something like $\text{softmax}$, and we have the logits, so these are the scores. You can really view this as a time step wise classification problem. That's exactly what in general, these language models are doing: time step wise classification based on $\text{softmax}$.



<p align="center"><img src="./lecture_07_slides/slide_68736_00-38-13.491.jpg" width="75%" alt="Lecture Video at 00:38:13.491" /></p>

<p align="center"><img src="./lecture_07_slides/slide_68960_00-38-20.965.jpg" width="75%" alt="Lecture Video at 00:38:20.965" /></p>

<p align="center"><img src="./lecture_07_slides/slide_69088_00-38-25.236.jpg" width="75%" alt="Lecture Video at 00:38:25.236" /></p>

<p align="center"><img src="./lecture_07_slides/slide_69124_00-38-26.437.jpg" width="75%" alt="Lecture Video at 00:38:26.437" /></p>

So you can actually create RNNs to do this basic language modeling task by operating at a character level, and it works quite well.



<p align="center"><img src="./lecture_07_slides/slide_69650_00-38-43.988.jpg" width="75%" alt="Lecture Video at 00:38:43.988" /></p>

<p align="center"><img src="./lecture_07_slides/slide_70648_00-39-17.288.jpg" width="75%" alt="Lecture Video at 00:39:17.288" /></p>

<p align="center"><img src="./lecture_07_slides/slide_71082_00-39-31.769.jpg" width="75%" alt="Lecture Video at 00:39:31.769" /></p>

Anyway, so we have $E$ here as our target character. In this case, you're correct that the model is actually getting it wrong, so we will want to penalize it heavily for this time step.



<p align="center"><img src="./lecture_07_slides/slide_71412_00-39-42.780.jpg" width="75%" alt="Lecture Video at 00:39:42.780" /></p>

<p align="center"><img src="./lecture_07_slides/slide_71436_00-39-43.581.jpg" width="75%" alt="Lecture Video at 00:39:43.581" /></p>

<p align="center"><img src="./lecture_07_slides/slide_71516_00-39-46.250.jpg" width="75%" alt="Lecture Video at 00:39:46.250" /></p>

One of the nice things about this implementation is also it's really simple. It's 112 lines of Python code, and you can train these models on a variety of different tasks.



<p align="center"><img src="./lecture_07_slides/slide_71780_00-39-55.059.jpg" width="75%" alt="Lecture Video at 00:39:55.059" /></p>

This was the pre-LLM era of what you could do; you can train it on sonnets by William Shakespeare. Could you explain again why you use an embedding layer? The basic idea for an embedding layer is that generally it's better to have vectors as input to our models, and we can learn what these embedding layers are too.



<p align="center"><img src="./lecture_07_slides/slide_72550_00-40-20.751.jpg" width="75%" alt="Lecture Video at 00:40:20.751" /></p>

We tend to favor spread out weights in general when we're trying to learn these. You can initialize your embedding layer to very small zero values with something like the Kaiming initialization, which we talked about. Then you're just looking at one row of it at a time as your input vector, rather. ...than it being a number as input, how you would have to represent that is basically a one with a bunch of zeros and optimization only the embedding works better.



<p align="center"><img src="./lecture_07_slides/slide_73462_00-40-51.182.jpg" width="75%" alt="Lecture Video at 00:40:51.182" /></p>

<p align="center"><img src="./lecture_07_slides/slide_73620_00-40-56.454.jpg" width="75%" alt="Lecture Video at 00:40:56.454" /></p>

So yeah, you can do it in 112 lines of Python code, which is pretty neat. You can train it on Sonnets by William Shakespeare, and it'll actually output reasonable text.



<p align="center"><img src="./lecture_07_slides/slide_73834_00-41-03.594.jpg" width="75%" alt="Lecture Video at 00:41:03.594" /></p>

We'll go through some examples. One of the cool things is you can see that as you train the model more, it becomes more and more coherent. At the beginning, it's basically just gibberish because it hasn't learned proper values for $W$. As you train it more and more, it becomes more like Stage III; it looks like English, at least some of the words are there.

Then as you train more, it actually starts working really well, which this is I guess was a bit of foreshadowing for what was to come in the era of AI, which is pretty cool.



<p align="center"><img src="./lecture_07_slides/slide_74606_00-41-29.353.jpg" width="75%" alt="Lecture Video at 00:41:29.353" /></p>

You can see full on, it learns things about the style, how you should have someone's name, and how something that seems fairly plausible. As you have it generating more and more, it starts making less and less sense, but it's pretty cool to see.



<p align="center"><img src="./lecture_07_slides/slide_75074_00-41-44.969.jpg" width="75%" alt="Lecture Video at 00:41:44.969" /></p>

You can train it on code, like I think in this example, they trained it on Linux.



<p align="center"><img src="./lecture_07_slides/slide_75374_00-41-54.979.jpg" width="75%" alt="Lecture Video at 00:41:54.979" /></p>

So there's just the source code for Linux. They trained one of these character-level RNNs, and you can see it generating C code, which looks pretty good. I don't know if this would compile, but it looks reasonable just looking at it. And this idea has really taken off over the past few years.

I mean, I'm sure you all know it, especially since a lot of you work in computer science or coding or your students in this area.



<p align="center"><img src="./lecture_07_slides/slide_75984_00-42-15.332.jpg" width="75%" alt="Lecture Video at 00:42:15.332" /></p>

Instead of trying to predict the next character, they're trying to predict the next token, which is a group of characters. How they define tokens depends on the model, and there are a lot of details we could get into there. But at a high level, it's a really similar thing; they're just predicting groups of characters autoregressively one after the next.

It has really seen a blow up in recent years with all these existing tools. What is the input to the model? Is it like a trigger?



<p align="center"><img src="./lecture_07_slides/slide_77248_00-42-57.508.jpg" width="75%" alt="Lecture Video at 00:42:57.508" /></p>

Oh, what like for this? Yeah. You could have the input be—you just maybe you start with a random character. Could be one way to do it, but you would need some initial input.

There could be... usually with language models they have a start token as predetermined. This is always what you see at the start of your sequence. So you could do similar things with RNNs. I don't know in this exact scenario what they did.

Maybe they just did a character, but it's hard to know. So the question is, how does labeling work with language models? And the neat thing about these pure language models, all they're doing is just predicting the next token. You don't need to label it; you just need to give it a lot of text.

That's why these models are so good—is because they scrape the internet for essentially all available text, and then they train the model on all of it. So that's why they're so good: it's because it's just generating the next token, and you don't need to label it. So the question is, if we're always taking the maximum probability output at each time step, are we always just going to be generating the same thing over and over again?

The answer is yes, actually. So if you just took the maximum probability, I guess this example is not so good.



<p align="center"><img src="./lecture_07_slides/slide_79244_00-44-04.108.jpg" width="75%" alt="Lecture Video at 00:44:04.108" /></p>

But imagine the probabilities are correct here, and you just took the maximum probability at each time step, you would always be getting the same output given the same input. In practice what people do is they don't do this; this is called greedy decoding, where you're always picking the maximum probability. In practice, they sample based on a distribution, the distribution given by the probabilities output by your softmax.

So you won't pick the max probability; you would pick, say in this case, probability $0.84$ for this one, or probability $0.13$ for this other output variable. Then you would run that for each sequence. And there are a bunch of different ways you can do it too. You can search ahead; it's called beam searching, where you're trying different ones and seeing which one has the highest overall probability for the sequence.

So there are a lot of—this is a whole active area of research on how do you sample from these models? But the simple answer is you don't always pick the highest probability.



<p align="center"><img src="./lecture_07_slides/slide_80804_00-44-56.160.jpg" width="75%" alt="Lecture Video at 00:44:56.160" /></p>

So the question is, in the case where we have many-to-one outputs, are we outputting something each time or do we have something to look at here? I think in practice, to save compute, you wouldn't want to output something that's never used. But you could feasibly output at each time step. And it might be interesting depending on your problem to look at that and understand if the output is converging over the course of training or not?

Something like that. So it might be useful to look at, but generally people wouldn't do it just to save compute. But it could be useful, actually. It could help you understand the way your model works—if there's certain triggers or things that help it predict the correct answer.

Cool, good questions. So we'll keep on chugging along.



<p align="center"><img src="./lecture_07_slides/slide_82032_00-45-37.134.jpg" width="75%" alt="Lecture Video at 00:45:37.134" /></p>

We talked about these RNNs, how good they are at generating characters. We related them to some of these modern coding tools, which are really neat.



<p align="center"><img src="./lecture_07_slides/slide_82180_00-45-42.072.jpg" width="75%" alt="Lecture Video at 00:45:42.072" /></p>

For example, in our little toy example, we looked at the output activations, and you would see it's the current value and the previous value. That was what the RNN states or cells were tracking.



<p align="center"><img src="./lecture_07_slides/slide_82842_00-46-04.161.jpg" width="75%" alt="Lecture Video at 00:46:04.161" /></p>

You can also give it basically a sequence here. The models I'll show in these slides are using a $\tanh$ activation, so this is from 1 to minus 1. Minus 1 means it's visualized as red here, and very close to one would be blue; we get the whole spectrum here. You can look at, for each character coming in, what is the activation of that cell at that time step?

That's how they are color-coding these plots here. This visualization isn't always informative—it can be random, and a lot of activations won't be interpretable.



<p align="center"><img src="./lecture_07_slides/slide_83744_00-46-34.258.jpg" width="75%" alt="Lecture Video at 00:46:34.258" /></p>

However, some have pretty cool things you can track. For example, one is a quote detector, so it turns on basically as soon as the quote starts and ends when the quote ends.



<p align="center"><img src="./lecture_07_slides/slide_84358_00-46-54.745.jpg" width="75%" alt="Lecture Video at 00:46:54.745" /></p>

Another cool thing is the line length tracking cell. These activations are just single layers of this model mapping to each character, so they are highly interpretable.



<p align="center"><img src="./lecture_07_slides/slide_85192_00-47-22.573.jpg" width="75%" alt="Lecture Video at 00:47:22.573" /></p>

<p align="center"><img src="./lecture_07_slides/slide_85416_00-47-30.047.jpg" width="75%" alt="Lecture Video at 00:47:30.047" /></p>

We also see an 'if statement' cell, which tracks anything within an if statement—which is pretty cool. It can even detect quotes or comments because it needs to know when to output the end-of-comment character here; it has to track that transition.



<p align="center"><img src="./lecture_07_slides/slide_85784_00-47-42.326.jpg" width="75%" alt="Lecture Video at 00:47:42.326" /></p>

Finally, there's the code depth cell. As you have nesting in your code, it activates more and more at each time step, representing the indentation into your code hierarchy. The RNN is internally doing a very similar process to what we might manually assign.



<p align="center"><img src="./lecture_07_slides/slide_87036_00-48-24.101.jpg" width="75%" alt="Lecture Video at 00:48:24.101" /></p>

I'll talk about now some of the trade-offs on why you might want to use an RNN and when it is helpful. A major advantage is that they can process any length of input. Many modern language models rely on transformers, which have something called a context length or maximum context window. RNNs do not have this; they can take a sequence of essentially infinite length as long as you keep running the model on it.

There is no context length limit. The computation for the time step $t$ can, in theory, use information from many steps back if it's captured in the hidden state. In practice, however, there might be some issues with this, which we will go into detail later. Also, the model size does not increase for a longer input.

This differs from an example where you might have to use a different layer for each input time step. We apply the same weights at each time step; basically, the update rule for calculating outputs is the same every single time, which provides nice symmetry. Conceptually, you are always doing the same thing at every single time step, which helps with both understanding and implementation.

What are the main disadvantages? You need to compute the previous hidden state to compute the next one every single time. Each hidden state is determined and conditioned on all the previous ones, meaning this recurrence computation can take a lot of time. This poses challenges for scaling up to large amounts of data.

As your sequence goes longer and longer. I'll talk about some applications more specific to computer vision, where RNNs have seen success now.



<p align="center"><img src="./lecture_07_slides/slide_91616_00-50-56.920.jpg" width="75%" alt="Lecture Video at 00:50:56.920" /></p>

One of them is image captioning, which we talked about. The basic thing here is we mentioned there's this start token or start character which begins the sequence. You will terminate when you have this end character or end token. In this case, it seems like it's word-level tokens.



<p align="center"><img src="./lecture_07_slides/slide_92164_00-51-15.205.jpg" width="75%" alt="Lecture Video at 00:51:15.205" /></p>

So you could have a model like this. So we have two stages here.



<p align="center"><img src="./lecture_07_slides/slide_92824_00-51-37.227.jpg" width="75%" alt="Lecture Video at 00:51:37.227" /></p>

More concretely, how would you combine the CNN and RNN?



<p align="center"><img src="./lecture_07_slides/slide_92882_00-51-39.162.jpg" width="75%" alt="Lecture Video at 00:51:39.162" /></p>

You can imagine you have this test image. It comes in, so your model's going downwards here, starting at the first layers at the top and then moving downwards.



<p align="center"><img src="./lecture_07_slides/slide_93234_00-51-50.907.jpg" width="75%" alt="Lecture Video at 00:51:50.907" /></p>

You can imagine this is something that was trained on ImageNet or something. We are not going to use the class labels, but we are going to use this second to last layer.



<p align="center"><img src="./lecture_07_slides/slide_93614_00-52-03.587.jpg" width="75%" alt="Lecture Video at 00:52:03.587" /></p>

<p align="center"><img src="./lecture_07_slides/slide_93742_00-52-07.858.jpg" width="75%" alt="Lecture Video at 00:52:07.858" /></p>

Then we can start using this as input to our hidden state. Now our hidden state is also a function of this $W_{ih}$ value here. So we don't necessarily have just a hidden state; we are also tracking the visual components here.



<p align="center"><img src="./lecture_07_slides/slide_94756_00-52-41.691.jpg" width="75%" alt="Lecture Video at 00:52:41.691" /></p>

<p align="center"><img src="./lecture_07_slides/slide_94854_00-52-44.961.jpg" width="75%" alt="Lecture Video at 00:52:44.961" /></p>

We use the sampling process—either greedy sampling or some other version of sampling—to calculate the tokens at each time step.



<p align="center"><img src="./lecture_07_slides/slide_94886_00-52-46.029.jpg" width="75%" alt="Lecture Video at 00:52:46.029" /></p>

<p align="center"><img src="./lecture_07_slides/slide_94920_00-52-47.164.jpg" width="75%" alt="Lecture Video at 00:52:47.164" /></p>

<p align="center"><img src="./lecture_07_slides/slide_94986_00-52-49.366.jpg" width="75%" alt="Lecture Video at 00:52:49.366" /></p>

We end it when we have this end token.



<p align="center"><img src="./lecture_07_slides/slide_95120_00-52-53.837.jpg" width="75%" alt="Lecture Video at 00:52:53.837" /></p>

Whenever we sample the end token, that's how we know when to finish. And these models actually worked very well for the time; I think they had a lot of great successes. You can see here a lot of nice examples of where the model is outputting very reasonable captions based on the input image. But also, these models would struggle in a lot of scenarios too.



<p align="center"><img src="./lecture_07_slides/slide_95624_00-53-10.654.jpg" width="75%" alt="Lecture Video at 00:53:10.654" /></p>

A lot of this has to do with the distribution of where these images are commonly seen in the training data. For example, someone holding something with their hands cupped like this—it very much looks how they might hold a mouse. But obviously, we can tell this is a phone because it's a flat object they're holding, and their hand is facing up, not downwards.

This thing is interesting to see. Also, I guess they think the woman's holding a cat where she's just wearing some fur clothing. They see a beach, so they assume there's a surfboard. In the data set, there's high co-occurrence of these actions or objects with the particular scene.

The model learns to associate them, but it doesn't learn to disentangle why this is happening in this scene. We know they are not throwing because the glove is here and the ball is going into the glove, not on the other hand. And the way we train these models, we're training them just to output the caption. So we're not doing any explanation there, and that's why it's part of the reason you see this co-occurrence issue.



<p align="center"><img src="./lecture_07_slides/slide_98624_00-54-50.754.jpg" width="75%" alt="Lecture Video at 00:54:50.754" /></p>

For visual question answering (VQA), this is another really common task where RNNs were used, and there are two formulations for visual question answering that were commonly used. One is to basically, say you have a model that is a captioning model, and you want to see how well it could answer questions. You could give it the question, and then have it output text and look at the probabilities of each of the answer sequences.

You have a probability for each character or token, and you could multiply them together to get the probability of the overall answer. This is one way you could use one of these RNN style models to do question answering. A more common way people did it is they would have basically a question as input to the model, and multiple different answers also as separate inputs to your model.

And then it's outputting essentially a probability per question. In this case, it would be a four-way classifier where you have four different classes: Answer 1, Answer 2, Answer 3, Answer 4, and you're just outputting the probabilities.



<p align="center"><img src="./lecture_07_slides/slide_100810_00-56-03.693.jpg" width="75%" alt="Lecture Video at 00:56:03.693" /></p>

Also visual dialogue. At the time, these were all considered very separate tasks. These days, you have one model that can do almost all of these, such as how can you have a chat about an image? We've really seen an explosion in the capabilities of these kinds of models in the last two years.



<p align="center"><img src="./lecture_07_slides/slide_101352_00-56-21.778.jpg" width="75%" alt="Lecture Video at 00:56:21.778" /></p>

Maybe one other type of model that RNNs were commonly used for is this visual navigation task. You have these images coming in, and you want to output a sequence of directions to move on some 2D floor plan. How do you get to the target destination? There's another application for you all to be aware of where these sequence models are used.



<p align="center"><img src="./lecture_07_slides/slide_102118_00-56-47.337.jpg" width="75%" alt="Lecture Video at 00:56:47.337" /></p>

And in practice, most of the RNNs I showed were multi-layer RNNs. The main difference is that you treat each layer separately. The hidden state of say Layer 1 depends on the hidden state of the previous time step of Layer 1. In the depth dimension, each of these layers, you're only looking at the hidden states from that layer in the previous time steps.

To calculate this top-right value, we need to calculate all of the different values, all of the different hidden states in this entire computational graph beforehand. You can get a feel for how as you start training, this gets to be a very involved process and not very efficient.



<p align="center"><img src="./lecture_07_slides/slide_105072_00-58-25.902.jpg" width="75%" alt="Lecture Video at 00:58:25.902" /></p>

<p align="center"><img src="./lecture_07_slides/slide_105988_00-58-56.466.jpg" width="75%" alt="Lecture Video at 00:58:56.466" /></p>

We talked about how by default $\tanh$ is a really commonly used activation function. You can also formulate this as we have our weights here, and you're stacking the vectors like this. Sometimes for shorthand, people will just combine both of these $W$'s together to form one big $W$. But this is a shorthand way to notate it where it makes thinking about it and writing down the math easier.

You will see all three variants here. This one is maybe the most explicit about where the actual values, the non-zero values, and the weight matrices lie. One way to think of it is you stack these vectors together, which is shown here. We're multiplying by this $W$ and then we pass it through $\tanh$.

This gives us our output $h_t$, which we pass to the next RNN. You can imagine these are stacked. You may also have either the output directly $\mathbf{y}_t$, or we have this layer where it's a weight matrix times $h_t$ with the activation function around it too. Here, we have multi-layer RNNs, and the weights are shared within the layers for multi-layer RNN.



<p align="center"><img src="./lecture_07_slides/slide_108950_01-00-35.298.jpg" width="75%" alt="Lecture Video at 01:00:35.298" /></p>

All of these hidden state updates will use the same weights. And then each layer, which you stack vertically in this diagram, each layer will have a separate set of weights.



<p align="center"><img src="./lecture_07_slides/slide_109670_01-00-59.322.jpg" width="75%" alt="Lecture Video at 01:00:59.322" /></p>

<p align="center"><img src="./lecture_07_slides/slide_109746_01-01-01.858.jpg" width="75%" alt="Lecture Video at 01:01:01.858" /></p>

<p align="center"><img src="./lecture_07_slides/slide_109758_01-01-02.258.jpg" width="75%" alt="Lecture Video at 01:01:02.258" /></p>

This is the way that it works. When you have backpropagation, if you don't have a loss for each time step, you need to only calculate your loss based on what the losses of your output $h_t$. When you do this backpropagation, you are multiplying by $W$, and then you are also taking the derivative of $\tanh$.



<p align="center"><img src="./lecture_07_slides/slide_110462_01-01-25.748.jpg" width="75%" alt="Lecture Video at 01:01:25.748" /></p>

Both of these can actually have some issues. This is what this gradient is calculating. We need the derivative of $\tanh$ because this is our activation function. And then we have $W h$, which is the multiplication here for converting the previous hidden state to the next one.

So, this is actually how we calculate the gradient.



<p align="center"><img src="./lecture_07_slides/slide_111480_01-01-59.716.jpg" width="75%" alt="Lecture Video at 01:01:59.716" /></p>

And here we can run into issues. If we are calculating the loss at each time step, and we have the total loss, we just sum over all weights.



<p align="center"><img src="./lecture_07_slides/slide_111992_01-02-16.799.jpg" width="75%" alt="Lecture Video at 01:02:16.799" /></p>

The total loss is just the $\sum$ of the loss at each time step with respect to this reused $W$ matrix. You end up getting this product of these $\frac{\partial L}{\partial h_{t-1}}$ to calculate the loss of $L_t$ at the final step $L_t$ with respect to $h_t$. To do this, you need to calculate each of the intermediate hidden states and how that affects $W$ in order to calculate this final loss here by using the chain rule.



<p align="center"><img src="./lecture_07_slides/slide_112848_01-02-45.361.jpg" width="75%" alt="Lecture Video at 01:02:45.361" /></p>

<p align="center"><img src="./lecture_07_slides/slide_113286_01-02-59.976.jpg" width="75%" alt="Lecture Video at 01:02:59.976" /></p>

<p align="center"><img src="./lecture_07_slides/slide_113612_01-03-10.853.jpg" width="75%" alt="Lecture Video at 01:03:10.853" /></p>

Why is this an issue? First of all, this is the derivative of $\tanh$ plotted here. The maximum value is 1, and so almost always you are getting less than 1. You can thus have vanishing gradients from this term here.



<p align="center"><img src="./lecture_07_slides/slide_114072_01-03-26.202.jpg" width="75%" alt="Lecture Video at 01:03:26.202" /></p>

<p align="center"><img src="./lecture_07_slides/slide_114298_01-03-33.743.jpg" width="75%" alt="Lecture Video at 01:03:33.743" /></p>

We will either have a large singular value; the vectors are coming in, what is the maximum they will be stretched?



<p align="center"><img src="./lecture_07_slides/slide_115252_01-04-05.575.jpg" width="75%" alt="Lecture Video at 01:04:05.575" /></p>

Or if it is very small, you can have this vanishing gradient issue. If you have exploding gradients, we have a fix which is scaling the gradient. You can just divide or clip it somehow so that you don't have too big of a gradient; it's not too much of an issue.



<p align="center"><img src="./lecture_07_slides/slide_115622_01-04-17.920.jpg" width="75%" alt="Lecture Video at 01:04:17.920" /></p>

I think these are the main reasons why they motivated a change in RNN architectures—and why a lot of the reasons why people don't use RNNs. This is one of the main issues.



<p align="center"><img src="./lecture_07_slides/slide_116608_01-04-50.820.jpg" width="75%" alt="Lecture Video at 01:04:50.820" /></p>

So how do you resolve this? The way that people did it was through the creation of the LSTM. The high-level idea, which I won't go into too many details because it's actually quite complicated, is that you have four of these different gates that are tracking different values.



<p align="center"><img src="./lecture_07_slides/slide_116906_01-05-00.763.jpg" width="75%" alt="Lecture Video at 01:05:00.763" /></p>

Instead of just having one hidden state, you have multiple of these values. You precompute to determine how to change your hidden state and then also what information to pass through a different pathway. You have the regular hidden state pathway. You have a different pathway where it's easier to pass information.



<p align="center"><img src="./lecture_07_slides/slide_117556_01-05-22.451.jpg" width="75%" alt="Lecture Video at 01:05:22.451" /></p>

<p align="center"><img src="./lecture_07_slides/slide_117590_01-05-23.586.jpg" width="75%" alt="Lecture Video at 01:05:23.586" /></p>

<p align="center"><img src="./lecture_07_slides/slide_117772_01-05-29.659.jpg" width="75%" alt="Lecture Video at 01:05:29.659" /></p>

<p align="center"><img src="./lecture_07_slides/slide_117906_01-05-34.130.jpg" width="75%" alt="Lecture Video at 01:05:34.130" /></p>

<p align="center"><img src="./lecture_07_slides/slide_118002_01-05-37.333.jpg" width="75%" alt="Lecture Video at 01:05:37.333" /></p>

<p align="center"><img src="./lecture_07_slides/slide_118274_01-05-46.409.jpg" width="75%" alt="Lecture Video at 01:05:46.409" /></p>

You can see this is really involved, a lot of design choices here. They put it all together into this fairly complicated diagram, but the basic idea is that this part is the same where we're... Doing this weight multiply, but now we have four different values we're computing instead of just the $h_t$. We have the input gate, and the gate to determine how much to write here, and we have our output that's passed to the next hidden state.

You can think of this top section here as a highway where the goal is to not have any activation functions—so no $\tanh$. So we avoid the issues we had where $\tanh$ made the gradients vanish, and all we're applying is this forget gate.



<p align="center"><img src="./lecture_07_slides/slide_119504_01-06-27.450.jpg" width="75%" alt="Lecture Video at 01:06:27.450" /></p>

So as long as we're not basically forgetting all the information at each time step, we're able to pass information more easily.



<p align="center"><img src="./lecture_07_slides/slide_119746_01-06-35.524.jpg" width="75%" alt="Lecture Video at 01:06:35.524" /></p>

This is the high-level explanation. And then more importantly, in practice, people seem to see that this worked very well. Again, you won't be implementing this for the course at all, but I think this is a really commonly used baseline still and some deep learning papers. You need to cram everything into this hidden state, so you have really long-term dependencies.

Those are lost. So they created a separate pathway to pass over this more long-term information through the top here.



<p align="center"><img src="./lecture_07_slides/slide_120924_01-07-14.830.jpg" width="75%" alt="Lecture Video at 01:07:14.830" /></p>

So do LSTMs solve the vanishing gradient problem completely? It definitely helps. So it makes the RNN easier to preserve this information over many time steps by using this top pathway diagram. So it doesn't guarantee it, but it makes it significantly easier and it helps improve learning long-term dependencies and works very well empirically.

So people generally don't train RNNs so much, and more often train LSTMs, if you were going to go with this recurrent modeling route.



<p align="center"><img src="./lecture_07_slides/slide_122866_01-08-19.628.jpg" width="75%" alt="Lecture Video at 01:08:19.628" /></p>

So you have multiple in ResNets. You have multiples of these convolution layers stacked together, and then you add skip connections where the value just gets added here. It's very long sequences of time steps. So this is parallel, but it's a little different because one is the number of layers and the other is the number of time steps.



<p align="center"><img src="./lecture_07_slides/slide_124430_01-09-11.814.jpg" width="75%" alt="Lecture Video at 01:09:11.814" /></p>

<p align="center"><img src="./lecture_07_slides/slide_124476_01-09-13.349.jpg" width="75%" alt="Lecture Video at 01:09:13.349" /></p>

But there are actually a lot of nice advantages they have. The main one is this unlimited context length. So one of the main issues with transformers is they have a limited context length, as people are really pushing the boundaries for what these models are capable of. This context length is becoming more and more of an issue.

So there have been various workarounds in the transformer space. People do things like RoPE and some other techniques to try to extend the context length, but it's a pretty significant limitation of the model. So there's no operation that looks across the entire input sequence like you have for transformers. So these are really big advantages, and there have been a couple of papers.

To shout out a few, there's this RWKV model. You can check out the arXiv link here, and also Mamba are both mainly highlighting this idea of we're able to achieve linear time sequence modeling. So as you scale up your input sequence, the compute also scales linearly as opposed to quadratically with transformers. And so it's better for long context problems sometimes.

In terms of compute it works better, and it has these main advantages. So people try to get the best of both worlds, and there's been a lot of research in this area. How can you get the performance of transformers with the scaling of RNNs?



<p align="center"><img src="./lecture_07_slides/slide_127698_01-11-00.856.jpg" width="75%" alt="Lecture Video at 01:11:00.856" /></p>

That's all for today in class. We basically talked about how there are a lot of different ways you can design architectures with RNNs. Vanilla RNNs are simple, but they don't work that well. There have been more complex variants that people have proposed that introduce ways to selectively pass information.

This backward flow of gradients in RNNs can either explode or vanish, depending on the activation function you use or what the properties of your weight matrix are. So you often need this back propagation through time to actually compute the gradient as well. These better architectures are a hot topic of research right now, as well as generally new paradigms for reasoning over sequences.

I think that's it for today.



<p align="center"><img src="./lecture_07_slides/slide_128954_01-11-42.765.jpg" width="75%" alt="Lecture Video at 01:11:42.765" /></p>

Next time we'll talk about attention and transformers.



