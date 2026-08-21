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

# Stanford CS231N Deep Learning for Computer Vision | Spring 2025 | Lecture 17: Robot Learning


<p align="center"><img src="./lecture_17_slides/slide_4_00-00-00.133.jpg" width="75%" alt="Lecture Video at 00:00:00.133" /></p>

We're here with our final guest lecturer for the course, and today we have Dr. Yunzhu Li. He is an assistant professor of computer science at Columbia University, where he leads the Robotic Perception, Interaction, and Learning Lab. He is also a former instructor of CS231N, like all of our guest lecturers. He taught the course in 2023 while he completed his postdoc here at Stanford with professors Fei-Fei Li and Jiajun Wu.

His research lies at the intersection of robotics, computer vision, and machine learning. Specifically, his work focuses on robot learning and aims to significantly expand robots' perception and physical interaction capabilities.



<p align="center"><img src="./lecture_17_slides/slide_1406_00-00-46.913.jpg" width="75%" alt="Lecture Video at 00:00:46.913" /></p>

In today's lecture, he'll be discussing exactly that: topic robot learning. I'll now hand it off to Yunzhu for today's lecture. "Yeah, thank you, Dean, for the very kind introduction. I'm super excited to be here.

The last time I was here giving lectures was two years ago, in 2023. Lately, I was going through many of the lectures. Today, I'm going to be talking about some of the things that I have been working on. It's also a very coherent piece of this overall picture on deep learning for computer vision, and this is specifically on robot learning.



<p align="center"><img src="./lecture_17_slides/slide_2952_00-01-38.498.jpg" width="75%" alt="Lecture Video at 00:01:38.498" /></p>

<p align="center"><img src="./lecture_17_slides/slide_3648_00-02-01.721.jpg" width="75%" alt="Lecture Video at 00:02:01.721" /></p>

So first of all, you guys have already learned a lot about supervised learning. The scene and the setup for supervised learning is that you have data $x$ and $y$. $X$ is the input, and $y$ is the label. You are trying to learn a mapping that maps from the input $x$ to the output $y$.

There are examples you have already learned, like classification, regression, object detection, et cetera. You have also learned about self-supervised learning, where instead of having labels, in this case, you are just having the data without any labels.



<p align="center"><img src="./lecture_17_slides/slide_4718_00-02-37.423.jpg" width="75%" alt="Lecture Video at 00:02:37.423" /></p>

So it's not just you have the input and outputs and mapping from the input $x$ to $y$, or some kind of latent representations. It's really about you are influenced by evolutions of the environment. So, no matter what action you decide to take in the real world, the world will change as a result of that actions. The world will give you some kind of new observations or reward telling you how the environment has been changing and how good you are in executing certain tasks.

The goal is trying to actually come up with a sequence of actions with feedback from the environments—that is, to maximize some reward or minimize some cost. Robot learning, especially in recent years, has attracted significant attention both within academia and also within industries.



<p align="center"><img src="./lecture_17_slides/slide_7294_00-04-03.376.jpg" width="75%" alt="Lecture Video at 00:04:03.376" /></p>

This field, like I mentioned, has attracted a lot of attention and also a lot of investments.



<p align="center"><img src="./lecture_17_slides/slide_7854_00-04-22.061.jpg" width="75%" alt="Lecture Video at 00:04:22.061" /></p>

<p align="center"><img src="./lecture_17_slides/slide_8390_00-04-39.946.jpg" width="75%" alt="Lecture Video at 00:04:39.946" /></p>

For today's lecture, I'm going to give you some kind of overviews on some of the key techniques enabling factors of the current success and boom of robot learning.



<p align="center"><img src="./lecture_17_slides/slide_8744_00-04-51.758.jpg" width="75%" alt="Lecture Video at 00:04:51.758" /></p>

<p align="center"><img src="./lecture_17_slides/slide_9046_00-05-01.834.jpg" width="75%" alt="Lecture Video at 00:05:01.834" /></p>

I will then discuss the more perception side; I will talk about the different considerations."



<p align="center"><img src="./lecture_17_slides/slide_9500_00-05-16.983.jpg" width="75%" alt="Lecture Video at 00:05:16.983" /></p>

<p align="center"><img src="./lecture_17_slides/slide_9538_00-05-18.251.jpg" width="75%" alt="Lecture Video at 00:05:18.251" /></p>

<p align="center"><img src="./lecture_17_slides/slide_9614_00-05-20.787.jpg" width="75%" alt="Lecture Video at 00:05:20.787" /></p>

<p align="center"><img src="./lecture_17_slides/slide_9660_00-05-22.322.jpg" width="75%" alt="Lecture Video at 00:05:22.322" /></p>

<p align="center"><img src="./lecture_17_slides/slide_9784_00-05-26.459.jpg" width="75%" alt="Lecture Video at 00:05:26.459" /></p>

<p align="center"><img src="./lecture_17_slides/slide_10036_00-05-34.867.jpg" width="75%" alt="Lecture Video at 00:05:34.867" /></p>

<p align="center"><img src="./lecture_17_slides/slide_10132_00-05-38.071.jpg" width="75%" alt="Lecture Video at 00:05:38.071" /></p>

We'll start with problem formulation. In general, how the problem should look like is illustrated by an agent that receives a task objective. This task objective could be, for example, language instructions from a human or some kind of objective function measuring how good this agent is at doing a specific task. The agent takes states from the physical world or some kind of environment.



<p align="center"><img src="./lecture_17_slides/slide_13242_00-07-21.841.jpg" width="75%" alt="Lecture Video at 00:07:21.841" /></p>

<p align="center"><img src="./lecture_17_slides/slide_14278_00-07-56.409.jpg" width="75%" alt="Lecture Video at 00:07:56.409" /></p>

<p align="center"><img src="./lecture_17_slides/slide_15074_00-08-22.969.jpg" width="75%" alt="Lecture Video at 00:08:22.969" /></p>

<p align="center"><img src="./lecture_17_slides/slide_15806_00-08-47.393.jpg" width="75%" alt="Lecture Video at 00:08:47.393" /></p>

<p align="center"><img src="./lecture_17_slides/slide_16258_00-09-02.475.jpg" width="75%" alt="Lecture Video at 00:09:02.475" /></p>

<p align="center"><img src="./lecture_17_slides/slide_16992_00-09-26.966.jpg" width="75%" alt="Lecture Video at 00:09:26.966" /></p>

<p align="center"><img src="./lecture_17_slides/slide_17308_00-09-37.510.jpg" width="75%" alt="Lecture Video at 00:09:37.510" /></p>

The agent decides what action to take, which needs to be executed in the physical world. This physical world will then be updated and given this state $s_{t+1}$, as well as rewards, telling the agent how good it is doing its task. This is generally how the framework looks like. You have to be very clear on this type of formulation that consists of goals, states, actions, and also rewards.

The problems of robot learning scenarios are very different from computer vision. Computer vision is mostly about trying to learn some kind of representations of the environment based on the inputs, like high-dimensional data. That is a key difference between robot learning and what people typically consider in computer vision. Some specific instantiations of this problem include cart-pole, where the goal is to balance a pole on top of a movable cart.

The states of this environment essentially describe the physical status of the system, which can include the angle, angular speed, positions, horizontal velocities, etc. The action would be the horizontal force applied to the carts. You can have rewards $r_t = 1$ indicating at each time step if the pole is being kept in an upright position. Another example could be robot locomotion, where the goal is to make these robots move forwards.

The states could include the angle, positions, and velocities of all joints within this robot. The action could be the torque applied to each one of the joints. The reward can be $r_t = 1$ at each time step if the robot makes a step forward and remains in an upright position. Some interesting examples include Atari games.

The goal can be to complete the game with the highest score possible. The states will be the raw pixel inputs of the gaming screen, and the action could be a gaming control like up, down, left, and right. The reward could be the score increase or decrease at each time step. Another famous example, especially with the developments of AlphaGo, is the game of Go.

The problem can also be defined in similar ways, where the goal is to win the game. The states will be all the pieces that are currently on the Go board. The action could be where to put the next piece down on this board. The reward could be: if you win on the last turn, you get a reward of $1$; and if you lose, you get a reward of $0$.

This not only applies to gaming domains. The states could be the current words in the sentence. The action will be what specific next word you want to put there. If it is correct, you get the reward; if it is incorrect, you get a reward of $0$.



<p align="center"><img src="./lecture_17_slides/slide_17736_00-09-51.791.jpg" width="75%" alt="Lecture Video at 00:09:51.791" /></p>

Similarly, you have probably played with many chatbots. You can also define a problem similarly, where the goal is to be a good companion to the human user. The states could be the current conversation, and the action that should be generated by the chatbot will be the next sentence given to the human user.



<p align="center"><img src="./lecture_17_slides/slide_18122_00-10-04.670.jpg" width="75%" alt="Lecture Video at 00:10:04.670" /></p>

And according to human evaluations, we could define the reward if the person is happy. If they are satisfied, you get a reward of 1. And if you are not happy or neutral, you get some other rewards.



<p align="center"><img src="./lecture_17_slides/slide_18840_00-10-28.628.jpg" width="75%" alt="Lecture Video at 00:10:28.628" /></p>

More specifically, for example, in the robotics domain, the task could be to fold clothes.



<p align="center"><img src="./lecture_17_slides/slide_19080_00-10-36.636.jpg" width="75%" alt="Lecture Video at 00:10:36.636" /></p>

We want the clothes folded nicely. The robot needs to decide its actions—how to move its end-effectors. Should it close or open its grippers in order to manipulate this cloth? According to human evaluations, if the cloth is properly folded, it gives the robot a reward of 1, and if the cloth is not folded, you give the reward of 0.

This is how you want to more concretely think about robot learning problems. It really is a way that allows the agents to interact with the world; it considers the effect of an action and also sequential decision-making problems. That is different from what people typically consider in computer vision, where we just need to predict outputs.



<p align="center"><img src="./lecture_17_slides/slide_20928_00-11-38.297.jpg" width="75%" alt="Lecture Video at 00:11:38.297" /></p>

This is about problem formulation. The question is how specific the reward needs to be designed. In many tasks, the reward can have many different types of specifications. Even for clothes folding, depending on the user's preference, the clothes can be folded in many different ways.

Some want the total area to be as small as possible; some want it to be as smooth as possible. There could be different types of rewards here. I am just talking in generic terms: if a person looks at the clothes, do they think this is folded or not? But more specifically, in terms of the reward design, there are actually a lot of nuances satisfying a specific need for a specific application.

This is how we are thinking about those robot learning problems that allow the agents to interact with the physical world.



<p align="center"><img src="./lecture_17_slides/slide_22736_00-12-38.624.jpg" width="75%" alt="Lecture Video at 00:12:38.624" /></p>

<p align="center"><img src="./lecture_17_slides/slide_23044_00-12-48.901.jpg" width="75%" alt="Lecture Video at 00:12:48.901" /></p>

<p align="center"><img src="./lecture_17_slides/slide_23258_00-12-56.041.jpg" width="75%" alt="Lecture Video at 00:12:56.041" /></p>

The physical world can give you, for example, high-dimensional RGB observation or RGBD observations. It could also include some other sensory data like tactile sensings.



<p align="center"><img src="./lecture_17_slides/slide_24190_00-13-27.139.jpg" width="75%" alt="Lecture Video at 00:13:27.139" /></p>

Essentially, the question we are trying to tackle is making sense of this unstructured real world. The real world can be very messy. The observations the robots are getting from the environments can only contain incomplete knowledge of the objects and the environments. There could be occlusions.

There could also be errors from the sensory data. And imperfect actions may also lead to failure; for example, the robots can try to grasp some objects, but their grasping behavior may not always be successful. Sometimes you will accidentally drop that object, which will also cause evolutions and unexpected changes of this environment. They will also need to have a perception system that is able to handle those scenarios.

Furthermore, this environment can change—it is dynamic, consisting of not just rigid objects but deformable objects like clothes, ropes, or granular media. There could be other agents, like dogs, or other kids, or other humans that are also in the same environments, messing up with the world.



<p align="center"><img src="./lecture_17_slides/slide_26308_00-14-37.810.jpg" width="75%" alt="Lecture Video at 00:14:37.810" /></p>

Your perception system needs to be able to cope with all those kinds of changes. It is considered, for example, tactile sensings, the audio information, the depth information, et cetera.



<p align="center"><img src="./lecture_17_slides/slide_26996_00-15-00.766.jpg" width="75%" alt="Lecture Video at 00:15:00.766" /></p>

The tactile information might tell you about whether a graph is stable or not; and the camera information tells you... About something that is more on the higher level, on the grand scheme of things, about the overall state of this environment.



<p align="center"><img src="./lecture_17_slides/slide_28100_00-15-37.603.jpg" width="75%" alt="Lecture Video at 00:15:37.603" /></p>

On the left is a very typical example you have already seen in computer vision, which is trying to do instance segmentation. What you are given is this 2D image. You have segments—different instances from this 2D image, like by drawing a contour over these 2D pixels. But what is different in the robotics domain, for example, on the right, is that the robot can, for example, be given one object.

This object seems to be maybe just one object or maybe a lot of pieces that are stacked into each other. So the robot has to know what type of actions will allow it to have better understanding, better perceptions about this environment. Is this one piece of an object or multiple pieces composed together?



<p align="center"><img src="./lecture_17_slides/slide_29934_00-16-38.797.jpg" width="75%" alt="Lecture Video at 00:16:38.797" /></p>

That is why robot vision is embodied, active, and also environmentally situated.



<p align="center"><img src="./lecture_17_slides/slide_30162_00-16-46.405.jpg" width="75%" alt="Lecture Video at 00:16:46.405" /></p>

By embodied, what we mean is robots have this kind of physical body that is directly experiencing the physical world. Their actions are part of a dynamic with the world that has immediate feedback on their own sensations.



<p align="center"><img src="./lecture_17_slides/slide_30646_00-17-02.554.jpg" width="75%" alt="Lecture Video at 00:17:02.554" /></p>

And active means the robots are active perceivers; it knows why it wishes to sense and chooses what to perceive and determines how and when and where to achieve that perception. You can move your head around. If you want to know what's behind this table, you can just move around to see what's behind the table. This is the active part, which is different from what people typically consider in computer vision; they are mostly working with a passively collected data set.



<p align="center"><img src="./lecture_17_slides/slide_31446_00-17-29.248.jpg" width="75%" alt="Lecture Video at 00:17:29.248" /></p>

The third point is about situated, where the robots are situated in the world. They do not deal with abstract descriptions but with the here and now of the world directly influencing the behavior of the systems.



<p align="center"><img src="./lecture_17_slides/slide_31866_00-17-43.262.jpg" width="75%" alt="Lecture Video at 00:17:43.262" /></p>

<p align="center"><img src="./lecture_17_slides/slide_31900_00-17-44.396.jpg" width="75%" alt="Lecture Video at 00:17:44.396" /></p>

Robots really have to understand, especially in clothing, there's a perception and action loop. It sees the world, understands its goals, and is able to act in the environment upon its perceptions. Sometimes, the robot does not have to know the full state of the environment. For example, if I am buttoning my shirt, I only have to know the local regions near that button for me to button that shirt.



<p align="center"><img src="./lecture_17_slides/slide_33004_00-18-21.233.jpg" width="75%" alt="Lecture Video at 00:18:21.233" /></p>

<p align="center"><img src="./lecture_17_slides/slide_33040_00-18-22.434.jpg" width="75%" alt="Lecture Video at 00:18:22.434" /></p>

<p align="center"><img src="./lecture_17_slides/slide_33558_00-18-39.718.jpg" width="75%" alt="Lecture Video at 00:18:39.718" /></p>

<p align="center"><img src="./lecture_17_slides/slide_33666_00-18-43.322.jpg" width="75%" alt="Lecture Video at 00:18:43.322" /></p>

Remember, earlier, we saw this image: the robot has to act upon this environment and get rewards from this environment. One very typical way of trying to solve this optimization problem is allowing the robots to interact with the world as extensively and as massively as possible.



<p align="center"><img src="./lecture_17_slides/slide_34392_00-19-07.546.jpg" width="75%" alt="Lecture Video at 00:19:07.546" /></p>

You just collect all the experience data and do these types of trials and errors.



<p align="center"><img src="./lecture_17_slides/slide_34418_00-19-08.413.jpg" width="75%" alt="Lecture Video at 00:19:08.413" /></p>

This allows the robots to understand that certain actions lead to high reward, while other actions lead to lower rewards. We can then pivot the agent's behaviors toward the actions that give the agent some higher rewards.



<p align="center"><img src="./lecture_17_slides/slide_35244_00-19-35.974.jpg" width="75%" alt="Lecture Video at 00:19:35.974" /></p>

I also want to be more specific in discussing the difference between reinforcement learning and supervised learning. This is a typical framework of how reinforcement learning looks like: you have the environment, which gives the agent some states. Agents generate actions, and environments give the agent feedback, which is the reward.



<p align="center"><img src="./lecture_17_slides/slide_36246_00-20-09.408.jpg" width="75%" alt="Lecture Video at 00:20:09.408" /></p>

Here is a typical image for supervised learning. You have the data set; the data set will input into the model as $x$, and this model will generate the prediction $y$. You will be able to calculate the loss according to the model's predictions versus the ground truth from this data set. So, this is a typical setup of supervised learning.



<p align="center"><img src="./lecture_17_slides/slide_37112_00-20-38.303.jpg" width="75%" alt="Lecture Video at 00:20:38.303" /></p>

This means there can be uncertainties and stochasticity in the environments, which leads to a stochastic behavior of these environments. This also gives these agents stochastic rewards, where the same action may not always lead to the same rewards. So, this is very different from supervised learning because we are dealing with an uncertain dynamical system.



<p align="center"><img src="./lecture_17_slides/slide_38468_00-21-23.548.jpg" width="75%" alt="Lecture Video at 00:21:23.548" /></p>

The second difference concerns the question of credit assignment. For supervised learning, you give the inputs, predict the outputs, and calculate the loss. You directly know what the mistakes are and what errors you are making by making a specific prediction.



<p align="center"><img src="./lecture_17_slides/slide_40184_00-22-20.806.jpg" width="75%" alt="Lecture Video at 00:22:20.806" /></p>

<p align="center"><img src="./lecture_17_slides/slide_41616_00-23-08.587.jpg" width="75%" alt="Lecture Video at 00:23:08.587" /></p>

<p align="center"><img src="./lecture_17_slides/slide_42662_00-23-43.488.jpg" width="75%" alt="Lecture Video at 00:23:43.488" /></p>

<p align="center"><img src="./lecture_17_slides/slide_43396_00-24-07.979.jpg" width="75%" alt="Lecture Video at 00:24:07.979" /></p>

But in the reinforcement learning or sequential decision-making domain, the rewards can be delayed. If you play the game of Go, only until the very end of this episode do you realize whether you are winning or losing. How do you properly assign the credits for all actions along this sequential decision making process? This is also another very tricky and important question people hope to answer using reinforcement learning.

The third thing is the differentiability of these dynamical systems. For example, for supervised learning, you have the inputs; you feed the inputs through the model, get the outputs, and calculate the loss. Everything along this process is differentiable, so you can directly gather gradients of the loss functions with respect to the parameters within the model.

However, that's typically not the case for reinforcement learning where environments can oftentimes be non-differentiable. That is also another difference. For supervised learning, whatever you predict doesn't influence other data points you are getting from this data set. But your actions will influence the next state you are getting in these sequential decision-making problems.

That is what makes this kind of reinforcement learning problem more nuanced than supervised learning. Here are some more specific examples. For example, playing Atari games, as I mentioned earlier. The goal could be to complete the game with the highest score.

The states would be the raw pixel inputs from the gaming screen, and the actions could be up, down, left, and right from the keyboard. We are trying to maximize rewards, which are the score increases and decreases at each time step. Some typical algorithms within this domain lie in fields like Q-learning or policy iterations. Here is an example of trying to learn this $Q$ function.

The $Q$ function essentially measures the discounted expected future accumulated rewards when you apply a specific action $a$ at a specific state $s$. You will be able to get these $Q$ functions through interactions with the gaming environment. Given these four actions, you can look at their $Q$ values and just execute the actions that give you the highest $Q$ values.

That is what allows you to do this type of decision making in this domain. If you are interested, you are very welcome to look at those algorithms in detail.



<p align="center"><img src="./lecture_17_slides/slide_45928_00-25-32.464.jpg" width="75%" alt="Lecture Video at 00:25:32.464" /></p>

This was developed by Google DeepMind, which is trying to develop these agents that are playing the game Breakout in this kind of Atari world. Just after 10 minutes of training, the robot, the agent can already touch the ball but oftentimes can still miss the ball quite often. After full hours of training, something interesting happens: the agents come up with a novel strategy, which possibly is not known to many of you.

This involves trying to push or bounce the ball back to create a tunnel on the left side of this wall.



<p align="center"><img src="./lecture_17_slides/slide_48196_00-26-48.139.jpg" width="75%" alt="Lecture Video at 00:26:48.139" /></p>

<p align="center"><img src="./lecture_17_slides/slide_48216_00-26-48.807.jpg" width="75%" alt="Lecture Video at 00:26:48.807" /></p>

It then pushes this ball on the upper side of the wall to achieve very efficient reductions of those bricks.



<p align="center"><img src="./lecture_17_slides/slide_48538_00-26-59.551.jpg" width="75%" alt="Lecture Video at 00:26:59.551" /></p>

This is the type of strategy that can be discovered by reinforcement learning. This is what's nice about it, because you allow the agents to do very extensive and comprehensive exploration and interactions with the world. It is totally possible for these reinforcement learning agents to discover some strategies that are better than even the best human players.



<p align="center"><img src="./lecture_17_slides/slide_49094_00-27-18.103.jpg" width="75%" alt="Lecture Video at 00:27:18.103" /></p>

A very typical example would be the game of Go. When AlphaGo came out in January 2016, it was also about the time when I was trying to decide what type of research directions I was going to go. The question then was: how does this $Q$-function specifically work?



<p align="center"><img src="./lecture_17_slides/slide_50598_00-28-08.286.jpg" width="75%" alt="Lecture Video at 00:28:08.286" /></p>

You can see this function takes as input the state $s$ and also the action $a$. This data represents the parameters of this $Q$ function, where $Q$ is instantiated as a neural network.



<p align="center"><img src="./lecture_17_slides/slide_50988_00-28-21.299.jpg" width="75%" alt="Lecture Video at 00:28:21.299" /></p>

<p align="center"><img src="./lecture_17_slides/slide_51174_00-28-27.505.jpg" width="75%" alt="Lecture Video at 00:28:27.505" /></p>

In this specific case, like I mentioned earlier, the state is the raw pixel inputs that you are actually getting from the gaming screen. The inputs could be these four steps—four frames—that are directly inputted to this $Q$-function. If you are dealing with images, a very straightforward way of instantiating this $Q$ function is to use convolutional neural networks.

You have convolutional layers, like shown in these orange blocks, and then you'll go through fully connected layers to directly derive the $Q$ value. In this case, because there are four discrete actions—maybe just left and right—but let's say there are four discrete actions: up and down, left and right. You will be able to have different $Q$ value estimations that are the results of this specific action $a$.

That is how you can use these $Q$ values to make decisions on what action to take that is most effective and maximizing this $Q$ value.



<p align="center"><img src="./lecture_17_slides/slide_52688_00-29-18.022.jpg" width="75%" alt="Lecture Video at 00:29:18.022" /></p>

<p align="center"><img src="./lecture_17_slides/slide_52996_00-29-28.299.jpg" width="75%" alt="Lecture Video at 00:29:28.299" /></p>

Later, we have AlphaGo Zero, which is essentially a simplified version of AlphaGo, no longer using any imitation learning for initialization. It was able to beat, at that time, the number one player, Ke Jie.



<p align="center"><img src="./lecture_17_slides/slide_54386_00-30-14.679.jpg" width="75%" alt="Lecture Video at 00:30:14.679" /></p>

<p align="center"><img src="./lecture_17_slides/slide_54736_00-30-26.357.jpg" width="75%" alt="Lecture Video at 00:30:26.357" /></p>

They then designed MuZero that not just does this kind of model-free reinforcement. learning, but it's able to learn a latent space dynamics model to plan over that gives you an even better performance.



<p align="center"><img src="./lecture_17_slides/slide_55622_00-30-55.920.jpg" width="75%" alt="Lecture Video at 00:30:55.920" /></p>

In November 2019, Lee Sedol, who was beaten by AlphaGo, announced his retirement. He realized there's just not possible, at that time, for any human players to beat the best AI agents out there.



<p align="center"><img src="./lecture_17_slides/slide_56196_00-31-15.073.jpg" width="75%" alt="Lecture Video at 00:31:15.073" /></p>

<p align="center"><img src="./lecture_17_slides/slide_57516_00-31-59.117.jpg" width="75%" alt="Lecture Video at 00:31:59.117" /></p>

This on the left is a work from ETH that was published in Science Robotics 2020. That essentially changed my mind about how useful reinforcement learning can be for real physical robots because before it was just mostly games like [INAUDIBLE]. In games, there's a lot of—you can just spawn as many games as possible. But for the real world, there's always sim-to-real gap, where you are training on one environment and you are also testing on another.

But for robots, if you train on a simulation, how much does the sim-to-real gap matter for the agents to generalize to the real environments? This paper really convinced me that sometimes, the sim-to-real gap just may not matter that much. We are not simulating the bushes; we are not simulating the snows. They can navigate into some very rough and challenging terrains.



<p align="center"><img src="./lecture_17_slides/slide_60342_00-33-33.411.jpg" width="75%" alt="Lecture Video at 00:33:33.411" /></p>

This next domain is about manipulations, where the robot has to manipulate objects in the real physical world. In 2019, when OpenAI touched upon robotics, they designed systems that were trying to do dexterous manipulations of Rubik's Cube. They were able to do the reinforcement learning in simulation and perform sim-to-real transfer that allowed these kind of robots to solve this Rubik's Cube.

One caveat is that their success rate was very low. Given that number, possibly the reliability is not very satisfying. It's all thanks to the developments of reinforcement learning. For manipulation, it is still limited to these kinds of very isolated domains, like working with this kind of isolated environment.



<p align="center"><img src="./lecture_17_slides/slide_63022_00-35-02.834.jpg" width="75%" alt="Lecture Video at 00:35:02.834" /></p>

This is actually some of the key challenges and bottlenecks of existing model-free reinforcement learning. It is mostly learned from trial and error with the environments, and it requires extensive interactions with the world.



<p align="center"><img src="./lecture_17_slides/slide_63478_00-35-18.049.jpg" width="75%" alt="Lecture Video at 00:35:18.049" /></p>

For example, AlphaGo Zero learns from 3,000 years of human knowledge in 40 days, which is amazing. But it still requires many, many years of computation—years of equivalent computations for the agents to learn.



<p align="center"><img src="./lecture_17_slides/slide_64326_00-35-46.344.jpg" width="75%" alt="Lecture Video at 00:35:46.344" /></p>

<p align="center"><img src="./lecture_17_slides/slide_64342_00-35-46.878.jpg" width="75%" alt="Lecture Video at 00:35:46.878" /></p>

Also, of course, if there is a sim-to-real gap, and you only can learn the model in the real environments, there are a lot of safety concerns.



<p align="center"><img src="./lecture_17_slides/slide_64552_00-35-53.885.jpg" width="75%" alt="Lecture Video at 00:35:53.885" /></p>

For example, here is an example showing this kind of learning progression of an agent that is controlling these humanoid robots to move forward.



<p align="center"><img src="./lecture_17_slides/slide_65274_00-36-17.975.jpg" width="75%" alt="Lecture Video at 00:36:17.975" /></p>

<p align="center"><img src="./lecture_17_slides/slide_65284_00-36-18.309.jpg" width="75%" alt="Lecture Video at 00:36:18.309" /></p>

It also has a very limited interpretability, and sometimes, it's very hard to correct things when things go wrong.



<p align="center"><img src="./lecture_17_slides/slide_65512_00-36-25.917.jpg" width="75%" alt="Lecture Video at 00:36:25.917" /></p>

We can imagine how the environment is going to change if we apply a specific action. So it's exactly this predictive capability that allows us humans to plan our behavior in achieving some specific targets. This predictive capability is also actually learned from we human's physical interactions and everyday experiences with the real physical world.



<p align="center"><img src="./lecture_17_slides/slide_66658_00-37-04.155.jpg" width="75%" alt="Lecture Video at 00:37:04.155" /></p>

<p align="center"><img src="./lecture_17_slides/slide_66728_00-37-06.490.jpg" width="75%" alt="Lecture Video at 00:37:06.490" /></p>

<p align="center"><img src="./lecture_17_slides/slide_66994_00-37-15.366.jpg" width="75%" alt="Lecture Video at 00:37:15.366" /></p>

For specific examples, we have a simulation. Typically, the simulation people use would be, for example, Isaac Gym or Isaac Sim developed by NVIDIA. There are essentially a bunch of rigid body simulations where the robot is just touching this kind of polygon type of representation, like a representation of the floor. It is not simulating, for example, the bushes; it is not simulating those snows.

But what people do is to randomize the simulated environments a lot and randomize the friction, the geometry, and many other physical parameters inside this environment. So the question is about what is the actual command? Conditioned on that high-level actions provided by human, the robot has to decide these kind of low-level actions. The low-level actions are typically, for example, the joint torque that are applied to each and every one of the joints on top of this robot.

As I mentioned, one biggest lesson I learned from this line of work on locomotion is that the simulation doesn't have to be perfect. As long as you randomize it enough, it can generalize very robustly in the real environments. But such a lesson hasn't really been generalized very well in the manipulation domain. In the manipulation domain, how accurate the simulation needs to be and how much does the sim-to-real gap matter is still a research question people hope to answer.

So there are regions where sim-to-real gap matters. There are other regions sim-to-real gap may not matter that much in the manipulation domain. If I understood your questions correctly, you are asking—there are still a person providing high-level commands to the robots. So the question is, can the robot come up with better plans than the human?

I can actually give you a more nuanced perspective. Although many of these videos seem very nice, there is a human operator operating the robots to choose which route to go. For example, let's say there's some kind of rough terrain or a pile of rocks. Humans can try to command the robot to go forward, trying to climb those kinds of rocks.

If that fails, humans can provide other high-level commands to get around this pile of rocks. There can be some kind of learning also on the human side in understanding the capabilities of those robots. This is why some videos look very nice: because humans select routes that show both the limits and the capabilities of low-level controllers. How to do that autonomously is a very interesting question, and people are doing research upon it.



<p align="center"><img src="./lecture_17_slides/slide_74470_00-41-24.815.jpg" width="75%" alt="Lecture Video at 00:41:24.815" /></p>

I've discussed some successful examples and the power of reinforcement learning, as well as its limitations. We still haven't seen very successful and wide-scale deployments of reinforcement learning in manipulation yet. Humans do not just learn from trials and error; we actually build a type of internal model. We are asking the question: can we actually learn models from the robot's interactions with environments, and use that model for better physical interactions?



<p align="center"><img src="./lecture_17_slides/slide_75384_00-41-55.312.jpg" width="75%" alt="Lecture Video at 00:41:55.312" /></p>

<p align="center"><img src="./lecture_17_slides/slide_75406_00-41-56.046.jpg" width="75%" alt="Lecture Video at 00:41:56.046" /></p>

Specifically, what we are touching upon is how we can learn approximations of the real physical world.



<p align="center"><img src="./lecture_17_slides/slide_75992_00-42-15.599.jpg" width="75%" alt="Lecture Video at 00:42:15.599" /></p>

We can use this essentially as a forward model that predicts the next state, given the current state and action.



<p align="center"><img src="./lecture_17_slides/slide_77282_00-42-58.642.jpg" width="75%" alt="Lecture Video at 00:42:58.642" /></p>

<p align="center"><img src="./lecture_17_slides/slide_78184_00-43-28.739.jpg" width="75%" alt="Lecture Video at 00:43:28.739" /></p>

<p align="center"><img src="./lecture_17_slides/slide_78394_00-43-35.746.jpg" width="75%" alt="Lecture Video at 00:43:35.746" /></p>

<p align="center"><img src="./lecture_17_slides/slide_78472_00-43-38.349.jpg" width="75%" alt="Lecture Video at 00:43:38.349" /></p>

<p align="center"><img src="./lecture_17_slides/slide_78562_00-43-41.352.jpg" width="75%" alt="Lecture Video at 00:43:41.352" /></p>

<p align="center"><img src="./lecture_17_slides/slide_78644_00-43-44.088.jpg" width="75%" alt="Lecture Video at 00:43:44.088" /></p>

<p align="center"><img src="./lecture_17_slides/slide_78890_00-43-52.296.jpg" width="75%" alt="Lecture Video at 00:43:52.296" /></p>

<p align="center"><img src="./lecture_17_slides/slide_79460_00-44-11.315.jpg" width="75%" alt="Lecture Video at 00:44:11.315" /></p>

The problem for us is planning, which is essentially an inverse of this forward model. Planning is to give the current state and the target states and come up with the action that allows the robot to achieve those target states. Given the current state shown in blue, we have targets shown in red. This allows us to know what actions can guide us closer to the targets.

Obviously, the model may not be accurate enough, so we typically only execute the first action and obtain new states from the environment. We can then re-optimize the action sequence using gradient descent or any other optimization technique for these trajectory optimizations. One key question has always been: what should be the right representation of the environment?

What is the right and most effective state representation $s$? And how can we learn this model based on the state representation? Over the years, there have been many different investigations into choosing or investigating different types of state representations.



<p align="center"><img src="./lecture_17_slides/slide_80482_00-44-45.416.jpg" width="75%" alt="Lecture Video at 00:44:45.416" /></p>

This was called Deep Visual Foresight, which set off some initial works in the domain of world models.



<p align="center"><img src="./lecture_17_slides/slide_81150_00-45-07.705.jpg" width="75%" alt="Lecture Video at 00:45:07.705" /></p>

<p align="center"><img src="./lecture_17_slides/slide_81710_00-45-26.390.jpg" width="75%" alt="Lecture Video at 00:45:26.390" /></p>

This is about pixel dynamics.



<p align="center"><img src="./lecture_17_slides/slide_81872_00-45-31.795.jpg" width="75%" alt="Lecture Video at 00:45:31.795" /></p>

<p align="center"><img src="./lecture_17_slides/slide_81990_00-45-35.733.jpg" width="75%" alt="Lecture Video at 00:45:35.733" /></p>

People can also use keypoints as a representation of environments to learn keypoint dynamics models. Here, we can track the movement of the keypoints on top of this box over 3D space and also model the dynamics of those keypoints resulting from some Actions. So besides using keypoints, what if you encountered some objects with even higher degrees of freedom?



<p align="center"><img src="./lecture_17_slides/slide_82974_00-46-08.565.jpg" width="75%" alt="Lecture Video at 00:46:08.565" /></p>

If you go one level finer, you can also represent those objects using a set of particles, essentially a set of points.



<p align="center"><img src="./lecture_17_slides/slide_86396_00-48-02.746.jpg" width="75%" alt="Lecture Video at 00:48:02.746" /></p>

This forward model can allow the robots to do inverse decision making that handles a wide range of granular objects of different granular sizes. Obviously, this model not just generalizes to different granular pieces of different granular sizes; it can also change to different target configurations. Here, you will very quickly realize what the target configurations are.

The robot has to come up with a strategy that performs nontrivial redistributions of the granular pieces. After the redistribution, it has to align the fine-grained details with the targets, like shape. You want to accomplish this like a pile rearrangement task. The task here is actually to rearrange these granular pieces into different letter shapes, all the way from letter A to letter Z.

This is actually a highly nontrivial task. Going beyond that, we also have subsequent work, which I was also involved in and done when I was here at Stanford. We designed these kind of dumpling-making robots, equipping the robots with 15 different 3D-printed tools.



<p align="center"><img src="./lecture_17_slides/slide_86736_00-48-14.091.jpg" width="75%" alt="Lecture Video at 00:48:14.091" /></p>

<p align="center"><img src="./lecture_17_slides/slide_86824_00-48-17.027.jpg" width="75%" alt="Lecture Video at 00:48:17.027" /></p>

We have four RGBD cameras looking at the environment to do a reconstruction of the geometry of the store.



<p align="center"><img src="./lecture_17_slides/slide_87012_00-48-23.300.jpg" width="75%" alt="Lecture Video at 00:48:23.300" /></p>

The robots will have to decide what tool to use and what action to take to get this dough into a dumpling.



<p align="center"><img src="./lecture_17_slides/slide_87326_00-48-33.777.jpg" width="75%" alt="Lecture Video at 00:48:33.777" /></p>

<p align="center"><img src="./lecture_17_slides/slide_87456_00-48-38.115.jpg" width="75%" alt="Lecture Video at 00:48:38.115" /></p>

The key enabling factor is also this forward predictive model represented using particles.



<p align="center"><img src="./lecture_17_slides/slide_87508_00-48-39.850.jpg" width="75%" alt="Lecture Video at 00:48:39.850" /></p>

<p align="center"><img src="./lecture_17_slides/slide_87538_00-48-40.851.jpg" width="75%" alt="Lecture Video at 00:48:40.851" /></p>

<p align="center"><img src="./lecture_17_slides/slide_87568_00-48-41.852.jpg" width="75%" alt="Lecture Video at 00:48:41.852" /></p>

<p align="center"><img src="./lecture_17_slides/slide_87598_00-48-42.853.jpg" width="75%" alt="Lecture Video at 00:48:42.853" /></p>

<p align="center"><img src="./lecture_17_slides/slide_87628_00-48-43.854.jpg" width="75%" alt="Lecture Video at 00:48:43.854" /></p>

Here, the red dots are representing the shape of the tool, and the blue dots are representing the shape of the objects. The first row is our model's open-loop prediction, and the second row is what actually happens in the real environment.



<p align="center"><img src="./lecture_17_slides/slide_87934_00-48-54.064.jpg" width="75%" alt="Lecture Video at 00:48:54.064" /></p>

<p align="center"><img src="./lecture_17_slides/slide_88016_00-48-56.800.jpg" width="75%" alt="Lecture Video at 00:48:56.800" /></p>

<p align="center"><img src="./lecture_17_slides/slide_88250_00-49-04.608.jpg" width="75%" alt="Lecture Video at 00:49:04.608" /></p>

What's interesting about this video is there's a person constantly perturbing the robot from doing its job. The robot takes real-time visual feedback from this environment to achieve real-time understanding of the shape of the dough. Humans are just so annoying, adding pieces, folding the dough. The robot is very robust to this external disturbance, continuing its progress in doing the task.

Here's what's interesting: after the robot casts a circle, the human shows no mercy, destroying everything. This really shows the patience and also the robustness of our systems with this type of external disturbance. In the end, the robot will place the skin on top of this dumpling clip, move the fillings on top of this dumpling skin, and use a hook to close the dumpling clip in order.

To use this general purpose robot equipped with 15 general purpose tools to make a dumpling out of a dough.



<p align="center"><img src="./lecture_17_slides/slide_91638_00-50-57.654.jpg" width="75%" alt="Lecture Video at 00:50:57.654" /></p>

This is about how we can learn the model and how that model can be useful for downstream model-based planning. For this specific case, if we want to describe it more rigorously, we are not using reinforcement learning. Some people also call it model-based reinforcement learning, depending on which background you are coming from. You can either call it model learning and model-based planning, or you can also call it model-based reinforcement learning.



<p align="center"><img src="./lecture_17_slides/slide_93118_00-51-47.037.jpg" width="75%" alt="Lecture Video at 00:51:47.037" /></p>

In this specific case, the high-level planning and low-level decision making are done by two different models. Back then, this work was done in 2023, at a time when visual language models weren't very powerful. What we did was allow a human operator to do the data collection—the demonstration of the task for 10 times. We used that data to train this kind of classifier to classify what tool to use.

This allows us to actually jump back and forth over this chain. In this specific case, what we have been doing is a combination of sampling-based trajectory optimization versus policy learning.



<p align="center"><img src="./lecture_17_slides/slide_95906_00-53-20.063.jpg" width="75%" alt="Lecture Video at 00:53:20.063" /></p>

We are given the current state of the dough and our forward predictive models. This allows us to sample a bunch of actions and a bunch of tools to predict what the evolution of the shape of that dough will be. We then compare the model's prediction with the targets we hope to achieve, which is similar to what I showed earlier. For example, our model predicts the shape of the dough will go into these green dots, but the target is these red dots.

We do this by comparing their distance, and that allows us to select the most effective actions that can get us to the target as close as possible. We can perform a lot of samples like this, but sampling during test time is very time-consuming. Therefore, we do this type of sampling in an offline fashion, which gives us a dataset. We use that dataset to train a policy that can be inferred using a very short period of time when doing inference during the test time.

Even though the policy is learned by distilling from our model's predictions over a huge amount of samples, there is still a neural network as the policies. For this specific work, there was no physics-based simulation at all. We actually have a baseline that uses a state-of-the-art deformable object simulator, which is called MPM (Material Point Methods).



<p align="center"><img src="./lecture_17_slides/slide_98034_00-54-31.067.jpg" width="75%" alt="Lecture Video at 00:54:31.067" /></p>

<p align="center"><img src="./lecture_17_slides/slide_98056_00-54-31.801.jpg" width="75%" alt="Lecture Video at 00:54:31.801" /></p>

Our model's prediction aligns very well with the ground truth, which is much more accurate than whatever physics-based simulator is out there.



<p align="center"><img src="./lecture_17_slides/slide_98582_00-54-49.352.jpg" width="75%" alt="Lecture Video at 00:54:49.352" /></p>

What we have discussed here is model learning and how this learned model can be effective for downstream model-based planning.



<p align="center"><img src="./lecture_17_slides/slide_98820_00-54-57.294.jpg" width="75%" alt="Lecture Video at 00:54:57.294" /></p>

The next category of algorithms is imitation learning. You use that data to do supervised learning to train this model and then using this model to do model-based planning. Instead of just using supervised learning to train the model, people are also asking if we can do supervised learning for the policies.



<p align="center"><img src="./lecture_17_slides/slide_100196_00-55-43.206.jpg" width="75%" alt="Lecture Video at 00:55:43.206" /></p>

This is trying to learn this kind of policy, taking the states as inputs that predict the actions. All of these kinds of learning signals and learning procedures are done through large-scale collected data, from human demoing to the robots how a task needs to be done.



<p align="center"><img src="./lecture_17_slides/slide_100960_00-56-08.698.jpg" width="75%" alt="Lecture Video at 00:56:08.698" /></p>

Learning from demonstration is, of course, not new. It has been investigated for decades.



<p align="center"><img src="./lecture_17_slides/slide_101442_00-56-24.781.jpg" width="75%" alt="Lecture Video at 00:56:24.781" /></p>

One of the earliest classic imitation learning algorithms is called behavior cloning, essentially trying to learn a mapping that maps an observation $o$ into the action $a$. This policy is represented using a function $\pi$ parameterized by $\theta$. One key issue for behavior cloning is called cascading error. In this case, your error can accumulate and be amplified over time.

This deviation will then lead the policy to make an even larger error, and this error will be amplified over the temporal horizon. This results in a trajectory that can deviate quite a lot from the original demonstration trajectories.



<p align="center"><img src="./lecture_17_slides/slide_103570_00-57-35.785.jpg" width="75%" alt="Lecture Video at 00:57:35.785" /></p>

When people are trying to make imitation learning work, they often follow a specific pipeline: First, we have the demonstrations collected by experts. Then we use that as training data for supervised learning to train this policy. We roll out the policy in the real environment and observe potential failure cases. In response to these failures, we either collect additional data or provide corrective behaviors.

This is a typical lifecycle when developing any imitation learning agents or algorithms in the real physical world.



<p align="center"><img src="./lecture_17_slides/slide_105556_00-58-42.051.jpg" width="75%" alt="Lecture Video at 00:58:42.051" /></p>

This led to a class of algorithms called Inverse Reinforcement Learning (IRL).



<p align="center"><img src="./lecture_17_slides/slide_106238_00-59-04.807.jpg" width="75%" alt="Lecture Video at 00:59:04.807" /></p>

<p align="center"><img src="./lecture_17_slides/slide_106312_00-59-07.277.jpg" width="75%" alt="Lecture Video at 00:59:07.277" /></p>

Some earlier success examples were developed at Stanford by Pieter Abbeel and Andrew Ng.



<p align="center"><img src="./lecture_17_slides/slide_106512_00-59-13.950.jpg" width="75%" alt="Lecture Video at 00:59:13.950" /></p>

These systems allowed them to control helicopters to perform very complex behaviors.



<p align="center"><img src="./lecture_17_slides/slide_106582_00-59-16.286.jpg" width="75%" alt="Lecture Video at 00:59:16.286" /></p>

<p align="center"><img src="./lecture_17_slides/slide_107232_00-59-37.974.jpg" width="75%" alt="Lecture Video at 00:59:37.974" /></p>

<p align="center"><img src="./lecture_17_slides/slide_107312_00-59-40.643.jpg" width="75%" alt="Lecture Video at 00:59:40.643" /></p>

Instead of learning an explicit policy shown on the left, these methods directly map from the observation $o$ through to the action $a$.



<p align="center"><img src="./lecture_17_slides/slide_108306_01-00-13.810.jpg" width="75%" alt="Lecture Video at 01:00:13.810" /></p>

The robot is then able to come up with strategies, distilling these policies from the demonstrations in performing content-rich manipulation tasks.



<p align="center"><img src="./lecture_17_slides/slide_108952_01-00-35.365.jpg" width="75%" alt="Lecture Video at 01:00:35.365" /></p>

Some of the recent success in robot learning as a whole is the result of work called Diffusion Policy, which again takes advances from generative models.



<p align="center"><img src="./lecture_17_slides/slide_109398_01-00-50.246.jpg" width="75%" alt="Lecture Video at 01:00:50.246" /></p>

<p align="center"><img src="./lecture_17_slides/slide_109418_01-00-50.913.jpg" width="75%" alt="Lecture Video at 01:00:50.913" /></p>

For this implicit behavior cloning, people are drawing inspiration from the development of energy-based models. Energy-based models are a type of generative model developed in the deep learning community.



<p align="center"><img src="./lecture_17_slides/slide_109700_01-01-00.323.jpg" width="75%" alt="Lecture Video at 01:01:00.323" /></p>

Another class of more powerful models is called diffusion models. People are also trying to use diffusion models as a $\text{policy}$ function class to allow agents to inherit the benefits and properties from those diffusion models. This work was originally done at Columbia, where I am right now. The leads, like the PI of this work, moved to Stanford.

You can see many of the work I selected has a lot of roots here at Stanford. One department is currently at Stanford. Meaning you collect the data in the morning, you train the $\text{policy}$ in the afternoon, and you can have a working $\text{policy}$ in the real physical world. Still, imitation learning is the most efficient way for you to get a $\text{policy}$ that can do something interesting in the real physical world.



<p align="center"><img src="./lecture_17_slides/slide_113126_01-02-54.637.jpg" width="75%" alt="Lecture Video at 01:02:54.637" /></p>

<p align="center"><img src="./lecture_17_slides/slide_113498_01-03-07.049.jpg" width="75%" alt="Lecture Video at 01:03:07.049" /></p>

So this is about imitation learning. So any questions?



<p align="center"><img src="./lecture_17_slides/slide_113994_01-03-23.599.jpg" width="75%" alt="Lecture Video at 01:03:23.599" /></p>

And of course, this is a very involved domain; actually, for each one of these items, you can have a whole course around it. For today's lecture, I am just skimming through them very quickly. I will only tell you the gist, the high-level knowledge needed by looking at those terms.



<p align="center"><img src="./lecture_17_slides/slide_114708_01-03-47.423.jpg" width="75%" alt="Lecture Video at 01:03:47.423" /></p>

<p align="center"><img src="./lecture_17_slides/slide_114720_01-03-47.824.jpg" width="75%" alt="Lecture Video at 01:03:47.824" /></p>

<p align="center"><img src="./lecture_17_slides/slide_114756_01-03-49.025.jpg" width="75%" alt="Lecture Video at 01:03:49.025" /></p>

For robotic foundation models, it is a type of model that is very similar to reinforcement learning or imitation learning in its function class. There is no explicit representation for states or this kind of model.



<p align="center"><img src="./lecture_17_slides/slide_115278_01-04-06.442.jpg" width="75%" alt="Lecture Video at 01:04:06.442" /></p>

<p align="center"><img src="./lecture_17_slides/slide_115436_01-04-11.714.jpg" width="75%" alt="Lecture Video at 01:04:11.714" /></p>

That is still like a representative—it can be very nicely represented using these figures. You have this agent, which is a $\text{policy}$ that takes the current state and also the goal as inputs. You are trying to generate these actions that can be executed in the real physical world.



<p align="center"><img src="./lecture_17_slides/slide_115888_01-04-26.796.jpg" width="75%" alt="Lecture Video at 01:04:26.796" /></p>

But you might say that this is very similar to imitation learning and reinforcement learning. So what is special about robotic foundation models?



<p align="center"><img src="./lecture_17_slides/slide_116524_01-04-48.017.jpg" width="75%" alt="Lecture Video at 01:04:48.017" /></p>

<p align="center"><img src="./lecture_17_slides/slide_116616_01-04-51.087.jpg" width="75%" alt="Lecture Video at 01:04:51.087" /></p>

Meaning it is a $\text{policy}$, but it needs to generalize much better than a $\text{policy}$ that just works for one specific task.



<p align="center"><img src="./lecture_17_slides/slide_116686_01-04-53.422.jpg" width="75%" alt="Lecture Video at 01:04:53.422" /></p>

<p align="center"><img src="./lecture_17_slides/slide_116702_01-04-53.956.jpg" width="75%" alt="Lecture Video at 01:04:53.956" /></p>

<p align="center"><img src="./lecture_17_slides/slide_117172_01-05-09.639.jpg" width="75%" alt="Lecture Video at 01:05:09.639" /></p>

What we hope to achieve with robotic foundation models is that the synthesized action may not always be the optimal actions conditioned by the observation and the task. But the general trajectory will always be beautiful and reasonable to execute in the real physical world. Beautiful means you shouldn't use any jiggling motions; it should be smooth and continuous.

Reasonable means you should listen to the language instructions given to the robots.



<p align="center"><img src="./lecture_17_slides/slide_117994_01-05-37.066.jpg" width="75%" alt="Lecture Video at 01:05:37.066" /></p>

Obviously, there are also many different names describing exactly the same thing. Some people call it Vision-Language Action Models like VLAs. Some people call it large behavior models.



<p align="center"><img src="./lecture_17_slides/slide_118706_01-06-00.823.jpg" width="75%" alt="Lecture Video at 01:06:00.823" /></p>

So this area is actually quite noisy. Noisy meaning it's very, very hard to quantify the progress of different kinds of robotic foundation models. Because you're calling it a foundation model, what does that mean? That means you expect this model to generalize very broadly over a wide range of scenarios.

If that's your expectation, you actually need significant evidence to show it actually generalizes broadly. So that's why evaluation and quantitative measurements of their progress is very challenging.



<p align="center"><img src="./lecture_17_slides/slide_119596_01-06-30.519.jpg" width="75%" alt="Lecture Video at 01:06:30.519" /></p>

But still, by looking at their empirical videos, you can still see a lot of very interesting and concrete progress over the past two years.



<p align="center"><img src="./lecture_17_slides/slide_119854_01-06-39.128.jpg" width="75%" alt="Lecture Video at 01:06:39.128" /></p>

A lot of the earlier investigation starts with $\text{RT-1}$, which was released in December of 2022.



<p align="center"><img src="./lecture_17_slides/slide_120102_01-06-47.403.jpg" width="75%" alt="Lecture Video at 01:06:47.403" /></p>

<p align="center"><img src="./lecture_17_slides/slide_120274_01-06-53.142.jpg" width="75%" alt="Lecture Video at 01:06:53.142" /></p>

<p align="center"><img src="./lecture_17_slides/slide_120308_01-06-54.276.jpg" width="75%" alt="Lecture Video at 01:06:54.276" /></p>

<p align="center"><img src="./lecture_17_slides/slide_120346_01-06-55.544.jpg" width="75%" alt="Lecture Video at 01:06:55.544" /></p>

<p align="center"><img src="./lecture_17_slides/slide_120674_01-07-06.489.jpg" width="75%" alt="Lecture Video at 01:07:06.489" /></p>

And actually, this year there's a boost. There's an [INAUDIBLE] first of a lot of foundation models, like Helix, Hi-Robot, Gemini Robotics, $\pi\text{-0.5}$, et cetera. Due to the time, I clearly cannot go into the details of all these models.



<p align="center"><img src="./lecture_17_slides/slide_121932_01-07-48.464.jpg" width="75%" alt="Lecture Video at 01:07:48.464" /></p>

If you are interested, please go and watch it.



<p align="center"><img src="./lecture_17_slides/slide_122308_01-08-01.010.jpg" width="75%" alt="Lecture Video at 01:08:01.010" /></p>

$\pi\text{-Zero}$ was first released in October 2024. It can handle cloth folding and box folding and many other different types of manipulation tasks at a very reliable manner.



<p align="center"><img src="./lecture_17_slides/slide_123042_01-08-25.501.jpg" width="75%" alt="Lecture Video at 01:08:25.501" /></p>

And here is how the framework actually looks like on a high level.



<p align="center"><img src="./lecture_17_slides/slide_123170_01-08-29.772.jpg" width="75%" alt="Lecture Video at 01:08:29.772" /></p>

On the left are data sets. For any model to be called the foundation model, it needs fuel for that foundation model. And that fuel is data.



<p align="center"><img src="./lecture_17_slides/slide_123826_01-08-51.660.jpg" width="75%" alt="Lecture Video at 01:08:51.660" /></p>

They use this data to do pre-training.



<p align="center"><img src="./lecture_17_slides/slide_125040_01-09-32.167.jpg" width="75%" alt="Lecture Video at 01:09:32.167" /></p>

And this is the pre-training stage. With a base model, base model can give you some reasonable baseline performance. They are evaluating their whole systems over three different categories.



<p align="center"><img src="./lecture_17_slides/slide_126106_01-10-07.736.jpg" width="75%" alt="Lecture Video at 01:10:07.736" /></p>

The first one is directly using their base model.



<p align="center"><img src="./lecture_17_slides/slide_126542_01-10-22.284.jpg" width="75%" alt="Lecture Video at 01:10:22.284" /></p>

For different tasks, but slightly more complicated, you can do post-training to allow the base model to further improve on those in-distribution tasks.



<p align="center"><img src="./lecture_17_slides/slide_126842_01-10-32.294.jpg" width="75%" alt="Lecture Video at 01:10:32.294" /></p>

And for unseen tasks, typically you have to do post-training by collecting those task-specific data and fine-tune your pre-trained model on these tasks for you to be performant.



<p align="center"><img src="./lecture_17_slides/slide_127162_01-10-42.972.jpg" width="75%" alt="Lecture Video at 01:10:42.972" /></p>

The $\pi\text{-Zero}$ model is actually open sourced, and you can just download the checkpoints. The students in my lab have already been starting to play with their models and trying to do post-training. We're starting to see some very promising results. If you are interested, I highly encourage you to try it.

You are essentially asking about the efficiency of existing robotic foundation models. There is a lot of reasons why the $\text{policy}$ is actually slower than humans. One of the major reasons is adapted from how the data was collected, specifically how the demonstration data was collected. Human teleoperation is actually slower than humans just using their hands to do these tasks, even if you have gone through hours of training.

There will be occlusion. There are a lot of caveats and inefficiencies of the current data collection regimes. That is why the $\text{policy}$ directly trained on those data turned out to be slower than human speeds. That's why there are a lot of investigations in how we can do these kinds of data collections to be even more efficient, to be at human speeds.

This is actually a very active research direction. In those types of scenarios, I don't believe one gigantic $\text{policy}$ is able to adapt to those scenarios.



<p align="center"><img src="./lecture_17_slides/slide_131832_01-13-18.794.jpg" width="75%" alt="Lecture Video at 01:13:18.794" /></p>

This process started with a pre-trained vision language model. That's why there is already a lot of semantic knowledge that are learned through this large-scale pre-training using this vision language data. I just have to fine-tune this model with those robot data to make sure it can also generalize not as a semantic level, but also as an action level.



<p align="center"><img src="./lecture_17_slides/slide_132840_01-13-52.428.jpg" width="75%" alt="Lecture Video at 01:13:52.428" /></p>

<p align="center"><img src="./lecture_17_slides/slide_132892_01-13-54.163.jpg" width="75%" alt="Lecture Video at 01:13:54.163" /></p>

<p align="center"><img src="./lecture_17_slides/slide_133042_01-13-59.168.jpg" width="75%" alt="Lecture Video at 01:13:59.168" /></p>

At the end, because we are running out of time, I will discuss some of the remaining challenges, especially along the development of robot learning models. One of the major challenges the whole community recognizes is evaluation.



<p align="center"><img src="./lecture_17_slides/slide_133218_01-14-05.040.jpg" width="75%" alt="Lecture Video at 01:14:05.040" /></p>

<p align="center"><img src="./lecture_17_slides/slide_133302_01-14-07.843.jpg" width="75%" alt="Lecture Video at 01:14:07.843" /></p>

Evaluation currently is primarily done in the real world.



<p align="center"><img src="./lecture_17_slides/slide_133612_01-14-18.187.jpg" width="75%" alt="Lecture Video at 01:14:18.187" /></p>

For example, this picture shows Google Robotics, which has a grid of teleoperating Alpha systems that they are doing data collection and also evaluation.



<p align="center"><img src="./lecture_17_slides/slide_133708_01-14-21.390.jpg" width="75%" alt="Lecture Video at 01:14:21.390" /></p>

Real-world evaluation is both costly and noisy. Their words to me was, for evaluation, they have large enough budget such that they can still make progress. Even the friction parameters of any factor can make a huge difference in how robust your downstream $\text{policy}$ is.



<p align="center"><img src="./lecture_17_slides/slide_134532_01-14-48.884.jpg" width="75%" alt="Lecture Video at 01:14:48.884" /></p>

This is very costly, and they have to wait for two days for the results to come back. Currently, there's very weak correlation between the training loss and real-world success rates. This is another important caveat and difference between supervised learning and this kind of sequential decision making, or $\text{policy}$ learning. For supervised learning, your training loss directly measures how good your model is.



<p align="center"><img src="./lecture_17_slides/slide_135730_01-15-28.857.jpg" width="75%" alt="Lecture Video at 01:15:28.857" /></p>

For long horizon task execution, your $\text{policy}$ can actually be worse.



<p align="center"><img src="./lecture_17_slides/slide_136240_01-15-45.874.jpg" width="75%" alt="Lecture Video at 01:15:45.874" /></p>

<p align="center"><img src="./lecture_17_slides/slide_136272_01-15-46.942.jpg" width="75%" alt="Lecture Video at 01:15:46.942" /></p>

People have to rely on real-world evaluations. So the question is, what about doing the evaluation in simulated environments?



<p align="center"><img src="./lecture_17_slides/slide_136824_01-16-05.360.jpg" width="75%" alt="Lecture Video at 01:16:05.360" /></p>

Now, people are trying to come up with these expensive simulated environments, trying to do evaluation and measurements of robot policies. Obviously, there are their own issues, especially with regard to sim-to-real gap. How can you do very accurate simulation of rigid body, deformable object, and clothes?



<p align="center"><img src="./lecture_17_slides/slide_137216_01-16-18.440.jpg" width="75%" alt="Lecture Video at 01:16:18.440" /></p>

They have good correlations with real-world performance. Asset is also another major issue, where large-scale generalization and generation of those assets is a huge pain.



<p align="center"><img src="./lecture_17_slides/slide_137580_01-16-30.586.jpg" width="75%" alt="Lecture Video at 01:16:30.586" /></p>

I can elaborate, but maybe after the lectures.



<p align="center"><img src="./lecture_17_slides/slide_137666_01-16-33.455.jpg" width="75%" alt="Lecture Video at 01:16:33.455" /></p>

How do you digitize the real world is an issue.



<p align="center"><img src="./lecture_17_slides/slide_137910_01-16-41.597.jpg" width="75%" alt="Lecture Video at 01:16:41.597" /></p>

How to do procedural generations of realistic and diverse things are all issues when using simulation to do evaluations for robot learning policies.



<p align="center"><img src="./lecture_17_slides/slide_138006_01-16-44.800.jpg" width="75%" alt="Lecture Video at 01:16:44.800" /></p>

We want to have this platform, meaning any progress on that benchmark or platform, means progress in robot learnings.



<p align="center"><img src="./lecture_17_slides/slide_138734_01-17-09.091.jpg" width="75%" alt="Lecture Video at 01:17:09.091" /></p>

<p align="center"><img src="./lecture_17_slides/slide_138876_01-17-13.829.jpg" width="75%" alt="Lecture Video at 01:17:13.829" /></p>

<p align="center"><img src="./lecture_17_slides/slide_138888_01-17-14.229.jpg" width="75%" alt="Lecture Video at 01:17:14.229" /></p>

We talk about how to build these foundational policies that can also be investigations into how to build a foundational world model.



<p align="center"><img src="./lecture_17_slides/slide_138916_01-17-15.163.jpg" width="75%" alt="Lecture Video at 01:17:15.163" /></p>

People are especially now collecting large-scale, action-conditioned robot interaction data to train this foundation policy. But there is a lot of dynamics knowledge embedded in those data.



<p align="center"><img src="./lecture_17_slides/slide_139360_01-17-29.978.jpg" width="75%" alt="Lecture Video at 01:17:29.978" /></p>

If we just use those data for policy learning, that would be such a waste.



<p align="center"><img src="./lecture_17_slides/slide_139686_01-17-40.856.jpg" width="75%" alt="Lecture Video at 01:17:40.856" /></p>

<p align="center"><img src="./lecture_17_slides/slide_139782_01-17-44.059.jpg" width="75%" alt="Lecture Video at 01:17:44.059" /></p>

There are existing works that are thinking about building this kind of foundational world models along a direction. There are some very interesting characteristics you might think about: Do you want it to be 3D? Do you want structural prior? How much learning versus how much physics?

And how you can correlate with the real physical world. I think we are about time, so I will end it here.



<p align="center"><img src="./lecture_17_slides/slide_140516_01-18-08.550.jpg" width="75%" alt="Lecture Video at 01:18:08.550" /></p>

<p align="center"><img src="./lecture_17_slides/slide_140540_01-18-09.351.jpg" width="75%" alt="Lecture Video at 01:18:09.351" /></p>

<p align="center"><img src="./lecture_17_slides/slide_140880_01-18-20.696.jpg" width="75%" alt="Lecture Video at 01:18:20.696" /></p>

Next lectures will be human-centered AI. That will be the end of today's lecture.



<p align="center"><img src="./lecture_17_slides/slide_141084_01-18-27.502.jpg" width="75%" alt="Lecture Video at 01:18:27.502" /></p>

Thank you so much.



