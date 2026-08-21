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

# Stanford CS231N Deep Learning for Computer Vision | Spring 2025 | Lecture 18: Human-Centered AI


<p align="center"><img src="./lecture_18_slides/slide_4_00-00-00.133.jpg" width="75%" alt="Lecture Video at 00:00:00.133" /></p>

Welcome to the last lecture of the quarter for CS231N, and it was great to see you guys at the beginning and now at the end. This lecture is a little bit of a departure. We're not going to teach any new materials in terms of algorithms.



<p align="center"><img src="./lecture_18_slides/slide_1898_00-01-03.266.jpg" width="75%" alt="Lecture Video at 00:01:03.266" /></p>

The title of this slide or this lecture is what we see and what we value: AI with the human perspective. I know that some of you have already heard about this. It's really the beginning—the origin of vision, both in terms of evolution as well as in terms of our technology. We did talk about the first slide that came to the animal world back 540 million years ago.

That was when animals, or trilobites to be specific, developed photosensitive cells to glean what the outer world is about.



<p align="center"><img src="./lecture_18_slides/slide_4212_00-02-20.400.jpg" width="75%" alt="Lecture Video at 00:02:20.400" /></p>

You wouldn't be surprised that vision is still to this day a primary sensory intelligent system in many, many animals. Not all animals use vision, admittedly, but many do. That's also one of the primary sensory systems for humans. We use vision to do everything from survival to work to entertainment to socialization to learning, development, and many other things.

So that's the recap or summary of evolution.



<p align="center"><img src="./lecture_18_slides/slide_5454_00-03-01.800.jpg" width="75%" alt="Lecture Video at 00:03:01.800" /></p>

This was very in line with the history of AI, where we tend to have clarity of the North Star but underestimate how long it would take. We are still probably experiencing that today, but a lot has happened.



<p align="center"><img src="./lecture_18_slides/slide_6522_00-03-37.400.jpg" width="75%" alt="Lecture Video at 00:03:37.400" /></p>

From enabling self-driving cars to understanding images to the generative AI revolution, we're seeing vision is playing a huge role, and also in many parts, leading the wave.



<p align="center"><img src="./lecture_18_slides/slide_7138_00-03-57.933.jpg" width="75%" alt="Lecture Video at 00:03:57.933" /></p>

Maybe it's time to just take a different look at this, both historically and going towards the future: where have we come from, and where are we going? This is an important topic to discuss because a lot of what has happened will inform what will happen. I'm organizing this talk in three chunks.



<p align="center"><img src="./lecture_18_slides/slide_8672_00-04-49.066.jpg" width="75%" alt="Lecture Video at 00:04:49.066" /></p>

Next, we'll talk about building AI to see what humans don't see. Finally, we'll finish with building AI to see what humans would like to see. Let's start with the first one: Building AI to see what humans see. Again, just a little bit of a review.

Humans are so good at seeing; we know this. This is a very half-a-century-old experiment showing us that even watching a video you've never watched played at 10 Hz, which means every frame is only about... On the screen for 100 milliseconds, you've never seen that; it's still no problem for human eyes to detect a target—in this case, a person.



<p align="center"><img src="./lecture_18_slides/slide_10204_00-05-40.133.jpg" width="75%" alt="Lecture Video at 00:05:40.133" /></p>

This is remarkable speed given the wetware we have under our skulls.



<p align="center"><img src="./lecture_18_slides/slide_11652_00-06-28.400.jpg" width="75%" alt="Lecture Video at 00:06:28.400" /></p>

<p align="center"><img src="./lecture_18_slides/slide_12748_00-07-04.933.jpg" width="75%" alt="Lecture Video at 00:07:04.933" /></p>

All this built up the history for the field of computer vision: that a few decades ago, object recognition became a fundamental building block for visual intelligence.



<p align="center"><img src="./lecture_18_slides/slide_13240_00-07-21.333.jpg" width="75%" alt="Lecture Video at 00:07:21.333" /></p>

We want to empower machines with that. That's such an effortless task for humans. So this is actually fundamentally a difficult task.



<p align="center"><img src="./lecture_18_slides/slide_14726_00-08-10.866.jpg" width="75%" alt="Lecture Video at 00:08:10.866" /></p>

The history pre-deep learning is also very interesting. There were some pretty heroic attempts at solving the problem of generic, generalizable object recognition. The first wave of attempts was actually very inspired by psychology itself. We self-introspect sometimes even to the detriment of over-self-introspection; we think that humans compose parts.

We look at objects, we can see geometric parts, and then we can compose them into different objects. That idea of using predesignated parts or shapes and composing them in specific ways was the first wave of object recognition. These were different works or models coming from the '70s, '80s, or even going all the way to '90s of using different parts and configurations to recognize objects.

Of course, it didn't really work.



<p align="center"><img src="./lecture_18_slides/slide_16940_00-09-24.666.jpg" width="75%" alt="Lecture Video at 00:09:24.666" /></p>

It was mathematically beautiful and simple, but it didn't work. It was very hard to use hand-tuned models to achieve good learning. We now know we need data, even though we didn't at that time. But we also know that we need to design or architect statistical models so that they have the capability of learning through different learning rules.

Because of that, we saw a blossoming of models in that era where we're learning random fields, Bayes nets, or support vector machines and all that.



<p align="center"><img src="./lecture_18_slides/slide_19388_00-10-46.266.jpg" width="75%" alt="Lecture Video at 00:10:46.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_20106_00-11-10.200.jpg" width="75%" alt="Lecture Video at 00:11:10.200" /></p>

The last unlock for object recognition, as we have learned, again goes back to cognitive science. This particular psychologist, Irvin Biederman, had long conjectured that humans can recognize a huge number of objects. This is intuitive for our common knowledge, but he actually put a number on it. He used this number combination of looking at dictionary's number of nouns, as well as visual studies of how kids recognize different objects.



<p align="center"><img src="./lecture_18_slides/slide_22028_00-12-14.266.jpg" width="75%" alt="Lecture Video at 00:12:14.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_22360_00-12-25.333.jpg" width="75%" alt="Lecture Video at 00:12:25.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_23202_00-12-53.400.jpg" width="75%" alt="Lecture Video at 00:12:53.400" /></p>

Of course, that's the beginning point that you start to come into this class.



<p align="center"><img src="./lecture_18_slides/slide_23510_00-13-03.666.jpg" width="75%" alt="Lecture Video at 00:13:03.666" /></p>

<p align="center"><img src="./lecture_18_slides/slide_23558_00-13-05.266.jpg" width="75%" alt="Lecture Video at 00:13:05.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_23924_00-13-17.466.jpg" width="75%" alt="Lecture Video at 00:13:17.466" /></p>

I'm going to skip the generic slide on this because you all know this. Now, we have algorithms that we can take to look at any picture in the world and be able to recognize objects, big or small, and in any kind of orientation. Is it 100% sure? No.

There are always long tail problems we can solve. But as far as industrial application goes, this has come a long way and really has been a matured problem.



<p align="center"><img src="./lecture_18_slides/slide_25594_00-14-13.133.jpg" width="75%" alt="Lecture Video at 00:14:13.133" /></p>

The three ingredients came together and brought the moment of deep learning—the birth of deep learning.



<p align="center"><img src="./lecture_18_slides/slide_26406_00-14-40.200.jpg" width="75%" alt="Lecture Video at 00:14:40.200" /></p>

That's the beginning, really, about the deep learning revolution.



<p align="center"><img src="./lecture_18_slides/slide_27178_00-15-05.933.jpg" width="75%" alt="Lecture Video at 00:15:05.933" /></p>

In terms of the quest for visual intelligence, we are not going to stop at just being able to label objects in a scene. For example, in these two scenes, if you just label objects, you'll think it's just a llama and a person.



<p align="center"><img src="./lecture_18_slides/slide_27680_00-15-22.666.jpg" width="75%" alt="Lecture Video at 00:15:22.666" /></p>

But if I show you the second scene... With the llama and the person, the story is completely different. Even though you have the same object, you have very different relationships.



<p align="center"><img src="./lecture_18_slides/slide_27972_00-15-32.400.jpg" width="75%" alt="Lecture Video at 00:15:32.400" /></p>

<p align="center"><img src="./lecture_18_slides/slide_28044_00-15-34.800.jpg" width="75%" alt="Lecture Video at 00:15:34.800" /></p>

<p align="center"><img src="./lecture_18_slides/slide_28992_00-16-06.400.jpg" width="75%" alt="Lecture Video at 00:16:06.400" /></p>

Inspired by that work, the field of computer vision started to look at how do we understand relationships. This was early work. Last week, you guys got a lecture from Ranjay; this was his PhD thesis, looking at learning object relationships using scene graph as a representation. Even something as simple as this, with mostly just two people one feeding a cake to the other, you can form a very dense scene graph because of the richness of the visual scene.



<p align="center"><img src="./lecture_18_slides/slide_30712_00-17-03.733.jpg" width="75%" alt="Lecture Video at 00:17:03.733" /></p>

<p align="center"><img src="./lecture_18_slides/slide_31384_00-17-26.133.jpg" width="75%" alt="Lecture Video at 00:17:26.133" /></p>

One of the works that Ranjay did—I thought this was really fun—was zero-shot learning of unusual object relationships. For example, it's not unusual to see a person riding a horse. It's not unusual to see a person wearing a hat, but it's unusual in general to see a horse wearing a hat. In the era of big data training, it is hard to get this kind of data repeatedly because you just don't have too many of that.

But using this compositional scene graph representation, we are able to learn more common relationships and then derive uncommon relationships within that representation.



<p align="center"><img src="./lecture_18_slides/slide_32588_00-18-06.266.jpg" width="75%" alt="Lecture Video at 00:18:06.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_33238_00-18-27.933.jpg" width="75%" alt="Lecture Video at 00:18:27.933" /></p>

And we were able to do that to make that happen. This figure from the paper shows that Ranjay's work at that time achieved state-of-the-art recognition rate compared to many other methods.



<p align="center"><img src="./lecture_18_slides/slide_33598_00-18-39.933.jpg" width="75%" alt="Lecture Video at 00:18:39.933" /></p>

But relationships are not enough. The ability to actually tell a story, which is a lot more richer, or also using natural language, is actually the next big goal. Around 2014, we started working on that problem and thinking about it. That's just two years after the image moment at Alex.

But the field was starting to evolve so fast. We were inspired by what we can do using a combination of convolutional neural network as well as a language model called LSTM. I know he's one of the co-instructors of this course. That was around the time between 2015 and 2018.

A lot of work has happened to solve the problem. Of course, today using multimodal LLMs, we have taken the solution of this problem even to another notch. But this is the beginning of that line of work.



<p align="center"><img src="./lecture_18_slides/slide_36918_00-20-30.600.jpg" width="75%" alt="Lecture Video at 00:20:30.600" /></p>

...as neural network algorithms. But a much harder problem is actually in dynamic scenes. In dynamic scenes, we tend to have much more complex relationships—much more complex movements. Also, the camera movement or the entity, the actors within the scene can do a lot of different things.

This is a much newer work; we only published this a couple of years ago. To capture the relationship between these actors and their activities in dynamic scenes is still an unsolved problem. And this will have profound implications. You know that you're in Silicon Valley, so you're hearing so much excitement about robots, for example.

This is an unsolved problem.



<p align="center"><img src="./lecture_18_slides/slide_39402_00-21-53.400.jpg" width="75%" alt="Lecture Video at 00:21:53.400" /></p>

And of course, generative AI and generative models. This is just to show you that the field of computer vision, since the rebirth of modern AI, has been moving extraordinarily fast. The take-home message in this section for me is that two things: One is that data, compute, and neural network algorithms truly converged about 10 years ago or 13 years ago.

That was the moment that the modern AI or deep learning revolution happened. The history of that, and so much of the problem that we have been working on, is truly inspired by cognitive science, psychology, and neuroscience. That is going to continue to happen. We will continue to be inspired by what the brain can do or how the brain does things, and also will continue to use AI to help our brain research.

So there is a very intimate relationship between today's AI and cognitive science, neuroscience, brain science, and all that. Of course, a lot of people—students and collaborators—have contributed to what I have just presented. Now, let's talk about going beyond just building AI to see what humans don't see. This is where pushing AI beyond the capability of humans, or you can call it superhumans.



<p align="center"><img src="./lecture_18_slides/slide_42982_00-23-52.733.jpg" width="75%" alt="Lecture Video at 00:23:52.733" /></p>

For example, most people don't recognize a ton of dinosaurs. You can probably name a few.



<p align="center"><img src="./lecture_18_slides/slide_43300_00-24-03.333.jpg" width="75%" alt="Lecture Video at 00:24:03.333" /></p>

Some kids really can name a lot. Well, let alone thousands and tens of thousands of bird species or tens of thousands of car categories. This is the line of work that I call fine-grained object categorization. Humans are just not that good at it, and this is still a problem that I don't think we're fully solved yet, to be honest.

In this generative AI era, especially, we're talking a lot about multimodal LLMs. In this early work of fine-grained bird species recognition, we put together a dataset; actually, we used a dataset of 4,000 birds. The algorithm is still not totally ready. Another work that I find fascinating is that a few years ago, a group of students in my lab trained a fine-grained car classifier in terms of make, model, and year.

It turns out after 1970s, there are thousands of car models that are defined by different make, model, and year. We took Google Street View images from 200 or 100, I think, major cities across the country. We used fine-grained car detectors to detect what were the cars on the street of these cities. We use it as a lens to study social patterns.

For example, I showed education patterns; car models and education patterns are highly correlated, and so are income patterns. In that paper, we show voting patterns, highly correlated, or even environmental patterns, highly correlated. So it's a really interesting way of using computer vision as a lens to study our society. No individual human, not even a collection of humans can do this easily at all.

AI is really pushing the boundary of what humans can see. To drive home this idea, let's do a couple of tests.



<p align="center"><img src="./lecture_18_slides/slide_49494_00-27-29.800.jpg" width="75%" alt="Lecture Video at 00:27:29.800" /></p>

Humans actually have limitations. I talked about humanity's ability of seeing, but we also have our limitations. This is a very famous visual illusion test called the Stroop test. Try to read it: red, yellow, green, purple, blue, black, orange.

It's fighting with you. This is the fight between visual attention and all that.



<p align="center"><img src="./lecture_18_slides/slide_50838_00-28-14.600.jpg" width="75%" alt="Lecture Video at 00:28:14.600" /></p>

Here's another example. There are two alternating images of a picture, and there's one change—a pretty big change happening between the two alternating pictures. I don't know if you spot the change. Do you spot it?

Yes, it's the engine. So it takes a while to spot it. This is a very famous psychology experiment called change blindness.



<p align="center"><img src="./lecture_18_slides/slide_51814_00-28-47.133.jpg" width="75%" alt="Lecture Video at 00:28:47.133" /></p>

Now, all this is fun; the Stroop test is fun, and this is fun. But this is not fun—that human attention is limited. In some situations in our working life, that kind of attention limit can be dire. For example, medical errors are the third leading cause of death in America's health care system.



<p align="center"><img src="./lecture_18_slides/slide_53144_00-29-31.466.jpg" width="75%" alt="Lecture Video at 00:29:31.466" /></p>

One has to be very careful. For example, in surgery rooms, honestly, scissors don't get left in the bodies typically, but much smaller things like sutures, needles, or piece of gauze and all that. Today, most of this is still just tracked by hands. We have these checklists to track in the surgery rooms.

If something is missing, the surgery has to be paused. On average, that pause is close to an hour. Think about the danger for the patient—the exposure to bacteria and the bleeding and all that, just because we have to search for that item. So if there is a way to use AI to help our doctors, surgeons, to track items, that would be so powerful.

This is just a demo; this is not a deploy system. We're not there in terms of fidelity. But this is a demo to show that we can use AI to count, in this case, gauze and all that. And this is just an example of pushing AI to see what humans don't see.



<p align="center"><img src="./lecture_18_slides/slide_55576_00-30-52.533.jpg" width="75%" alt="Lecture Video at 00:30:52.533" /></p>

Here's another example that is really fun; I don't know if I showed this before, but this is one of my favorite visual illusions. If you look at the two squares, A and B, on a checkerboard at the top, it is so hard to believe they have the same grayscale or luminance. And then you look at the bottom, you're like, "Oh, of course, they do." But why?

Even though you have the bottom picture in front of you, seeing the top still gives you the illusion. Why? This is so deep in our visual development that it's hard for us to see it another way.



<p align="center"><img src="./lecture_18_slides/slide_57594_00-31-59.800.jpg" width="75%" alt="Lecture Video at 00:31:59.800" /></p>

So what I'm trying to get at is there's bias in our human visual system. The bias might come from evolutionary construct; the bias can come from our social experience; the bias can come from the data we're exposed to. But some of these biases can be harmful. When the bias happens, that became unfair to a group of people—a community.

And we have to be aware of this. A few years ago, face recognition algorithms were not good, and they tended to recognize certain skin colors and even genders better than others. This has consequences. Think about self-driving cars or many other medical use cases.



<p align="center"><img src="./lecture_18_slides/slide_59254_00-32-55.133.jpg" width="75%" alt="Lecture Video at 00:32:55.133" /></p>

So we have to be vigilant about this. I do believe that AI bias has been a problem that people are now carrying. A few years ago, this problem was so new that many people were not even paying attention.



<p align="center"><img src="./lecture_18_slides/slide_60162_00-33-25.400.jpg" width="75%" alt="Lecture Video at 00:33:25.400" /></p>

And then there's another kind of "not seen," and this is interesting. Sometimes not seeing is exactly what we want because you want to respect privacy. So how do you create AI that helps people to see yet you still want it—not to see what people don't want you to see? This is a very deep technical problem as well as a human problem.



<p align="center"><img src="./lecture_18_slides/slide_61052_00-33-55.066.jpg" width="75%" alt="Lecture Video at 00:33:55.066" /></p>

So from a technical point of view, there are many ways to consider ML, machine learning privacy. But even there, we have to recognize issues like faces or just full body information and even homes. And this is a list of potential solutions. For example, you can do blurring, or you can do masking; you can do dimensionality reduction.

But you can also try to do different approaches, for example, federated learning so that you don't send all the data to the server, or encryption and other things.



<p align="center"><img src="./lecture_18_slides/slide_62848_00-34-54.933.jpg" width="75%" alt="Lecture Video at 00:34:54.933" /></p>

I'm not going to belabor this, but there's one work I want to show you. It's not even my work, but I really like this work. It is a work about taking videos of people and trying to recognize the action of people, but yet respecting the privacy of people.



<p align="center"><img src="./lecture_18_slides/slide_63394_00-35-13.133.jpg" width="75%" alt="Lecture Video at 00:35:13.133" /></p>

How do you do that? For example, in this case, you want to take a video of this kid moving in the scene. There are ways to do this. If you blur this or defocus this or do some of these, you can protect privacy, but you also lose enough information that you might not even know what this person is doing.

And for many applications, the whole goal... ...is to know what this person is doing. So particular that if you look at the top row, what the lens captures into the camera protects the privacy a lot. You don't see the person's face, you don't see the body, and so on.

So that's a really interesting approach.



<p align="center"><img src="./lecture_18_slides/slide_66628_00-37-00.933.jpg" width="75%" alt="Lecture Video at 00:37:00.933" /></p>

So that's a work I really like; I really like the spirit of that work. In this part of the lecture, I shared with you a number of things—just considerations for building AI to see what humans don't see. Sometimes we're pushing AI, like fine-grained recognition of birds, to go beyond human ability; those are superhuman abilities. Sometimes we know humans are not good.

We have bias or we have attention issues, and then we want to use AI to help us. And sometimes we genuinely have situations where we don't want anyone to see. How do you use AI to continue to help without violating those privacy concerns? You can see that AI is a very interesting, powerful tool.

It can both help but amplify us. If we have bias or if we have issues, AI can amplify us too. That's the second take-home message. Now, let's talk about building AI to see what humans want to see.



<p align="center"><img src="./lecture_18_slides/slide_69322_00-38-30.733.jpg" width="75%" alt="Lecture Video at 00:38:30.733" /></p>

In fact, we're going to go beyond seeing; we're going to connect seeing and doing together. If you think about today's societal anxiety about AI, one of the biggest anxieties is labor.



<p align="center"><img src="./lecture_18_slides/slide_69862_00-38-48.733.jpg" width="75%" alt="Lecture Video at 00:38:48.733" /></p>

A lot of headline news will say labor is under threat—robots taking over jobs. The truth is, the picture is complex. Denying job change is wrong. Every technological shift in human history has caused labor market change, and some of them are very painful.

Some of them can lead to even civil wars and wars, but that change sometimes is inevitable. And a tiny digression: A lot of the labor threat rhetoric that we have been hearing tends to think about physical labors. But today, in the past two years, if you look at GenAI's impact, it is white-collar jobs that are drastically being impacted, especially software engineering and analytical work in offices.

So there's just definitely a labor change.



<p align="center"><img src="./lecture_18_slides/slide_71890_00-39-56.333.jpg" width="75%" alt="Lecture Video at 00:39:56.333" /></p>

But in the meantime, we also need to recognize that AI can be helpful. We actually fundamentally have human labor shortages in many situations, especially in elderly care as well as health. First of all, as modern medicine improves, human life expectancy increases. That inevitably pushes society towards longer living, and that's a good thing.

But in the meantime, we have shortages of laborers. Young people need to work; that's how to make this society vibrant, economy vibrant. But who is taking care of our elderlies? Who are taking care of our chronically ill?

Even in America's hospitals, we have such an attrition... ...of health care workers, especially nurses, that we don't have enough hands, ears, eyes to help our patients.



<p align="center"><img src="./lecture_18_slides/slide_73670_00-40-55.666.jpg" width="75%" alt="Lecture Video at 00:40:55.666" /></p>

<p align="center"><img src="./lecture_18_slides/slide_73716_00-40-57.200.jpg" width="75%" alt="Lecture Video at 00:40:57.200" /></p>

Instead of thinking about this word "replace," we can actually think about AI augmenting.



<p align="center"><img src="./lecture_18_slides/slide_74022_00-41-07.400.jpg" width="75%" alt="Lecture Video at 00:41:07.400" /></p>

You got a glimpse of that in my surgery room example. Indeed, in health, there are so many spaces where we don't have enough pairs of eyes. That's what I call the "dark spaces" of health. They range from the surgery room to the patient room, to pharmaceuticals, to homes, and so on.



<p align="center"><img src="./lecture_18_slides/slide_74698_00-41-29.933.jpg" width="75%" alt="Lecture Video at 00:41:29.933" /></p>

So how do we make AI help? This is an area that Ihsan has been leading a lot of work. This allows us to alert the patients, or family members, or doctors in time to help patients.



<p align="center"><img src="./lecture_18_slides/slide_75682_00-42-02.733.jpg" width="75%" alt="Lecture Video at 00:42:02.733" /></p>

Again, the full paper is in a particular paper we published a couple of years ago.



<p align="center"><img src="./lecture_18_slides/slide_75836_00-42-07.866.jpg" width="75%" alt="Lecture Video at 00:42:07.866" /></p>

Let me just give you a couple of examples. One example is this hand hygiene project, which actually started way before COVID. Hand hygiene turns out to be really important for keeping hospital infection low.



<p align="center"><img src="./lecture_18_slides/slide_76326_00-42-24.200.jpg" width="75%" alt="Lecture Video at 00:42:24.200" /></p>

Hospital-acquired infection is actually one of the leading causes of American patient fatality in our hospitals. It kills three times more people per year than car accidents nationwide, and it is really hard to control. Most of these germs are passed from patient room to patient room, and then they just brew together. So what do we do?

The hospitals try to use human auditors, but we just talked about—we don't even have enough nurses, let alone hiring auditors. Also, you cannot hire enough of them; there is human fatigue. We talk about the human attention problem, so this is not a pretty prohibitive solution. There were some technological solutions like RFID: put the badge.

But that's very nonspecific; you cannot guarantee that. The hospital rooms are pretty small, corridors are small, and just standing next to something doesn't mean you're doing it. So a few years ago, we did this project where we put smart sensors that protect privacy by only gleaning depth information, like the blue screen or the video there. We then use computer vision algorithms to classify actions: is the person washing hands or not washing hands?



<p align="center"><img src="./lecture_18_slides/slide_79378_00-44-05.933.jpg" width="75%" alt="Lecture Video at 00:44:05.933" /></p>

The result is that if you compare ground truth with the algorithm output versus human outputs—or human detection results—you can see the algorithm is so much better and more consistent than humans. You have to almost show the same video to four humans to get almost as good as AI, and this is just not plausible. If it's one person, you can see how sparse the detection is, and that's not good.



<p align="center"><img src="./lecture_18_slides/slide_80520_00-44-44.000.jpg" width="75%" alt="Lecture Video at 00:44:44.000" /></p>

So this is one application. Another application we worked on is in ICUs.



<p align="center"><img src="./lecture_18_slides/slide_80746_00-44-51.533.jpg" width="75%" alt="Lecture Video at 00:44:51.533" /></p>

ICU is where patients fight life and death; ICU is also where $1\%$ of US GDP is spent. So making ICU as effective and safe as possible is a top priority. One of the goals, or the goal of ICU, is to get our patients safely out of ICU and go into step-down units or even go home. So one of the most important things people have learned in ICU is to help patients to move.

Proper movement, which we call mobilization, is actually important for recovery. But this is a very dicey situation. You have to get nurses to help; doctors have to give orders, and you have to move properly, and it has to be at different times, like designated time. And you have to assess the movement...

All this is not easy, right? So we collaborated with Stanford as well as Utah's Intermountain Hospital to put these smart sensors in ICU units and help doctors to monitor patient movement. In this particular case, they monitored four different kinds of movements: getting out of bed, getting in bed, getting out of a chair, and getting in a chair. These things are so important for ICU patients.

I know that for us it's a no-brainer, but this really is critical. You can see how this kind of detection and prediction helps doctors, especially when there is a labor shortage.



<p align="center"><img src="./lecture_18_slides/slide_83736_00-46-31.200.jpg" width="75%" alt="Lecture Video at 00:46:31.200" /></p>

Another example is aging in place, which is just so important for many reasons. People are seniors who want to live at home independently and healthily. During the beginning of COVID, we saw such high fatality among aging seniors; a lot was due to hospital overrun and overtaxed systems.



<p align="center"><img src="./lecture_18_slides/slide_84660_00-47-02.000.jpg" width="75%" alt="Lecture Video at 00:47:02.000" /></p>

Therefore, keeping seniors safe and well in their homes is really critical.



<p align="center"><img src="./lecture_18_slides/slide_85508_00-47-30.266.jpg" width="75%" alt="Lecture Video at 00:47:30.266" /></p>

This brings us to the last technical topic: embodied AI. A large part of embodied AI is robotics. I find this extremely exciting because it closes the loop between perception and action. If you think about the Cambrian explosion of evolution, when there was an onset of eyes, animals started to move.



<p align="center"><img src="./lecture_18_slides/slide_87282_00-48-29.400.jpg" width="75%" alt="Lecture Video at 00:48:29.400" /></p>

The area of robotics is where we can close the loop between seeing and doing. However, robots are not easy. As much as we are excited by them, they are still very slow and very clumsy. It is difficult for them to adapt to a generalizable situation.



<p align="center"><img src="./lecture_18_slides/slide_87780_00-48-46.000.jpg" width="75%" alt="Lecture Video at 00:48:46.000" /></p>

<p align="center"><img src="./lecture_18_slides/slide_87870_00-48-49.000.jpg" width="75%" alt="Lecture Video at 00:48:49.000" /></p>

<p align="center"><img src="./lecture_18_slides/slide_87928_00-48-50.933.jpg" width="75%" alt="Lecture Video at 00:48:50.933" /></p>

<p align="center"><img src="./lecture_18_slides/slide_87980_00-48-52.666.jpg" width="75%" alt="Lecture Video at 00:48:52.666" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88006_00-48-53.533.jpg" width="75%" alt="Lecture Video at 00:48:53.533" /></p>

In today's robotic research, we have made immense progress, and Stanford is definitely one of the centers of robotic learning.



<p align="center"><img src="./lecture_18_slides/slide_88060_00-48-55.333.jpg" width="75%" alt="Lecture Video at 00:48:55.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88136_00-48-57.866.jpg" width="75%" alt="Lecture Video at 00:48:57.866" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88184_00-48-59.466.jpg" width="75%" alt="Lecture Video at 00:48:59.466" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88232_00-49-01.066.jpg" width="75%" alt="Lecture Video at 00:49:01.066" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88242_00-49-01.400.jpg" width="75%" alt="Lecture Video at 00:49:01.400" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88262_00-49-02.066.jpg" width="75%" alt="Lecture Video at 00:49:02.066" /></p>

<p align="center"><img src="./lecture_18_slides/slide_88732_00-49-17.733.jpg" width="75%" alt="Lecture Video at 00:49:17.733" /></p>

Let me share a couple of works from our lab. One project was a few years ago where we looked at how to bring robots into the wild. If we have to predesignate the set of tasks, it is unsatisfying. Conversely, if you look at today's LLMs, they are totally in the wild; you can talk about anything.

My students want to learn, and a few students want to close this gap.



<p align="center"><img src="./lecture_18_slides/slide_89606_00-49-46.866.jpg" width="75%" alt="Lecture Video at 00:49:46.866" /></p>

The idea is: how do we give an open instruction to a robot—any instruction without pretraining everything in a closed world—and have the robot perform tasks? Suppose your training set is defined by "open a door." In the wild, you might encounter doors like that.



<p align="center"><img src="./lecture_18_slides/slide_90476_00-50-15.866.jpg" width="75%" alt="Lecture Video at 00:50:15.866" /></p>

The goal is achieving generalization in the wild.



<p align="center"><img src="./lecture_18_slides/slide_90726_00-50-24.200.jpg" width="75%" alt="Lecture Video at 00:50:24.200" /></p>

<p align="center"><img src="./lecture_18_slides/slide_90754_00-50-25.133.jpg" width="75%" alt="Lecture Video at 00:50:25.133" /></p>

<p align="center"><img src="./lecture_18_slides/slide_90798_00-50-26.600.jpg" width="75%" alt="Lecture Video at 00:50:26.600" /></p>

<p align="center"><img src="./lecture_18_slides/slide_90856_00-50-28.533.jpg" width="75%" alt="Lecture Video at 00:50:28.533" /></p>

<p align="center"><img src="./lecture_18_slides/slide_90922_00-50-30.733.jpg" width="75%" alt="Lecture Video at 00:50:30.733" /></p>

What was demonstrated was telling a robot arm to open a drawer by planning a motion path that avoids knocking down a flower.



<p align="center"><img src="./lecture_18_slides/slide_91526_00-50-50.866.jpg" width="75%" alt="Lecture Video at 00:50:50.866" /></p>

Crucially, all these instructions were not pretrained.



<p align="center"><img src="./lecture_18_slides/slide_91560_00-50-52.000.jpg" width="75%" alt="Lecture Video at 00:50:52.000" /></p>

We actually borrowed the latest advances... in LLM, as well as in visual language model. And the idea is that we use LLM and VLM to give us an instruction set. Then we use a visual language model to help us to recognize or understand the environment.

We then turn that into a motion planning map so that the robotic arm can execute. Because we're using LLMs as well as VLMs, we get rid of the problem of training robots in a closed world and bring them to a more generalizable—or in the wild setting.



<p align="center"><img src="./lecture_18_slides/slide_93014_00-51-40.466.jpg" width="75%" alt="Lecture Video at 00:51:40.466" /></p>

The details involve the instruction "open top drawer" coming in. An LLM turns this into like literally code.



<p align="center"><img src="./lecture_18_slides/slide_93334_00-51-51.133.jpg" width="75%" alt="Lecture Video at 00:51:51.133" /></p>

And then because of these instructions like "draw" or "handle," we send this information to a VLM model. That model detects "draw" and "handle" in the scene.



<p align="center"><img src="./lecture_18_slides/slide_93842_00-52-08.066.jpg" width="75%" alt="Lecture Video at 00:52:08.066" /></p>

Because of that, it updates its information, and then it updates a motion map.



<p align="center"><img src="./lecture_18_slides/slide_94100_00-52-16.666.jpg" width="75%" alt="Lecture Video at 00:52:16.666" /></p>

This is presented by a heat map to show you where the robot arm should focus, where it should not focus.



<p align="center"><img src="./lecture_18_slides/slide_94310_00-52-23.666.jpg" width="75%" alt="Lecture Video at 00:52:23.666" /></p>

With that, then you give it another instruction: "but watch out for the vase."



<p align="center"><img src="./lecture_18_slides/slide_94594_00-52-33.133.jpg" width="75%" alt="Lecture Video at 00:52:33.133" /></p>

Again, it goes through the same thing with an LLM, writes the code or generates the code, and sends it through a VLM model.



<p align="center"><img src="./lecture_18_slides/slide_94690_00-52-36.333.jpg" width="75%" alt="Lecture Video at 00:52:36.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_94890_00-52-43.000.jpg" width="75%" alt="Lecture Video at 00:52:43.000" /></p>

The VLM model detects the object and then updates the motion planning map. In this case, it's the negative, not the positive, because you want to avoid that.



<p align="center"><img src="./lecture_18_slides/slide_95124_00-52-50.800.jpg" width="75%" alt="Lecture Video at 00:52:50.800" /></p>

Then, combining with the previous map, you get a heat map of knowing where to avoid and where to go. Eventually, what we do is we do this for the motion planning map.



<p align="center"><img src="./lecture_18_slides/slide_95506_00-53-03.533.jpg" width="75%" alt="Lecture Video at 00:53:03.533" /></p>

We do it for rotation to gripper velocity, and then this is the result.



<p align="center"><img src="./lecture_18_slides/slide_95680_00-53-09.333.jpg" width="75%" alt="Lecture Video at 00:53:09.333" /></p>

Actually, let me just show you this. This is the actual result of the robot.



<p align="center"><img src="./lecture_18_slides/slide_95982_00-53-19.400.jpg" width="75%" alt="Lecture Video at 00:53:19.400" /></p>

And then we do this for many different tasks. We can do it for articulated object manipulation.



<p align="center"><img src="./lecture_18_slides/slide_96314_00-53-30.466.jpg" width="75%" alt="Lecture Video at 00:53:30.466" /></p>

<p align="center"><img src="./lecture_18_slides/slide_96658_00-53-41.933.jpg" width="75%" alt="Lecture Video at 00:53:41.933" /></p>

<p align="center"><img src="./lecture_18_slides/slide_96698_00-53-43.266.jpg" width="75%" alt="Lecture Video at 00:53:43.266" /></p>

So this is one work.



<p align="center"><img src="./lecture_18_slides/slide_96998_00-53-53.266.jpg" width="75%" alt="Lecture Video at 00:53:53.266" /></p>

Another work I want to just show you quickly is that overall robotics research is still lacking good benchmarks.



<p align="center"><img src="./lecture_18_slides/slide_97368_00-54-05.600.jpg" width="75%" alt="Lecture Video at 00:54:05.600" /></p>

<p align="center"><img src="./lecture_18_slides/slide_97914_00-54-23.800.jpg" width="75%" alt="Lecture Video at 00:54:23.800" /></p>

<p align="center"><img src="./lecture_18_slides/slide_98042_00-54-28.066.jpg" width="75%" alt="Lecture Video at 00:54:28.066" /></p>

We also know that both natural language and computer vision have benefited a lot from setting up important large-scale data sets for both training and benchmarks.



<p align="center"><img src="./lecture_18_slides/slide_98198_00-54-33.266.jpg" width="75%" alt="Lecture Video at 00:54:33.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_98320_00-54-37.333.jpg" width="75%" alt="Lecture Video at 00:54:37.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_98896_00-54-56.533.jpg" width="75%" alt="Lecture Video at 00:54:56.533" /></p>

And that's the Behavior of Benchmark, which is benchmark for everyday household activities in virtual interactive and ecological environments. Now, here's a question because this lecture has a lot to do with human values: Who is to say which tasks robots should do? I know that every graduate student who is working on robotics just wants two tasks: one is laundry, the other one is dishwasher.

That's great, but moving beyond grad school, what are the tasks we should get robots to do for us?



<p align="center"><img src="./lecture_18_slides/slide_100086_00-55-36.200.jpg" width="75%" alt="Lecture Video at 00:55:36.200" /></p>

<p align="center"><img src="./lecture_18_slides/slide_100592_00-55-53.066.jpg" width="75%" alt="Lecture Video at 00:55:53.066" /></p>

Let me test this. Would you like a robot to help you to clean the kitchen floor? Say yes or no.



<p align="center"><img src="./lecture_18_slides/slide_100880_00-56-02.666.jpg" width="75%" alt="Lecture Video at 00:56:02.666" /></p>

OK, good. Normal people would say yes: shoveling snow?



<p align="center"><img src="./lecture_18_slides/slide_100990_00-56-06.333.jpg" width="75%" alt="Lecture Video at 00:56:06.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_101094_00-56-09.800.jpg" width="75%" alt="Lecture Video at 00:56:09.800" /></p>

OK. Folding laundry? Yes. OK, good. Cooking breakfast?

No. Yes. See, we're getting mixture answers, right?



<p align="center"><img src="./lecture_18_slides/slide_101332_00-56-17.733.jpg" width="75%" alt="Lecture Video at 00:56:17.733" /></p>

What about opening Christmas gift? No. Exactly. People are different.

I actually think a robot can do this pretty well, but we don't want it. One of the tasks we ask is buying wedding rings—can you imagine that?



<p align="center"><img src="./lecture_18_slides/slide_101830_00-56-34.333.jpg" width="75%" alt="Lecture Video at 00:56:34.333" /></p>

So what we did is that we wanted to respect human preference. We took a bunch of government surveys from a labor office in the US and Europe, and so on, and put together thousands of everyday activity tasks. Then we went online to find people. We want to be as diverse as possible, but I think we have room to improve.



<p align="center"><img src="./lecture_18_slides/slide_103036_00-57-14.533.jpg" width="75%" alt="Lecture Video at 00:57:14.533" /></p>

But we found 1,400 people to answer these tasks and tell us which tasks they want robots to help with, and then we rank that. There are a lot of tasks that matter to us as humans, emotionally, or socially, or whatever. Our goal is first to have a principled way to decide which are the 1,000 tasks that we want to train robots for; those are the tasks that humans prefer to get help with.



<p align="center"><img src="./lecture_18_slides/slide_104434_00-58-01.133.jpg" width="75%" alt="Lecture Video at 00:58:01.133" /></p>

<p align="center"><img src="./lecture_18_slides/slide_105358_00-58-31.933.jpg" width="75%" alt="Lecture Video at 00:58:31.933" /></p>

<p align="center"><img src="./lecture_18_slides/slide_105544_00-58-38.133.jpg" width="75%" alt="Lecture Video at 00:58:38.133" /></p>

<p align="center"><img src="./lecture_18_slides/slide_105760_00-58-45.333.jpg" width="75%" alt="Lecture Video at 00:58:45.333" /></p>

With that in mind, we had to build virtual environments. We scanned or acquired 3D scenes from 50 different real-world environments—from restaurants to apartments to grocery stores to offices and so on. We also built a simulation environment.



<p align="center"><img src="./lecture_18_slides/slide_105918_00-58-50.600.jpg" width="75%" alt="Lecture Video at 00:58:50.600" /></p>

A lot of people have built simulation environments. Our particular simulation environment was a collaboration with NVIDIA's Omniverse group.



<p align="center"><img src="./lecture_18_slides/slide_106758_00-59-18.600.jpg" width="75%" alt="Lecture Video at 00:59:18.600" /></p>

We also tested our behavior environment against other environments in terms of perceptual realism from human user study. Here are some examples of physical interaction, such as cloth or liquids and so on.



<p align="center"><img src="./lecture_18_slides/slide_107480_00-59-42.666.jpg" width="75%" alt="Lecture Video at 00:59:42.666" /></p>

There's a lot of nuance that has gone into this work.



<p align="center"><img src="./lecture_18_slides/slide_107756_00-59-51.866.jpg" width="75%" alt="Lecture Video at 00:59:51.866" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108644_01-00-21.466.jpg" width="75%" alt="Lecture Video at 01:00:21.466" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108730_01-00-24.333.jpg" width="75%" alt="Lecture Video at 01:00:24.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108772_01-00-25.733.jpg" width="75%" alt="Lecture Video at 01:00:25.733" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108788_01-00-26.266.jpg" width="75%" alt="Lecture Video at 01:00:26.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108842_01-00-28.066.jpg" width="75%" alt="Lecture Video at 01:00:28.066" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108870_01-00-29.000.jpg" width="75%" alt="Lecture Video at 01:00:29.000" /></p>

<p align="center"><img src="./lecture_18_slides/slide_108896_01-00-29.866.jpg" width="75%" alt="Lecture Video at 01:00:29.866" /></p>

One thing I want to share with you is these numbers: today's algorithm still cannot do behavior tasks. Of all these roles, the top role is what we wish robots can do. We give them no privileged information; they have to be dropped in an environment and do these tasks. We benchmarked three behavior tasks using today's robotic algorithm, and the performance is just zero.

And once you start to give more privileged information or make assumptions that make the task... Simpler, like MagicMotion or Perfect Memory, and all that, things start to get better. If you look at it, only look at the top row, you get pretty depressed by today's robots. But as a grad student, I hope you're inspired because that means we have a lot of room to grow.



<p align="center"><img src="./lecture_18_slides/slide_110818_01-01-33.933.jpg" width="75%" alt="Lecture Video at 01:01:33.933" /></p>

<p align="center"><img src="./lecture_18_slides/slide_110958_01-01-38.600.jpg" width="75%" alt="Lecture Video at 01:01:38.600" /></p>

These are just different papers from our lab.



<p align="center"><img src="./lecture_18_slides/slide_111002_01-01-40.066.jpg" width="75%" alt="Lecture Video at 01:01:40.066" /></p>

I'm going to actually fast forward because I think we've talked enough about this.



<p align="center"><img src="./lecture_18_slides/slide_111124_01-01-44.133.jpg" width="75%" alt="Lecture Video at 01:01:44.133" /></p>

Again, this is an unsolved problem, and there's a long way to go.



<p align="center"><img src="./lecture_18_slides/slide_111752_01-02-05.066.jpg" width="75%" alt="Lecture Video at 01:02:05.066" /></p>

<p align="center"><img src="./lecture_18_slides/slide_111818_01-02-07.266.jpg" width="75%" alt="Lecture Video at 01:02:07.266" /></p>

<p align="center"><img src="./lecture_18_slides/slide_111876_01-02-09.200.jpg" width="75%" alt="Lecture Video at 01:02:09.200" /></p>

<p align="center"><img src="./lecture_18_slides/slide_111976_01-02-12.533.jpg" width="75%" alt="Lecture Video at 01:02:12.533" /></p>

And in this particular case, we're showing you that this robot, without speeding up, you can see how slow it is.



<p align="center"><img src="./lecture_18_slides/slide_112182_01-02-19.400.jpg" width="75%" alt="Lecture Video at 01:02:19.400" /></p>

It's trying to clean up this room.



<p align="center"><img src="./lecture_18_slides/slide_112194_01-02-19.800.jpg" width="75%" alt="Lecture Video at 01:02:19.800" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112262_01-02-22.066.jpg" width="75%" alt="Lecture Video at 01:02:22.066" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112502_01-02-30.066.jpg" width="75%" alt="Lecture Video at 01:02:30.066" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112550_01-02-31.666.jpg" width="75%" alt="Lecture Video at 01:02:31.666" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112594_01-02-33.133.jpg" width="75%" alt="Lecture Video at 01:02:33.133" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112700_01-02-36.666.jpg" width="75%" alt="Lecture Video at 01:02:36.666" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112776_01-02-39.200.jpg" width="75%" alt="Lecture Video at 01:02:39.200" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112886_01-02-42.866.jpg" width="75%" alt="Lecture Video at 01:02:42.866" /></p>

<p align="center"><img src="./lecture_18_slides/slide_112960_01-02-45.333.jpg" width="75%" alt="Lecture Video at 01:02:45.333" /></p>

So there's still a lot of mistakes. Let me fast forward.



<p align="center"><img src="./lecture_18_slides/slide_113110_01-02-50.333.jpg" width="75%" alt="Lecture Video at 01:02:50.333" /></p>

<p align="center"><img src="./lecture_18_slides/slide_113160_01-02-52.000.jpg" width="75%" alt="Lecture Video at 01:02:52.000" /></p>

<p align="center"><img src="./lecture_18_slides/slide_113496_01-03-03.200.jpg" width="75%" alt="Lecture Video at 01:03:03.200" /></p>

There's no invasive brain control; this is from electrical signals. What we have to do is to pretrain these thoughts. You have to pretrain the robotic arm with, say, lift or place or drop or whatever. Once you do that, this is an entire meal cooked based on the wave.

This is really sci-fi. This has happened last year. So I'm pretty excited by where all this is going, combining vision and perception and robotics and also helping people in clinical settings. This is really the future; it is helping severely paralyzed patients.

The Behavior project is really aimed at augmenting people. It's a large-scale diverse benchmark, and it has realistic and ecological physics and perception. And the last take-home message is that we not only want to build AI to just do things or see things, we really want to build it to help people.



<p align="center"><img src="./lecture_18_slides/slide_117312_01-05-10.400.jpg" width="75%" alt="Lecture Video at 01:05:10.400" /></p>

AI being an augmentation tool or enhancing tool for humanity is very important instead of a tool that replaces.



