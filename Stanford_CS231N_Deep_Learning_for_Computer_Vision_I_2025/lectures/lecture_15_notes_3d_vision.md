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

# Stanford CS231N Deep Learning for Computer Vision | Spring 2025 | Lecture 15: 3D Vision


<p align="center"><img src="./lecture_15_slides/slide_4_00-00-00.133.jpg" width="75%" alt="Lecture Video at 00:00:00.133" /></p>

I'm happy to announce our next guest speaker for the course, Professor Jiajun Wu. He is an assistant professor at Stanford in the Department of Computer Science and a faculty member of the Stanford Vision and Learning Lab.



<p align="center"><img src="./lecture_15_slides/slide_1038_00-00-34.634.jpg" width="75%" alt="Lecture Video at 00:00:34.634" /></p>

I will now turn it over to Jiajun to begin. I am an assistant professor here, and a few years ago I used to teach this class. Today, we are going to talk about 3D vision. For 3D, I plan to first introduce what the 3D representations are.

We will also look into a few different applications regarding 3D generation, reconstruction, and related topics.



<p align="center"><img src="./lecture_15_slides/slide_3478_00-01-56.049.jpg" width="75%" alt="Lecture Video at 00:01:56.049" /></p>

<p align="center"><img src="./lecture_15_slides/slide_4080_00-02-16.136.jpg" width="75%" alt="Lecture Video at 00:02:16.136" /></p>

Let's start by looking at the possible ways to represent objects in 3D. In 2D, it is straightforward; you just have pixels. You load a file of a PNG or a JPEG file—it's like $200 \times 200$ pixels. But how can we represent 3D objects?

This is the first thing we want to look into. 3D objects are diverse and can exist at different scales. They could be huge, like large buildings and trees with complex structures, and if you zoom in, you see all the fine details. The challenge is determining the best 3D representations that cover these types of objects across various scales and features.

Let's begin by focusing on geometry. Even just for 3D object geometry, there are many ways to represent them.



<p align="center"><img src="./lecture_15_slides/slide_5024_00-02-47.634.jpg" width="75%" alt="Lecture Video at 00:02:47.634" /></p>

<p align="center"><img src="./lecture_15_slides/slide_5172_00-02-52.572.jpg" width="75%" alt="Lecture Video at 00:02:52.572" /></p>

<p align="center"><img src="./lecture_15_slides/slide_5224_00-02-54.307.jpg" width="75%" alt="Lecture Video at 00:02:54.307" /></p>

<p align="center"><img src="./lecture_15_slides/slide_5534_00-03-04.651.jpg" width="75%" alt="Lecture Video at 00:03:04.651" /></p>

<p align="center"><img src="./lecture_15_slides/slide_5636_00-03-08.054.jpg" width="75%" alt="Lecture Video at 00:03:08.054" /></p>

<p align="center"><img src="./lecture_15_slides/slide_5666_00-03-09.055.jpg" width="75%" alt="Lecture Video at 00:03:09.055" /></p>

<p align="center"><img src="./lecture_15_slides/slide_6170_00-03-25.872.jpg" width="75%" alt="Lecture Video at 00:03:25.872" /></p>

<p align="center"><img src="./lecture_15_slides/slide_6522_00-03-37.617.jpg" width="75%" alt="Lecture Video at 00:03:37.617" /></p>

<p align="center"><img src="./lecture_15_slides/slide_6604_00-03-40.353.jpg" width="75%" alt="Lecture Video at 00:03:40.353" /></p>

We can generally categorize these into two different types: explicit representations and implicit representations. Implicit shape representations are a different category and we will talk about those as well. I will explain them in detail later, including level sets, algebraic surfaces, and distance functions. These represent 3D objects or their geometries as functions, which is not as intuitive as simply collecting points.

However, as we will see, they have advantages and weaknesses when using implicit representations. Every choice has a suitable task and type of geometry. In the context of deep learning, these methods also have unique strengths and weaknesses when applying deep learning techniques on top of them. When do we choose a representation?

We must consider storage: pixels are easy to store because they form a matrix, but 3D point clouds are more irregular. Furthermore, if you use implicit representations—like representing an object as a function—how would you store that in a computer?



<p align="center"><img src="./lecture_15_slides/slide_7138_00-03-58.171.jpg" width="75%" alt="Lecture Video at 00:03:58.171" /></p>

<p align="center"><img src="./lecture_15_slides/slide_7166_00-03-59.105.jpg" width="75%" alt="Lecture Video at 00:03:59.105" /></p>

We also need to know how the method supports creating new shapes.



<p align="center"><img src="./lecture_15_slides/slide_7328_00-04-04.510.jpg" width="75%" alt="Lecture Video at 00:04:04.510" /></p>

For images, you want to edit them using language or strokes.



<p align="center"><img src="./lecture_15_slides/slide_7878_00-04-22.862.jpg" width="75%" alt="Lecture Video at 00:04:22.862" /></p>

How do you perform any type of operation on 3D objects?



<p align="center"><img src="./lecture_15_slides/slide_7890_00-04-23.263.jpg" width="75%" alt="Lecture Video at 00:04:23.263" /></p>

And how can you render that 3D object into 2D pixels? In a sense, you can. 3D vision is the process of inverting this: going from 2D images to reconstruct the 3D objects.



<p align="center"><img src="./lecture_15_slides/slide_8266_00-04-35.808.jpg" width="75%" alt="Lecture Video at 00:04:35.808" /></p>

Something that connects all these representations is their integration with deep learning methods for shape editing, rendering, inverse rendering, and animation.



<p align="center"><img src="./lecture_15_slides/slide_8896_00-04-56.829.jpg" width="75%" alt="Lecture Video at 00:04:56.829" /></p>

<p align="center"><img src="./lecture_15_slides/slide_9030_00-05-01.301.jpg" width="75%" alt="Lecture Video at 00:05:01.301" /></p>

To quickly go through some of these representations, point clouds are probably the simplest: they only contain 3D points. It doesn't have connectivity, so it doesn't capture how these points are connected.



<p align="center"><img src="./lecture_15_slides/slide_9750_00-05-25.325.jpg" width="75%" alt="Lecture Video at 00:05:25.325" /></p>

<p align="center"><img src="./lecture_15_slides/slide_9762_00-05-25.725.jpg" width="75%" alt="Lecture Video at 00:05:25.725" /></p>

<p align="center"><img src="./lecture_15_slides/slide_9788_00-05-26.592.jpg" width="75%" alt="Lecture Video at 00:05:26.592" /></p>

You have a number of points. Sometimes, you can represent the surface normals of the point as well so that you have not only where the point is in the 3D space but also to which direction it's facing. You have the surface normals, which give you a bit more information. Sometimes, people call them surfels, which are points with orientations.



<p align="center"><img src="./lecture_15_slides/slide_10366_00-05-45.878.jpg" width="75%" alt="Lecture Video at 00:05:45.878" /></p>

Why do you need surface normals? Because if you want to render them—you want to say how the object looks like—then that means you have to often specify a lighting source. Where's the lighting coming from? But to make the rendering look realistic, you have to consider how the lighting, coming from a certain direction, is going to interact with the point.

This is where the surface normals are used to help you to make the rendering look realistic, like you can see here.



<p align="center"><img src="./lecture_15_slides/slide_11166_00-06-12.572.jpg" width="75%" alt="Lecture Video at 00:06:12.572" /></p>

How do you get points? A benefit of the point cloud is that it is often a raw format that you will get from a lot of 3D sensors, including depth sensors and some 3D scanners. Nowadays, I think if you use an iPhone, they have an AR kit or these kinds of software allow you to scan 3D objects. But the raw output of those sensors are still 3D point clouds.

Of course, after that, you have to process them and fuse them to make it into objects with textures.



<p align="center"><img src="./lecture_15_slides/slide_12048_00-06-42.001.jpg" width="75%" alt="Lecture Video at 00:06:42.001" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12076_00-06-42.935.jpg" width="75%" alt="Lecture Video at 00:06:42.935" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12108_00-06-44.003.jpg" width="75%" alt="Lecture Video at 00:06:44.003" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12194_00-06-46.873.jpg" width="75%" alt="Lecture Video at 00:06:46.873" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12342_00-06-51.811.jpg" width="75%" alt="Lecture Video at 00:06:51.811" /></p>

In this part, you have to consider how these different pictures can be registered to give you the shared point cloud.



<p align="center"><img src="./lecture_15_slides/slide_12560_00-06-59.085.jpg" width="75%" alt="Lecture Video at 00:06:59.085" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12574_00-06-59.552.jpg" width="75%" alt="Lecture Video at 00:06:59.552" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12588_00-07-00.019.jpg" width="75%" alt="Lecture Video at 00:07:00.019" /></p>

<p align="center"><img src="./lecture_15_slides/slide_12616_00-07-00.953.jpg" width="75%" alt="Lecture Video at 00:07:00.953" /></p>

They're very flexible because you can move points here and there. You can use them to represent basically any type of object geometry; you're not constrained by the topology or stuff like that. It is useful for large data sets because sometimes, you have to consider a very diverse set of objects.



<p align="center"><img src="./lecture_15_slides/slide_13756_00-07-38.991.jpg" width="75%" alt="Lecture Video at 00:07:38.991" /></p>

<p align="center"><img src="./lecture_15_slides/slide_13772_00-07-39.525.jpg" width="75%" alt="Lecture Video at 00:07:39.525" /></p>

<p align="center"><img src="./lecture_15_slides/slide_14248_00-07-55.408.jpg" width="75%" alt="Lecture Video at 00:07:55.408" /></p>

Other limitations are that it's not obvious how we can directly perform sometimes very useful operations, like simplification or subdivisions, on these objects.



<p align="center"><img src="./lecture_15_slides/slide_14384_00-07-59.946.jpg" width="75%" alt="Lecture Video at 00:07:59.946" /></p>

It doesn't directly allow you to do smooth rendering; there's no topological information.



<p align="center"><img src="./lecture_15_slides/slide_14474_00-08-02.949.jpg" width="75%" alt="Lecture Video at 00:08:02.949" /></p>

It is partial information about what the object is if you just have the point clouds.



<p align="center"><img src="./lecture_15_slides/slide_15250_00-08-28.841.jpg" width="75%" alt="Lecture Video at 00:08:28.841" /></p>

That naturally goes to the polygonal meshes. These represent the objects still as a collection of points, but they also describe how these points are connected. Now you have not only the points, but also the faces, the surfaces. This is arguably the most widely used representation for 3D objects in all graphics engines and in computer games; basically, it is all represented as polygon meshes.

But to represent faces, it is more complex because often, especially if you're looking at raw meshes, every face may have a different number of points. You may have three points and have four points or five points. But here you have what I would say is a variable dimension of this raw information. How does that integrate with deep learning?

That has been some big challenge.



<p align="center"><img src="./lecture_15_slides/slide_17538_00-09-45.184.jpg" width="75%" alt="Lecture Video at 00:09:45.184" /></p>

But meshes are really widely used, and they can be very complex meshes that capture all the details. For example, you have scanners; you get points, and then you fuse them, and you apply some algorithm. You can get a very large mesh. This one has 56 million triangles and 28 million vertices to represent the sculpture.



<p align="center"><img src="./lecture_15_slides/slide_18094_00-10-03.736.jpg" width="75%" alt="Lecture Video at 00:10:03.736" /></p>

And you can have even larger ones, let's say from Google Earth. They have trillions of triangles. Try to represent basically all of the buildings on Earth.



<p align="center"><img src="./lecture_15_slides/slide_18372_00-10-13.012.jpg" width="75%" alt="Lecture Video at 00:10:13.012" /></p>

The nice thing about meshes is that it supports a lot of operations, like subdivisions.



<p align="center"><img src="./lecture_15_slides/slide_18624_00-10-21.420.jpg" width="75%" alt="Lecture Video at 00:10:21.420" /></p>

Or I want to have more details, and how can I use more meshes to capture more details of the shape?



<p align="center"><img src="./lecture_15_slides/slide_18656_00-10-22.488.jpg" width="75%" alt="Lecture Video at 00:10:22.488" /></p>

You can do simplification as well. Sometimes you want to process things very fast, so you don't need that many meshes.



<p align="center"><img src="./lecture_15_slides/slide_18958_00-10-32.565.jpg" width="75%" alt="Lecture Video at 00:10:32.565" /></p>

<p align="center"><img src="./lecture_15_slides/slide_19002_00-10-34.033.jpg" width="75%" alt="Lecture Video at 00:10:34.033" /></p>

You just want to simplify it, and there are existing algorithms that allow you to do that as well. They have roughly the same size, and so that it's easier for processing.



<p align="center"><img src="./lecture_15_slides/slide_19626_00-10-54.854.jpg" width="75%" alt="Lecture Video at 00:10:54.854" /></p>

This provides obvious good properties that support future processing of different graphics algorithms and meshes.



<p align="center"><img src="./lecture_15_slides/slide_20150_00-11-12.338.jpg" width="75%" alt="Lecture Video at 00:11:12.338" /></p>

So this is one type of shape representation.



<p align="center"><img src="./lecture_15_slides/slide_20288_00-11-16.942.jpg" width="75%" alt="Lecture Video at 00:11:16.942" /></p>

They are very general, but sometimes we lose a lot of information. If you look at, let's say, your chairs or your tables, you have all these straight lines. How can you represent these kinds of straight lines?



<p align="center"><img src="./lecture_15_slides/slide_20908_00-11-37.630.jpg" width="75%" alt="Lecture Video at 00:11:37.630" /></p>

<p align="center"><img src="./lecture_15_slides/slide_20922_00-11-38.097.jpg" width="75%" alt="Lecture Video at 00:11:38.097" /></p>

<p align="center"><img src="./lecture_15_slides/slide_20934_00-11-38.497.jpg" width="75%" alt="Lecture Video at 00:11:38.497" /></p>

When people design them, they often use some of these parametric representations. You can represent shapes as a function. Think about it: if I want to represent a surface or represent a curve, the underlying degree of freedom is actually lower.



<p align="center"><img src="./lecture_15_slides/slide_22302_00-12-24.143.jpg" width="75%" alt="Lecture Video at 00:12:24.143" /></p>

<p align="center"><img src="./lecture_15_slides/slide_22318_00-12-24.677.jpg" width="75%" alt="Lecture Video at 00:12:24.677" /></p>

<p align="center"><img src="./lecture_15_slides/slide_22332_00-12-25.144.jpg" width="75%" alt="Lecture Video at 00:12:25.144" /></p>

<p align="center"><img src="./lecture_15_slides/slide_22920_00-12-44.764.jpg" width="75%" alt="Lecture Video at 00:12:44.764" /></p>

<p align="center"><img src="./lecture_15_slides/slide_23368_00-12-59.712.jpg" width="75%" alt="Lecture Video at 00:12:59.712" /></p>

<p align="center"><img src="./lecture_15_slides/slide_23380_00-13-00.112.jpg" width="75%" alt="Lecture Video at 00:13:00.112" /></p>

<p align="center"><img src="./lecture_15_slides/slide_23392_00-13-00.513.jpg" width="75%" alt="Lecture Video at 00:13:00.513" /></p>

<p align="center"><img src="./lecture_15_slides/slide_23816_00-13-14.660.jpg" width="75%" alt="Lecture Video at 00:13:14.660" /></p>

<p align="center"><img src="./lecture_15_slides/slide_23826_00-13-14.994.jpg" width="75%" alt="Lecture Video at 00:13:14.994" /></p>

<p align="center"><img src="./lecture_15_slides/slide_23842_00-13-15.528.jpg" width="75%" alt="Lecture Video at 00:13:15.528" /></p>

<p align="center"><img src="./lecture_15_slides/slide_24020_00-13-21.467.jpg" width="75%" alt="Lecture Video at 00:13:21.467" /></p>

<p align="center"><img src="./lecture_15_slides/slide_24252_00-13-29.208.jpg" width="75%" alt="Lecture Video at 00:13:29.208" /></p>

<p align="center"><img src="./lecture_15_slides/slide_24694_00-13-43.956.jpg" width="75%" alt="Lecture Video at 00:13:43.956" /></p>

<p align="center"><img src="./lecture_15_slides/slide_24704_00-13-44.290.jpg" width="75%" alt="Lecture Video at 00:13:44.290" /></p>

<p align="center"><img src="./lecture_15_slides/slide_24768_00-13-46.425.jpg" width="75%" alt="Lecture Video at 00:13:46.425" /></p>

<p align="center"><img src="./lecture_15_slides/slide_24948_00-13-52.431.jpg" width="75%" alt="Lecture Video at 00:13:52.431" /></p>

Often, if I have a curve, there's only one underlying degree of freedom. That's why I can represent a curve using a function $f(x)$. I just vary $x$ and get a value of $y$. And this allows you to represent 3D objects in a parametric representation using basically a set of functions.

You can do that for curves, let's say, in circles. Another way is you just represent the curve of the circle as this function. Basically, there's a sine function and a cosine function, and you just vary one variable, which is $t$. You can think about it as the degrees or angles, and it will map it to all the points on the circles.

So here now, you can use a function to represent parametric representations for curves in 2D. And of course, you can do that in 3D as well. If you want to represent a sphere, all you need is just two degrees of freedom, $u$ and $v$. Then you can go through these functions so that you can map them to every point in 3D space for this sphere.

They also allow you to do things like subdivision, so you can try to get more details into the surfaces and make it more fine-grained, and stuff like that. Or you can represent them in a parametric way, where you have a function, basically. By varying a few parameters that are underlying the true degree of freedoms of the object geometry, you can map them into more complex shapes.

And all of them fall into this category of being quite explicit. It's like I have points, and points are just directly points on objects. For the surfaces or for parametric curves as well, they directly map it to the points on the objects.



<p align="center"><img src="./lecture_15_slides/slide_26450_00-14-42.548.jpg" width="75%" alt="Lecture Video at 00:14:42.548" /></p>

<p align="center"><img src="./lecture_15_slides/slide_26528_00-14-45.150.jpg" width="75%" alt="Lecture Video at 00:14:45.150" /></p>

So there are explicit representations. They have a lot of benefits. First, you map all the points directly, so you can get all these points.



<p align="center"><img src="./lecture_15_slides/slide_26720_00-14-51.557.jpg" width="75%" alt="Lecture Video at 00:14:51.557" /></p>

In general, every point I have in, let's say, a sample, I have Bézier surface representations. I can basically sample two points and $u$ and $v$ in this underlying lower dimensional space and then go into that function and map it to a point in the 3D space. So I directly get a point in a 3D space; so all points are given, in some sense, you can say directly.



<p align="center"><img src="./lecture_15_slides/slide_27436_00-15-15.447.jpg" width="75%" alt="Lecture Video at 00:15:15.447" /></p>

I can directly get other points.



<p align="center"><img src="./lecture_15_slides/slide_27502_00-15-17.650.jpg" width="75%" alt="Lecture Video at 00:15:17.650" /></p>

So it's very easy for us to sample points. Let's say I have this torus, and I have represented using this $f$ function. Now my question is, can you just sample some points on the surface of the object for me?



<p align="center"><img src="./lecture_15_slides/slide_27798_00-15-27.526.jpg" width="75%" alt="Lecture Video at 00:15:27.526" /></p>

<p align="center"><img src="./lecture_15_slides/slide_27812_00-15-27.993.jpg" width="75%" alt="Lecture Video at 00:15:27.993" /></p>

This is so easy because I would just randomly put in some $u$ and $v$ values.



<p align="center"><img src="./lecture_15_slides/slide_27940_00-15-32.264.jpg" width="75%" alt="Lecture Video at 00:15:32.264" /></p>

Sampling is much easier.



<p align="center"><img src="./lecture_15_slides/slide_28412_00-15-48.013.jpg" width="75%" alt="Lecture Video at 00:15:48.013" /></p>

What is hard about these explicit representations? The hard thing is it's very hard, in some sense, to test whether a point is inside or outside the object.



<p align="center"><img src="./lecture_15_slides/slide_28656_00-15-56.155.jpg" width="75%" alt="Lecture Video at 00:15:56.155" /></p>

Similarly, if I represent a sphere as this function, and it's easy for me to sample points on the sphere.



<p align="center"><img src="./lecture_15_slides/slide_28914_00-16-04.763.jpg" width="75%" alt="Lecture Video at 00:16:04.763" /></p>

But it is hard for me to say, now I have a different—now I have a query. I say, at this point $(3/4, 1/2, 1/4)$. At this point in 3D space, is it the inside object or the outside object? It is actually hard to test whether a certain point is inside or outside an object.



<p align="center"><img src="./lecture_15_slides/slide_29564_00-16-26.452.jpg" width="75%" alt="Lecture Video at 00:16:26.452" /></p>

All these representations have their own strengths and weaknesses. For explicit representations, it's actually pretty easy to sample points, which are very useful because sometimes you want to convert them into, let's say, a collection of points. You then want to apply whatever your point neural networks on it. But it is hard to test if a certain point is inside or outside the object, which may have some issues.

What would be the geometry or density of the object at that particular point? What would be the material? Or what would be the radiance or color of the object at that particular point? Essentially, explicit representations are not very supportive of running these operations.



<p align="center"><img src="./lecture_15_slides/slide_31040_00-17-15.701.jpg" width="75%" alt="Lecture Video at 00:17:15.701" /></p>

Naturally, people thought, "Okay, maybe we can come up with a different type of way to represent geometry." And here I say "implicit" representations for geometry. In deep learning methods, they just extend these implicit representations for not only geometry but also for colors and appearance of objects in 3D.



<p align="center"><img src="./lecture_15_slides/slide_31542_00-17-32.451.jpg" width="75%" alt="Lecture Video at 00:17:32.451" /></p>

The idea of these implicit representations is that you want to classify these points.



<p align="center"><img src="./lecture_15_slides/slide_31970_00-17-46.732.jpg" width="75%" alt="Lecture Video at 00:17:46.732" /></p>

<p align="center"><img src="./lecture_15_slides/slide_32516_00-18-04.950.jpg" width="75%" alt="Lecture Video at 00:18:04.950" /></p>

<p align="center"><img src="./lecture_15_slides/slide_33972_00-18-53.532.jpg" width="75%" alt="Lecture Video at 00:18:53.532" /></p>

<p align="center"><img src="./lecture_15_slides/slide_34184_00-19-00.606.jpg" width="75%" alt="Lecture Video at 00:19:00.606" /></p>

<p align="center"><img src="./lecture_15_slides/slide_34728_00-19-18.757.jpg" width="75%" alt="Lecture Video at 00:19:18.757" /></p>

<p align="center"><img src="./lecture_15_slides/slide_35138_00-19-32.437.jpg" width="75%" alt="Lecture Video at 00:19:32.437" /></p>

I assume if the points are on the object, or on the surface of the object, then they satisfy some certain relationship. For example, for a sphere, what would be the points on the unit sphere? The constraint they satisfy is the square of $x$, and the square of $y$, and square $z$. When we sum them up, they equal to 1.

This is the constraint satisfied for all the points on the sphere. More generally, you can write it down as the constraint will be some function of $x$ and $y$ and $z$ equals $0$. In this case, the function will be $x^2 + y^2 + z^2 - 1$. But more generally, you can think about it even for complex shapes.

Sometimes these functions can be so complex that you don't even have a closed form. How can I represent an $f$? I just write it as a neural network. My hope is a neural network will be able to represent it.

This is called implicit representations, which started with geometry. But as I said, now we are using all these different ways representing textures, materials, appearance, and all these things. The bad thing about implicit representation is that it's actually much harder to sample points. I tell you, "Okay, this is a constraint; let's say, this torus satisfies."

If I put every $x$, $y$, and $z$ into this function, and the output is $0$, then yeah, they must be on the surface of this object. But how can I get a couple of these $(x, y, z)$ tuples? That would be very hard because you are required to solve this function. When the function gets really complex for arbitrary shapes, it becomes much harder to solve these functions.

So it's not easy to actually sample points on the surface of objects if you are representing objects implicitly. The benefit, the strength of that is that it's actually pretty easy to test whether a point is inside an object or outside an object.



<p align="center"><img src="./lecture_15_slides/slide_35386_00-19-40.712.jpg" width="75%" alt="Lecture Video at 00:19:40.712" /></p>

<p align="center"><img src="./lecture_15_slides/slide_35522_00-19-45.250.jpg" width="75%" alt="Lecture Video at 00:19:45.250" /></p>

<p align="center"><img src="./lecture_15_slides/slide_35632_00-19-48.921.jpg" width="75%" alt="Lecture Video at 00:19:48.921" /></p>

Because if I want to do testing, I just have a query; this is so easy because it's either inside or outside.



<p align="center"><img src="./lecture_15_slides/slide_35702_00-19-51.256.jpg" width="75%" alt="Lecture Video at 00:19:51.256" /></p>

I just send it into that function and get a value, or the value is less than $0$. And if the output value is positive, then the point must be outside the object. So now it becomes much easier to test whether a certain point is inside or outside an object, although it becomes much harder to sample a number of points on the surface of an object.



<p align="center"><img src="./lecture_15_slides/slide_36736_00-20-25.757.jpg" width="75%" alt="Lecture Video at 00:20:25.757" /></p>

You can see now there is a clear trade-off between these implicit and explicit representations. Here, again we talk about geometry, but this distinction and the contrast between explicit and implicit representations, I think, is very important and fundamental. It is behind deep neural networks when they apply to 3D data in general, as we'll see later.

Before we talk about how deep learning can be applied to 3D representations in general, a little bit more on implicit representations—some other features of them.



<p align="center"><img src="./lecture_15_slides/slide_37934_00-21-05.731.jpg" width="75%" alt="Lecture Video at 00:21:05.731" /></p>

The good thing about them is it's easy to compose them.



<p align="center"><img src="./lecture_15_slides/slide_38020_00-21-08.600.jpg" width="75%" alt="Lecture Video at 00:21:08.600" /></p>

<p align="center"><img src="./lecture_15_slides/slide_38422_00-21-22.014.jpg" width="75%" alt="Lecture Video at 00:21:22.014" /></p>

So if I want to represent the shape of a cow, how would I represent that? What would be the function I can write for the shape of a cow?



<p align="center"><img src="./lecture_15_slides/slide_38680_00-21-30.622.jpg" width="75%" alt="Lecture Video at 00:21:30.622" /></p>

It's just not obvious.



<p align="center"><img src="./lecture_15_slides/slide_38778_00-21-33.892.jpg" width="75%" alt="Lecture Video at 00:21:33.892" /></p>

<p align="center"><img src="./lecture_15_slides/slide_39452_00-21-56.381.jpg" width="75%" alt="Lecture Video at 00:21:56.381" /></p>

<p align="center"><img src="./lecture_15_slides/slide_40314_00-22-25.143.jpg" width="75%" alt="Lecture Video at 00:22:25.143" /></p>

<p align="center"><img src="./lecture_15_slides/slide_40334_00-22-25.811.jpg" width="75%" alt="Lecture Video at 00:22:25.811" /></p>

<p align="center"><img src="./lecture_15_slides/slide_40352_00-22-26.411.jpg" width="75%" alt="Lecture Video at 00:22:26.411" /></p>

<p align="center"><img src="./lecture_15_slides/slide_40810_00-22-41.693.jpg" width="75%" alt="Lecture Video at 00:22:41.693" /></p>

<p align="center"><img src="./lecture_15_slides/slide_40952_00-22-46.431.jpg" width="75%" alt="Lecture Video at 00:22:46.431" /></p>

<p align="center"><img src="./lecture_15_slides/slide_41696_00-23-11.256.jpg" width="75%" alt="Lecture Video at 00:23:11.256" /></p>

<p align="center"><img src="./lecture_15_slides/slide_41920_00-23-18.730.jpg" width="75%" alt="Lecture Video at 00:23:18.730" /></p>

<p align="center"><img src="./lecture_15_slides/slide_42542_00-23-39.484.jpg" width="75%" alt="Lecture Video at 00:23:39.484" /></p>

<p align="center"><img src="./lecture_15_slides/slide_43364_00-24-06.912.jpg" width="75%" alt="Lecture Video at 00:24:06.912" /></p>

<p align="center"><img src="./lecture_15_slides/slide_43418_00-24-08.713.jpg" width="75%" alt="Lecture Video at 00:24:08.713" /></p>

<p align="center"><img src="./lecture_15_slides/slide_43462_00-24-10.182.jpg" width="75%" alt="Lecture Video at 00:24:10.182" /></p>

But the nice thing about implicit representation is you don't have to write everything in one shot because it's so easy to compose them. You can actually perform logical operations on these implicit functions. Let's say you have two objects and you find unions or intersections or differences; again, they are just values. So you put $x$, $y$, $z$ onto this function, you get a value.

You put $x$, $y$, $z$ onto that function, you get a value. You can just do arithmetic operations on top of these values, and that allows you to compute the unions or intersections or differences between these objects. Eventually, you can compose them to define pretty complex shapes. So you can even add them up, and this allows you to smoothly blend the shapes.

You can see that here, if I have a distance function and here I just want to represent a vertical line, okay, this is here. Anything that's minus 0 is to the left of the line; anything that is positive is to the right of the line. And then you have another line representing using a different function. So what happens if you add them up?

If you add them up, then it naturally becomes an interpolation between these two shapes. This is an example of doing things in 1D, but you can imagine you can even similarly do things in 3D in a sense. Now you can actually even blend these different shapes. These distance functions can be arbitrarily composed and allow you to create actually pretty complex worlds, like this.

They are actually very expressive if you're very good at it. So we said, okay, we have parametric representation that can be explicit, that directly give you points on the 3D surface. Or we can have parametric representations, like these functions, but they are implicit. So they are just like, okay, now you can only try to verify if the point is inside or outside an object.

But then you can also compose them to build more complex shapes. And is that possible for us to also have implicit representation and nonparametric, like that point style, but then you also querying functions? Well, sometimes they actually do have things like that. This eventually goes to methods like level set methods.

So implicit surfaces are very nice because as we said, it's easy to merge them; it's easy to split them. But sometimes it's hard to describe, as we said, complex shapes in closed forms. You have a cow—how would you represent it? You can compose them.

But if every time I have to query whether a certain point is inside a cow, you have to have hundreds of functions and you perform all these AND or OR, plus/minus operations. Then it takes a long time. So what if I just prequery? So I have a 3D space, and I just sample, let's say, a $100$ by $100$ by $100$ grid.

So I have now a million points sampled. You can precompute them and then you can store all the values in a matrix. This is in 2D, but this is for visualization; but in practice, it's in 3D. So you have a 3D matrix that stores all these precomputed values of the distance functions.

Now, in some sense, you still have an implicit representation, but because you have prequeried them, you turned it into nonparametric representations.



<p align="center"><img src="./lecture_15_slides/slide_45536_00-25-19.384.jpg" width="75%" alt="Lecture Video at 00:25:19.384" /></p>

And even if you just look at this matrix in 2D, You can now still find where the boundaries are. So, where are the boundaries? They're just basically where you have two adjacent values: one is positive, and one is negative.

That means there must be somewhere in between. That is the point here. They satisfy the function $f(x) = 0$, which means the point must be on the surface. This allows you to have actually more explicit controls because you can now visualize them.



<p align="center"><img src="./lecture_15_slides/slide_46740_00-25-59.558.jpg" width="75%" alt="Lecture Video at 00:25:59.558" /></p>

You can say, "I have this matrix." And I can visualize them based on their values. This is used a lot in things like CTs and MRIs and all these medical data.



<p align="center"><img src="./lecture_15_slides/slide_47100_00-26-11.570.jpg" width="75%" alt="Lecture Video at 00:26:11.570" /></p>

A related thing is people may ask, "What if I don't care about all these distance values? I can prequery what's going on at all these points, but then I compute all the values. I say plus 5, minus 5." But all I care about is whether this is inside or outside object.

So if it's positive, I'll just treat them as one. If it's negative, which means they're inside object, treat them as 0, let's say.



<p align="center"><img src="./lecture_15_slides/slide_47730_00-26-32.591.jpg" width="75%" alt="Lecture Video at 00:26:32.591" /></p>

<p align="center"><img src="./lecture_15_slides/slide_47768_00-26-33.858.jpg" width="75%" alt="Lecture Video at 00:26:33.858" /></p>

<p align="center"><img src="./lecture_15_slides/slide_47888_00-26-37.862.jpg" width="75%" alt="Lecture Video at 00:26:37.862" /></p>

If you binarize them, then this gives you a final representation, which is arguably the easiest to understand, and this is called voxels. You can prequery where the implicit functions are, and then you have all these density-sampled grids. Instead of storing their distance functions—how far they are from the surface by going through the functions and giving you plus 5, minus 5—you just binarize it.

You only care about whether a certain point is inside objects or outside objects. Then you have a voxel representation, which is again like a $3D$ matrix, maybe $100 \times 100 \times 100$. But for every point, you have to go through this function and query whether it's inside and outside objects. You have 1 or 0, and you can represent objects in a binarized way.

This gives you the final representation I'm going to talk about for objects in $3D$. I introduced voxels in a kind of complex way. Now you have $3D$ matrices, and voxels is basically just a $3D$ matrix. Although, you can see that they have connections with all the other ways that we can represent shapes.

The way I'm introducing it this way is actually when deep learning comes in. First, deep learning: when did it start? 2010. Deep learning has been there for a long time, but the modern deep learning thing started around 2010.

Geoff Hinton started doing that on speech recognition. In 2012, they have AlexNet, which was written on ImageNet. So you've learned all these, and they're all in $2D$. People then asked, "What if I want to do this in $3D$?"

This is a very natural thought. So I want to go from $2D$ Convolutional Neural Networks (CNNs). In 2012, there was no transformer. How can I apply a $2D$ CNN on $3D$ data?



<p align="center"><img src="./lecture_15_slides/slide_50896_00-28-18.229.jpg" width="75%" alt="Lecture Video at 00:28:18.229" /></p>

Everyone knows we have all these different $3D$ representations, but which one do you begin with? It turns out that the people who started doing deep learning on $3D$ data were computer vision people, not graphics people. They thought, "I've been working with pixels, and maybe the easiest thing I can do is just to scale up." Instead of working on $2D$ matrices, they made it work on $3D$ matrices.

So that would be the simplest thing to do: instead of having a $2D$ Convolutional Network, they have a volumetric Convolutional Neural Network. Which of these representations supports a volumetric convolution? It turned out to be this voxel representation. This is basically the easiest you can imagine.

However, graphics people don't agree with that because they argue that voxels are really bad because it's very slow to compute. That is, instead of doing a $2D$ convolution, now do a $3D$ convolution. In some sense, this is how things got started.



<p align="center"><img src="./lecture_15_slides/slide_53098_00-29-31.703.jpg" width="75%" alt="Lecture Video at 00:29:31.703" /></p>

Before I talk about deep learning methods for $3D$ data, another aspect that's very important is the data—sorry, for $3D$ data. Beyond methods, data sets are also very important. ImageNet really prompted AlexNet and stuff like that.



<p align="center"><img src="./lecture_15_slides/slide_53482_00-29-44.516.jpg" width="75%" alt="Lecture Video at 00:29:44.516" /></p>

<p align="center"><img src="./lecture_15_slides/slide_53496_00-29-44.983.jpg" width="75%" alt="Lecture Video at 00:29:44.983" /></p>

For $3D$, similarly, we have to collect a lot of data as well. Prior to deep learning, the common dataset or popular dataset people often use is this thing called the Princeton Shape Benchmark, which has 1,800 models in 180 categories.



<p align="center"><img src="./lecture_15_slides/slide_53758_00-29-53.725.jpg" width="75%" alt="Lecture Video at 00:29:53.725" /></p>

<p align="center"><img src="./lecture_15_slides/slide_53834_00-29-56.261.jpg" width="75%" alt="Lecture Video at 00:29:56.261" /></p>

You can see they actually have quite a lot of categories—180 categories—but only 1,800 models, which means there are basically ten models per year, which was so small back then. But it was considered pretty large. And people thought, "This is already enough to work on."



<p align="center"><img src="./lecture_15_slides/slide_54398_00-30-15.079.jpg" width="75%" alt="Lecture Video at 00:30:15.079" /></p>

There was very little machine learning there. So prior to 2014, all these data sets were more or less small. They may have a certain number of models—even up to $10,000$, $9,000$, $10,000$. But they're also divided into so many different classes.

Each class, you only have 10 models each or less than $100$, I would say.



<p align="center"><img src="./lecture_15_slides/slide_54924_00-30-32.630.jpg" width="75%" alt="Lecture Video at 00:30:32.630" /></p>

<p align="center"><img src="./lecture_15_slides/slide_54936_00-30-33.031.jpg" width="75%" alt="Lecture Video at 00:30:33.031" /></p>

So after that, people started by saying, "OK, if we have ImageNet, can we also have the 3D data sets for shapes?" This is behind efforts of a few concurrent works.



<p align="center"><img src="./lecture_15_slides/slide_55470_00-30-50.849.jpg" width="75%" alt="Lecture Video at 00:30:50.849" /></p>

Stanford, through people like Leo Guibas and Silvio Savarese, led these large data sets called ShapeNet, which has $3$ million models.



<p align="center"><img src="./lecture_15_slides/slide_55636_00-30-56.387.jpg" width="75%" alt="Lecture Video at 00:30:56.387" /></p>

<p align="center"><img src="./lecture_15_slides/slide_55664_00-30-57.322.jpg" width="75%" alt="Lecture Video at 00:30:57.322" /></p>

But in practice, just ImageNet, you have this large image. There's a smaller data set that people often use: the ShapeNet core data set, which is what people typically use as basically $50,000$ models across $55$ categories. For every category, you have $1,000$ models on average. But in practice, it's not a balance.

For chairs, for example, you actually have a lot more. This allows people to say, "Oh, now I have finally... I have thousands of models on chairs. I can train some deep networks on it."

Before this, you might only have $10$ models, and you couldn't do anything.



<p align="center"><img src="./lecture_15_slides/slide_56910_00-31-38.897.jpg" width="75%" alt="Lecture Video at 00:31:38.897" /></p>

People felt like that wasn't enough, so they should move even bigger. In the past few years, this work has come from AI2, the Allen Institute from Seattle. What they did is collect much larger data sets called Objaverse and Objaverse Extra Large, which have roughly $1$ million or $10$ million models for different $3$D assets.



<p align="center"><img src="./lecture_15_slides/slide_57710_00-32-05.590.jpg" width="75%" alt="Lecture Video at 00:32:05.590" /></p>

<p align="center"><img src="./lecture_15_slides/slide_57762_00-32-07.325.jpg" width="75%" alt="Lecture Video at 00:32:07.325" /></p>

<p align="center"><img src="./lecture_15_slides/slide_57818_00-32-09.193.jpg" width="75%" alt="Lecture Video at 00:32:09.193" /></p>

<p align="center"><img src="./lecture_15_slides/slide_57974_00-32-14.399.jpg" width="75%" alt="Lecture Video at 00:32:14.399" /></p>

<p align="center"><img src="./lecture_15_slides/slide_58322_00-32-26.010.jpg" width="75%" alt="Lecture Video at 00:32:26.010" /></p>

<p align="center"><img src="./lecture_15_slides/slide_58344_00-32-26.744.jpg" width="75%" alt="Lecture Video at 00:32:26.744" /></p>

<p align="center"><img src="./lecture_15_slides/slide_58362_00-32-27.345.jpg" width="75%" alt="Lecture Video at 00:32:27.345" /></p>

<p align="center"><img src="./lecture_15_slides/slide_58960_00-32-47.298.jpg" width="75%" alt="Lecture Video at 00:32:47.298" /></p>

<p align="center"><img src="./lecture_15_slides/slide_58984_00-32-48.099.jpg" width="75%" alt="Lecture Video at 00:32:48.099" /></p>

<p align="center"><img src="./lecture_15_slides/slide_59150_00-32-53.638.jpg" width="75%" alt="Lecture Video at 00:32:53.638" /></p>

<p align="center"><img src="./lecture_15_slides/slide_59234_00-32-56.441.jpg" width="75%" alt="Lecture Video at 00:32:56.441" /></p>

<p align="center"><img src="./lecture_15_slides/slide_59602_00-33-08.720.jpg" width="75%" alt="Lecture Video at 00:33:08.720" /></p>

<p align="center"><img src="./lecture_15_slides/slide_59686_00-33-11.522.jpg" width="75%" alt="Lecture Video at 00:33:11.522" /></p>

<p align="center"><img src="./lecture_15_slides/slide_60436_00-33-36.547.jpg" width="75%" alt="Lecture Video at 00:33:36.547" /></p>

<p align="center"><img src="./lecture_15_slides/slide_61912_00-34-25.797.jpg" width="75%" alt="Lecture Video at 00:34:25.797" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62094_00-34-31.869.jpg" width="75%" alt="Lecture Video at 00:34:31.869" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62104_00-34-32.203.jpg" width="75%" alt="Lecture Video at 00:34:32.203" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62114_00-34-32.537.jpg" width="75%" alt="Lecture Video at 00:34:32.537" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62124_00-34-32.870.jpg" width="75%" alt="Lecture Video at 00:34:32.870" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62136_00-34-33.271.jpg" width="75%" alt="Lecture Video at 00:34:33.271" /></p>

They also have many more categories. These models are on average of higher quality, including textures, because they are synthetic data sets. But there are also real data sets being produced, some coming from $3$D scans. Back in $2016$, people were working on a data set called the Redwood data set, where you have $10,000$ scans of real-world objects.

More recently, people have been building larger data sets by encouraging people to take data for them. This is an effort co-led by Meta and Oxford; they pay people to do this. Using just an iPhone, if you place an object on a table and take a $360$ video around it, you can get $1$ or something like that. This first version has $19,000$ videos of objects.

These are real objects because capturing real objects is much harder than synthetic ones. While the previous work involved synthetic objects, these are real. Because of a lot of development in $3$D vision algorithms, you can actually take these $360$ videos and try to reconstruct the $3$D objects. You now have paired data consisting of the videos or images of objects, as well as their $3$D geometries and textures.

The first version is available, and I believe they have a more recent version, V2, or maybe even V3 right now, which is supposed to be a little larger. But it is hard to scale up. Right now, you might have $90,000$ videos—basically $90,000$ objects—and while they are scaling it up, I don't think it will exceed $100,000$. So basically, for real objects, you have $100,000$ models.

But if you look at the data set size of the images, it is $[\text{?} \text{ in the } ? ]$ lie on $5$ B or whatever—that's like $5$ billion images. Google and OpenAI must have much larger data sets.

There is still a huge gap between the number of data points that you can have for $2$D images or videos versus what you can have for $3$D objects. I think that's a big challenge regarding how we can move forward with $3$D vision. People have different ideas, but these datasets are much larger than what we had before. At least it is possible to still, more or less, train some deep learning models on these data sets now.

Additionally, there are other data sets being built focusing on parts. This also comes from Stanford, where they try to annotate a little bit of object parts and their correspondence and hierarchies.



<p align="center"><img src="./lecture_15_slides/slide_62588_00-34-48.352.jpg" width="75%" alt="Lecture Video at 00:34:48.352" /></p>

For example, a laptop that can be opened and closed. There are also data sets for $3$D scenes, so not just objects and parts, but also the rooms themselves.



<p align="center"><img src="./lecture_15_slides/slide_62750_00-34-53.758.jpg" width="75%" alt="Lecture Video at 00:34:53.758" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62760_00-34-54.092.jpg" width="75%" alt="Lecture Video at 00:34:54.092" /></p>

<p align="center"><img src="./lecture_15_slides/slide_62782_00-34-54.826.jpg" width="75%" alt="Lecture Video at 00:34:54.826" /></p>

There have been things like scan data sets, where people actually go inside your home or into an office. They use a $3$D scanner to scan the home, and they include some annotations.



<p align="center"><img src="./lecture_15_slides/slide_63210_00-35-09.107.jpg" width="75%" alt="Lecture Video at 00:35:09.107" /></p>

More recently, you can do that even with your iPhone right now. But still, these kinds of data sets are much smaller. For example, the first version scanner has $1,500$. I think there is a second version ($\text{V}2$), which is roughly the same size, maybe $2,000$ or $3,000$ rooms.

The amount of data you have for $3$D data, and for $3$D scenes in particular, is even much smaller than... The amount of data you have for 3D objects.



<p align="center"><img src="./lecture_15_slides/slide_64478_00-35-51.415.jpg" width="75%" alt="Lecture Video at 00:35:51.415" /></p>

There are attempts being made in trying to collect more data. And finally, if you want to apply deep learning to 3D vision, what are the tasks we care about?



<p align="center"><img src="./lecture_15_slides/slide_64702_00-35-58.890.jpg" width="75%" alt="Lecture Video at 00:35:58.890" /></p>

There is generative modeling, just like—just like what Justin said. You can generate 2D images or videos; you can also generate 3D shapes. You can generate 3D scenes and make them conditioned. The condition can be on language, condition on image, or if you have an input image, how can you reconstruct the 3D object?



<p align="center"><img src="./lecture_15_slides/slide_65236_00-36-16.707.jpg" width="75%" alt="Lecture Video at 00:36:16.707" /></p>

<p align="center"><img src="./lecture_15_slides/slide_65246_00-36-17.041.jpg" width="75%" alt="Lecture Video at 00:36:17.041" /></p>

And you have to learn the shape priors. You have to do shape generation and completion.



<p align="center"><img src="./lecture_15_slides/slide_65584_00-36-28.319.jpg" width="75%" alt="Lecture Video at 00:36:28.319" /></p>

<p align="center"><img src="./lecture_15_slides/slide_66862_00-37-10.962.jpg" width="75%" alt="Lecture Video at 00:37:10.962" /></p>

<p align="center"><img src="./lecture_15_slides/slide_66874_00-37-11.362.jpg" width="75%" alt="Lecture Video at 00:37:11.362" /></p>

<p align="center"><img src="./lecture_15_slides/slide_67568_00-37-34.518.jpg" width="75%" alt="Lecture Video at 00:37:34.518" /></p>

<p align="center"><img src="./lecture_15_slides/slide_67604_00-37-35.720.jpg" width="75%" alt="Lecture Video at 00:37:35.720" /></p>

<p align="center"><img src="./lecture_15_slides/slide_68496_00-38-05.483.jpg" width="75%" alt="Lecture Video at 00:38:05.483" /></p>

<p align="center"><img src="./lecture_15_slides/slide_69140_00-38-26.971.jpg" width="75%" alt="Lecture Video at 00:38:26.971" /></p>

<p align="center"><img src="./lecture_15_slides/slide_69164_00-38-27.772.jpg" width="75%" alt="Lecture Video at 00:38:27.772" /></p>

<p align="center"><img src="./lecture_15_slides/slide_69178_00-38-28.239.jpg" width="75%" alt="Lecture Video at 00:38:28.239" /></p>

<p align="center"><img src="./lecture_15_slides/slide_69792_00-38-48.726.jpg" width="75%" alt="Lecture Video at 00:38:48.726" /></p>

Sometimes, you have a partial object, and you want to repair it; you want to fix it. So there is geometric data processing as well. Other tasks include discriminative models. For example, you have a 3D shape.

How can you classify what category of object it belongs to? Is it a chair or a table? So you just take a 3D object; you can render it into a picture. You can upload a picture to GPT, and they can do it for you.

So that is, in some sense, one way of solving these discriminative problems. But there are also more specific things that are not very easy to solve. For example, you have different types of cells and 3D scans. How can you classify the cell?

And all these specialized domains where you don't have that much data—how can you solve these discriminative problems? Or joint modeling of 2D and 3D data, which is becoming more and more important because we have so much more data in both. We have so many images and videos; we have very good foundation models trained on them. So how can we leverage the priors in our 2D foundation models—like what an image looks like, how to make an image look realistic, how to make a video look realistic?

How can we use that information to help our 3D reconstruction be more realistic? Joint modeling in 2D and 3D data is important because there are so many large-scale 2D datasets and very good pretrained models. Also, there have been a lot of advances in neural rendering or differential rendering methods that basically connect the 3D world and 2D world.

Because you have a 3D world, you have a 3D model; you can render them into 2D. The rendering process can be made differentiable or approximated with neural networks. Sometimes, you even want to do some joint multimodal beyond visual data, including text data. Sometimes, you have other data.

Let's say in robotics, you often have tactile data; how to fuse them as well? And sometimes for autonomous driving, maybe you have LiDAR data or depth data; how can you fuse them as well? I want to use deep learning on 3D data to solve all these different problems. We spend all the time talking about representations.

So how do we begin with? As I suggested, people who are initially doing that are computer vision people who work on pixels. They work on images. Naturally, they say why don't we start with voxels?

But even before that, they say, "Okay, this is an old idea." And this is the very first idea that people tried in applying deep learning to 3D vision. In some sense, it's coming back. But the very first idea they tried was: let's don't even worry about voxels.

Let's just say you have a 3D shape—it's a mesh, it's voxel, whatever. And I want you to learn to recognize what an object is. What is the object here? It's a chair.

But what if the input is 3D data?



<p align="center"><img src="./lecture_15_slides/slide_70384_00-39-08.479.jpg" width="75%" alt="Lecture Video at 00:39:08.479" /></p>

How can we process that before we had a 3D deep learning method? What if I just render it into images because I have very good image models? I just take the 3D objects; I would just put cameras at different places. I can render all these images, the object from different views.



<p align="center"><img src="./lecture_15_slides/slide_70810_00-39-22.693.jpg" width="75%" alt="Lecture Video at 00:39:22.693" /></p>

And then this becomes a 2D problem.



<p align="center"><img src="./lecture_15_slides/slide_70902_00-39-25.763.jpg" width="75%" alt="Lecture Video at 00:39:25.763" /></p>

I will just apply a convolutional neural network on each of these views and find some ways to fuse them using pooling or whatever.



<p align="center"><img src="./lecture_15_slides/slide_70978_00-39-28.299.jpg" width="75%" alt="Lecture Video at 00:39:28.299" /></p>

Then I just do image classification. So this becomes an image classification problem. The only difference is that now you have multiple views. This is like, in some sense, one of the very first ideas people applied to 3D vision: they just used 2D networks.

Why did you want to use 2D networks? Because back then, they were pushing on ImageNet, and they're very good. So ImageNet is much larger than 3D datasets. Any model that are pretrained on ImageNet have very good performance.

The easiest way to solve this 3D recognition problem is to first render it into 2D. Later, people moved away from it because people were like, "Oh, we have more 3D data; we should try to do 3D native methods." People also came up with ideas about connecting 3D and 2D through neural rendering. But now I feel like this trend is coming back because all these image and video models are getting so great.

I don't know if many of you may have seen the VLA or whatever was released yesterday. So how can we incorporate that? But coming back, this is the very first method. In some sense, people try to apply deep learning on $3D$ data just by converting them into $2D$.



<p align="center"><img src="./lecture_15_slides/slide_73286_00-40-45.309.jpg" width="75%" alt="Lecture Video at 00:40:45.309" /></p>

<p align="center"><img src="./lecture_15_slides/slide_73538_00-40-53.717.jpg" width="75%" alt="Lecture Video at 00:40:53.717" /></p>

<p align="center"><img src="./lecture_15_slides/slide_73550_00-40-54.118.jpg" width="75%" alt="Lecture Video at 00:40:54.118" /></p>

<p align="center"><img src="./lecture_15_slides/slide_73564_00-40-54.585.jpg" width="75%" alt="Lecture Video at 00:40:54.585" /></p>

<p align="center"><img src="./lecture_15_slides/slide_73658_00-40-57.721.jpg" width="75%" alt="Lecture Video at 00:40:57.721" /></p>

You can leverage a lot of literature on $2D$ image pre-trained models.



<p align="center"><img src="./lecture_15_slides/slide_73708_00-40-59.390.jpg" width="75%" alt="Lecture Video at 00:40:59.390" /></p>

<p align="center"><img src="./lecture_15_slides/slide_73786_00-41-01.992.jpg" width="75%" alt="Lecture Video at 00:41:01.992" /></p>

But the issue is that you need some projections, but sometimes the input can be very noisy. People ask: "What if my input is too noisy? The point clouds or whatever are just not very good. If I render them, they look kind of bad."

So, is it possible for us to come up with more $3D$ native methods?



<p align="center"><img src="./lecture_15_slides/slide_74218_00-41-16.407.jpg" width="75%" alt="Lecture Video at 00:41:16.407" /></p>

Later, people tried a number of $3D$ native methods that just applied deep learning directly on $3D$ data. As I said, the easiest way to do this is just to apply your pixel convolution neural network into a voxel, volumetric convolution neural network.



<p align="center"><img src="./lecture_15_slides/slide_74642_00-41-30.554.jpg" width="75%" alt="Lecture Video at 00:41:30.554" /></p>

<p align="center"><img src="./lecture_15_slides/slide_74764_00-41-34.625.jpg" width="75%" alt="Lecture Video at 00:41:34.625" /></p>

This is actually a Deep Belief Network, which is a generative network, but it still uses some $3D$ convolutional filters. This was done in 2015 by Princeton, and you can see that their generative model could actually synthesize $3D$ shapes in the form of $3D$ voxels at relatively lower resolution. But this is ten years ago now.



<p align="center"><img src="./lecture_15_slides/slide_75392_00-41-55.579.jpg" width="75%" alt="Lecture Video at 00:41:55.579" /></p>

Back then, this was considered pretty impressive. You can do all these conditional generations, conditional semantic labels at bats and a desk and tables; you can synthesize these different shapes.



<p align="center"><img src="./lecture_15_slides/slide_75770_00-42-08.192.jpg" width="75%" alt="Lecture Video at 00:42:08.192" /></p>

Because this is a generative network, you can also use it for classification, so you can do image shape classification as well.



<p align="center"><img src="./lecture_15_slides/slide_75934_00-42-13.664.jpg" width="75%" alt="Lecture Video at 00:42:13.664" /></p>

<p align="center"><img src="./lecture_15_slides/slide_76002_00-42-15.933.jpg" width="75%" alt="Lecture Video at 00:42:15.933" /></p>

<p align="center"><img src="./lecture_15_slides/slide_76022_00-42-16.600.jpg" width="75%" alt="Lecture Video at 00:42:16.600" /></p>

Later, something that we actually did was to apply GANs—this Generative Adversarial Network. You can use GANs to generate $2D$ pixels. There's no reason you cannot use GANs to generate $3D$ voxels, so we just applied a GAN to $3D$ voxels and actually achieved a pretty good generation of $3D$ objects. This was eight or nine years ago.



<p align="center"><img src="./lecture_15_slides/slide_76678_00-42-38.489.jpg" width="75%" alt="Lecture Video at 00:42:38.489" /></p>

<p align="center"><img src="./lecture_15_slides/slide_76814_00-42-43.027.jpg" width="75%" alt="Lecture Video at 00:42:43.027" /></p>

Later, with training from CMU, we also did an extension. This meant you can use GANs to not only generate $3D$ shapes but also render them into $2D$. You can project them into $2D$ surfaces so that you can get the depth map of the $3D$ objects you generated. Then, you can use a CycleGAN to convert this depth map into a color image.

Now, you can have adversarial losses not only on $3D$ shapes but also on $2D$ pictures. You want $3D$ shapes to look realistic so that they are indistinguishable from the $3D$ object data you have.



<p align="center"><img src="./lecture_15_slides/slide_77878_00-43-18.529.jpg" width="75%" alt="Lecture Video at 00:43:18.529" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78216_00-43-29.807.jpg" width="75%" alt="Lecture Video at 00:43:29.807" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78226_00-43-30.140.jpg" width="75%" alt="Lecture Video at 00:43:30.140" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78244_00-43-30.741.jpg" width="75%" alt="Lecture Video at 00:43:30.741" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78328_00-43-33.544.jpg" width="75%" alt="Lecture Video at 00:43:33.544" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78344_00-43-34.078.jpg" width="75%" alt="Lecture Video at 00:43:34.078" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78354_00-43-34.411.jpg" width="75%" alt="Lecture Video at 00:43:34.411" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78400_00-43-35.946.jpg" width="75%" alt="Lecture Video at 00:43:35.946" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78414_00-43-36.413.jpg" width="75%" alt="Lecture Video at 00:43:36.413" /></p>

<p align="center"><img src="./lecture_15_slides/slide_78424_00-43-36.747.jpg" width="75%" alt="Lecture Video at 00:43:36.747" /></p>

You can even transfer the texture of one car onto the shape of another car.



<p align="center"><img src="./lecture_15_slides/slide_78544_00-43-40.751.jpg" width="75%" alt="Lecture Video at 00:43:40.751" /></p>

This was in 2018. People started trying applying deep networks like convolutional neural networks, generative adversarial networks, on $3D$ voxels instead of $2D$ pixels. But we wondered if we could do a little bit better with voxels because one thing people complained about is that they are just really slow; you have to pre-sample them.



<p align="center"><img src="./lecture_15_slides/slide_79376_00-44-08.512.jpg" width="75%" alt="Lecture Video at 00:44:08.512" /></p>

There is also a lot of wasted effort because many sample points are like empty space, or they are inside objects and give no information. Naturally, people thought: "OK, can we actually make it better?" So there were improvements to voxels, like octave trees.



<p align="center"><img src="./lecture_15_slides/slide_79572_00-44-15.052.jpg" width="75%" alt="Lecture Video at 00:44:15.052" /></p>

<p align="center"><img src="./lecture_15_slides/slide_79612_00-44-16.387.jpg" width="75%" alt="Lecture Video at 00:44:16.387" /></p>

<p align="center"><img src="./lecture_15_slides/slide_80032_00-44-30.401.jpg" width="75%" alt="Lecture Video at 00:44:30.401" /></p>

<p align="center"><img src="./lecture_15_slides/slide_80046_00-44-30.868.jpg" width="75%" alt="Lecture Video at 00:44:30.868" /></p>

When I'm in this empty space or inside objects where I don't care too much about what's going on, I can have huge voxels.



<p align="center"><img src="./lecture_15_slides/slide_80784_00-44-55.492.jpg" width="75%" alt="Lecture Video at 00:44:55.492" /></p>

So you can recursively partition the space, and you can have different sizes of voxels at different spaces.



<p align="center"><img src="./lecture_15_slides/slide_81042_00-45-04.101.jpg" width="75%" alt="Lecture Video at 00:45:04.101" /></p>

<p align="center"><img src="./lecture_15_slides/slide_81080_00-45-05.369.jpg" width="75%" alt="Lecture Video at 00:45:05.369" /></p>

This allows you to really scale up. Compared with just directly using voxels, this is circa 2019.



<p align="center"><img src="./lecture_15_slides/slide_81612_00-45-23.120.jpg" width="75%" alt="Lecture Video at 00:45:23.120" /></p>

<p align="center"><img src="./lecture_15_slides/slide_81626_00-45-23.587.jpg" width="75%" alt="Lecture Video at 00:45:23.587" /></p>

<p align="center"><img src="./lecture_15_slides/slide_81646_00-45-24.254.jpg" width="75%" alt="Lecture Video at 00:45:24.254" /></p>

With octave trees, you can do $256$, and you can even use that for generation as well. You can generate objects; they look like voxels but are higher resolution because you are more efficient in representing the space.



<p align="center"><img src="./lecture_15_slides/slide_81998_00-45-35.999.jpg" width="75%" alt="Lecture Video at 00:45:35.999" /></p>

These were the very early attempts in applying deep learning to $3D$ space. When we ask, "Why don't we just try voxels?" this is the moment where people have gained more interest thinking, "Oh no, now what if you graphics people feel like—you are doing all this wrong." Why do you want to use these kind of pretty inefficient, ugly-looking representations, like voxels or octave trees?

Now we have all these good representations—point clouds, meshes, splines. Why aren't we using these representations? But as we said, the challenge is that points are here and there. How can you even apply convolutional operations to points and stuff like that?



<p align="center"><img src="./lecture_15_slides/slide_83074_00-46-11.902.jpg" width="75%" alt="Lecture Video at 00:46:11.902" /></p>

It's just not very obvious. However, people start to look into it.



<p align="center"><img src="./lecture_15_slides/slide_83392_00-46-22.513.jpg" width="75%" alt="Lecture Video at 00:46:22.513" /></p>

<p align="center"><img src="./lecture_15_slides/slide_83404_00-46-22.913.jpg" width="75%" alt="Lecture Video at 00:46:22.913" /></p>

<p align="center"><img src="./lecture_15_slides/slide_83874_00-46-38.595.jpg" width="75%" alt="Lecture Video at 00:46:38.595" /></p>

It is called PointNet. The idea is that for points, you have to be permutation invariant. There's no guaranteed ordering in the sense that our top left might be $(1, 1)$ and the bottom right might be $(100, 100)$. If the points are unordered, you have to be permutation invariant.



<p align="center"><img src="./lecture_15_slides/slide_85176_00-47-22.039.jpg" width="75%" alt="Lecture Video at 00:47:22.039" /></p>

Second, you also need to be sampling invariant. Sometimes, you sample, say, $10$ points on the head of a bunny or rabbit and $5$ points on the tail of the rabbit. Other times, you might sample $10$ points on the tail of the rabbit and only $5$ points on the head of the rabbit. How can you also be invariant to that?

Because there's no guarantee regarding your sample points.



<p align="center"><img src="./lecture_15_slides/slide_86108_00-47-53.136.jpg" width="75%" alt="Lecture Video at 00:47:53.136" /></p>

<p align="center"><img src="./lecture_15_slides/slide_86298_00-47-59.476.jpg" width="75%" alt="Lecture Video at 00:47:59.476" /></p>

The most important point, which they used, is applying a symmetric function on the embeddings of the points. Basically, for all the points, I first compute some embeddings for them, just like computing embeddings for different regions or windows of an image. Then I compute the features for each point and have to fuse them.



<p align="center"><img src="./lecture_15_slides/slide_87100_00-48-26.236.jpg" width="75%" alt="Lecture Video at 00:48:26.236" /></p>

It can also be a sum function; I just add them up.



<p align="center"><img src="./lecture_15_slides/slide_87456_00-48-38.115.jpg" width="75%" alt="Lecture Video at 00:48:38.115" /></p>

This is very simple: you have $N$ number of points—$1, 2, 3$, or whatever index—and then you compute embeddings for them and they aggregate them. You can compute the max for each dimension or sum them up like that. You end up with these aggregated embeddings for all the points. Then you go through maybe a few layers of fully connected networks, and you use it to classify: are these points representing a chair or a table?

That is basically what's going on, and it turned out to be quite powerful. Of course, there have been many improvements on top of that. People have come up with new methods that improve on PointNet, such as PointNet++.



<p align="center"><img src="./lecture_15_slides/slide_88400_00-49-09.613.jpg" width="75%" alt="Lecture Video at 00:49:09.613" /></p>

<p align="center"><img src="./lecture_15_slides/slide_88414_00-49-10.080.jpg" width="75%" alt="Lecture Video at 00:49:10.080" /></p>

<p align="center"><img src="./lecture_15_slides/slide_88592_00-49-16.019.jpg" width="75%" alt="Lecture Video at 00:49:16.019" /></p>

<p align="center"><img src="./lecture_15_slides/slide_88818_00-49-23.560.jpg" width="75%" alt="Lecture Video at 00:49:23.560" /></p>

But the original idea in the PointNet paper is so simple, and it turned out to be also very powerful.



<p align="center"><img src="./lecture_15_slides/slide_89210_00-49-36.640.jpg" width="75%" alt="Lecture Video at 00:49:36.640" /></p>

Something else you want to consider is measurement. For pixels, it's easy: I have an output image, and I have the ground truth image. I just compute the differences between the two; I have a loss function. For points, how would you compare the output point cloud and the input point cloud, especially if you are dealing with a generation task?

If you do classification, that's fine: you have an input point cloud, and the output is "chair" or "table," etc. You use a cross-entropy loss; that's all you need. However, if your output is a point cloud (a set of points), how would you compare the output point cloud versus the ground truth point cloud?



<p align="center"><img src="./lecture_15_slides/slide_90480_00-50-19.016.jpg" width="75%" alt="Lecture Video at 00:50:19.016" /></p>

<p align="center"><img src="./lecture_15_slides/slide_90492_00-50-19.416.jpg" width="75%" alt="Lecture Video at 00:50:19.416" /></p>

You have to design distance metrics. The two common distance metrics that people used include the Chamfer distance. A Chamfer distance is easy to understand: you have two sets of points. For each point on either side, you just basically find the nearest neighbor.

So you have a collection of red points and a collection of blue points. For each red point, you find its nearest neighbor in the blue set. And for each of the blue points, you just... Find the nearest neighbor in the red set.

You want to minimize the distance, minimize the distance of each point to its nearest neighbor in the other set. Another loss function that people may use is called Earth Mover's distance. These are the two common metrics that people use when they're comparing the distance between point clouds.



<p align="center"><img src="./lecture_15_slides/slide_92268_00-51-18.675.jpg" width="75%" alt="Lecture Video at 00:51:18.675" /></p>

So we have moved from voxels to point clouds. People were like, "Okay, this is great. Now I can process the points, I can output points." But we also have other beautiful partitions like splines.

They are very good at capturing the surfaces of objects. So how can we have a neural network that can output or understand objects but also represent beautiful surfaces? People thought about how I could integrate neural networks with things like splines or functions like that.



<p align="center"><img src="./lecture_15_slides/slide_93392_00-51-56.179.jpg" width="75%" alt="Lecture Video at 00:51:56.179" /></p>

<p align="center"><img src="./lecture_15_slides/slide_93456_00-51-58.315.jpg" width="75%" alt="Lecture Video at 00:51:58.315" /></p>

A notable example is this thing called AtlasNet.



<p align="center"><img src="./lecture_15_slides/slide_93748_00-52-08.058.jpg" width="75%" alt="Lecture Video at 00:52:08.058" /></p>

What's going on here is they try to use deep learning, but instead of directly outputting a set of 3D point clouds, it learns a transformation function. I have latent shape representations. When we say you have these parametric representation of object shapes, you're basically transforming, let's say, a 2D space of $u$ and $v$ into a 3D space, like a sphere.

For simple things like a sphere, it is easy; you can write it down using sine and cosine or whatever. But for complex objects, it is very hard to write a function, and often, there is no closed form. The idea here is: if there is no closed form, then why don't we just use a neural network to represent that? Here you can see this neural network, which is implemented as MLP; it just learns that function $f$.

You can take the two values, $u$ and $v$, as the input to the function $f$. The neural network is performing the computation of the function $f(u, v)$ and outputs a point in 3D space.



<p align="center"><img src="./lecture_15_slides/slide_95296_00-52-59.709.jpg" width="75%" alt="Lecture Video at 00:52:59.709" /></p>

<p align="center"><img src="./lecture_15_slides/slide_95306_00-53-00.043.jpg" width="75%" alt="Lecture Video at 00:53:00.043" /></p>

It's basically learning how we are able to transform this 2D space into the 3D space. Think about it as now you have a piece of paper; you can fold it in different ways, and you can fold it multiple times. All these things get put together to form the final shapes you care about.



<p align="center"><img src="./lecture_15_slides/slide_96120_00-53-27.204.jpg" width="75%" alt="Lecture Video at 00:53:27.204" /></p>

You can see the differences between three different representations here. You have an input image, and if you want to represent it reconstructed using voxels, you can see it's doing something, but you are really bounded by limited resolution voxels. For point clouds, you are no longer bounded by resolutions, and it gives you maybe a bit more details.

But the points are really unordered; you cannot really get any smooth surfaces out of the point clouds. For this thing called AtlasNet, which is basically learning transform pieces, you can see that they have actually smoother surfaces. When they are combined, they give you the final output geometries conditioned on 2D images.



<p align="center"><img src="./lecture_15_slides/slide_97606_00-54-16.786.jpg" width="75%" alt="Lecture Video at 00:54:16.786" /></p>

Finally, in some sense, we can put it this way. What is deep network doing when they're doing ImageNet classification? That function is really complex, and the output space is really small. The output space is 1,000 dimensions.

It's like: "Okay, is it a cat or dog?" You have 1,000 way classification. Output space is so small. Input space is much larger because you have $500 \times 500$ pixels, which is 250,000 or something.

The input space is much larger. The output space is really small. The function is really hard to write. I cannot do that.

The function is so hard to write; there's no closed form. That's why I need a deep network. Input space is large, output space is small. What are the representations that really map them—map it the best?

What's the optimal representation? That seems to really fit into the paradigm? If we think more carefully, around 2019, people realized that deep network is an implicit function. Why don't we just use it to represent an implicit function for object $\text{3D}$ geometry?

Finally, people moved to take the leap from specific representations on point clouds or splines into implicit representations, but not directly working on voxels. Instead, think about it as a level set or some implicit functions that use deep network to represent that. That's the final step. Going from this AtlasNet or whatever, you're learning the transformation from a $\text{2D}$ space to $\text{3D}$ space.

But now we can directly do an implicit query using the deep networks.



<p align="center"><img src="./lecture_15_slides/slide_103008_00-57-17.033.jpg" width="75%" alt="Lecture Video at 00:57:17.033" /></p>

They all argued that before, we had been using voxels and point clouds and meshes, or whatever; they have their own strengths or weaknesses.



<p align="center"><img src="./lecture_15_slides/slide_103230_00-57-24.441.jpg" width="75%" alt="Lecture Video at 00:57:24.441" /></p>

<p align="center"><img src="./lecture_15_slides/slide_103242_00-57-24.841.jpg" width="75%" alt="Lecture Video at 00:57:24.841" /></p>

<p align="center"><img src="./lecture_15_slides/slide_103260_00-57-25.442.jpg" width="75%" alt="Lecture Video at 00:57:25.442" /></p>

But really, the right thing to do is just send the query into the deep network. What it should do is take the input—let's say $\mathbf{x}, y, z$-coordinate—and output whether that point is inside and outside the object. This was one of the final ideas proposed in 2019. Even right now, in 2025, a lot of people are still using this same idea: just use deep network to tell us whether a point is inside or outside an object.

Or what are the density values of the point? Later, what will be the color? What will be the radiance values of the point?



<p align="center"><img src="./lecture_15_slides/slide_105142_00-58-28.238.jpg" width="75%" alt="Lecture Video at 00:58:28.238" /></p>

<p align="center"><img src="./lecture_15_slides/slide_105158_00-58-28.771.jpg" width="75%" alt="Lecture Video at 00:58:28.771" /></p>

<p align="center"><img src="./lecture_15_slides/slide_106168_00-59-02.472.jpg" width="75%" alt="Lecture Video at 00:59:02.472" /></p>

Then, one or two years later, people came up with this thing called NeRF.



<p align="center"><img src="./lecture_15_slides/slide_106700_00-59-20.223.jpg" width="75%" alt="Lecture Video at 00:59:20.223" /></p>

<p align="center"><img src="./lecture_15_slides/slide_106722_00-59-20.957.jpg" width="75%" alt="Lecture Video at 00:59:20.957" /></p>

The difference here is now that we use deep knowledge to query not only what will be the signed distance function or density of the objects, but also querying the radiance. Here you can see what's going on: you query NeRF about $\mathbf{x}, y, z$-coordinate in the $\text{3D}$ space. In addition to that, because we are trying to model the appearance as well, you also query the viewing directions and the camera viewing directions.

The output of the neural network is not just like $1$ or $0$, inside/outside; it is the density values in addition to the color values or the radiance.



<p align="center"><img src="./lecture_15_slides/slide_107458_00-59-45.515.jpg" width="75%" alt="Lecture Video at 00:59:45.515" /></p>

But here, you want to train on $\text{2D}$ images. That's what's going on with NeRF. They also put that together with the neural rendering—volume rendering function.



<p align="center"><img src="./lecture_15_slides/slide_108316_01-00-14.143.jpg" width="75%" alt="Lecture Video at 01:00:14.143" /></p>

They made this volume rendering function differentiable, in the sense that you can have a rendering model; you can query all these different points in $\text{3D}$ space.



<p align="center"><img src="./lecture_15_slides/slide_108564_01-00-22.418.jpg" width="75%" alt="Lecture Video at 01:00:22.418" /></p>

This is basically volume rendering as in computer graphics. There's very minimal change made because you can see, even directly from the volume-rendering equations, that everything here is an approximation. But with approximation, everything here is differentiable. So you can compute how much light...

You can think of it as opacity of a point in 3D space, and it also gives you the color. Then you can compute how much light has been blocked by the points that are sampled ahead of that point.



<p align="center"><img src="./lecture_15_slides/slide_109800_01-01-03.660.jpg" width="75%" alt="Lecture Video at 01:01:03.660" /></p>

You can also compute how much light is contributing to what I'm going to see in this ray from any particular point. Now you have a few things: you have neural networks to represent implicit functions for the colors or radiance, and the densities.



<p align="center"><img src="./lecture_15_slides/slide_110342_01-01-21.744.jpg" width="75%" alt="Lecture Video at 01:01:21.744" /></p>

And then you have this volume render equations, which are made differentiable so that you can learn directly from 2D images. These are the two things that have changed. One is, I no longer have to train on 3D shapes; I can train on 2D images with these volume rendering equations. The second change is that instead of just looking into geometry or density of objects in 3D, I also look into their radiance or appearance in 3D.

These two changes lead to a big jump from NeRF, or from implicit functions or deep [SDF] and all these other methods to NeRF. A lot of people feel like, "Oh yeah, NeRF has been great."



<p align="center"><img src="./lecture_15_slides/slide_111370_01-01-56.045.jpg" width="75%" alt="Lecture Video at 01:01:56.045" /></p>

It seems like out of nowhere.



<p align="center"><img src="./lecture_15_slides/slide_111714_01-02-07.523.jpg" width="75%" alt="Lecture Video at 01:02:07.523" /></p>

Although, they focus only on geometry, but now I do both geometry and appearance.



<p align="center"><img src="./lecture_15_slides/slide_111882_01-02-13.129.jpg" width="75%" alt="Lecture Video at 01:02:13.129" /></p>

And I do learning from 2D images instead of 3D shapes. Here are some results of NeRF. You may have seen many times.



<p align="center"><img src="./lecture_15_slides/slide_112220_01-02-24.407.jpg" width="75%" alt="Lecture Video at 01:02:24.407" /></p>

<p align="center"><img src="./lecture_15_slides/slide_112290_01-02-26.743.jpg" width="75%" alt="Lecture Video at 01:02:26.743" /></p>

If you remember, we said in the past that we had been working on something like generating 3D shapes and then also generating their 2D appearances. At the very beginning, we used a representation that is voxels. But now, as we said, NeRF is great. If we have implicit representations, there's no need to really represent it as voxels.



<p align="center"><img src="./lecture_15_slides/slide_112976_01-02-49.632.jpg" width="75%" alt="Lecture Video at 01:02:49.632" /></p>

What if we just replace that with radiance fields? We also did that. We have a neural network that captures the implicit radiance fields and densities, but it is generative neural network.



<p align="center"><img src="./lecture_15_slides/slide_113210_01-02-57.440.jpg" width="75%" alt="Lecture Video at 01:02:57.440" /></p>

<p align="center"><img src="./lecture_15_slides/slide_113300_01-03-00.443.jpg" width="75%" alt="Lecture Video at 01:03:00.443" /></p>

You can even still apply the same GAN rendering framework so that you can render objects in 3D as well as their 2D pictures. You can also do the same thing as controllability. You can change the camera viewpoint. You can change object identity, but you can keep the viewpoint; you can do all the things that you can do before.

But now with NeRF, you can learn directly from images. So you don't have to restrict yourself to categories of cars or chairs where you have a lot of 3D data because you can learn directly from images. You can see that now the output becomes much more realistic. This is what we did called [pigeon] with Eric Chen as a first author, and also with mostly people from Gordon's group.



<p align="center"><img src="./lecture_15_slides/slide_114542_01-03-41.884.jpg" width="75%" alt="Lecture Video at 01:03:41.884" /></p>

NeRF is great, but NeRF has an issue: you have to sample a lot of points in 3D. You are no longer pre-sampling them and then applying a volumetric convolution.



<p align="center"><img src="./lecture_15_slides/slide_115884_01-04-26.662.jpg" width="75%" alt="Lecture Video at 01:04:26.662" /></p>

<p align="center"><img src="./lecture_15_slides/slide_115970_01-04-29.532.jpg" width="75%" alt="Lecture Video at 01:04:29.532" /></p>

<p align="center"><img src="./lecture_15_slides/slide_116796_01-04-57.093.jpg" width="75%" alt="Lecture Video at 01:04:57.093" /></p>

But still, just like a level set, you have to sample all the points and create neural network all the time. Now you can learn from 2D, and you can do all these great things. But because you still have to do all the sampling, it is very slow. So people thought about this again, from the graphics people.

They were like, "OK, I have this good idea about points and meshes." The nice thing about them is they are free in space; they are very efficient. Is it possible for us to integrate the two? Can I have implicit representations but maybe I don't have to use a fixed sampling grid?

I don't have to sample all the time because it takes so much time. Maybe I really should put them together. You can argue that NeRF tried to parameterize densities—sorry, parameterize the scenes very densely; you have to sample all the points density in 3D. A lot of points are wasted, just like in voxels.

You have all the points that are representing empty space. You don't want that. In NeRF, a lot of sampling, a lot of queries are also querying empty space. The network may give you a density of $0$ or something like that, but it's taking a lot of time.

How can we address that? What if I just try to sample things more sparsely? I still have implicit representations, but instead of sampling empty spaces all the time, I only sample at places where I know there are stuff. But how can I know that?

What if I have point representations? This is the idea behind this thing called Gaussian splats, which you may have heard of. It still has the same implicit functions; you're querying neural network for densities and for appearance and stuff like that. But instead of creating neural network all the time, I have a point representation of the 3D Gaussian blobs in the 3D space, which I think sometimes you can think about them as point clouds.

The points are not like a single point; they're like a blob. They're like some regions. And because you know where these blobs are, when you're sending out a ray from your camera to the 3D space and sampling points, you don't have to sample all the time. You just look at where these blobs are, and then you can know—based on the radius...

of these different Gaussians, you will only sample at regions where you know there is some stuff.



<p align="center"><img src="./lecture_15_slides/slide_118728_01-06-01.557.jpg" width="75%" alt="Lecture Video at 01:06:01.557" /></p>

So this makes rendering much more efficient. And so here are some of the reconstruction results using 3D Gaussian splats.



<p align="center"><img src="./lecture_15_slides/slide_119116_01-06-14.503.jpg" width="75%" alt="Lecture Video at 01:06:14.503" /></p>

<p align="center"><img src="./lecture_15_slides/slide_119130_01-06-14.971.jpg" width="75%" alt="Lecture Video at 01:06:14.971" /></p>

You can see that in terms of quality, they're actually not—they're comparable. I would say they're comparable to NeRFs. This is different metrics: PSNR, SSIM. They're like rendering qualities.

And I think the $y$-axis doesn't start from 0, so this is a little misleading. But basically, you can see these numbers are really close.



<p align="center"><img src="./lecture_15_slides/slide_119822_01-06-38.060.jpg" width="75%" alt="Lecture Video at 01:06:38.060" /></p>

So in terms of quality and rendering quality, Gaussian splats and NeRFs are similar, at least when they were first proposed. However, Gaussian splats are just much more efficient. This is FPS, Frames Per Second. You can render 150 pictures per second.

For NeRF, it takes you maybe 20 seconds to render just a single picture. So now this thing is made 1,000 times faster, at least that's what they argued. Because you no longer waste all your computing power on sampling empty space and querying neural networks all the time about these points that are in the empty space.



<p align="center"><img src="./lecture_15_slides/slide_120790_01-07-10.359.jpg" width="75%" alt="Lecture Video at 01:07:10.359" /></p>

<p align="center"><img src="./lecture_15_slides/slide_121242_01-07-25.441.jpg" width="75%" alt="Lecture Video at 01:07:25.441" /></p>

<p align="center"><img src="./lecture_15_slides/slide_121540_01-07-35.384.jpg" width="75%" alt="Lecture Video at 01:07:35.384" /></p>

Because often, there could be—the chairs are symmetric. So we talk a little bit about it, where there's a parametric surface. And you can parameterize part of the surface using a sphere or stuff like that, using these closed-form equations. And that gives you a little bit of symmetry.

People also come up with different representations for it as well.



<p align="center"><img src="./lecture_15_slides/slide_122464_01-08-06.215.jpg" width="75%" alt="Lecture Video at 01:08:06.215" /></p>

<p align="center"><img src="./lecture_15_slides/slide_122474_01-08-06.549.jpg" width="75%" alt="Lecture Video at 01:08:06.549" /></p>

None of them is directly capturing things like regularities, like symmetry and repetition.



<p align="center"><img src="./lecture_15_slides/slide_122878_01-08-20.029.jpg" width="75%" alt="Lecture Video at 01:08:20.029" /></p>

So how can we capture that?



<p align="center"><img src="./lecture_15_slides/slide_123194_01-08-30.573.jpg" width="75%" alt="Lecture Video at 01:08:30.573" /></p>

<p align="center"><img src="./lecture_15_slides/slide_123258_01-08-32.708.jpg" width="75%" alt="Lecture Video at 01:08:32.708" /></p>

<p align="center"><img src="./lecture_15_slides/slide_123558_01-08-42.718.jpg" width="75%" alt="Lecture Video at 01:08:42.718" /></p>

<p align="center"><img src="./lecture_15_slides/slide_123702_01-08-47.523.jpg" width="75%" alt="Lecture Video at 01:08:47.523" /></p>

This is even more so the case for scenes; let's say, a bed. A headboard is usually next to the wall. Chairs are usually next to the tables, and stuff like that. So you not only want to represent them as unrelated collection of parts or objects.



<p align="center"><img src="./lecture_15_slides/slide_124510_01-09-14.483.jpg" width="75%" alt="Lecture Video at 01:09:14.483" /></p>

You want to capture their relationships as well. In the hierarchies—when you are constructing, when you're building—you're doing some construction, your architecture. You're an architect. You're designing a building, then, of course, you're not just representing objects or their relationships.

You have to consider hierarchies: what you build first. There's a classroom, and a classroom has—there are some tables and chairs in it, and chairs has parts.



<p align="center"><img src="./lecture_15_slides/slide_125320_01-09-41.510.jpg" width="75%" alt="Lecture Video at 01:09:41.510" /></p>

So you have hierarchical graph, where, let's say, for chairs, you have different level hierarchy for bases, for seats, for backs. And the bases may have different legs. But then also, the legs themselves are related. The left leg of the chair and the right leg of the chair, they are supposed to be symmetric.

And they should have the identical shape. There are constraints on where these legs are; they have to be really aligned, otherwise the chair is going to fall. So there are all these constraints that are pretty useful. And how can we represent them?

People come up with all these different representations.



<p align="center"><img src="./lecture_15_slides/slide_126636_01-10-25.421.jpg" width="75%" alt="Lecture Video at 01:10:25.421" /></p>

I think this is also from Leonidas's group, from 2019.



<p align="center"><img src="./lecture_15_slides/slide_127176_01-10-43.439.jpg" width="75%" alt="Lecture Video at 01:10:43.439" /></p>

Between these object parts. That's also an important topic. What are the constraints of chairs to satisfy? So is it possible for me to use a large language model to output programs?

But then maybe I can use some implicit functions, or whatever, to capture the specific geometric details of the parts of the objects, like the chairs. There's some kind of new emerging trend of research that is happening right now in these days.



<p align="center"><img src="./lecture_15_slides/slide_129024_01-11-45.100.jpg" width="75%" alt="Lecture Video at 01:11:45.100" /></p>

I think that's all I have. Thank you.



