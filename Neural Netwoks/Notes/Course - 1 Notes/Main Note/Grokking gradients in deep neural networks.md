What factors influence the gradients of the weights of a layer in a deep neural network? Having an intuition for that can help understand why some networks are slow to converge.
[[Step-by-Step Derivation of the Backward Pass in a Neural Network]]
[[Mathematics for Forward and Backward Propagation with Multiple Hidden Layers]]
[[Neural Networks and Deep Learning]]

“To grok” is to fully understand something. And that’s what I want for you dear, reader. In this article, I’m going to explain how different factors in the network affect the computation of gradients at each layer. I’m going to start with an _intuitive_ sense of how the gradients are being calculated which will be easy to remember and to infer from. And I’m going to follow that up with the detailed math.

The following is a visual summary of what we’ll cover in this article. It will serve as a reference that you can come back to whenever you need a refresher … and hopefully it will also serve to pike your interest.

Cheat-sheet summary of this article’s contents

![https://miro.medium.com/v2/resize:fit:1000/1*ibFO1psmxdcAjPFAFm9m_g.png](https://miro.medium.com/v2/resize:fit:1000/1*ibFO1psmxdcAjPFAFm9m_g.png)

# **Gradients and Computation Graphs**

Anyone who’s read anything on neural networks knows that you can’t escape the math, but unless you’ve done a PhD in linear algebra you probably find the math hard to understand at times. The underlying ideas of neural networks are often quite simple, but they become complex due to the variety of notation needed to deal with the quantity of numbers, and they get even worse once you add multiple layers.

The various tutorials and courses out there almost always provide some details about the computation of the gradients in a single-layer network. But to understand the gradients in multi-layer networks requires understanding the entire graph of computations across the entire network, spanning both its forward and backward passes. This is often omitted from those courses for a simple reason — it gets complex to explain.

![https://miro.medium.com/v2/resize:fit:700/1*9sgbOwh_-UUM2fBOJbe0vQ.png](https://miro.medium.com/v2/resize:fit:700/1*9sgbOwh_-UUM2fBOJbe0vQ.png)

Computation graphs take a lot of work to explain

To get to something simple, intuitive, and easily remembered, I’ll initially skip all of that. For those who like deep mathematical explanations, later sections cover the details.

In what follows, you will find two explanations of gradient propagation. The first will use just the bare minimal amount of math needed to explain the high-level concepts and to provide you with useful intuitions. To make that work, I’ll be ignoring some mathematical rigor and working with “intuitive equations” — equations that capture the basic ideas, without worrying about the extra things (like transposes and matching up matrix shapes) that would be needed in the real world.

The later sections will drill deep into the math, explaining everything from first principles.

# **An Intuitive Explanation**

We’ll start with a classic deep network containing some number of fully-connected layers. We’ll represent the number of layers with the letter _L_, the input training data with the letter _X_, and the output predictions as _Ŷ_ (pronounced “Y hat”).

![https://miro.medium.com/v2/resize:fit:700/1*hznqDcYwVLfiMeIkTFFgHQ.png](https://miro.medium.com/v2/resize:fit:700/1*hznqDcYwVLfiMeIkTFFgHQ.png)

Classic deep network

During supervised training, we also know what we want the network to output, which we’ll indicate with the letter _Y_ (no extra “hat” on top). This forms our desired output, and the difference between the predicted and the desired is the _prediction error_.

We usually don’t train against a single data sample; rather, we have a batch of _n_ samples. And what we care about is the _mean prediction error_ across that training batch:

![https://miro.medium.com/v2/resize:fit:89/0*Bhn2B4x4CvKOSCnm.png](https://miro.medium.com/v2/resize:fit:89/0*Bhn2B4x4CvKOSCnm.png)

Now, training uses a loss function, which we’ll denote as _J,_ and there are a few different options. In logistic regression, a Mean Squared Error (MSE) loss function usually works well. For classification tasks, you often use Binary or Categorical Cross-Entry Loss, both of which involve some form of sigmoid or softmax activation on the output and a loss function involving logs. It turns out that all you care about is the _mean prediction error_ when it comes to computing gradients, at least for all those most common configurations. So in the interests of keeping things simple, we can skip considerations of different loss functions for now (don’t worry, I’ll explain this in more detail later).

When computing gradients we start with the mean prediction error and propagate it backwardsthrough the same network:

![https://miro.medium.com/v2/resize:fit:700/1*ifNEqgEdYQxYyMA3NtS9kg.png](https://miro.medium.com/v2/resize:fit:700/1*ifNEqgEdYQxYyMA3NtS9kg.png)

Backprop through a network. Highlights how the weights are used during the forward pass (from left to right), and then the gradients of the weights are computed during backprop (from right to left). (source: author)

The weights, biases, and activations that were used at each layer during the forward pass from left to right, are used again when computing the gradients.

# **Simplified Model without Biases or Activation Functions**

We gain our first set of intuitions by focusing on just the weights. We ignore the biases and activation functions completely by assuming a network where they don’t exist. This gives us the following equation for the gradients of the weights at any layer _l_:

![https://miro.medium.com/v2/resize:fit:700/0*cvyd3grOvS57yRMs.png](https://miro.medium.com/v2/resize:fit:700/0*cvyd3grOvS57yRMs.png)

Intuitive equation for the gradients of the weights at any layer

This equation lacks some of the mathematical rigor required to be used in real calculations, but what we can learn from it still holds.

Technically, what we’re computing here is a matrix, $dJ/dW$, that captures how each of the weights at layer _l_ affects the final loss, _J_. The individual weight values that have a stronger effect on the loss are the ones that we want to focus on updating during training. They also happen to have a larger gradient. So this gradient matrix naturally provides the right coordinates to shift the weights by, during the update step.

So, what influences these gradients?

Let’s focus on the outer ends of the network first.

The _X_ in the equation shows us that training data influences the gradients. Now that’s not surprising, but this equation applies to all layers so the significant point is that the input training data influences the weights of all layers. Furthermore, notice that there’s no squaring, square-rooting, logging, or exponentiating of _X_. It has an equal and linear effect on each layer throughout the depth of the network (we’re ignoring some nuances from the effects of biases, activation functions, and the problems of diminishing gradients).

On the other end is not _Ŷ,_ the network output, or _Y_, the desired output, but their difference. In other words, the gradients of each _W_ are not influenced by the absolute values of the network output, but by the _prediction error_. Or more accurately, the _mean prediction error_. The number of training samples used in each batch doesn’t typically change the size of the gradients. Notice also that, like _X_, the mean prediction error has an equal and linear effect on each layer, regardless of depth.

Looking to the layers now, we see that the weights of almost all layers are represented within the equation. All the early layers 1, 2, etc. are included. All the final layers up to layer L are also included. There’s only one layer missing — the target layer _l_ itself. I’ve made that a little more visible by including an identity matrix, _**I**_, in its place, but it’s only superficial and is not present during real calculations. So the gradients of the weights at layer _l_ are influenced by the weights of all other layers.

If we were looking at the gradients at the first or last layer, the same logic would follow. The gradients of the weights at layer _1_ are influenced by all layers after it. The gradients of the weights at the last layer _L_ are influenced by all layers before it.

Lastly, once again notice that there’s no squaring, square-rooting, logging, exponentiating, or any other non-linearity applied against these weights. They each have a linear effect on the gradients of all other layers, and their effects are equal across the depth of the network.

This outcome will hold, to an extent, as we progress towards more realistic networks with biases and activation functions.

![https://miro.medium.com/v2/resize:fit:700/1*7VUtrsE42MAmzqQILjaUVQ.png](https://miro.medium.com/v2/resize:fit:700/1*7VUtrsE42MAmzqQILjaUVQ.png)

Influences on gradient of weights at target layer in simplified model without bias or activation function

But first, let’s briefly summarize what we’ve found. The gradients of the weights at any layer are influenced by:

- the input data, X
- the mean prediction error, (_Ŷ — Y)/n_
- the weights of all layers except the target layer,
- and those influences all have an equal and linear effect.

# **Simplified Model with Activation Functions**

Now let’s extend our intuitive equation to include activation functions.

In the literature, the activation function is typically indicated by a sigma, σ. It is applied as a function against the results of the weights and the biases against the layer input. Each layer can potentially have a different activation function, and so you’ll often see indices 1, 2, etc. against each σ.

![https://miro.medium.com/v2/resize:fit:700/1*aPeIuWA9A0Pe_xRIAGq-1A.png](https://miro.medium.com/v2/resize:fit:700/1*aPeIuWA9A0Pe_xRIAGq-1A.png)

Backprop through a network with activation functions

In practice, most deep networks today use the ReLU activation function for all of the hidden layers. That function simply passes positive values through without changes, but zeros-out negative values. It’s the fact that the ReLU activation function is linear for half of its range that we could apply our first approximation and drop it entirely. It also makes our next approximation easier to understand.

Thanks to something called the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), the computation of the gradients in a layer with an activation function can be split into two parts. One part is the same as we’ve already seen, with the weights from each layer (except the target layer) being included as is within the equation. The second part requires us to compute _dJ/dA_, the gradient of the loss function w.r.t. to the result of the activation function.

That would require us to calculate the result of applying a function, which is quite achievable but it makes the equation messier and harder to glean useful intuitions from. Instead, it’s possible to replace a σ as an activation function with a matrix (a diagonal matrix, to be precise, with length equal to the number of features in the layer).

![https://miro.medium.com/v2/resize:fit:105/1*J_RcAeG3pPQoFIJMdRiwGw.png](https://miro.medium.com/v2/resize:fit:105/1*J_RcAeG3pPQoFIJMdRiwGw.png)

An example S activation matrix

For a ReLU activation function, our equivalent matrix, which we shall call _S_ (upper-case version of s, for sigma), contains a 1 for every layer output value that should be passed through unchanged, and a 0 for every value that should be zeroed out. The same principle even works for arbitrary activation functions — the values of the _S_ matrix just need to be adjusted.

Now, because _S_ acts like a constant multiplier, it gets included as the same constant multiplier when computing the gradients. Thus we can easily extend our intuitive form of the gradient computation to include the effect of activation functions:

![https://miro.medium.com/v2/resize:fit:700/0*71hAOMz6VYGXBdNA.png](https://miro.medium.com/v2/resize:fit:700/0*71hAOMz6VYGXBdNA.png)

Notice that every weight matrix _W_ is now paired with its activation matrix _S_. This is true even of the target layer _l_, which has its _S_ matrix represented even though its weights are not. For those mathematically inclined, you’ll notice that if we set every S to the identity matrix, then they fall out of the equation and we’re back to our previous model without an activation function.

It’s important to keep in mind that _S_ reflects the output activation of a layer at a point in time. That output activation depends on the values passed to the layer and on its pre-activation results, which in turn depends on the values of the weights in the layer. This also varies for each data sample. So _S_ is not truly a constant matrix, but it can be thought of as constant for a single training batch, which is what matters for us.

This enables us to make some further observations.

Firstly, even when we incorporate activation functions, the observations that we made earlier still largely hold. For example, the weights in each layer still have an approximately equal effect on all other layers.

Secondly, as stated already, while the target layer’s weights do not have a direct effect on its gradients, its activation does. And because the target layer’s activation is a result of its weights, those weights have an indirect effect on their own gradients.

Lastly, this provides a way to make sense of the non-linearities. The linear simplicity of the equation above w.r.t. to the different _W_ matrices belies the fact that the _S_ matrices capture non-linearities in the effect of the _W_s. In other words, it splits the computations into a linear component (the various _W_ matrices themselves) and a non-linear adjustment component (the various _S_ matrices). In most cases that non-linear component has only an _attenuation_ effect against the weights — ie: it might _reduce_ the effect of the weights by some percentage. For example, if all ReLU units are active (ie: positive), then all weights are propagated as is without any non-linear adjustment. If some ReLU units are inactive (ie: zeroed out), then less of the weights are propagated.

# **Intuitive Effect of Inactive Units**

It’s possible to visualize more precisely what I’m getting at. This will give us an important insight into the effect of the activation functions.

I’ve mentioned that each _S_ is a diagonal matrix, but I haven’t explained it in much detail. Consider a single data sample as it passes through the network. After the weights and bias (if any) have been applied, you have a pre-activation vector, _z_, having values for, say, 3 features. For a ReLU activation function, the diagonal of _S_ is made up of 1’s where the _z_ element is positive and 0’s otherwise. The _S_ matrix means that each element in _z_ is either “active” (_z_ element is positive and corresponding _S_ element is 1) or “inactive” (_z_ element is zero or negative and corresponding _S_ element is 0).

When applied against a batch of data, this simply means that we are working against the matrix _Z_ as the pre-activation output from the weights, which is then multiplied by our 3x3 _S_ matrix:

![https://miro.medium.com/v2/resize:fit:700/1*WvORltFjkz3rWMWKrsPN4w.png](https://miro.medium.com/v2/resize:fit:700/1*WvORltFjkz3rWMWKrsPN4w.png)

If any of those S diagonal values are zero, then the outputs for that unit are set to zero. How does that affect the gradients of the layer in question and of other layers?

It turns out that the gradients in all layers are attenuated. The diagram below quantifies this effect. It compares what the gradients will be due to having a single inactive unit in the middle layer, against what they would have been if all units in all layers were active.

![https://miro.medium.com/v2/resize:fit:700/1*X_PhnPVaPyE1Aau3OGfCQA.png](https://miro.medium.com/v2/resize:fit:700/1*X_PhnPVaPyE1Aau3OGfCQA.png)

How weight matrix gradients are attenuated by an inactive unit

For the layer with the inactive unit, the entire corresponding column of weight gradients becomes zero. The remaining gradients in that layer are 2/3 of what they otherwise would have been if all units were active. The very next layer gets the same treatment but transposed. By the time the gradients are propagated into the next set of layers, in either direction, the single zeroed-out column/row gets blended into the other values so that the gradients of all other weights are attenuated to 4/9 = 0.4444 of what they otherwise would have been.

The results are the same with larger and differently shaped weight matrices. The immediate layers next to the inactivated unit output have a row or column zeroed out, while all other layers are equally attenuated. The proportions are just different for different-sized matrices.

One way of looking at this is that inactive units slow down learning. Now, we can’t simply remove the activation function to improve learning speed. We need the non-linearities provided by inactive units to learn complex functions. But this does suggest a relationship between the rate of learning and what I shall call the average “activation rate” (the percentage of units that are non-zero for any given data sample). For example, sparsely activated networks (where most units tend to be inactive for any given sample) will tend to learn slower than others.

In general, you don’t need to worry that gradients will be zero for some weights. I’ve so far considered only the case for a single data sample as it passes through the layers, and assumed only a single unit is inactive. In practice, multiple units will be inactive at any given time, with their attenuation effects overlapping. However, with each different data sample, different activations will result in a different pattern of attenuations. For a training batch, those different overlapping activation patterns are averaged out. Thus you will still generally get non-zero gradients for almost all weights. But, as mentioned, those gradients will be attenuated in proportion to the number of inactive units.

I’ll leave it to another day or perhaps someone else to explain whether the Adam optimizer can counteract the effects of different average activation rates in different network architectures.

![https://miro.medium.com/v2/resize:fit:700/1*ya6MB51BV0x4SNBgYId3qA.png](https://miro.medium.com/v2/resize:fit:700/1*ya6MB51BV0x4SNBgYId3qA.png)

Influences on gradient of weights at target layer in simplified model with activation function but no bias

Let’s summarize again what we’ve found, now that we have considered activation functions. The gradients of the weights at any layer are influenced by:

- the input data, X
- the mean prediction error, (_Ŷ — Y)/n_
- the weights of all layers except the target layer (the weights of the target layer do have some effect, but it’s only indirect)
- the pattern of unit activations at every layer including the target layer

Additionally, of those influences:

- they each have an equal effect relative to the others
- the weights have a linear component plus a non-linear component that attenuates the gradients (never amplifies them) in proportion to the percentage of inactive units across the network
- the attenuation effect of inactive units in one layer affects all layers.

# **Intuitive Effect of Biases**

I’ll only briefly cover biases. From the point of view of developing our intuition, their main effect is to make the math harder. But there are some generic statements we can make about biases in terms of the intuitive equations that we’ve been working with.

![https://miro.medium.com/v2/resize:fit:700/1*7jlVZTNWcKTU0tRWDh446g.png](https://miro.medium.com/v2/resize:fit:700/1*7jlVZTNWcKTU0tRWDh446g.png)

The first is that, in terms of computing the gradients in layer _l_, only the biases in the layers _before_ have any effect. Due to the nature of differentiation, the biases in the later layers do not affect any of the layers before them.

The second is that we can’t easily represent the full equation once we take biases into account. Rather, at that point, it is better to start with the output from the immediately prior layer as if it is the input to our network. That is also more consistent with how gradients are computed by common frameworks— gradient computation stops at the target layer and its final step is to take into account the input to that layer.

A last statement is that the calculation of the gradients of the biases only includes the layers _after_ the target layer. Calculation proceeds from the last layer towards the target layer and then stops, without any consideration of the earlier layers:

![https://miro.medium.com/v2/resize:fit:700/0*Chw64aIlZF4rhnGJ.png](https://miro.medium.com/v2/resize:fit:700/0*Chw64aIlZF4rhnGJ.png)

# **Vanishing and Exploding Gradients**

With what we’ve covered so far, it is possible to get a very simple and intuitive understanding of the vanishing gradient and exploding gradient problems as they pertain to deep neural networks with the ReLU activation function.

We’ve already talked about one form of that problem — in that inactive units attenuate the gradients across the network. If you have very few active units, then that attenuation will become quite extreme, leading to a vanishing gradient problem.

But there is another problem that is caused by the weights themselves.

![https://miro.medium.com/v2/resize:fit:700/0*cvyd3grOvS57yRMs.png](https://miro.medium.com/v2/resize:fit:700/0*cvyd3grOvS57yRMs.png)

The simple intuitive equation again without activation function or bias

If all weight matrices are identity matrices, then they do not affect the values being passed through during the forward pass, nor propagation of gradients during the backprop. It would be the equivalent of multiplying a scalar value by 1.0 multiple times.

One way of looking at the weight matrices in this vein is to consider their mean. This turns each weight matrix into a single number. Indeed, if they were all identity matrices, as discussed they would be the equivalent of multiplying the gradients by 1.0 multiple times. Now, if the mean weight of each layer is a little less or a little man than 1.0 then the net result after _L_ layers is more significant. Specifically, after _L_ layers of a mean weight _w_, the effect is _w^L_ (_w_ to the power of _L_). Here are just a few examples of how that can play out for different network depths and different mean weights:

![https://miro.medium.com/v2/resize:fit:700/0*USEkkdxWTacSFruY.png](https://miro.medium.com/v2/resize:fit:700/0*USEkkdxWTacSFruY.png)

Does the attenuation effect of the activation function have anything to add here? Yes, very much so.

Recall the intuitive gradient equation with the activation functions included as _S_ matrices. Conceptually, we can group each _W_ with its _S_ as follows:

![https://miro.medium.com/v2/resize:fit:700/0*Yyt_Eqe2hXuOVeWL.png](https://miro.medium.com/v2/resize:fit:700/0*Yyt_Eqe2hXuOVeWL.png)

Now we know that the effect of each _S_ is to attenuate the _W_ gradients in proportion to the number of inactive units. So we can say, as an approximation, that we can compute w (the mean weight) for each layer by multiplying _W_ and _S_ together and then taking the mean of that. This enables us to apply the same rule for estimating the amount of gradient vanishing or explosion.

Now, the percentage of active units may vary a lot across different network architectures and different problem domains, but it seems reasonable to pick 50% as a suitable sort of target. So even if our weights are perfectly tuned, _w_ becomes 1*50% = 0.5, and in a 10-layer network that causes a gradient multiplier of 0.00098.

One can see from this that, while exploding gradients too happen, they occur far less frequently than vanishing gradient problems, leading to very slow learning.

By the way, it’s for this reason that weight initialization has proven to be so important. The He and Glorot weight initialization schemes attempt to avoid the power curse by picking random weight values that balance out in some way similar to what I’ve described above.

I wonder if anyone has looked at regularization terms that keep the mean weight * activation rate close to 1.0 to maintain good gradient propagation.

# **Summary of Intuitive Observations**

It’s now time to pull everything together and provide one final summary of intuitions for the gradients in the network.

![https://miro.medium.com/v2/resize:fit:700/1*7uuSRXK_sIVEERxUDR5jfQ.png](https://miro.medium.com/v2/resize:fit:700/1*7uuSRXK_sIVEERxUDR5jfQ.png)

Influences on gradient of weights at target layer in full model

The gradients of the weights at any layer are influenced by:

- the input data, X
- the mean prediction error, (_Ŷ — Y)/n_
- the weights of all layers except the target layer (the weights of the target layer do have some effect, but it’s only indirect).
- the pattern of unit activations at every layer including the target layer
- the biases of all earlier layers, but not of the target layer or later layers

Additionally, of those influences:

- they each have (the potential for) equal effect relative to the others (though layer-to-layer differences in the various attenuation/vanishing/explosion effects can shift this)
- the weights have a linear component plus a non-linear component that attenuates the gradients (never amplifies them) in proportion to the percentage of inactive units across the network
- the mean values of the weights can have a strong vanishing or exploding effect to the gradients if it is either far from 1.0 or if there are many layers.

The gradients of the biases at any layer are influenced by:

- the mean prediction error, (_Ŷ — Y)/n_
- the weights of all later layers
- the pattern of unit activations at all later layers and at the target layer

That’s it for the easy intuitive explanations. In the next part I’m going to dig much deeper into how these equations come about. This is a great place to stop for those who don’t like the math. I hope that what I’ve presented so far is useful to you. I hope that you now have a better idea of how each layer’s weights contribute to the gradients at the other layers. And I hope that it helps your troubleshooting when network training doesn’t feel like it’s progressing as well as you’d expect.

In preparing for this article I kept finding ways in which my understanding was insufficient. Simple things threw up roadblocks, like: is the loss a scalar or a vector when your network output is vector-valued? And I discovered just how weird gradients of matrices are.

So I wanted to try to expand on the details for those who have understand the above and have had similar questions, and perhaps also for those who just want a more detailed explanation.

There’s a few ways to tackle this. I’m going to jump around a little, by starting with a single-layer network, then talking about computational graphs in the general case, and ending with a simplified multi-layer network.

Let’s get started…

# **Notation**

We’re about to get math-heavy, so I need to properly introduce the notation first:

![https://miro.medium.com/v2/resize:fit:700/1*b56o5KRasdzkIea8BmnSzg.png](https://miro.medium.com/v2/resize:fit:700/1*b56o5KRasdzkIea8BmnSzg.png)

This comes with a few special cases to be aware of:

![https://miro.medium.com/v2/resize:fit:700/1*CpGbjoISV39CcrrSRvXvdA.png](https://miro.medium.com/v2/resize:fit:700/1*CpGbjoISV39CcrrSRvXvdA.png)

# **Detailed Computation Graph**

When we look at a multi-layer network we need to consider the full computation graph, which is best represented visually by a somewhat daunting diagram:

![https://miro.medium.com/v2/resize:fit:700/1*LiFf82OxQpZGWbp4YJHP6g.png](https://miro.medium.com/v2/resize:fit:700/1*LiFf82OxQpZGWbp4YJHP6g.png)

Full computation graph for forward and backward passes during training. Matrix sizes indicated in red (source: author)

I’ve included the shapes of every matrix (in red) as I find that helps a lot to understand what’s happening.

Let’s break this down…

## **Forward pass**

The forward pass is represented by the green boxes along the top of the diagram.

Notice that you start with an input data matrix _X_ with _n_ samples and that at each step you still have a data matrix with _n_ samples. What changes as you progress forwards is the number of features, with the input matrix having _f0_ features, and the final output having _fL_ features. That is the result of the weight matrices at each layer having shape (_f(l-1) × fl_).

The various σ represent the activation functions at each layer. For example, the ReLU activation function. These operate element-wise against the _Z_ matrices, changing their values without changing their shape.

## **Loss**

At the far right the loss _J_ is computed, producing a scalar value that represents the aggregate across all data samples — usually the mean. In the diagram above I’ve assumed use of MSE loss, but the gradients of the loss function usually work out the same for cross-entropy losses (discussed further in a section below).

The scalar value itself is useful for reporting and plotting so that we know how the training is going. But other than that, it’s promptly discarded.

What really matters is the gradient of that loss.

## **Backward pass**

The backprop algorithm proceeds in reverse order, starting from the output of the network proceeding backwards towards the input. To start with, we’ll follow along the first row of purple back-prop boxes in the diagram.

![https://miro.medium.com/v2/resize:fit:700/1*g9DvlAZrgMzGZ7zFLAE2Vg.png](https://miro.medium.com/v2/resize:fit:700/1*g9DvlAZrgMzGZ7zFLAE2Vg.png)

Gradient of loss w.r.t. final pre-activation output (most common networks)

The first gradient to be computed is directly affected by the loss function used to compute J, in conjunction with the final output activation function. As discussed earlier, this usually expands to the mean of the prediction error _(Ŷ — Y)_. For other less common combinations of final output activation function and loss function, this first gradient will take other forms but will almost always be some function of the mean prediction error. Thus we’ll assume the more common case as we proceed.

Notice that the mean prediction error has the same shape as the output, (_n × fL_). Also notice that the scalar loss value is not present anymore. Indeed, it is not used in the backprop algorithm at all, as we are only interested in gradients w.r.t. to that loss, and those gradients almost all appear as matrices.

![https://miro.medium.com/v2/resize:fit:700/1*rCQJJgg-29M9ZU2siiQ6fw.png](https://miro.medium.com/v2/resize:fit:700/1*rCQJJgg-29M9ZU2siiQ6fw.png)

The “running product” of the backprop algorithm

The mean prediction error forms the first component of a “running product” (like a running sum, but with matrix multiplication). Each successive step of the backprop algorithm computes a matrix that is multiplied by the running product, with the result forming the new value to be propagated.

The shape of the running product always has the form (_n × fl_). It varies from layer to layer, depending on the number of features at each layer; but it always has _n_ rows, corresponding to the _n_ samples. In other words, the running product represents the propagated gradients for each sample individually.

## **Backprop through layers**

The key machinery of the backprop algorithm is the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), which enables us to compute the _partial_ influences at each layer independently. As each layer has two computations (the linear component with the weights and bias, followed by the activation function), the chain rule enables us to break the layer’s contribution into two separate partial derivatives.

![https://miro.medium.com/v2/resize:fit:700/1*S8Ztn4wTjJrg7kNxS30RhQ.png](https://miro.medium.com/v2/resize:fit:700/1*S8Ztn4wTjJrg7kNxS30RhQ.png)

Partial derivative of activation output w.r.t pre-activation values, as it is used in backprop (example given for 2nd layer). Matrix shapes given in red. Also shows how the middle two f2 lengths in the chain rule cancel out.

The first of those computes how its activation output varies w.r.t. to its intermediate _Z_ value. For arbitrary activation functions this requires analytically computing the derivative of the activation function and then applying it against the _Z_ input. For common activation functions like ReLU, in typical ML packages, I suspect it’s implementation is heavily optimized beyond just [AutoDiff](https://en.wikipedia.org/wiki/Automatic_differentiation). For example, notice that the math suggests a multiplication with an (_f2 × f2_) matrix. But most activation functions are applied element-wise. This suggests that a more efficient implementation would apply an element-wise operation directly against _dJ/dA_ rather than computing _dA/dZ_ and then matrix-multiplying it with _dJ/dA_.

However, the fact that it can be represented as a matrix multiplication can be handy. For example, when I used an _S_ matrix to represent the activation function, its derivative is simply its transpose. And thus I was able to capture that activation-derivative conveniently in the larger equation that showed how all the weights influence the gradients (more on this later).

![https://miro.medium.com/v2/resize:fit:700/1*CRCu_KI1ihCdFYbDFNimVQ.png](https://miro.medium.com/v2/resize:fit:700/1*CRCu_KI1ihCdFYbDFNimVQ.png)

Partial derivative of pre-activation output w.r.t layer input, as it is used in backprop (example given for 2nd layer). Matrix shapes given in red. Also shows how the middle two f2 lengths in the chain rule cancel out, giving an (n x f1) matrix as the new running product.

The second derivative of each layer computes how its pre-activation output, _Z_, varies w.r.t. the layer input, _A(l-1)_, the activation output from the layer before. While that might be a mouthful to say, it turns out to be simply the transpose of the layer’s weight matrix.

Why the transpose? One way to think about it is in terms of the matrix shapes. During the forward pass, a weight matrix of shape (_f1 × f2_) takes a dataset of shape (_n × f1_), having _n_ samples with _f1_ features, and maps it onto a new dataset shape (_n × f2_), having _n_ samples with _f2_ features. When we do backprop, we’re following the computation graph in reverse, and so we want to map from (_n × f2_) back to (_n × f1_). A transposed _W_ does just that.

## **The Gradients of the Input Data**

![https://miro.medium.com/v2/resize:fit:700/1*DKlKdJa3BEr-1YnSu_IUxA.png](https://miro.medium.com/v2/resize:fit:700/1*DKlKdJa3BEr-1YnSu_IUxA.png)

The gradient w.r.t. to the input data

The backprop algorithm proceeds almost all the way along the first purple row back to the input data. Generally you omit this last computation, which would give you the gradient of the loss w.r.t. to the input data. That doesn’t help you update any of the networks parameters. But it can be helpful on occasion when you want to identify how the input data affects the loss. For example, in image processing networks you can use this result to identify which regions of the image are deemed more significant for classification task.

But enough about that. It’s time to look at the branches off from the main backprop row.

## **The Gradients of the Weights**

![https://miro.medium.com/v2/resize:fit:700/1*xPAJ2EzlBUF-7kCS-ITdKg.png](https://miro.medium.com/v2/resize:fit:700/1*xPAJ2EzlBUF-7kCS-ITdKg.png)

Final calculation of the gradients w.r.t. the weights at layer 2

The backprop algorithm branches off briefly at each layer to compute the final gradients of the parameters at that layer. Thanks to the chain rule we only need to calculate the last part, being the partial derivative of the layer’s pre-activation output, _Z_, w.r.t. its weight matrix, _W,_ and it’s bias vector, _b_. As these are linear equations, we obtain a now familiar result: that the derivative w.r.t. to one variable is the transpose of the other. And so the partial derivative w.r.t. to _W_ is simply the transpose of the input data to the layer, _A(l-1)_.

When computing the gradients for the first layer, that is the network input data matrix, _X_.

There is one thing here that will seem unusual, however. In computing our “running product” so far we have always done the matrix-multiplication with the newly computed value on the right. In this case, we place the new value on the left. The reason for that is most easily seen in the shapes of the matrices. Up until this point, the running product has always had a shape of (_n × fl_), containing the gradients _per-sample_. But in order to apply the gradients as an update rule against _W_, we need the gradient matrix to have the same shape. As seen in the figure above, if you put the transposed A matrix first, then the _n_’s cancel out, and you get the shape you need.

Another way of thinking about it is in terms of the data samples. The loss is a scalar representing the mean of the individual per-sample losses. For parameter updates, we only care about that big-picture mean. The individual samples are just noise. But the math of the backprop algorithm works such that we propagate per-sample gradients. Thus, before we finish all our computations, we need to revert back to a mean. The matrix-multiplication of the layer input data, transposed, with the running product achieves just that. It computes a sum across the samples. The normalization factor to turn the sum into a mean is already incorporated from the _1/n_ in the original loss gradient.

When considering the very first layer, the input is the matrix _X_. Thus, one can see that the gradients at the first layer are influenced by both the input data, _X,_ and all the prediction errors at the output, _(Ŷ — Y)_. Intuitively, a similar rule applies to all layers. Given that the number of samples may be large, one may think it wise to do that computation first. The fully expanded equation for the gradients supports this idea, and it does lead to the same results. In practice, networks are usually trained with mini-batches where _n_ is usually small, around 32 to 64, or 1024 at most. The numbers of units in fully connected layers often range from 128 to 1024. So you don’t gain anything from trying to pull the _X_ and _(Ŷ— Y)_ multiplication earlier.

## **The Gradients of the Biases**

![https://miro.medium.com/v2/resize:fit:700/1*3-69yqGNh1KHeoJl0VHHWA.png](https://miro.medium.com/v2/resize:fit:700/1*3-69yqGNh1KHeoJl0VHHWA.png)

Final calculation of the gradients w.r.t. the biases at layer 2 (I think)

The gradients of the bias vectors are similar to the above for weights. The difference is that the biases are just an additive constant, which has 1 as its derivative. So it’s pretty much a terminal operation as far as propagation goes: when computing the gradients of the biases at some middle layer, you don’t care about any of the earlier layers.

I haven’t spent much time myself trying to understand the exact equations here. Logically the 1 vector would be broadcast to a matrix with _n_ columns and multiplied by the running product. In practice, the running product is simply summed over its samples. With the mean-normalization factor that was included in the derivative of the loss function, that sum turns into a mean over the samples.

## **Example**

The computation graph presented earlier can be used to expand the equations to compute the gradients of any variable.

Before moving on I wanted to give one concrete example. Here I will show the progression through backprop in order to calculate the weight gradients at layer 2.

![https://miro.medium.com/v2/resize:fit:700/1*O3OGcSdU7VGKzp-UfvnTgQ.png](https://miro.medium.com/v2/resize:fit:700/1*O3OGcSdU7VGKzp-UfvnTgQ.png)

Path through computation graph to computes gradient of weights at layer 2

At the end of the forward pass, we have _Ŷ_, _Y_, and our loss function. We start the backprop with our initial running product:

![https://miro.medium.com/v2/resize:fit:519/1*Hljf7bSG1Mmkh7FY7AC6BQ.png](https://miro.medium.com/v2/resize:fit:519/1*Hljf7bSG1Mmkh7FY7AC6BQ.png)

This is then multiplied by the weights in the second to last layer:

![https://miro.medium.com/v2/resize:fit:591/1*X48oUOkYTnS_7smosmDAmg.png](https://miro.medium.com/v2/resize:fit:591/1*X48oUOkYTnS_7smosmDAmg.png)

To retrieve those weights you could just look up the layer configuration. In practice, ML toolsets using generic AutoDiff capabilities that don’t know anything about layers. Instead, they maintain a cache of values used in every computation executed during the forward pass. Thus, during the backward pass, the weights required at this point are retrieved from the cache. Those familiar with TensorFlow will recognize this as the role of the [GradientTape](https://www.tensorflow.org/guide/autodiff).

Skipping a few steps we come to the second layer, with the propagated running product so far given by:

![https://miro.medium.com/v2/resize:fit:573/1*nswgnt1bqe49IQ47aDk2LQ.png](https://miro.medium.com/v2/resize:fit:573/1*nswgnt1bqe49IQ47aDk2LQ.png)

Notice that we need the originally computed _Z_ value when applying the derivative of the activation function. The computation cache provides that important value. Also remember that in practice the derivates of the activation functions are likely applied via an element-wise operation against the running product, rather than as a matrix-multiplication. However, it’s hard to represent that in an equation — we’d need to represent recursive operations and it just gets messy. So we’ll represent this as a matrix-multiplication. This is still mathematically accurate, even if not computationally efficient.

Propagating that through the derivative of layer 2’s activation function gives us:

![https://miro.medium.com/v2/resize:fit:505/1*dPlpTpgtyjQ6WR4J6B_a3g.png](https://miro.medium.com/v2/resize:fit:505/1*dPlpTpgtyjQ6WR4J6B_a3g.png)

We are now ready to branch off from the main direction of backprop so that we can calculate the final gradient of the weights at layer 2. For this we need the input data provided to layer 2, which happens to be output activations from layer 1. Just like the weights and Z values before, this is provided by caching the value during the forward pass. Finally, as discussed earlier, we also break with the trend so far and apply the new value at the front:

![https://miro.medium.com/v2/resize:fit:572/1*i0a940wf_IDDLQ8h9RbO0A.png](https://miro.medium.com/v2/resize:fit:572/1*i0a940wf_IDDLQ8h9RbO0A.png)

This gives us a final gradient matrix with the same shape as the weights: (f1 _× f2_). It can be directly applied with the learning rate to update the value of the weights at this layer.

# **Effect of Different Loss Functions**

So far I’ve described the gradients under an MSE loss function regime. How might a different loss function affect things? For example, cross-entropy loss is the most common loss function for classification scenarios.

It it turns out that the most common configurations of output activation function + loss function all lead to the same basic form for the gradient of the loss w.r.t. to the last layer’s output:

![https://miro.medium.com/v2/resize:fit:609/1*q3siC9-U3WGPx00PoliHXQ.png](https://miro.medium.com/v2/resize:fit:609/1*q3siC9-U3WGPx00PoliHXQ.png)

This is true even for those cases with sigmoid or softmax activation functions. When you combine the activation function with the appropriate loss function, the loss w.r.t. to the network output combines with the gradient of the final activation function to form the same equation:

![https://miro.medium.com/v2/resize:fit:700/1*KvyBLLt8u2I-iQ_ofNogHA.png](https://miro.medium.com/v2/resize:fit:700/1*KvyBLLt8u2I-iQ_ofNogHA.png)

This seems like one of those weird and unexpected beautiful outcomes of math. But, more than that, I expect it’s the outcome of the original engineers of these output activation and loss functions intentionally making the math make sense. For example, Goodfellow, Yoshua and Courville (2016, p. 219) attribute the switch from MSE to cross-entry loss against sigmoid/softmax outputs as one of the factors that made deep learning possible. I think that’s because the mean error gradient is linear.

This outcome holds for the following common combinations…

- Linear activation with Mean Squared Error (MSE) loss:

![https://miro.medium.com/v2/resize:fit:700/0*UsRNexobsUV7O-Lk.png](https://miro.medium.com/v2/resize:fit:700/0*UsRNexobsUV7O-Lk.png)

- Sigmoid activation with Binary Cross-Entropy Loss:

![https://miro.medium.com/v2/resize:fit:700/0*-hcAcFY_ExWmHNSe.png](https://miro.medium.com/v2/resize:fit:700/0*-hcAcFY_ExWmHNSe.png)

- Softmax activation with Categorical Cross-Entry loss:

![https://miro.medium.com/v2/resize:fit:700/0*m5oUfCYc0TpyPqph.png](https://miro.medium.com/v2/resize:fit:700/0*m5oUfCYc0TpyPqph.png)

Note that where the output _Ŷ_ has multiple features, the loss function will typically take the mean across the feature axis too. This is why you see it being normalized by _1/fL_ in the equations above. I have sometimes omitted that detail for simplicity.

It’s also important to remember that the derivative of the loss function won’t always become the mean prediction error, but for many cases it does. And since we’re trying to get an intuitive understanding of the more common cases, I’ve taken this rule as granted for this article.

# **Detailed Gradient Derivation**

We are now in a position where we can fully describe the gradient equation for weights at any layer by simply expanding the computation graph described above.

The full chain rule for the gradients of the weights at layer _l_ is given by:

![https://miro.medium.com/v2/resize:fit:700/0*LGYFMAXiKW_sliQw.png](https://miro.medium.com/v2/resize:fit:700/0*LGYFMAXiKW_sliQw.png)

The very first component listed, _dZ/dW_, is the last component to be computed and incorporated into the running product. It expands as the input to layer _l_. The next two components form _dJ/dZ_ of the last layer, which becomes the mean prediction error with extra attached normalizing factors. The various _dZ/dA_ become the weights at each other layer following layer _l_.

The various _dA/dZ_ need a little more work, as they reflect the derivative of the activation function.

## **Matrix approximation of activation function**

The first step is to notice that if we remove the activation functions altogether (ie: linear activation functions), this is the equivalent of multiplying the _Z_ by the identity matrix at each layer during the forward pass. Its derivative is also the identity matrix:

![https://miro.medium.com/v2/resize:fit:700/0*4Y4ozqDD0KGLow6m.png](https://miro.medium.com/v2/resize:fit:700/0*4Y4ozqDD0KGLow6m.png)

Now, replace the identity matrix with a generic matrix _S_ (as the upper-case version of the lower-case s, for sigma). If we’re operating on a single input vector _z_, a ReLU activation function can be represented by an _S_ matrix that is almost an identity matrix, but has some of the diagonal values zeroed out wherever the corresponding values in _z_ are negative:

![https://miro.medium.com/v2/resize:fit:700/0*eD4eagkxnTQdMmAc.png](https://miro.medium.com/v2/resize:fit:700/0*eD4eagkxnTQdMmAc.png)

In fact, since almost all activation functions operate element-wise, this holds for any activation function — it can be turned into a diagonal matrix where its trace is constructed according to the values of _z_. Even the softmax activation function can be represented by an _S_ matrix if you include the right values in the off-diagonals.

There are a couple of problems though. The first is to be aware that _S_ is not a constant. Because it directly reflects the activations, and they change for every batch, _S_ changes for every batch and every gradient update step. That’s OK because we only need it to be constant within a single update.

The second problem is more problematic. This activation function to _S_ matrix substitution is mathematically equivalent for a single vector. It isn’t for a whole matrix that represents multiple data samples. The problem is that the value of _S_ depends on the individual _z_ row for each sample, and they are different for each row. For our purpose, we’re simply going to gloss over that issue on the basis that it’s OK if _S_ represents the mean activation across all data samples in a batch. I don’t have any mathematical proof — I just hope it’s a good enough approximation.

## **Expanding chain rule**

With that in place, we can now expand the chain rule. I’ll list it again, along with the replacements that we know about:

![https://miro.medium.com/v2/resize:fit:700/0*dG9VHbLZweM-Wqsw.png](https://miro.medium.com/v2/resize:fit:700/0*dG9VHbLZweM-Wqsw.png)

Plugging those expansions into the chain rule and using the product symbol (∏) to write the result compactly gives us:

![https://miro.medium.com/v2/resize:fit:700/1*2Ube7IBC7UA3I00RWKyTOA.png](https://miro.medium.com/v2/resize:fit:700/1*2Ube7IBC7UA3I00RWKyTOA.png)

Full and accurate gradient equation for the weights at any layer

## **Expanding target layer input**

There’s one further expansion that we can apply if we take some simplifying assumptions.

In the equation above, the activation _A_ from layer _l-1_ forms the input. There’s no simple way to accurately represent the computation of _A_ at an arbitrary layer as a single equation because its value is computed recursively, with the weights, biases and activation function in one layer forming the input to the next layer. However, by eliminating the bias (ie: assuming it to be zero), and using our prior substitution of an _S_ matrix for the activation function, we can arrive at a useful approximation:

![https://miro.medium.com/v2/resize:fit:700/0*9n-UVlz-xsp6239c.png](https://miro.medium.com/v2/resize:fit:700/0*9n-UVlz-xsp6239c.png)

Through successive expansions we turn what would otherwise have been recursive equation into a product:

![https://miro.medium.com/v2/resize:fit:700/0*7uUNtQIolGpnMSNv.png](https://miro.medium.com/v2/resize:fit:700/0*7uUNtQIolGpnMSNv.png)

Plugging that into our gradient equation from above gives us an approximation of the gradient of the weights at layer _l_ with a form that clearly shows the contribution from each of the layers, the input data X, and the mean prediction error:

![https://miro.medium.com/v2/resize:fit:700/1*-nGD5j1t9pSoG-3Fv-ojcg.png](https://miro.medium.com/v2/resize:fit:700/1*-nGD5j1t9pSoG-3Fv-ojcg.png)

Full gradient equation with approximate expansion for input to target layer

Writing in long form we can see more clearly what we’ve arrived at:

![https://miro.medium.com/v2/resize:fit:700/0*zbK-boab91E6DkI2.png](https://miro.medium.com/v2/resize:fit:700/0*zbK-boab91E6DkI2.png)

Notice that the order of components in the equation doesn’t follow the order of layers. This follows from the fact that the equation lists things in a different order to the order in which they are computed. The actual computation through the forward pass first, using X, W1, S1, etc. in the order listed. The backward pass then starts a new running product starting with the mean prediction error, followed by the S and W of the last layer, and proceeding backwards to the target layer l. Finally, it then combines that with the result of the forward pass up to layer l.

## **Gradients of biases**

The equation for the gradients of the biases at each layer can be easily constructed in the same way as above. In fact most of the hard word is already done. The chain rule is almost entirely the same:

![https://miro.medium.com/v2/resize:fit:700/1*gTEKi9KBFc-KOk1Bf1IC_A.png](https://miro.medium.com/v2/resize:fit:700/1*gTEKi9KBFc-KOk1Bf1IC_A.png)

The derivative of _Z_ w.r.t. to the bias solves quite simply to the unit vector. As we ultimately need a gradient in the same shape as the bias vector this turns into a sum over the rows:

![https://miro.medium.com/v2/resize:fit:700/1*FBVh8GNX6ChXHF9etIPzew.png](https://miro.medium.com/v2/resize:fit:700/1*FBVh8GNX6ChXHF9etIPzew.png)

Full equation for gradient of biases at any layer

## **Example**

For a concrete example, consider a network with 3 layers. The gradients for the first and last layer are given by:

![https://miro.medium.com/v2/resize:fit:700/1*7Lw070COl1Uv3XLzc-NcFQ.png](https://miro.medium.com/v2/resize:fit:700/1*7Lw070COl1Uv3XLzc-NcFQ.png)

I’ve included the dimensions of each component. The way that matrix multiplication works is that adjacent lengths cancel out. So, for example, the first _X^T dot (Ŷ — Y)_ multiplication in the first layer is a multiplication between _(f0_ x _n)_ and _(n_ x _f3)_, and the two n’s in the middle cancel out, giving you a _(fo_ x _f3)_ matrix. When that’s subsequently multiplied by the _(f3_ x _f2)_ matrix next to it, that becomes _(f0_ x _f2)_. This repeats until finally you get a _(f0_ x _f1)_ matrix, the same shape as your target weight matrix.