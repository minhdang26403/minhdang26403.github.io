---
layout: post
title: Why Not Newton? The Real Reason We're Stuck with SGD for Deep Learning
date: 2025-10-20 20:30:00
description: It's not just the petabyte-sized Hessian. The real reason we don't use Newton's.
tags: machine-learning, deep-learning
categories: deep-learning
featured: false
---

If you come from a classic optimization background, one of the first things you ask in deep learning is, "Why are we using this slow, simple algorithm called Stochastic Gradient Descent (SGD)?"

In a university course, you learn that Newton's method is the king. It's a second-order method that converges quadratically—meaning the number of correct digits in your answer *doubles* at every step. SGD, a first-order method, just plods along with sub-linear convergence.

So, why are we using a horse-and-buggy when a spaceship is available?

It's not just one reason. It's a three-level cascade of "it's a bad idea," ranging from the physically impossible to the philosophically wrong.

### The "Level 1" Answer: The Obvious Dealbreaker

This is the one everyone learns first. Using Newton's method requires computing and inverting the **Hessian matrix** ($H$), which is the $N \times N$ matrix of all possible second-order partial derivatives.

$N$ is the number of parameters in your model.

Let's do the math on a "small" model like ResNet-50, which has about $N = 25 \text{ million}$ parameters.

- **To store the Hessian:** You would need $N \times N = (25,000,000)^2 = 625 \text{ trillion}$ entries.
- At 32-bit (4 bytes) per entry, that's **2.5 Petabytes**.

Your top-of-the-line H100 GPU has 80 gigabytes of memory. You would need around 312,500 GPUs just to *store* this matrix, let alone run the $O(N^3)$ operation to *invert* it.

This is a full-stop, hard-no. It's physically impossible.

### The "Level 2" Answer: The Real Reason (Even if We Had the Memory)

This is where it gets interesting. A smart person will say, "Okay, but we have **Quasi-Newton** methods like **L-BFGS**. They're brilliant! They *approximate* the inverse Hessian using just the last $k$ gradient updates, so the memory cost is only $O(N \cdot k)$, not $O(N^2)$. Problem solved!"

So why don't we use L-BFGS?

The real reason is that we do **stochastic** optimization.

L-BFGS and other classic methods are for **deterministic** optimization. They assume you are calculating the *true* gradient and *true* Hessian over the *entire* dataset.

We don't do that. We use a tiny **mini-batch** (e.g., 64 samples) to get a *noisy guess* of the gradient.

And this is the core of the problem:

1. **A Stochastic Hessian is Garbage:** The first derivative (gradient) is noisy but stable. If you average the "downhill" direction from 64 random samples, you get a pretty good estimate of the true "downhill." But the *second* derivative (curvature) is *incredibly* sensitive to noise. The curvature from a tiny mini-batch is a wildly unstable, garbage approximation of the true landscape's curvature. Taking a "perfect" Newton step based on this garbage map is often *worse* than just taking a dumb SGD step.
2. **The Prize (Quadratic Convergence) is a Lie:** Even if you *could* get a good mini-batch Hessian, the very act of using mini-batches (stochasticity) introduces noise that *dominates* the convergence. The best-case speed for *any* stochastic method is sub-linear. You simply *cannot* achieve quadratic convergence.

So, you'd be paying the massive computational cost of a Quasi-Newton method... without even getting the prize.

### The "Level 3" Answer: The Bug is Actually a Feature

This is the final nail in the coffin. Let's say we solved the first two problems. It's *still* a bad idea.

- **Classic ML (like SVMs)** are **convex** problems. The loss landscape is a single, perfect bowl. You *want* a high-precision tool like Newton's to find the one, true global minimum at the bottom.
- **Deep Learning** is **wildly non-convex**. The landscape is a hellish, high-dimensional mountain range with millions of bad local minima, saddle points, and sharp, narrow "overfitting" valleys.

We don't *want* to find the "sharpest" minimum. A model that perfectly memorizes the training data (overfits) lives in a very sharp, deep, narrow valley.

We want a **wide, flat valley**. Why? Because a wide valley means that if the test data is a little different from the training data, the loss is still low. This is **generalization**.

This is where SGD's "bug" becomes its greatest feature:

- The **noise** in the stochastic gradient acts as a regularizer.
- It bounces the optimizer out of those sharp, overfitted valleys.
- It's *too dumb and noisy* to settle into a bad, sharp minimum and prefers to find the wide, flat, generalizable ones.

### The Takeaway

So, why do we use SGD and not Newton?

1. **It's computationally impossible:** The $O(N^2)$ Hessian is petabytes large.
2. **The "fix" (Quasi-Newton) doesn't work:** The stochastic Hessian is too noisy to trust, and the stochastic noise makes the prize of quadratic convergence impossible anyway.
3. **It's the wrong tool for the job:** We're in a non-convex world where the "dumb" noise of SGD is a *feature* that helps us find generalizable solutions.

We don't want a scalpel. We need a jackhammer. And that's SGD.

_(P.S. - So what is Adam? Adam is just a very, very clever jackhammer. It's still a first-order method, but it uses momentum and an adaptive step size to navigate the landscape more intelligently than plain SGD. It's a "fake" second-order method that uses first-order information to get a tiny, cheap approximation of the curvature. But that's a post for another day.)_
