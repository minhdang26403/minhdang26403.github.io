---
layout: post
title: The Heart of a Tensor Library: A "Thick" NDArray and the Control Plane vs. Data Plane Design
date: 2015-07-15 15:09:00
description: an example of a blog post with some code
tags: MLSys, Systems Design, Python, C++, Deep Learning
categories: needle
featured: true
---

When you first decide to build a deep learning framework, you immediately hit a fundamental fork in the road: how "smart" should your main `NDArray` object be?

On one side, you have the "thin" wrapper. This approach is tempting. The `NDArray` class is just a simple shell, and it delegates *everything*—all the math, all the striding logic, all the broadcasting—to its backend (NumPy, C++, Metal). This sounds clean, but it's a maintenance nightmare. It means you have to re-implement all that complex, error-prone view logic in C++, then *again* in Metal, and *again* in CUDA. This design doesn't scale.

Then, there's the "thick" wrapper. This is the design I'm building, and it's built on a clean separation of concerns I call the **Control Plane vs. Data Plane** model.

### ✈️ The Control Plane: A "Smart" Python Wrapper

The "Control Plane" is my Python `NDArray` class. It's "thick" because it handles **all the logical operations** of the array.

At its core, my `NDArray` is just a map. It's a Python object that holds metadata—`shape`, `strides`, and `offset`—which defines a "logical view" over a "physical" block of memory (which I call the `_handle`).

The key insight is that **a huge number of array operations are just math on this metadata.** They don't need to touch the data at all, making them zero-copy, free operations.

* **`transpose()` or `permute()`?** That's not a computation. I just swap the numbers in the `shape` and `strides` tuples. It's an O(1) operation.
* **Slicing (`a[1:5, ::2]`)?** That's just a math problem. I just calculate a new `offset` (to jump to the `[1, 0]` element) and new `strides` (to handle the `::2` step). It's free.
* **`broadcast_to()`?** That's the "zero-stride trick." By setting the stride of a new dimension to `0`, I can "stretch" an array from shape `(3,)` to `(10, 3)` without allocating any new memory.



The Control Plane is where all this "view" magic lives. It's written once, in pure Python, and it's easy to test and debug.

### ⚙️ The Data Plane: A "Dumb" C++ Engine

The "Data Plane" is my C++/Metal/CUDA backend. It's the "muscle." It is designed to be **brutally fast and incredibly simple.**

My C++ backend doesn't know what a "stride" is. It doesn't know what "broadcasting" is. It's a "dumb" compute engine that expects one thing: **a flat, 1D, contiguous block of memory.**

Its entire API is just a set of C-style functions that operate on these flat buffers:
* `ewise_add(a_handle, b_handle, out_handle)`
* `matmul(a_handle, b_handle, out_handle, M, K, N)`
* `reduce_sum(in_handle, out_handle, reduce_size)`

This makes my backend C++ code *radically* simpler. The `reduce_sum` kernel, for example, just loops over contiguous blocks of memory. It doesn't need to know *anything* about the original tensor's shape.

Best of all, the API for the C++ backend is now **identical** to the API for the Metal backend, which will be identical to the API for the CUDA backend. Adding new hardware is now trivial.

---

### 🌉 The Bridge: The `compact()` Function

This leads to the obvious question: what happens when the "smart" Control Plane (with its fragmented, non-contiguous view) needs to talk to the "dumb" Data Plane (which needs a flat array)?

This is where the most important function in the library comes in: `compact()`.

I call `compact()` the **"View Tax."**

It's the (necessary) performance hit you take to connect the two planes. When I call `a.transpose() @ b`, the `matmul` operation does this:

1.  **Check `a`:** The Control Plane sees that `a` is a `transpose()` view. Its `strides` are `(1, 10)` instead of `(10, 1)`, so it's non-contiguous.
2.  **Pay the Tax:** It calls `a_compact = a.compact()`.
3.  **`compact()`:** This function allocates a *new*, contiguous block of memory and (relatively slowly) copies the data from the "fragmented" view into this new, "defragmented" buffer.
4.  **Call the Data Plane:** The `matmul` kernel is now called with the *new*, *compact* `a_compact._handle`. The C++ code doesn't have to deal with the `(1, 10)` strides; it just gets a simple, flat array and runs at maximum speed.

This is the entire philosophy of the design. As I wrote in my original notes:

> We call `.compact()` (which copies memory) liberally in order to make the underlying C++ implementations simpler. Everything that can be done in Python, is done in Python.

This trade-off—accepting a few explicit data-copy "taxes" in exchange for a massive reduction in backend complexity—is the key to building a system that is sane, maintainable, and scalable.
