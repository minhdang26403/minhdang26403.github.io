---
layout: post
title: How to Build a Multi-Dimensional Array by Separating View Logic from Compute Kernels
date: 2025-11-11 11:30:00
description: A deep dive into my `NDArray`'s core design. I explain the "Control Plane vs. Data Plane" model–using "smart" Python to handle all the complex view and stride logic, which lets us build a "dumb," simple, and incredibly fast C++ and CUDA backend
tags: MLSys, Systems Design, Python, C++, Deep Learning
categories: needle
featured: true
---

### The Central Design Problem

When you build a tensor library, you immediately hit a fundamental fork in the road: how "smart" should your main `NDArray` (multi-dimensional array) object be?

On one side, you have the "thin" wrapper. This approach is tempting. The `NDArray` class is just a simple shell that delegates _everything_—all the math, all the striding logic, all the broadcasting—to its backend. This sounds clean, but it's a maintenance nightmare. It means you have to re-implement all that complex, error-prone view logic in C++, then _again_ in CUDA, and _again_ in the language of your favoriate hardware accelerator. This design doesn't scale.

Then, there's the "thick" wrapper. This is the design I'm building, and it's built on a clean separation of concerns. I call it the **"Control Plane vs. Data Plane"** model.

It's a "Python Brains, C++ Brawn" approach:

- **The Control Plane (The "Brains"):** A "smart," "thick" `NDArray` class written in pure Python that handles all the complex logical operations.
- **The Data Plane (The "Brawn"):** A set of "dumb," simple, and brutally fast compute backends (NumPy, C++, CUDA) that just do the number crunching.

### ✈️ The Control Plane: A "Smart" Python Wrapper

The "Control Plane" is my Python `NDArray` class. It's "thick" because it handles **all the logical operations** of the array.

At its core, my `NDArray` is just a map. It's a Python object that holds metadata—`shape`, `strides`, and `offset`—which defines a "logical view" over a "physical" block of memory (which I call the `_handle`).

The key insight is that **a huge number of array operations are just math on this metadata.** They don't need to touch the data at all, making them zero-copy, free operations.

Take `permute`. It's the perfect example of a "free" operation that just shuffles tuples and returns a new `NDArray` pointing to the _exact same_ data.

**Code Snippet 1: `permute` (The "Free" View Operation)**

```python
    def permute(self, new_axes: Union[tuple[int, ...], List[int]]) -> "NDArray":
        """Permute dimensions... (no data copy)."""

        # 1. Just calculate new logical shape and strides
        new_shape = tuple(self.shape[i] for i in new_axes)
        new_strides = tuple(self.strides[i] for i in new_axes)

        # 2. Return a NEW array view, pointing to the SAME data handle
        return NDArray.make(
            new_shape,
            new_strides,
            self.device,
            self._handle,  # <-- Pass the *original* handle
            self._offset,
        )
```

The Control Plane is where all this "view" magic lives. Slicing (`__getitem__`)? That's just a math problem to find a new `offset` and `strides`. Broadcasting (`broadcast_to`)? That's the "zero-stride trick." It's all handled once, in pure Python, and it's easy to test. Since all the `__getitem__` and `broadcast_to` are more complicated than `permute`, I don't include it here, but you can take a look at their implementation at my [repo](http://github.com/minhdang26403/needle).

### ⚙️ The Data Plane: A "Dumb" C++ Engine

The "Data Plane" is my C++/CUDA backend. It is designed to be **brutally fast and incredibly simple.**

For this project, I'm building three backends that all follow the same simple API:

1.  **NumPy Backend:** A pure Python backend using `numpy` for reference and testing.
2.  **C++ CPU Backend:** A high-performance backend using plain C++ for CPU-bound computation.
3.  **CUDA Backend:** A GPU-accelerated backend for high-performance training.

My C++ and CUDA backends don't know what "strides" or "broadcasting" are. They are "dumb" compute engines that expect one thing: **a flat, 1D, contiguous block of memory.**

The entire "data plane" API is just a set of C-style functions that operate on these flat buffers. Look at the NumPy backend's `reduce_sum` (the C++ version is nearly identical in concept):

**Code Snippet 2: `reduce_sum` (The "Dumb" Backend Kernel)**

```python
# From python/needle/backends/ndarray_backend_numpy.py

def reduce_sum(a: Array, out: Array, reduce_size: int) -> None:
    """Reduce the last logical dimension by sum."""

    # This kernel knows nothing about 3D shapes or axes.
    # It just gets a flat array 'a', a 'reduce_size',
    # and does its job.
    out.buffer[:] = np.sum(a.buffer.reshape(-1, reduce_size), axis=1)
```

This is the beauty of the design. The kernel is simple. It only knows how to sum "blocks" of data. This makes my C++ and CUDA code _radically_ simpler and easier to optimize.

---

### 🌉 The Bridge: The `compact()` Function

This leads to the obvious question: what happens when the "smart" Control Plane (with its fragmented, non-contiguous `permute` view) needs to talk to the "dumb" Data Plane (which needs a flat array)?

This is where the most important function in the library comes in: `compact()`.

I call `compact()` the **"View Tax."**

It's the (necessary) performance hit you take to connect the two planes. When I call `a.sum(axis=1)`, the `sum` method acts as the "manager" that orchestrates this whole process.

**Code Snippet 3: `sum()` (The "Manager" Connecting the Planes)**

```python
    # The "Control Plane" sum method
    def sum(self, axis=None, keepdims=False) -> "NDArray":

        # 1. Prepare the view: "permute-to-the-end" trick
        view, out, reduce_size = self.reduce_view_out(axis, keepdims=keepdims)

        # 2. Pay the "View Tax"
        #    This copies the fragmented 'view' into a new,
        #    flat, contiguous buffer.
        compact_view_handle = view.compact()._handle

        # 3. Call the "Dumb" Data Plane
        #    The backend kernel gets the simple, flat buffer it expects.
        self.device.reduce_sum(compact_view_handle, out._handle, reduce_size)
        return out
```

This is the whole philosophy in one method. The `sum` method calls `reduce_view_out` to get a "smart" permuted view, pays the `compact()` tax to "defragment" it, and then hands the resulting simple, flat buffer to the "dumb" `reduce_sum` kernel.

As I wrote in my original notes:

> We call `.compact()` (which copies memory) liberally in order to make the underlying C++ implementations simpler. Everything that can be done in Python, is done in Python.

This trade-off—accepting a few explicit data-copy "taxes" in exchange for a massive reduction in backend complexity—is the key to building a system that is sane, maintainable, and scalable.
