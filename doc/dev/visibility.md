<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Symbol Visibility

[Symbol visibility](https://gcc.gnu.org/wiki/Visibility) is a feature of
compilers that allows fine-grained control over which symbols are exposed from
a library. The [gcc wiki](https://gcc.gnu.org/wiki/Visibility) site is old, but
nevertheless highlights why we would want to do this:

* Faster load times for libraries, and
* decreased likelihood of symbol collisions

We default to "hidden" symbol visibility and then selectively mark symbols as
"default" visibility. The only reason to mark a symbol as visible is if it is
part of our API. The macro to mark a class, method, or function as part of our
API is is `NVF_API`, defined in `csrc/visibility.h`.

## FAQ

### Should I mark a method visible or the whole class?

Marking a class as visible applies recursively: both the class itself and the
individual methods are visible. You should prefer to mark individual methods as
visible instead of the whole class. This approach minimizes our surface area.

### I see an undefined reference to `typeinfo for <class>`. How do I fix this?

If you see an undefined reference to a `typeinfo for <class>`, then you will
need to make the entire class visible.

### I see an undefined reference to `vtable for <class>`. How do I fix this?

Mark one of the virtual functions as NVF_API. Use the destructor for this.

If the destructor is not already virtual, mark it so now. An object with
virtual functions should always have a virtual destructor.

### I see that `Foo` is visible but I do not think it needs to be.

You are probably correct! Please fix it. You can do this by removing the
`NVF_API` macro from the relevant place and compiling + linking. If linking
succeeds then you were likely correct. However to *guarantee* that everything is
setup properly, ensure `python3 -c "import nvfuser"` has an exit code of 0 and
prints nothing.

### Should I mark my new method or class as `NVF_API`?

Your default answer should be "no". If a client is trying to use your new
symbol and they see undefined symbol errors, then you should go back and
mark it public.

Due to the way our library is designed to be used, this happens rarely. Most
users are going through the Python API, and if something is only accessed
through Python then there should not be a need to mark it public.
