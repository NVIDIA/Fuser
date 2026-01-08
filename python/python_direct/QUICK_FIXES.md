# Quick Fix Examples - High Priority Nanobind Improvements

This document provides copy-paste ready code fixes for the highest priority improvements.

## Fix 1: Add `nb::keep_alive` to ir.cpp

### Location: `ir.cpp` around lines 67-90

**Before:**
```cpp
.def(
    "definition",
    &Val::definition,
    nb::rv_policy::reference,
    R"(
Get the definition of this expression.

Returns
-------
Expr
    The definition of this expression.
)")
.def(
    "uses",
    &Val::uses,
    nb::rv_policy::reference,
    R"(
Get the uses of this expression.

Returns
-------
Expr
    The uses of this expression.
)");
```

**After:**
```cpp
.def(
    "definition",
    &Val::definition,
    nb::rv_policy::reference,
    nb::keep_alive<0, 1>(),  // Keep 'self' alive as long as return is alive
    R"(
Get the definition of this expression.

Returns
-------
Expr
    The definition of this expression.
)")
.def(
    "uses",
    &Val::uses,
    nb::rv_policy::reference,
    nb::keep_alive<0, 1>(),  // Keep 'self' alive as long as return is alive
    R"(
Get the uses of this expression.

Returns
-------
Expr
    The uses of this expression.
)");
```

### Location: `ir.cpp` around lines 94-129

**Before:**
```cpp
.def(
    "input",
    &Expr::input,
    nb::arg("index"),
    nb::rv_policy::reference,
    R"(...)")
.def(
    "output",
    &Expr::output,
    nb::arg("index"),
    nb::rv_policy::reference,
    R"(...)")
```

**After:**
```cpp
.def(
    "input",
    &Expr::input,
    nb::arg("index"),
    nb::rv_policy::reference,
    nb::keep_alive<0, 1>(),
    R"(...)")
.def(
    "output",
    &Expr::output,
    nb::arg("index"),
    nb::rv_policy::reference,
    nb::keep_alive<0, 1>(),
    R"(...)")
```

### Location: `ir.cpp` around line 150-161 (IterDomain::extent)

**Before:**
```cpp
.def(
    "extent",
    &IterDomain::extent,
    nb::rv_policy::reference,
    R"(...)")
```

**After:**
```cpp
.def(
    "extent",
    &IterDomain::extent,
    nb::rv_policy::reference,
    nb::keep_alive<0, 1>(),
    R"(...)")
```

### Location: `ir.cpp` - TensorView methods that return internal pointers

Search for these methods and add `nb::keep_alive<0, 1>()` after `nb::rv_policy::reference`:
- `domain()`
- `logical_domain()`
- `root_domain()`
- `allocation_domain()`

## Fix 2: Add __repr__ methods

### Location: `ir.cpp` around line 37-42

**Before:**
```cpp
nb::class_<Statement>(nvfuser, "Statement")
    .def(
        "__str__",
        [](Statement* self) { return self->toString(); },
        R"(Get string representation of Statement.)");
```

**After:**
```cpp
nb::class_<Statement>(nvfuser, "Statement")
    .def(
        "__str__",
        [](Statement* self) { return self->toString(); },
        R"(Get string representation of Statement.)")
    .def(
        "__repr__",
        [](Statement* self) {
            return "<nvfuser.Statement at 0x" +
                   std::to_string(reinterpret_cast<uintptr_t>(self)) + ">";
        });
```

### Location: `ir.cpp` - Val class around line 44

**After the `__str__` or at the end of Val bindings, add:**
```cpp
.def("__repr__", [](Val* self) {
    return "<nvfuser.Val: " + self->toString() + ">";
})
```

### Location: `ir.cpp` - Expr class around line 93

**After the existing bindings, add:**
```cpp
.def("__repr__", [](Expr* self) {
    return "<nvfuser.Expr: " + self->toString() + ">";
})
```

### Location: `ir.cpp` - IterDomain class around line 137

**After the existing bindings, add:**
```cpp
.def("__repr__", [](IterDomain* self) {
    return "<nvfuser.IterDomain: " + self->toString(0) + ">";
})
```

### Location: `ir.cpp` - TensorView class around line 192

**After the existing `__str__`, add:**
```cpp
.def("__repr__", [](TensorView* self) {
    return "<nvfuser.TensorView: " + self->toString(0) + ">";
})
```

### Location: `runtime.cpp` - Fusion class

**After the existing bindings, add:**
```cpp
.def("__repr__", [](Fusion* self) {
    std::ostringstream oss;
    oss << "<nvfuser.Fusion with " << self->inputs().size() << " inputs, "
        << self->outputs().size() << " outputs>";
    return oss.str();
})
```

### Location: `runtime.cpp` - FusionExecutorCache class

**After the existing bindings, add:**
```cpp
.def("__repr__", [](FusionExecutorCache* self) {
    return "<nvfuser.FusionExecutorCache>";
})
```

## Fix 3: Optimize vector conversion in direct_utils.h

### Location: `direct_utils.h` around line 20-30

**Before:**
```cpp
template <typename T>
std::vector<T> from_pysequence(nb::sequence seq) {
  std::vector<T> result;
  std::transform(
      seq.begin(), seq.end(), std::back_inserter(result), [](nb::handle obj) {
        NVF_ERROR(nb::isinstance<T>(obj));
        return nb::cast<T>(obj);
      });
  return result;
}
```

**After:**
```cpp
template <typename T>
std::vector<T> from_pysequence(nb::sequence seq) {
  std::vector<T> result;
  result.reserve(seq.size());  // Pre-allocate to avoid reallocations
  std::transform(
      seq.begin(), seq.end(), std::back_inserter(result), [](nb::handle obj) {
        NVF_ERROR(nb::isinstance<T>(obj));
        return nb::cast<T>(obj);
      });
  return result;
}
```

## Fix 4: Improve error handling in tensor_caster.h

### Location: `tensor_caster.h` around line 23-43

**Before:**
```cpp
bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
  if (!src || !THPVariable_Check(src.ptr())) {
    return false;
  }

  // Unpack the C++ tensor from the Python object
  // THPVariable_Unpack returns 'at::Tensor' (which is a smart pointer
  // wrapper)
  value = THPVariable_Unpack(src.ptr());
  return true;
}

static handle from_cpp(
    const at::Tensor& src,
    rv_policy policy,
    cleanup_list* cleanup) {
  // Wrap the C++ tensor into a new Python object
  // THPVariable_Wrap creates a new reference (PyObject*)
  PyObject* obj = THPVariable_Wrap(src);
  return handle(obj);
}
```

**After:**
```cpp
bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
  if (!src || !THPVariable_Check(src.ptr())) {
    return false;
  }

  try {
    // Unpack the C++ tensor from the Python object
    // THPVariable_Unpack returns 'at::Tensor' (which is a smart pointer
    // wrapper)
    value = THPVariable_Unpack(src.ptr());
    return true;
  } catch (const std::exception& e) {
    // Let nanobind handle the exception
    return false;
  }
}

static handle from_cpp(
    const at::Tensor& src,
    rv_policy policy,
    cleanup_list* cleanup) {
  // Wrap the C++ tensor into a new Python object
  // THPVariable_Wrap creates a new reference (PyObject*)
  PyObject* obj = THPVariable_Wrap(src);
  if (!obj) {
    throw std::runtime_error("Failed to wrap torch::Tensor to Python object");
  }
  return handle(obj);
}
```

## Fix 5: Add argument names to macro-generated functions in ops.cpp

### Location: `ops.cpp` around line 58-71

**Before:**
```cpp
#define NVFUSER_DIRECT_BINDING_UNARY_OP(NAME, OP_NAME, DOCSTRING)      \
  ops.def(                                                             \
      NAME,                                                            \
      [](ScalarVariant v) -> Val* {                                    \
        return static_cast<Val* (*)(Val*)>(OP_NAME)(convertToVal(v));  \
      },                                                               \
      nb::rv_policy::reference);                                       \
  ops.def(                                                             \
      NAME,                                                            \
      [](TensorView* tv) -> TensorView* {                              \
        return static_cast<TensorView* (*)(TensorView*)>(OP_NAME)(tv); \
      },                                                               \
      DOCSTRING,                                                       \
      nb::rv_policy::reference);
```

**After:**
```cpp
#define NVFUSER_DIRECT_BINDING_UNARY_OP(NAME, OP_NAME, DOCSTRING)      \
  ops.def(                                                             \
      NAME,                                                            \
      [](ScalarVariant v) -> Val* {                                    \
        return static_cast<Val* (*)(Val*)>(OP_NAME)(convertToVal(v));  \
      },                                                               \
      nb::arg("x"),                                                    \
      nb::rv_policy::reference);                                       \
  ops.def(                                                             \
      NAME,                                                            \
      [](TensorView* tv) -> TensorView* {                              \
        return static_cast<TensorView* (*)(TensorView*)>(OP_NAME)(tv); \
      },                                                               \
      nb::arg("x"),                                                    \
      DOCSTRING,                                                       \
      nb::rv_policy::reference);
```

### Location: `ops.cpp` around line 73-102

**Before:**
```cpp
#define NVFUSER_DIRECT_BINDING_BINARY_OP(NAME, OP_NAME, DOCSTRING)       \
  ops.def(                                                               \
      NAME,                                                              \
      [](ScalarVariant lhs, ScalarVariant rhs) -> Val* { ... },         \
      nb::rv_policy::reference);                                         \
  ops.def(                                                               \
      NAME,                                                              \
      [](TensorView* lhs, ScalarVariant rhs) -> TensorView* { ... },    \
      nb::rv_policy::reference);                                         \
  // ... etc
```

**After:**
```cpp
#define NVFUSER_DIRECT_BINDING_BINARY_OP(NAME, OP_NAME, DOCSTRING)       \
  ops.def(                                                               \
      NAME,                                                              \
      [](ScalarVariant lhs, ScalarVariant rhs) -> Val* { ... },         \
      nb::arg("lhs"),                                                    \
      nb::arg("rhs"),                                                    \
      nb::rv_policy::reference);                                         \
  ops.def(                                                               \
      NAME,                                                              \
      [](TensorView* lhs, ScalarVariant rhs) -> TensorView* { ... },    \
      nb::arg("lhs"),                                                    \
      nb::arg("rhs"),                                                    \
      nb::rv_policy::reference);                                         \
  // ... add nb::arg to all overloads
```

## Testing Recommendations

After applying these fixes, test with:

```python
import nvfuser

# Test __repr__ works
fusion = nvfuser.Fusion()
print(repr(fusion))  # Should show nice representation

# Test that objects stay alive (keep_alive test)
def test_lifetime():
    fusion = nvfuser.Fusion()
    t0 = fusion.define_tensor(...)
    definition = t0.definition()  # Should not crash even after t0 goes out of scope
    return definition

d = test_lifetime()
print(d)  # Should not crash

# Test vector performance
import time
large_list = list(range(10000))
start = time.time()
result = nvfuser.some_function_taking_vector(large_list)
print(f"Time: {time.time() - start}")  # Should be faster with reserve()
```

## Summary of Changes

1. **ir.cpp**: Add `nb::keep_alive<0, 1>()` to ~10 methods
2. **ir.cpp**: Add `__repr__` to 5 classes
3. **runtime.cpp**: Add `__repr__` to 2 classes
4. **direct_utils.h**: Add `.reserve()` to 1 template function
5. **tensor_caster.h**: Add error handling to 2 methods
6. **ops.cpp**: Add `nb::arg()` to 3 macros (~12 overloads each)

Estimated time to apply all fixes: **30-45 minutes**

Expected impact:
- **Memory safety**: Improved with keep_alive (prevents crashes)
- **Developer experience**: Much better with __repr__
- **Performance**: Slightly better with reserve()
- **Robustness**: Better error messages with tensor_caster improvements
