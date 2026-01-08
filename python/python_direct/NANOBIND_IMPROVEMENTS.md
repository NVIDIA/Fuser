# Nanobind Optimization Recommendations

This document outlines improvements for the nvfuser python bindings after converting from pybind11 to nanobind.

## 1. Lambda Capture Optimization

### Current Issue
Many simple property getters use lambdas when direct member function pointers would suffice.

### Example - Current Code
```cpp
// In ir.cpp, line ~196
.def_prop_ro(
    "ndim",
    [](TensorView* self) {
        return std::ranges::distance(
            self->getLogicalDomain() | TensorDomain::kNoReductions);
    },
```

### Recommendation
This is fine for complex logic, but for simple property access, consider direct binding when possible. The lambda is necessary here due to the ranges operation, which is good.

However, check for simpler cases that could use direct function pointers:
```cpp
// Good: Direct function pointer (no lambda overhead)
.def_prop_ro("name", &ClassName::getName)

// Current: Unnecessary lambda wrapper
.def_prop_ro("name", [](ClassName* self) { return self->getName(); })
```

## 2. Missing `nb::keep_alive` Policies

### Current Issue
Functions returning raw pointers to objects with lifetime dependencies don't specify keep_alive policies.

### Example - Current Code
```cpp
// In ir.cpp
.def("definition", &Val::definition, nb::rv_policy::reference)
.def("uses", &Val::uses, nb::rv_policy::reference)
```

### Recommendation
Add `nb::keep_alive<0, 1>()` to ensure the returned object's lifetime is tied to the owner:

```cpp
.def("definition",
     &Val::definition,
     nb::rv_policy::reference,
     nb::keep_alive<0, 1>(),  // Keep 'this' (arg 1) alive as long as return value (0) is alive
     R"(Get the definition of this expression.)")
```

**Apply this pattern to:**
- `Val::definition()` and `Val::uses()` in ir.cpp
- `Expr::input()` and `Expr::output()` in ir.cpp
- `TensorView::domain()` and similar methods
- Any method returning pointers/references to internally owned objects

## 3. Unnecessary Lambda Wrappers for Static Casts

### Current Issue
Many overload resolutions use static_cast wrapped in lambdas.

### Example - Current Code
```cpp
// In ir.cpp, line ~449
.def(
    "merge",
    static_cast<TensorView* (TensorView::*)(int64_t, int64_t)>(
        &TensorView::merge),
    nb::arg("axis_o"),
    nb::arg("axis_i"),
    nb::rv_policy::reference)
```

### Recommendation
This is actually fine! This is the standard way to disambiguate overloads in both pybind11 and nanobind. However, nanobind also supports `nb::overload_cast<>`:

```cpp
// Alternative using nb::overload_cast (more readable)
.def(
    "merge",
    nb::overload_cast<int64_t, int64_t>(&TensorView::merge),
    nb::arg("axis_o"),
    nb::arg("axis_i"),
    nb::rv_policy::reference)
```

Note: `nb::overload_cast` may not be available in all nanobind versions. Check your version before adopting.

## 4. Enum Export Improvements

### Current Code is Good!
```cpp
// enum.cpp
nb::enum_<PrimDataType>(nvfuser, "DataType")
    .value("Double", DataType::Double)
    .value("Float", DataType::Float)
    // ...
```

### Optional Enhancement
Consider using `.export_values()` if you want enum values accessible without the class prefix:

```cpp
nb::enum_<PrimDataType>(nvfuser, "DataType")
    .value("Double", DataType::Double)
    .value("Float", DataType::Float)
    // ...
    .export_values();  // Allows: nvfuser.Double instead of nvfuser.DataType.Double
```

**Recommendation:** Only add `.export_values()` if the Python API design calls for it. It can cause namespace pollution.

## 5. Constructor Binding - Placement New Pattern

### Current Code is Good!
```cpp
// multidevice.cpp, line 76
.def(
    "__init__",
    [](DeviceMesh* self, at::Tensor devices) {
        new (self) DeviceMesh(std::move(devices));
    },
```

This placement-new pattern is correct for nanobind. However, ensure you're using it consistently.

### Check For
Some constructors use `nb::init<>()` while others use placement new. Both are valid, but be consistent:

**When to use `nb::init<>()`:**
- Simple constructors with direct argument passing
- No need for argument transformation

```cpp
.def(nb::init<size_t>(), nb::arg("max_fusions"))
```

**When to use placement new:**
- Need to transform arguments
- Need to std::move arguments
- Need complex initialization logic

```cpp
.def("__init__", [](MyClass* self, Arg arg) {
    new (self) MyClass(std::move(arg));
})
```

## 6. String Conversions - `__str__` vs `__repr__`

### Current Issue
Most classes only define `__str__`, not `__repr__`.

### Example - Current Code
```cpp
// ir.cpp
nb::class_<Statement>(nvfuser, "Statement")
    .def("__str__", [](Statement* self) { return self->toString(); })
```

### Recommendation
Add `__repr__` for better debugging experience:

```cpp
nb::class_<Statement>(nvfuser, "Statement")
    .def("__str__", [](Statement* self) { return self->toString(); })
    .def("__repr__", [](Statement* self) {
        return "<Statement: " + self->toString() + ">";
    })
```

**Apply to:** All classes that currently only have `__str__`.

## 7. `nb::sig()` for Better Type Annotations

### Current Issue
Complex function signatures don't have explicit type annotations for Python.

### Recommendation
Use `nb::sig()` to provide better type hints:

```cpp
.def("execute",
     [](FusionExecutorCache& self, const nb::iterable& iter, ...) { ... },
     nb::arg("inputs"),
     nb::kw_only(),
     nb::arg("device") = nb::none(),
     nb::sig("def execute(self, inputs: Iterable, *, device: Optional[int] = None) -> List[torch.Tensor]"),
     R"(...)")
```

**Priority:** Medium - improves IDE autocomplete and type checking.

## 8. Module Organization with Submodules

### Current Issue
All bindings are added to a single flat module.

### Example - Current Code
```cpp
// bindings.cpp
void initNvFuserPythonBindings(nb::module_& nvfuser) {
  bindEnums(nvfuser);
  bindHeuristicParams(nvfuser);
  bindFusionIr(nvfuser);
  // ... all in one module
}
```

### Recommendation
Consider organizing into submodules for better namespace organization:

```cpp
void initNvFuserPythonBindings(nb::module_& nvfuser) {
  // Core IR submodule
  nb::module_ ir = nvfuser.def_submodule("ir", "NvFuser IR nodes");
  bindFusionIr(ir);
  bindInternalIr(ir);

  // Operations submodule
  nb::module_ ops = nvfuser.def_submodule("ops", "NvFuser operations");
  bindOperations(ops);

  // Runtime submodule
  nb::module_ runtime = nvfuser.def_submodule("runtime", "NvFuser runtime");
  bindRuntime(runtime);

  // Keep some common items at top level for convenience
  bindEnums(nvfuser);
}
```

**Benefits:**
- Better API organization
- Clearer import paths (`from nvfuser.ir import TensorView`)
- Reduced naming conflicts

**Considerations:**
- Breaking change for existing users
- More complex import paths
- May want to keep common types at top level for convenience

## 9. Vector Conversion Optimization

### Current Code
```cpp
// direct_utils.h
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

### Recommendation
Add `.reserve()` for better performance:

```cpp
template <typename T>
std::vector<T> from_pysequence(nb::sequence seq) {
  std::vector<T> result;
  result.reserve(seq.size());  // Pre-allocate
  std::transform(
      seq.begin(), seq.end(), std::back_inserter(result), [](nb::handle obj) {
        NVF_ERROR(nb::isinstance<T>(obj));
        return nb::cast<T>(obj);
      });
  return result;
}
```

## 10. Return Value Policy Consistency

### Current Status
The code uses `nb::rv_policy::reference` extensively, which is correct for pointer returns.

### Verification Needed
Ensure all pointer-returning methods have appropriate return value policies:

- `nb::rv_policy::reference` - for pointers to objects with external ownership ✓
- `nb::rv_policy::reference_internal` - for pointers to internal objects (use with keep_alive)
- `nb::rv_policy::copy` - for returning copies
- `nb::rv_policy::take_ownership` - for transferring ownership (rare)

### Example Audit
```cpp
// Good: Returns pointer managed elsewhere
.def("fusion", &FusionExecutorCache::fusion, nb::rv_policy::reference)

// Consider: Returns internal state, might need reference_internal + keep_alive
.def("vals", [](Fusion& self) { return self.vals(); },
     nb::rv_policy::reference_internal,
     nb::keep_alive<0, 1>())
```

## 11. Type Caster for at::Tensor

### Current Code is Good!
```cpp
// tensor_caster.h
template <>
struct type_caster<at::Tensor> {
  NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
    if (!src || !THPVariable_Check(src.ptr())) {
      return false;
    }
    value = THPVariable_Unpack(src.ptr());
    return true;
  }

  static handle from_cpp(const at::Tensor& src, rv_policy policy, cleanup_list* cleanup) {
    PyObject* obj = THPVariable_Wrap(src);
    return handle(obj);
  }
};
```

### Recommendation
This is well-implemented! However, consider adding error handling:

```cpp
bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
  if (!src || !THPVariable_Check(src.ptr())) {
    return false;  // Let nanobind handle the error
  }
  try {
    value = THPVariable_Unpack(src.ptr());
    return true;
  } catch (...) {
    return false;
  }
}

static handle from_cpp(const at::Tensor& src, rv_policy policy, cleanup_list* cleanup) {
  PyObject* obj = THPVariable_Wrap(src);
  if (!obj) {
    throw std::runtime_error("Failed to wrap torch::Tensor");
  }
  return handle(obj);
}
```

## 12. Default Argument Optimization

### Current Code
```cpp
// runtime.cpp
.def("print_kernel",
     [](Fusion& f, const CompileParams& compile_params) { ... },
     nb::arg("compile_params") = CompileParams(),
```

### Potential Issue
Creating default CompileParams on every call. Consider:

```cpp
// Option 1: Static default
.def("print_kernel",
     [](Fusion& f, std::optional<CompileParams> compile_params) {
         static const CompileParams default_params;
         const auto& params = compile_params.value_or(default_params);
         // use params...
     },
     nb::arg("compile_params") = nb::none())

// Option 2: Keep current approach if CompileParams construction is cheap
```

## 13. Macro Improvements for Operation Binding

### Current Code is Excellent!
The macros in ops.cpp (`NVFUSER_DIRECT_BINDING_UNARY_OP`, etc.) are well-designed for reducing boilerplate.

### Minor Enhancement
Consider adding a doc parameter to the macros for inline documentation:

```cpp
#define NVFUSER_DIRECT_BINDING_BINARY_OP(NAME, OP_NAME, DOCSTRING) \
  ops.def( \
      NAME, \
      [](ScalarVariant lhs, ScalarVariant rhs) -> Val* { \
        return static_cast<Val* (*)(Val*, Val*)>(OP_NAME)( \
            convertToVal(lhs), convertToVal(rhs)); \
      }, \
      nb::arg("lhs"), \
      nb::arg("rhs"), \
      DOCSTRING, \
      nb::rv_policy::reference);
```

Add argument names to improve help() output.

## 14. Thread Safety Considerations

### Question
Are any of these bindings used in multi-threaded contexts?

### Recommendation
If so, consider using `nb::call_guard<>`:

```cpp
#include <nanobind/call_guard.h>
#include <mutex>

std::mutex my_mutex;

.def("thread_sensitive_method",
     &MyClass::method,
     nb::call_guard<nb::gil_scoped_release, std::scoped_lock<std::mutex>>())
```

## 15. Documentation String Format

### Current Status
Good use of numpy-style docstrings! This is excellent.

### Minor Suggestions
- Consider adding "See Also" sections for related functions
- Add "Examples" sections more liberally (you have some, add more)
- Consider "Raises" sections for error conditions

```python
R"(
Brief description.

Extended description with details.

Parameters
----------
param1 : type
    Description

Returns
-------
type
    Description

Raises
------
ValueError
    When X happens
RuntimeError
    When Y happens

See Also
--------
related_function : Brief description

Examples
--------
>>> import nvfuser
>>> # example code
)"
```

## 16. Suggested Priority Order

**High Priority (Do These):**
1. Add `nb::keep_alive` policies where needed (#2)
2. Add `.reserve()` to vector conversions (#9)
3. Add `__repr__` methods (#6)

**Medium Priority (Nice to Have):**
4. Review and add error handling to type casters (#11)
5. Add more examples to docstrings (#15)
6. Consider `nb::overload_cast` over static_cast (#3)

**Low Priority (Consider Later):**
7. Module organization with submodules (#8) - breaking change
8. Add `nb::sig()` for type hints (#7)
9. Thread safety guards if needed (#14)

## 17. Specific File Recommendations

### `ir.cpp`
- Add `nb::keep_alive<0, 1>()` to `Val::definition()`, `Val::uses()`, `Expr::input()`, `Expr::output()`
- Add `__repr__` methods to `Statement`, `Val`, `Expr`, `IterDomain`, `TensorDomain`, `TensorView`

### `runtime.cpp`
- Add `__repr__` to `Fusion`, `FusionExecutorCache`
- Consider keep_alive for `FusionExecutorCache::fusion()`

### `direct_utils.cpp`
- Add `.reserve()` in `from_pysequence()` template

### `enum.cpp`
- Consider `.export_values()` on a case-by-case basis
- Already well-implemented!

### `bindings.cpp`
- Consider module organization (low priority, breaking change)

### `ops.cpp`
- Excellent macro design, keep as-is
- Consider adding arg names to macros for better help()

## Conclusion

Your nanobind conversion is **very well done**! The code follows good practices and most patterns are correct. The recommendations above are optimizations and enhancements rather than bug fixes.

The most impactful improvements would be:
1. Adding `keep_alive` policies for memory safety
2. Adding `__repr__` methods for better debugging
3. Small performance optimizations (reserve, etc.)

The code is production-ready as-is, and these improvements can be made incrementally.
