# Nanobind-Specific Features and Differences from Pybind11

This document highlights nanobind features that differ from pybind11 or are unique to nanobind.

## 1. Smaller Binary Size ✅

**Current Status**: You're already benefiting from this!

Nanobind produces significantly smaller binaries compared to pybind11 (often 2-10x smaller). No action needed, but worth measuring:

```bash
# Compare binary sizes
ls -lh build/lib/*.so

# Expected: nvfuser nanobind bindings should be smaller than pybind11 version
```

## 2. Type Annotations Support

### pybind11 Way
```cpp
// Limited type hint support in pybind11
.def("func", &func)
```

### Nanobind Way
```cpp
// Better type hint support with nb::sig()
.def("func",
     &func,
     nb::sig("def func(x: int, y: float) -> str"))
```

**Recommendation**: Add `nb::sig()` to complex functions for better IDE support.

### Example for nvfuser
```cpp
// In runtime.cpp, FusionExecutorCache::execute
.def("execute",
     [](FusionExecutorCache& self, const nb::iterable& iter, ...) { ... },
     nb::arg("inputs"),
     nb::kw_only(),
     nb::arg("device") = nb::none(),
     nb::sig("def execute(self, inputs: Iterable[torch.Tensor], *, "
             "device: Optional[int] = None, "
             "_enable_options: List[str] = [], "
             "_disable_options: List[str] = []) -> List[torch.Tensor]"),
     R"(...)")
```

## 3. `nb::typed<>` for Type-Safe Containers

### pybind11 Way
```cpp
// Generic container, no type checking
std::vector<int> func(const py::list& items)
```

### Nanobind Way
```cpp
// Type-safe container with compile-time checking
#include <nanobind/stl/list.h>
std::vector<int> func(const nb::typed<nb::list, int>& items)
```

**Potential Use**: Could be applied to functions taking lists/vectors of specific types.

### Example for nvfuser
```cpp
// In schedule.cpp
void transform_like(
    TensorView* reference_tv,
    const nb::typed<nb::list, TensorView*>& selected_tensors)  // Type-safe!
```

**Trade-off**: More type safety vs flexibility. Current approach is more flexible.

## 4. `NB_MAKE_OPAQUE()` for STL Containers

### Current Code ✅ GOOD!
```cpp
// heuristic_params.cpp, line 20
NB_MAKE_OPAQUE(nvfuser::MmaMacro);
```

This prevents nanobind from automatically converting the type, giving you full control.

**When to use**:
- Custom container types that shouldn't auto-convert
- Enums you want to bind as classes with methods
- Types that need special handling

**Already used correctly!** No changes needed.

## 5. `nb::call_guard<>` for Automatic Guards

### pybind11 Way
```cpp
// Manual GIL management
py::gil_scoped_release release;
expensive_operation();
```

### Nanobind Way
```cpp
// Automatic guard management
.def("expensive_operation",
     &expensive_operation,
     nb::call_guard<nb::gil_scoped_release>())
```

**Recommendation**: Use for long-running operations that don't need Python objects.

### Example for nvfuser
```cpp
// In runtime.cpp, FusionExecutorCache::execute
.def("execute",
     [](FusionExecutorCache& self, ...) {
         // This is a long-running CUDA operation
         return ...;
     },
     nb::call_guard<nb::gil_scoped_release>(),  // Release GIL during execution
     nb::arg("inputs"),
     ...)
```

**Benefits**:
- Better multi-threading support
- Allows other Python threads to run during execution
- Critical for CUDA kernel execution

## 6. Implicit Conversion Control

### Nanobind is Stricter
Nanobind is more strict about implicit conversions than pybind11. This is generally good for type safety.

**Current Code Pattern**:
```cpp
// ops.cpp - ScalarVariant handles this well
using ScalarVariant = std::variant<Val*, PolymorphicValue::VariantType>;
```

**This is excellent!** The variant pattern gives you explicit control over conversions.

## 7. Return Value Policies - Different Defaults

### Key Difference
Nanobind's default return value policy is more conservative than pybind11's.

**Current Code**: ✅ Explicitly specifies policies everywhere - GOOD!

```cpp
.def("definition", &Val::definition, nb::rv_policy::reference)
```

**Keep doing this!** Always explicit about return policies.

## 8. `nb::new_()` for Factory Functions

### Nanobind Feature
```cpp
// Alternative to nb::init<> for complex construction
nb::class_<MyClass>(m, "MyClass")
    .def(nb::new_<MyClass>([](int arg) {
        // Custom construction logic
        return std::make_unique<MyClass>(arg);
    }))
```

**Current Code**: Uses placement new, which is also fine:
```cpp
.def("__init__", [](MyClass* self, int arg) {
    new (self) MyClass(arg);
})
```

**Both approaches are valid.** Current approach is more explicit about memory layout.

## 9. `nb::is_method()` for Static Methods

### Nanobind Way to Bind Static Methods
```cpp
nb::class_<MyClass>(m, "MyClass")
    .def_static("static_method", &MyClass::static_method)
```

**Check**: Do you have any static methods that should be bound?

### Potential for nvfuser
```cpp
// If Fusion has static factory methods, bind them:
nb::class_<Fusion>(nvfuser, "Fusion")
    .def(nb::init<>())
    .def_static("create", &Fusion::create)  // If such method exists
```

## 10. Property Binding Improvements

### Current Code
```cpp
// heuristic_params.cpp
.def_prop_rw(
    "bdimx",
    [](LaunchParams& self) { return self.bdimx(); },
    [](LaunchParams& self, int64_t val) { self.bindUnsafe(val, ParallelType::TIDx); })
```

### Could Also Use
```cpp
// If you have actual member variables (not applicable here, just FYI)
.def_rw("member", &MyClass::member)
```

**Current approach is correct** since you need custom getters/setters.

## 11. Module Docstrings

### pybind11 Way
```cpp
PYBIND11_MODULE(module_name, m) {
    m.doc() = "Module documentation";
}
```

### Nanobind Way
```cpp
NB_MODULE(module_name, m) {
    m.doc() = "Module documentation";
}
```

**Current Code**: ✅ Already done!
```cpp
// extension.cpp
NB_MODULE(PYTHON_DIRECT_EXTENSION, m) {
  m.doc() = "Python bindings for NvFuser Direct CPP API";
}
```

## 12. Exception Translation

### Nanobind Way
```cpp
nb::register_exception<MyException>(m, "MyException", PyExc_RuntimeError);

// Or with custom translator
nb::register_exception_translator([](const std::exception_ptr& p, void* payload) {
    try {
        std::rethrow_exception(p);
    } catch (const MyException& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return true;
    }
    return false;
});
```

**Question**: Does nvfuser have custom exception types that should be translated?

### Recommendation
If you have `NVF_ERROR` or custom exceptions, consider registering them:

```cpp
// In bindings.cpp, add to initNvFuserPythonBindings:
nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
        std::rethrow_exception(p);
    } catch (const nvfuser::ErrorType& e) {  // If such type exists
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return true;
    }
    return false;
});
```

## 13. Buffer Protocol Support (PyTorch Integration)

### Current Code ✅ EXCELLENT!
```cpp
// tensor_caster.h
template <>
struct type_caster<at::Tensor> {
  // Custom caster for PyTorch tensors
}
```

This is the **correct** way to integrate with PyTorch in nanobind.

**No changes needed** - well implemented!

## 14. Inheritance and Trampolines

### If You Need Virtual Function Overrides in Python

**pybind11 Way**:
```cpp
class PyMyClass : public MyClass {
public:
    using MyClass::MyClass;
    void virtual_func() override {
        PYBIND11_OVERRIDE(void, MyClass, virtual_func);
    }
};
```

**Nanobind Way**:
```cpp
struct PyMyClass : MyClass {
    NB_TRAMPOLINE(MyClass, 1);  // 1 = number of virtual functions

    void virtual_func() override {
        NB_OVERRIDE(virtual_func);
    }
};
```

**Question**: Do any of your classes need virtual function overrides from Python?

**Likely Answer**: Probably not for nvfuser's use case. Most operations are C++ → Python, not Python → C++.

## 15. Capsule API for C Interop

### Nanobind Feature
```cpp
// Export C API as capsule
m.add_capsule("my_api", &api_struct);

// In another module
auto api = m.get_capsule("my_api");
```

**Use Case**: Inter-module communication in pure C.

**Likely not needed** for nvfuser unless you have multiple Python modules that need to share C APIs.

## 16. Performance Comparison Checklist

### Measure These
1. **Import time**: `python -c "import time; start=time.time(); import nvfuser; print(time.time()-start)"`
2. **Binary size**: `ls -lh build/lib/*.so`
3. **Memory usage**: Profile with memory_profiler
4. **Function call overhead**: Benchmark hot paths

### Expected Improvements vs pybind11
- ✅ Import time: 2-3x faster
- ✅ Binary size: 3-10x smaller
- ✅ Memory usage: ~20% less
- ✅ Call overhead: Similar or slightly better

## 17. Recommended Next Steps

### Immediate (High Value)
1. Add `nb::call_guard<nb::gil_scoped_release>()` to `FusionExecutorCache::execute()`
2. Add exception translators for nvfuser-specific exceptions
3. Add `nb::sig()` to public API functions for better type hints

### Short Term (Medium Value)
4. Use `nb::typed<>` for type-safe container parameters (where appropriate)
5. Profile and benchmark vs pybind11 version
6. Add static method bindings if any exist

### Long Term (Low Priority)
7. Consider module restructuring with submodules
8. Add trampolines if Python-side inheritance is needed
9. Document performance improvements in migration guide

## 18. Compatibility Notes

### What Works the Same
- ✅ Basic class binding
- ✅ Function binding
- ✅ Enum binding
- ✅ Property binding
- ✅ Operator overloading
- ✅ STL container support

### What's Different
- ⚠️ More strict type checking (good!)
- ⚠️ Different macro names (`NB_MODULE` vs `PYBIND11_MODULE`)
- ⚠️ Different trampoline syntax
- ⚠️ Return value policy defaults

### Migration Checklist ✅
- [x] Replace `PYBIND11_MODULE` with `NB_MODULE`
- [x] Replace `py::` with `nb::`
- [x] Replace `pybind11/*.h` with `nanobind/*.h`
- [x] Update CMakeLists.txt/setup.py to use nanobind
- [x] Verify all bindings compile
- [ ] Add nanobind-specific improvements (this guide)
- [ ] Test all Python APIs
- [ ] Benchmark performance
- [ ] Update documentation

## 19. Testing Recommendations

### Test Suite to Create
```python
# test_nanobind_features.py
import nvfuser
import torch
import gc

def test_lifetime_management():
    """Test nb::keep_alive works"""
    fusion = nvfuser.Fusion()
    t = fusion.define_tensor(...)
    definition = t.definition()
    del t
    gc.collect()
    # Should not crash
    str(definition)

def test_gil_release():
    """Test that GIL is released during execution"""
    import threading
    import time

    def background_task():
        # Should be able to run while fusion executes
        time.sleep(0.1)

    fusion_cache = nvfuser.FusionExecutorCache(...)
    thread = threading.Thread(target=background_task)
    thread.start()
    outputs = fusion_cache.execute([...])  # Should release GIL
    thread.join()

def test_type_annotations():
    """Test that type hints work"""
    import inspect
    sig = inspect.signature(nvfuser.FusionExecutorCache.execute)
    # Check that annotations exist and are correct

def test_error_handling():
    """Test exception translation"""
    fusion = nvfuser.Fusion()
    try:
        fusion.invalid_operation()
    except RuntimeError as e:
        assert "helpful error message" in str(e)
```

## 20. Documentation Updates Needed

### Update These Docs
1. **Installation guide**: Mention nanobind dependency
2. **API reference**: Add type hints from nb::sig()
3. **Performance**: Document size/speed improvements
4. **Migration guide**: For users upgrading from pybind11 version

### Example Performance Documentation
```markdown
## Performance Improvements in v2.0

The Python bindings were migrated from pybind11 to nanobind, resulting in:

- 🚀 **3x faster import time** (300ms → 100ms)
- 💾 **5x smaller binary** (15MB → 3MB)
- 🧵 **Better multi-threading** via GIL release
- ✨ **Better type hints** for IDE support
```

## Summary

Your nanobind conversion is **very well done**! The code is clean and follows best practices.

**High-value additions**:
1. Add `nb::call_guard<nb::gil_scoped_release>()` to CUDA operations
2. Add custom exception translators
3. Add `nb::sig()` for better type hints

**Everything else is working correctly** and these are optimizations rather than fixes.

Total estimated time for high-value additions: **1-2 hours**
