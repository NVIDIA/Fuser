# Implementation Checklist

Quick reference checklist for applying the recommended nanobind improvements.

## ⏱️ Time Budget
- **Minimum (High Priority Only)**: 1-2 hours
- **Recommended (High + Medium)**: 3-4 hours
- **Complete (All improvements)**: 6-8 hours

---

## 🔴 Phase 1: High Priority (1-2 hours)

### Task 1: Add `nb::keep_alive` Policies ⏱️ 30 min

**File: `ir.cpp`**

- [ ] Line ~70: Add to `Val::definition()`
  ```cpp
  nb::keep_alive<0, 1>()
  ```
- [ ] Line ~80: Add to `Val::uses()`
- [ ] Line ~96: Add to `Expr::input()`
- [ ] Line ~114: Add to `Expr::output()`
- [ ] Line ~153: Add to `IterDomain::extent()`
- [ ] Search for `TensorView` methods returning pointers:
  - [ ] `domain()`
  - [ ] `logical_domain()`
  - [ ] `root_domain()`
  - [ ] `allocation_domain()`
  - [ ] Any other methods returning `nb::rv_policy::reference`

**File: `runtime.cpp`**

- [ ] Search for methods returning pointers with `nb::rv_policy::reference`
- [ ] Line ~368: Check `FusionExecutorCache::fusion()`

**File: `internal_ir.cpp`**

- [ ] Check all methods with `nb::rv_policy::reference`

**Test:**
```python
# Should not crash
fusion = nvfuser.Fusion()
t = fusion.define_tensor(...)
definition = t.definition()
del t
import gc; gc.collect()
print(definition)  # Should work
```

---

### Task 2: Add `__repr__` Methods ⏱️ 20 min

**File: `ir.cpp`**

- [ ] Line ~37: `Statement` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](Statement* self) {
      return "<nvfuser.Statement at 0x" +
             std::to_string(reinterpret_cast<uintptr_t>(self)) + ">";
  })
  ```

- [ ] Line ~44: `Val` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](Val* self) {
      return "<nvfuser.Val: " + self->toString() + ">";
  })
  ```

- [ ] Line ~93: `Expr` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](Expr* self) {
      return "<nvfuser.Expr: " + self->toString() + ">";
  })
  ```

- [ ] Line ~134: `IterDomain` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](IterDomain* self) {
      return "<nvfuser.IterDomain: " + self->toString(0) + ">";
  })
  ```

- [ ] Line ~181: `TensorDomain` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](TensorDomain* self) {
      return "<nvfuser.TensorDomain: " + self->toString(0) + ">";
  })
  ```

- [ ] Line ~189: `TensorView` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](TensorView* self) {
      return "<nvfuser.TensorView: " + self->toString(0) + ">";
  })
  ```

**File: `runtime.cpp`**

- [ ] Line ~54: `Fusion` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](Fusion* self) {
      std::ostringstream oss;
      oss << "<nvfuser.Fusion with " << self->inputs().size()
          << " inputs, " << self->outputs().size() << " outputs>";
      return oss.str();
  })
  ```

- [ ] Line ~236: `FusionExecutorCache` class - Add `__repr__`
  ```cpp
  .def("__repr__", [](FusionExecutorCache* self) {
      return "<nvfuser.FusionExecutorCache>";
  })
  ```

**Test:**
```python
import nvfuser
fusion = nvfuser.Fusion()
print(repr(fusion))  # Should show nice repr
```

---

### Task 3: Optimize Vector Conversion ⏱️ 5 min

**File: `direct_utils.h`**

- [ ] Line ~22: Add `.reserve()` before the loop
  ```cpp
  template <typename T>
  std::vector<T> from_pysequence(nb::sequence seq) {
    std::vector<T> result;
    result.reserve(seq.size());  // ADD THIS LINE
    std::transform(
        seq.begin(), seq.end(), std::back_inserter(result), [](nb::handle obj) {
          NVF_ERROR(nb::isinstance<T>(obj));
          return nb::cast<T>(obj);
        });
    return result;
  }
  ```

**Test:**
```python
# Performance test
import time
large_list = list(range(100000))
start = time.time()
result = nvfuser.some_function(large_list)
print(f"Time: {time.time() - start}")
```

---

### Task 4: Improve Tensor Caster ⏱️ 10 min

**File: `tensor_caster.h`**

- [ ] Line ~23: Wrap in try-catch
  ```cpp
  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
    if (!src || !THPVariable_Check(src.ptr())) {
      return false;
    }
    try {
      value = THPVariable_Unpack(src.ptr());
      return true;
    } catch (const std::exception& e) {
      return false;
    }
  }
  ```

- [ ] Line ~35: Add null check
  ```cpp
  static handle from_cpp(const at::Tensor& src, rv_policy policy, cleanup_list* cleanup) {
    PyObject* obj = THPVariable_Wrap(src);
    if (!obj) {
      throw std::runtime_error("Failed to wrap torch::Tensor to Python object");
    }
    return handle(obj);
  }
  ```

---

### ✅ Phase 1 Testing

- [ ] Compile successfully
  ```bash
  cd /opt/pytorch/nvfuser
  python setup.py build_ext --inplace
  ```

- [ ] Run existing tests
  ```bash
  pytest python/test/
  ```

- [ ] Quick manual test
  ```python
  import nvfuser
  fusion = nvfuser.Fusion()
  print(repr(fusion))  # Test __repr__
  ```

---

## 🟡 Phase 2: Medium Priority (1-2 hours)

### Task 5: Add GIL Release Guards ⏱️ 15 min

**File: `runtime.cpp`**

- [ ] Line ~276: `FusionExecutorCache::execute` - Add GIL release
  ```cpp
  .def("execute",
       [](FusionExecutorCache& self, ...) {
           // Convert inputs
           KernelArgumentHolder args = from_pyiterable(iter, device);

           // Setup guards
           EnableOptionsGuard enable_opt_guard;
           // ... setup code ...

           // Release GIL for CUDA execution
           nb::gil_scoped_release release;
           KernelArgumentHolder outputs = self.runFusionWithInputs(
               args, std::nullopt, args.getDeviceIndex());
           nb::gil_scoped_acquire acquire;

           return to_tensor_vector(outputs);
       },
       nb::arg("inputs"),
       nb::kw_only(),
       ...)
  ```

OR use `nb::call_guard<>`:
  ```cpp
  .def("execute",
       [](FusionExecutorCache& self, ...) { ... },
       nb::call_guard<nb::gil_scoped_release>(),  // ADD THIS
       nb::arg("inputs"),
       ...)
  ```

**Note**: Check if guards are already in place in the implementation!

**Test:**
```python
import threading
import nvfuser

def background_task():
    # Should run while CUDA executes
    print("Background thread running")

thread = threading.Thread(target=background_task)
thread.start()
outputs = fusion_cache.execute([...])
thread.join()
```

---

### Task 6: Add Exception Translators ⏱️ 20 min

**File: `bindings.cpp`**

- [ ] Add at end of `initNvFuserPythonBindings()`, before the commented cleanup section:
  ```cpp
  // Register exception translators for nvfuser-specific exceptions
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
      try {
          std::rethrow_exception(p);
      } catch (const nvfuser::nvfError& e) {  // Adjust type name
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return true;
      } catch (const c10::Error& e) {  // For PyTorch errors
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return true;
      }
      return false;
  });
  ```

**Note**: Check actual exception types in nvfuser codebase!

**Test:**
```python
import nvfuser
try:
    # Trigger an error
    fusion = nvfuser.Fusion()
    fusion.invalid_operation()
except RuntimeError as e:
    print(f"Caught: {e}")
    assert "nvfuser" in str(e).lower()
```

---

### Task 7: Add Type Hints with `nb::sig()` ⏱️ 30 min

**File: `runtime.cpp`**

- [ ] Line ~276: `FusionExecutorCache::execute` - Add signature
  ```cpp
  .def("execute",
       [](FusionExecutorCache& self, ...) { ... },
       nb::arg("inputs"),
       nb::kw_only(),
       nb::arg("device") = nb::none(),
       nb::arg("_enable_options") = nb::list(),
       nb::arg("_disable_options") = nb::list(),
       nb::sig("def execute(self, inputs: Iterable[torch.Tensor], *, "
               "device: Optional[int] = None, "
               "_enable_options: List[str] = [], "
               "_disable_options: List[str] = []) -> List[torch.Tensor]"),
       R"(...)")
  ```

**File: `ir.cpp`**

- [ ] Add signatures to major public API methods (optional, time permitting)

**File: `ops.cpp`**

- [ ] Consider adding to frequently-used operations (optional)

**Test:**
```python
import nvfuser
import inspect

sig = inspect.signature(nvfuser.FusionExecutorCache.execute)
print(sig)  # Should show typed signature

# Check IDE autocomplete
help(nvfuser.FusionExecutorCache.execute)
```

---

### Task 8: Add Argument Names to Macros ⏱️ 20 min

**File: `ops.cpp`**

- [ ] Line ~58: `NVFUSER_DIRECT_BINDING_UNARY_OP` - Add `nb::arg("x")`
  ```cpp
  #define NVFUSER_DIRECT_BINDING_UNARY_OP(NAME, OP_NAME, DOCSTRING)      \
    ops.def(                                                             \
        NAME,                                                            \
        [](ScalarVariant v) -> Val* { ... },                             \
        nb::arg("x"),                          /* ADD THIS */           \
        nb::rv_policy::reference);                                       \
    ops.def(                                                             \
        NAME,                                                            \
        [](TensorView* tv) -> TensorView* { ... },                       \
        nb::arg("x"),                          /* ADD THIS */           \
        DOCSTRING,                                                       \
        nb::rv_policy::reference);
  ```

- [ ] Line ~73: `NVFUSER_DIRECT_BINDING_BINARY_OP` - Add `nb::arg("lhs")`, `nb::arg("rhs")`
- [ ] Line ~104: `NVFUSER_DIRECT_BINDING_TERNARY_OP` - Add arg names
- [ ] Other macros - Add appropriate arg names

**Test:**
```python
import nvfuser
help(nvfuser.ops.add)  # Should show argument names
```

---

### ✅ Phase 2 Testing

- [ ] Compile successfully
- [ ] Run full test suite
- [ ] Test GIL release with threading
- [ ] Test exception handling
- [ ] Test type hints in IDE
- [ ] Verify help() shows argument names

---

## 🟢 Phase 3: Optional Improvements

### Task 9: Module Organization ⏱️ 2-3 hours
⚠️ **Breaking Change** - Consider carefully!

- [ ] Create submodules (ir, ops, runtime, etc.)
- [ ] Update all imports
- [ ] Update documentation
- [ ] Update tests
- [ ] Provide migration guide for users

---

### Task 10: Use `nb::overload_cast<>` ⏱️ 30 min

- [ ] Replace `static_cast` with `nb::overload_cast<>` where supported
- [ ] Check nanobind version compatibility

---

### Task 11: Enum `.export_values()` ⏱️ 15 min

- [ ] Review each enum on case-by-case basis
- [ ] Add `.export_values()` where appropriate
- [ ] Update documentation

---

## 📋 Final Checklist

### Code Quality
- [ ] All changes compile without warnings
- [ ] No new linter errors
- [ ] Code follows existing style

### Testing
- [ ] All existing tests pass
- [ ] New features tested manually
- [ ] Performance benchmarks run
- [ ] Memory safety verified (valgrind/sanitizers)

### Documentation
- [ ] README updated with improvements
- [ ] CHANGELOG updated
- [ ] API docs updated (if needed)
- [ ] Migration notes for users (if breaking changes)

### Measurements
- [ ] Binary size measured (before/after)
- [ ] Import time measured (before/after)
- [ ] Memory usage profiled
- [ ] Performance benchmarked

---

## 📊 Progress Tracking

### High Priority (Must Do)
- [ ] Task 1: `keep_alive` policies (30 min)
- [ ] Task 2: `__repr__` methods (20 min)
- [ ] Task 3: Vector `.reserve()` (5 min)
- [ ] Task 4: Tensor caster errors (10 min)

**Total: ~65 minutes**

### Medium Priority (Should Do)
- [ ] Task 5: GIL release (15 min)
- [ ] Task 6: Exception translators (20 min)
- [ ] Task 7: Type hints (30 min)
- [ ] Task 8: Macro arg names (20 min)

**Total: ~85 minutes**

### Grand Total
- [ ] All high priority tasks ✅
- [ ] All medium priority tasks ✅
- [ ] Testing completed ✅
- [ ] Documentation updated ✅

**Estimated Total Time: 3-4 hours**

---

## 🚀 Quick Start

```bash
# 1. Create a branch
git checkout -b nanobind-improvements

# 2. Open this checklist in editor
# Keep it visible while working

# 3. Start with Task 1
# Work through tasks in order

# 4. Test after each phase
python setup.py build_ext --inplace
pytest python/test/

# 5. Commit incrementally
git commit -m "Add keep_alive policies"
git commit -m "Add __repr__ methods"
# etc.

# 6. Final review and merge
git push origin nanobind-improvements
# Create PR
```

---

## 📞 If You Get Stuck

1. **Compilation errors**: Check include statements and nanobind version
2. **Linking errors**: Verify PyTorch integration in tensor_caster.h
3. **Runtime errors**: Test with small examples first
4. **Type errors**: Review type caster implementations

Refer back to detailed documentation:
- QUICK_FIXES.md for code examples
- NANOBIND_IMPROVEMENTS.md for explanations
- NANOBIND_SPECIFIC_FEATURES.md for advanced topics

---

**Good luck! 🎉**
