# PolymorphicValue and KernelArgumentHolder Notes

**Objective:** Understand `PolymorphicValue` and `KernelArgumentHolder` to determine how non-tensor outputs could be represented and potentially allocated, particularly in the context of `inferOutputSizes`.

## PolymorphicValue

Based on `csrc/polymorphic_value.h` and `lib/dynamic_type/src/dynamic_type/dynamic_type.h`.

1.  **Core Idea:** `PolymorphicValue` is an alias for `dynamic_type::DynamicType<...>`, a generic wrapper class based on `std::variant`. It allows storing values of different, predefined types within a single object and performing operations on them dynamically.

2.  **Supported Types:** The specific instantiation in `polymorphic_value.h` supports:
    *   **`std::monostate`**: Represents a null or uninitialized state (implicit base case from `dynamic_type`).
    *   **`at::Tensor`**: Standard PyTorch tensors.
    *   **`double`**: 64-bit floating-point numbers.
    *   **`int64_t`**: 64-bit integers.
    *   **`bool`**: Boolean values.
    *   **`std::complex<double>`**: Complex numbers.
    *   **`Pointer`**: A custom wrapper (`csrc/polymorphic_value.h`) to represent generic pointers (`void*` internally) along with the size of the pointed-to type. Used for passing raw device pointers.
    *   **`StructHandle`**: A custom wrapper (`csrc/polymorphic_value.h`) around `std::shared_ptr<Struct>` for handling user-defined C++ structs (like `TensorMetaData`).
    *   **`Opaque`**: A custom wrapper (`csrc/opaque_type.h`) for storing arbitrary byte sequences, useful for custom data types not directly supported.
    *   **`std::vector<PolymorphicValue>`**: Allows nesting, enabling representation of lists containing any mix of supported types.

3.  **Creating Instances / Defaults:**
    *   **Default:** `PolymorphicValue()` creates an instance holding `std::monostate` (the null/empty state).
    *   **Scalars:** `PolymorphicValue(3.14)`, `PolymorphicValue(123LL)`, `PolymorphicValue(true)`, `PolymorphicValue(std::complex<double>(1.0, 2.0))`.
    *   **`at::Tensor`:** Requires an existing `at::Tensor` object, e.g., `PolymorphicValue(at::randn({2, 2}))`. For placeholders, one might use a meta tensor: `PolymorphicValue(at::empty({2,2}, at::TensorOptions().device(c10::kMeta)))`.
    *   **`Pointer`:** `PolymorphicValue(Pointer(ptr_to_some_data))` or `PolymorphicValue(Pointer())` for a null pointer representation.
    *   **`StructHandle`:** Requires an existing `std::shared_ptr<Struct>` (or a derived class like `TensorMetaData`). Creating a default depends on the specific struct's constructor.
    *   **`Opaque`:** `PolymorphicValue(Opaque(byte_vector))` or `PolymorphicValue(Opaque())` for an empty opaque value.
    *   **`std::vector`:** `PolymorphicValue(std::vector<PolymorphicValue>{PolymorphicValue(1LL), PolymorphicValue(2.0)})`.

4.  **Operations:** `dynamic_type` provides mechanisms for type checking (`is<T>()`), access (`as<T>()`), and operator overloading that dispatches to the underlying type's implementation if available.

## KernelArgumentHolder

Based on `csrc/runtime/executor_kernel_arg.h` and `csrc/runtime/executor_kernel_arg.cpp`.

1.  **Purpose:** Acts as a specialized container (`std::vector<PolymorphicValue>`) to hold the arguments passed to a fusion kernel for both compilation (shape inference, scheduling) and execution.

2.  **Storage:** Directly stores `PolymorphicValue` objects in its internal `arguments_` vector.

3.  **Adding Arguments (`push`):**
    *   Provides numerous `push` overloads for convenience (e.g., `push(at::Tensor)`, `push(double)`, `push(const std::vector<at::Tensor>&)`, `push(const c10::ArrayRef<c10::IValue>&)`).
    *   The fundamental operation involves creating a `PolymorphicValue` from the input and adding it to the internal vector (e.g., `push(PolymorphicValue val)`).
    *   **`pushTensorProxy(...)`:** A crucial method for compilation/inference. It *doesn't* store a real tensor. Instead, it creates an `at::Tensor` on the `Meta` device with the given sizes, strides, and dtype, and pushes *that* `PolymorphicValue` into the holder. This acts as a placeholder carrying shape/type information without needing actual data or memory.

4.  **Metadata:** Also stores `device_index_` and an optional `cache_id_`.

## Interaction and Relevance to `inferOutputSizes`

1.  **Storage:** `KernelArgumentHolder` is the standard way arguments (inputs and outputs) are grouped and passed around during compilation and runtime setup. It stores these arguments as `PolymorphicValue`s.

2.  **Current `inferOutputSizes` Behavior:** As seen in `allocations.cpp`, the current `inferOutputSizes` function:
    *   Iterates through the fusion's outputs.
    *   Asserts that outputs *must* be `TensorView`s.
    *   Calculates the sizes and strides for these `TensorView`s.
    *   Creates **tensor proxies** using `output_tensor_proxies.pushTensorProxy(...)`.
    *   Returns a `KernelArgumentHolder` containing only these *meta-tensor proxies*.

3.  **Allocating Non-Tensor Outputs:**
    *   To handle non-tensor outputs (like scalars, structs, opaque types), `inferOutputSizes` would need to be modified.
    *   Instead of asserting `output->isA<TensorView>()`, it would need to check the actual `DataType` of the output `Val`.
    *   If the output is a scalar (e.g., `DataType::Double`, `DataType::Int`), it should create a `PolymorphicValue` holding a default or representative value of that type (e.g., `PolymorphicValue(double(0.0))` or perhaps `std::monostate()` if the value isn't known yet) and `push` *that* into the result `KernelArgumentHolder`.
    *   If the output is a `Struct`, it would need to construct an appropriate `StructHandle` (potentially default-constructed if possible for the specific struct) and push a `PolymorphicValue` holding that handle.
    *   If the output is `Opaque`, it might push `PolymorphicValue(Opaque())`.
    *   Crucially, it would return a `KernelArgumentHolder` containing a *mix* of `PolymorphicValue`s: meta-tensor proxies for tensor outputs and actual `PolymorphicValue`s (potentially with default/placeholder values) for non-tensor outputs.
    *   The downstream function `allocateOutputs` would then need corresponding logic to inspect the type held by each `PolymorphicValue` in the holder returned by `inferOutputSizes`. If it's a meta-tensor, it allocates; if it's a non-tensor `PolymorphicValue`, it might simply pass it through or perform a different kind of initialization/allocation based on its type.
    *   **Runtime Confirmation (`executor.cpp`):** The review of `KernelExecutor::computeArgs` confirms this approach is viable. This function uses `polymorphicValueToBytes` to convert non-tensor `PolymorphicValue`s (scalars, pointers, structs, opaque) directly into their byte representation for the kernel launch. This means the runtime path can correctly handle the proposed default/placeholder `PolymorphicValue` instances returned by a modified `inferOutputSizes`.
    *   **Alternative (`AllocationType::Evaluate`):** Note that `executor.cpp` also shows a mechanism (`AllocationType::Evaluate`) where certain outputs (often non-tensors) are computed directly by the `ExpressionEvaluator` on the host *instead* of being allocated/returned by the kernel. This is distinct from handling non-tensors *computed by* the kernel.

**In summary:** `PolymorphicValue` is capable of holding the necessary non-tensor types. `KernelArgumentHolder` can store these `PolymorphicValue`s. The primary change needed is within `inferOutputSizes` to create and return appropriate `PolymorphicValue` instances for non-tensor outputs, instead of assuming all outputs are tensors and only creating meta-tensor proxies. The runtime execution path is already equipped to handle these non-tensor `PolymorphicValue` arguments.

## Key Public APIs

### `PolymorphicValue` (Alias for `dynamic_type::DynamicType<...>`)

*   **Constructors:** `PolymorphicValue()`, `PolymorphicValue(T&& value)` (where `T` is one of the supported types like `double`, `at::Tensor`, `StructHandle`, `Pointer`, `Opaque`, `bool`, `int64_t`, `std::complex<double>`, `std::vector<PolymorphicValue>`).
*   **Type Checking:** `template <typename T> bool is() const;` (Checks if the held value is of type `T`). Includes `isNull()` and `hasValue()` for checking `std::monostate`.
*   **Value Access:** `template <typename T> T& as();`, `template <typename T> const T& as() const;` (Returns a reference to the held value, throws if type `T` doesn't match).
*   **Type Info:** `const std::type_info& type() const;` (Returns `typeid` of the held value).
*   **Operators:** Overloads many standard operators (`+`, `-`, `*`, `/`, `==`, `<`, `[]`, `->*`, `<<` etc.) which dispatch to the held type's implementation if compatible.
*   **Conversion (Explicit):** `explicit constexpr operator T() const;` (Allows explicit casting to compatible types).

### `KernelArgumentHolder`

*   **Constructors:** Default, Copy, Move, Variadic template `KernelArgumentHolder(Args&&... args)` (populates holder by calling `push` on arguments).
*   **`push(...)` Overloads:** For individual supported types (`at::Tensor`, `PolymorphicValue`, `std::optional<at::Tensor>`), containers (`const std::vector<at::Tensor>&`, `const c10::ArrayRef<c10::IValue>&`, `std::initializer_list`, `std::vector<PolymorphicValue>`), and specifically for another `KernelArgumentHolder`.
*   **`pushTensorProxy(...)`:** `void pushTensorProxy(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, at::ScalarType dtype);` (Adds a meta-tensor proxy).
*   **Element Access:** `PolymorphicValue& operator[](size_t ind);`, `const PolymorphicValue& operator[](size_t ind) const;`, `PolymorphicValue& back();`, `const PolymorphicValue& back() const;`.
*   **Iteration:** `begin()`, `end()`, `rbegin()`, `rend()`, `cbegin()`, `cend()`.
*   **Size/Capacity:** `size()`, `empty()`, `reserve(size_t)`, `resize(size_t)`.
*   **Modification:** `void erase(const PolymorphicValue& arg_to_delete);`.
*   **Metadata:** `void setDeviceIndex(std::optional<int8_t> index = std::nullopt);`, `int8_t getDeviceIndex() const;`, `void setCacheId(size_t id);`, `std::optional<size_t> getCacheId() const;`.
*   **Utility:** `PrimDataType getSmallestIndexTypeOfArguments() const;`, `std::string toString() const;`.
*   **Serialization:** `serialize(...)`, `deserialize(...)`.

### `Pointer`

*   **Constructors:** `Pointer()`, `Pointer(T* ptr)`, `Pointer(void* ptr, DataType dtype)`.
*   **Casting:** `explicit operator T*() const;`, `explicit operator bool() const;`, `explicit operator int64_t() const;`, etc.
*   **Arithmetic:** `operator+=`, `operator-=`, `operator++`, `operator--`, `operator+`, `operator-` (with offsets).
*   **Comparison:** `operator==`, `operator!=`, `operator<`, `operator>`, `operator<=`, `operator>=` (with `Pointer` and `nullptr_t`).
*   **Info:** `int64_t size() const;` (returns size of pointed-to type).

### `StructHandle`

*   **Constructor:** `StructHandle(std::shared_ptr<Struct> struct_ptr)`.
*   **Assignment:** `operator=(std::shared_ptr<Struct> struct_ptr)`.
*   **Type Checking:** `template <typename T> bool is() const;` (Checks if underlying struct is of type `T`).
*   **Value Access:** `template <typename T> T& as() const;` (Gets reference to underlying struct of type `T`).
*   **Type Info:** `StructType type() const;` (Gets the `StructType` definition).
*   **Member Access:** `operator->*` overloaded for direct member access (`obj->*Class::member`) and access by string key (`obj->*"member_name"`).
*   **Comparison:** `operator==`.

### `Opaque`

*   **Constructors:** `Opaque()`, `Opaque(std::vector<std::byte> bytes)`.
*   **Access:** `const std::vector<std::byte>& bytes() const;`, `size_t size() const;`.
*   **Comparison:** `operator==`. 