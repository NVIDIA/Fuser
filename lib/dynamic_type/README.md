# Introduction

**dynamic_type** is a header-only C++ template library that defines `DynamicType`.

`DynamicType` is a class template that can used to define polymorphic types.
It is similar to `std::variant`, but it is designed to be used in a way that
is more similar to how dynamic types are used in Python. For example, in Python,
you can do something like this:

```python
x = 1
y = 2.5
z = x + y
```

and `z` will be a dynamic float. However, in C++, you will not be able to do:

```C++
using IntOrFloat = std::variant<int, float>;
IntOrFloat x = 1;
IntOrFloat y = 2.5f;
IntOrFloat z = x + y;
```

because the `operator+` on `std::variant` is not defined. The goal of
`DynamicType` is to fill this gap. So you can do
(please ignore the `NoContainers` for now):

```C++
using IntOrFloat = DynamicType<NoContainers, int, float>;
IntOrFloat x = 1;
IntOrFloat y = 2.5f;
IntOrFloat z = x + y;
```

The design purpose of `DynamicType` is to allow the user to forget about the
actual type as much as possible, and use operators seamlessly just like if
they are using Python. `DynamicType` should support arbitrary types, including
user-defined types, pointers, but excluding references, due to the limitation
of the C++ standard. The definition of operators on `DynamicType` should be
automatic. For example, if you have:

```C++
struct CustomType {};
using IntOrFloatOrCustom = DynamicType<NoContainers, int, float, CustomType>;
```

Then the `operator+` on `IntOrFloatOrCustom` should be defined, and it should
be equivalent to one of the following:

- `operator+(int, int)`
- `operator+(float, float)`
- `operator+(int, float)`
- `operator+(float, int)`

depending on the actual type of the `DynamicType`. If the actual type is
`CustomType` which does not have `operator+``, or if the value is null,
then this is a runtime error. However, if have:

```C++
struct CustomType2 {};
using Custom12 = DynamicType<NoContainers, CustomType, CustomType2>;
```

Then the `operator+` on `Custom12` should not be defined at compile time,
and doing `Custom12{} + Custom12{}` results in a compilation error. It is
a compilation error because we know at compile time that none of them are
defined:

- `operator+(CustomType, CustomType)`
- `operator+(CustomType, CustomType2)`
- `operator+(CustomType2, CustomType)`
- `operator+(CustomType2, CustomType2)`

So we decided to not create the `operator+` for `Custom12`.

Also, besides requiring `operator+(T1, T2)` to be defined for some `T1` and
`T2` in the type list, it is also required that the result type of
`operator+(T1, T2)` can be used to construct the dynamic type. For example,
if you have:

```C++
struct bfloat16_zero {}; struct half_zero {};
float operator+(bfloat16_zero, half_zero) { return 0.0f; }
using BFloatOrHalfZero = DynamicType<NoContainers, bfloat16_zero, half_zero>;
```

Then the `operator+` on `BFloatOrHalfZero` should not be defined, because `BFloatOrHalfZero` can not be constructed from the result of `operator+`.
However, if you have:

```C++
using BFloatOrHalfZeroOrInt = DynamicType<NoContainers, bfloat16_zero, half_zero, int>;
```

Then the `operator+` on `BFloatOrHalfZeroOrInt` should be defined at compile time
because `int+int`` is defined, but
`BFloatOrHalfZeroOrInt(half_zero{}) + BFloatOrHalfZeroOrInt(bfloat16_zero{})`
should be a runtime error because `BFloatOrHalfZeroOrInt` can not be constructed
from the result of `half_zero+bfloat16_zero`(i.e. float).

Besides the operators within `DynamicType`, such as `DynamicType + DynamicType`,
`DynamicType` also supports operators with static type. For example, if you have

```C++
IntOrFloat x = 1;
float y = 2.5f;
```

then `x + y` or `y + x` should be an `IntOrFloat` with value `3.5f`. However, if you
have

```C++
IntOrFloat x = 1;
double y = 2.5;
```

then you will get a compilation error for doing `x + y` or `y + x`, because
`int + double` and `double + int` are `double`, which can not be used to construct
`IntOrFloat`.

All the above behaviors are handled by template meta-programming, so they are
automatic. Adding a new type to the list of types does not introduce any
extra work. All the behaviors mentioned in this note are tested in
`DynamicTypeTest.ExamplesInNote`, so if you want to change anything in this
doc, please make sure to update the test as well.

`DynamicType` also supports recursive types, that is, the type list can
contain `DynamicType`. For example, something like:

```C++
// Warning: this code does not compile!
using IntFloatVecList = DynamicType<
    NoContainers,
    int,
    float,
    std::vector<IntFloatVecList>,
    std::list<IntFloatVecList>>;
```

However, the above code doesn't work because `IntFloatVecList` can not appear
in the definition of itself. That's why we have the "`Containers`" mechanism,
where we reserve the first argument of `DynamicType` for recursive types. The
`Containers` accepts class templates as its template arguments, such as
`Containers<Template1, Template2>`, then `Template1<Self>` and `Template2<Self>`
will be in the type list. With this mechanism, the correct way for writing
the above code is:

```C++
using IntFloatVecList = DynamicType<Containers<std::vector, std::list>, int, float>;
```

with the above definition, the value contained in `IntFloatVecList` can be an `int`,
a `float`, a `std::vector<IntFloatVecList>`, or a `std::list<IntFloatVecList>`. For example, we can have:

```C++
IntFloatVecList x = std::vector<IntFloatVecList>{1, 2.0f};
IntFloatVecList y = std::list<IntFloatVecList>{3, x};
```

then y will have a structure like `{3, {1, 2.0f}}`.

TODO: document the following:
- ctor `IntFloatVecList(std::vector<int>)`
- casting operators `(std::vector<int>)IntFloatVecList`
- `is`, `as`
- `[]` for subscripting
- `->*` for member access

Also, operations on `DynamicType` are as `constexpr` as possible. So most
tests in `DynamicTypeTest` are `static_assert` tests.
