#include <iostream>
#include <type_traits>
#include <vector>

#if __has_include(<bits/c++config.h>)
#include <bits/c++config.h>
#endif

namespace std {

#if !__cpp_lib_type_identity
template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;
#endif

} // namespace std

// Implementation detail for opcheck. This implementation is very long, I
// recommend read the usage doc below first before reading this implementation.
namespace opcheck_impl {

// A type that is purposely made implicitly convertible from OperatorChecker
struct CastableFromOperatorChecker {};

template <typename T>
struct OperatorChecker {
  constexpr operator CastableFromOperatorChecker() const {
    return {};
  }
};

template <typename T1, typename T2>
constexpr auto operator<(OperatorChecker<T1>, OperatorChecker<T2>)
    -> decltype((std::declval<T1>() < std::declval<T2>()), true) {
  return true;
}

constexpr bool operator<(
    CastableFromOperatorChecker,
    CastableFromOperatorChecker) {
  return false;
}

} // namespace opcheck_impl

// Note [Operator checker]
//
// "opcheck" is a utility to check if an operator for certain type is defined.
template <typename T>
constexpr opcheck_impl::OperatorChecker<T> opcheck;

// Take the cartesion product of two tuples.
// For example:
// cartesian_product((1, 2), (3, 4)) = ((1, 3), (1, 4), (2, 3), (2, 4))
template <typename Tuple>
constexpr auto cartesian_product(Tuple t) {
  return std::apply(
      [](auto... ts) constexpr {
        return std::make_tuple(std::make_tuple(ts)...);
      },
      t);
}

template <typename Tuple1, typename... OtherTuples>
constexpr auto cartesian_product(Tuple1 first, OtherTuples... others) {
  auto c_first = cartesian_product(first);
  auto c_others = cartesian_product(others...);
  // cat one item in c_first with all the items in c_others
  auto cat_one_first_all_others = [c_others](auto first_item) {
    return std::apply(
        [first_item](auto... other_item) constexpr {
          return std::make_tuple(std::tuple_cat(first_item, other_item)...);
        },
        c_others);
  };
  return std::apply(
      [cat_one_first_all_others](auto... first_items) constexpr {
        return std::tuple_cat(cat_one_first_all_others(first_items)...);
      },
      c_first);
}

// Check if all the booleans in the arguments are true. There are two versions:
// one for variadic arguments, and one for std::tuple.

template <typename... Ts>
constexpr bool any(Ts... bs) {
  return (bs || ...);
}

// Can I find an x from tuple1 and a y from tuple12 such that f(x, y) is
// true? f(x, y) must be defined for all x in tuple1 and y in tuple2.
template <typename... Tuples, typename Fun>
constexpr bool any_check(Fun f, Tuples... tuples) {
  auto c = cartesian_product(tuples...);
  return std::apply(
      [f](auto... candidates) constexpr {
        return any(std::apply(f, candidates)...);
      },
      c);
}

template <typename T>
struct DynamicType {
  using TypeIdentitiesAsTuple = std::tuple<
      std::type_identity<T>,
      std::type_identity<std::vector<DynamicType>>>;
  static constexpr TypeIdentitiesAsTuple type_identities_as_tuple{};
};

constexpr auto lt_defined_checker = [](auto x, auto y) constexpr {
  using X = typename decltype(x)::type;
  using Y = typename decltype(y)::type;
  return opcheck<X> < opcheck<Y>;
};

template <
    typename DT,
    typename = std::enable_if_t<any_check(
        lt_defined_checker,
        DT::type_identities_as_tuple,
        DT::type_identities_as_tuple)>>
inline constexpr bool operator<(
    const DT& x,
    const std::type_identity_t<DT>& y) {
  return true;
}

int main() {
  using DT = DynamicType<int64_t>;
  static_assert(opcheck<std::vector<DT>> < opcheck<std::vector<DT>>);
}
