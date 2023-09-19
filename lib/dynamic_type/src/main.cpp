#include <algorithm>
#include <iostream>
#include <optional>
#include <ostream>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include "dynamic_type/C++20/type_traits"
#include "dynamic_type/error.h"
#include "dynamic_type/type_traits.h"

namespace dynamic_type {

template <template <typename...> typename Templates>
// Note: `Templates` is a list of templates, not a list of types.
// Just like std::vector is a template, std::vector<int> is a type.
struct Containers {
  template <typename DynamicType, typename... MemberTypes>
  using TypeIdentitiesAsTuple = std::tuple<
      std::type_identity<MemberTypes>...,
      std::type_identity<Templates<DynamicType>>>;
};

template <typename Containers, typename... Ts>
struct DynamicType {
  using TypeIdentitiesAsTuple =
      typename Containers::template TypeIdentitiesAsTuple<DynamicType, Ts...>;
  static constexpr TypeIdentitiesAsTuple type_identities_as_tuple{};
};

constexpr auto lt_defined_checker = [](auto x, auto y) constexpr {
  using X = typename decltype(x)::type;
  using Y = typename decltype(y)::type;
  return opcheck<X> < opcheck<Y>;
};

template <
    typename DT,
    typename = std::enable_if_t<
        any_check(
            lt_defined_checker,
            DT::type_identities_as_tuple,
            DT::type_identities_as_tuple)>>
inline constexpr bool operator<(
    const DT& x,
    const std::type_identity_t<DT>& y) {
  return true;
}

} // namespace dynamic_type

using namespace dynamic_type;

int main() {
  using DT = DynamicType<Containers<std::vector>, int64_t>;
  static_assert(opcheck<std::vector<DT>> < opcheck<std::vector<DT>>);
}
