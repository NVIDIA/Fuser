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

template <template <typename...> typename... Templates>
// Note: `Templates` is a list of templates, not a list of types.
// Just like std::vector is a template, std::vector<int> is a type.
struct Containers {
  template <typename DynamicType, typename... MemberTypes>
  using VariantType =
      std::variant<std::monostate, MemberTypes..., Templates<DynamicType>...>;

  template <typename DynamicType, typename... MemberTypes>
  using TypeIdentitiesAsTuple = std::tuple<
      std::type_identity<std::monostate>,
      std::type_identity<MemberTypes>...,
      std::type_identity<Templates<DynamicType>>...>;

  template <typename DynamicType, typename... MemberTypes>
  using ForAllTypes = dynamic_type::
      ForAllTypes<std::monostate, MemberTypes..., Templates<DynamicType>...>;

  // Check if T is one of the types in the type list MemberTypes..., or a
  // container
  template <typename T, typename DynamicType, typename... MemberTypes>
  static constexpr auto is_candidate_type = dynamic_type::
      belongs_to<T, std::monostate, MemberTypes..., Templates<DynamicType>...>;
};

using NoContainers = Containers<>;

template <typename Containers, typename... Ts>
struct DynamicType {
  using VariantType =
      typename Containers::template VariantType<DynamicType, Ts...>;
  VariantType value;

  using TypeIdentitiesAsTuple =
      typename Containers::template TypeIdentitiesAsTuple<DynamicType, Ts...>;
  static constexpr TypeIdentitiesAsTuple type_identities_as_tuple{};

  using ForAllTypes =
      typename Containers::template ForAllTypes<DynamicType, Ts...>;
  static constexpr ForAllTypes for_all_types{};
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
