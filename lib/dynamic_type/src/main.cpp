#include <cstdint>
#include <type_traits>
#include <vector>

template <typename T1, typename T2>
static auto hasLessThanHelper(int)
    -> decltype(std::declval<T1>() < std::declval<T2>(), std::true_type{});

template <typename, typename>
static auto hasLessThanHelper(long) -> std::false_type;

template <typename T1, typename T2>
struct hasLessThan : decltype(hasLessThanHelper<T1, T2>(0)) {};

struct DynamicType {
  using T1 = int64_t;
  using T2 = std::vector<DynamicType>;
};

template <
    typename DT,
    typename = std::enable_if_t<
        (hasLessThan<typename DT::T1, typename DT::T1>::value ||
         hasLessThan<typename DT::T1, typename DT::T2>::value ||
         hasLessThan<typename DT::T2, typename DT::T1>::value ||
         hasLessThan<typename DT::T2, typename DT::T2>::value)>>
inline constexpr bool operator<(const DT& x, const DT& y) {
  // implementation omitted
  return true;
}

int main() {
  using DT = DynamicType;
  static_assert(hasLessThan<std::vector<DT>, std::vector<DT>>::value);
}
