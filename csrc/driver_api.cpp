#include <iostream>
#include <utility>

int f(int, int) {
  std::cout << "in f" << std::endl;
  return 0;
}

// header

namespace nvfuser {

#define DECLARE_DRIVER_API_WRAPPER(funcName) \
  extern decltype(::funcName)* funcName;

DECLARE_DRIVER_API_WRAPPER(f);

} // namespace nvfuser

// cpp

#define DEFINE_DRIVER_API_WRAPPER(funcName)        \
  template <typename Ret, typename... Args>        \
  struct FunctionWrapper {                         \
    static Ret lazilyLoadAndInvoke(Args... args) { \
      std::cout << "lazy load" << std::endl;       \
      funcName = &::funcName;                      \
      return funcName(args...);                    \
    }                                              \
                                                   \
    FunctionWrapper(Ret(Args...)){};               \
  };                                               \
                                                   \
  template <typename ReturnType, typename... Args> \
  FunctionWrapper(ReturnType(Args...))             \
      -> FunctionWrapper<ReturnType, Args...>;     \
                                                   \
  decltype(::funcName)* funcName =                 \
      decltype(FunctionWrapper(::funcName))::lazilyLoadAndInvoke

namespace nvfuser {

DEFINE_DRIVER_API_WRAPPER(f);

} // namespace nvfuser

int main() {
  nvfuser::f(0, 0);
  nvfuser::f(0, 0);
}