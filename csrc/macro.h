#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult _result = x;                                       \
    if (_result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(_result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult _result = x;                                          \
    if (_result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(_result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)
