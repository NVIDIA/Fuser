#include <ATen/ATen.h>
#include <cute/tensor.hpp>

using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayout>> smem;
  cute::uint64_t bulk_copy_mbar[1];
};

template <class T, class GmemLayout, class SmemLayout>
__global__ void cute_bulk_copy(
    T const* g_in,
    T* g_out,
    GmemLayout gmem_layout,
    SmemLayout smem_layout) {
  // Use Shared Storage structure to allocate and distribute aligned SMEM
  // addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smem_layout);
  // Construct the GMEM tensor
  Tensor gA = make_tensor(make_gmem_ptr(g_in), gmem_layout);

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* bulk_copy_mbar = shared_storage.bulk_copy_mbar;

  //
  // Perform the BULK_COPY load
  //

  auto blkcp = Copy_Traits<SM90_BULK_COPY_AUTO>{};

#if 0
  if (thread0()) {
    print("sA: "); print(sA.data()); print(" o "); print(sA.layout()); print("\n");
    print("gA: "); print(gA.data()); print(" o "); print(gA.layout()); print("\n");
  }
#endif

  // Set the bytes transferred in this transaction (may involve multiple issues)
  constexpr int transaction_bytes = size(sA) * sizeof(T);

  if (threadIdx.x == 0) {
    /// Initialize shared memory barrier
    bulk_copy_mbar[0] = 0;
    initialize_barrier(bulk_copy_mbar[0], 1 /*numThreads*/);
    set_barrier_transaction_bytes(bulk_copy_mbar[0], transaction_bytes);

    copy(blkcp.with(bulk_copy_mbar[0]), gA, sA);
  }
  __syncthreads();

  /// Wait on the shared memory barrier until the phase bit flips from kPhaseBit
  /// value
  constexpr int kPhaseBit = 0;
  wait_barrier(bulk_copy_mbar[0], kPhaseBit);

#if 0
  if (thread0()) {
    print(sA);
  }
#endif

  //
  // Write out trivially
  //

  Tensor gA_out = make_tensor(make_gmem_ptr(g_out), gmem_layout);

  // Output smem -> gmem
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    gA_out(i) = sA(i);
  }
}

int main() {
  auto smem_layout = make_layout(Shape<_32, _32>{}, GenRowMajor{});
  auto gmem_layout = smem_layout;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto d_in = at::randn({32, 32}, options);
  auto d_out = at::empty({32, 32}, options);

  int32_t smem_size =
      static_cast<int32_t>(sizeof(SharedStorage<float, decltype(smem_layout)>));
  std::cout << "smem_size: " << smem_size << std::endl;
  std::cout << "cosize(gmem_layout): " << cosize(gmem_layout) << std::endl;
  cute_bulk_copy<<<1, 128, smem_size>>>(
      d_in.data_ptr<float>(),
      d_out.data_ptr<float>(),
      gmem_layout,
      smem_layout);

  // Flatten the tensors for easier access
  auto d_in_flat = d_in.flatten();
  auto d_out_flat = d_out.flatten();

  // Validate the results
  for (int i = 0; i < cute::size(gmem_layout); ++i) {
    int k = gmem_layout(i);
    // TODO: Replace with ASSERT_EQ if using gtest framework
    float in_val = d_in_flat[k].item<float>();
    float out_val = d_out_flat[k].item<float>();
    if (in_val != out_val) {
      printf("d_in[%d] = %f, d_out[%d] = %f\n", k, in_val, k, out_val);
    }
  }
  return 0;
}
