// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <manager.h>
#include <parser.h>
#include <partition.h>
#include <register_interface.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

/*
 * Registers function pointers in interface.h
 */

namespace nvfuser {

namespace {
class RegisterInterface {
 public:
  RegisterInterface() {
    auto ptr = torch::jit::fuser::cuda::getFuserInterface();
    ptr->fn_compile_n = &compileCudaFusionGroup;
    ptr->fn_run_n_s = &runCudaFusionGroup;
    ptr->fn_fuse_graph = &CudaFuseGraph;
    ptr->fn_can_fuse_n = &isFusibleCudaFusionGroup;
    ptr->fn_insert_profile_inodes = &InsertProfileNodes;
    ptr->fn_profile_n = &shouldProfileNode;
    ptr->fn_skip_n = &skipNodeKind;
  }
};

static RegisterInterface register_interface_;

class RegisterNVFuserPass {
 public:
  RegisterNVFuserPass() {
    NVFuserPassManager::registerPass(true);
  }
};

static RegisterNVFuserPass register_nvfuser_pass_;

} // namespace

//! [ Note -- type guard logic in CudaFusionGuard ]
//!
//! CudaFusionGuard is used to Guard input tensor to `CudaFusionGroup` so that
//! we would not feed inputs that violates the graph defined in `GraphCache`.
//!
//! see [ Note -- 2 level cache implementation ] for definition of unique
//! computational graph.
//! see [ Note -- CudaFusionGuard implementation] for details on how guard works
//! in profiling executor
//!
//! Type guard logic is used to query whether a runtime input `tensor` compiles
//! with profiled `guard_tensor_type`. `guard_tensor_type` is the observed
//! tensor type during profiling runs.
//!
//! At this moment, we only do single profiling run, so `guard_tensor_type` has
//! static shape / stride / scalarType. *This might be a little confusing as our
//! implementation is actually more relaxed.
//!
//! Things that we check:
//!   a. identical rank & scalar type
//!   b. stride check:
//!        b.1. identical stride order
//!        b.2. identical contiguity
//!             note that contiguity here is used for tensor collapsing. So
//!             extra attention should be paid to contiguity across size-1
//!             dimensions.
//!   c. size check:
//!        c.1 broadcast check:
//!        making sure that broadcast semantics are identical. So we want to
//!        make sure a given dimension either are both size-1 for `tensor` &
//!        `guard_tensor_type`, or are both non-size-1.
//!        This is due to the fact that we specialize size-1 dimension as
//!        broadcasted dimension while translating PyTorch tensor to Fusion IR.
//!        c.1 size-0 check:
//!        we don't specialize this on codegen, but we do specialize fusion
//!        logic for size-0 on reductoins, hence the check
//!
bool complyWith(
    const at::Tensor& tensor,
    const c10::TensorTypePtr& guard_tensor_type) {
  // guard broadcast semantics, contiguity & stride order;
  TORCH_INTERNAL_ASSERT(
      guard_tensor_type && guard_tensor_type->dim().has_value());

  // check a. if num_dimension check fails or scalar type check fails
  if (*guard_tensor_type->dim() != static_cast<size_t>(tensor.ndimension()) ||
      (guard_tensor_type->scalarType().has_value() &&
       (guard_tensor_type->scalarType().value() != tensor.scalar_type())) ||
      (guard_tensor_type->device().has_value() &&
       (guard_tensor_type->device().value() != tensor.device())) ||
      (guard_tensor_type->requiresGrad().has_value() &&
       guard_tensor_type->requiresGrad().value() !=
           (tensor.requires_grad() && at::GradMode::is_enabled()))) {
    return false;
  }

  // TODO: should we get symbolic_size instead and check for size
  // consistency across tensors as well?
  const auto& sizes = guard_tensor_type->sizes();
  // see [ Note -- stirde_properties in tensor type ]
  const auto& stride_properties = guard_tensor_type->stride_properties();

  const auto& t_sizes = tensor.sizes();
  const auto& t_strides = tensor.strides();
  int inner_dim = -1;
  for (const auto j : c10::irange(*guard_tensor_type->dim())) {
    // check b. for stride check, we go along dimensions from fastest stride to
    // slowest stride
    int sorted_index = stride_properties[j]->stride_index_
        ? static_cast<int>(*stride_properties[j]->stride_index_)
        : -1;

    // only apply stride check when we have stride_properties
    if (sorted_index != -1) {
      // check b.1. stride order [current dimension has stride larger
      // than its inner dimension(s)], check only applies when both:
      //     i. already encountered an inner dimension
      //    ii. not at the fastest dimension
      if (j != 0 && inner_dim != -1) {
        // we are not looking at dim-j, but dim-sorted_index, which
        // is the j-th fastest dim;
        // Note: we ignore 0-stride dimension, since eager logic on stride
        // indices is ambiguous
        if (t_strides[sorted_index] != 0 && t_strides[inner_dim] != 0 &&
            t_strides[sorted_index] < t_strides[inner_dim]) {
          return false;
        }
      }

      // check b.2. contiguity, we only check when it's marked as
      // contiguous.
      if (stride_properties[j]->contiguous_ &&
          *stride_properties[j]->contiguous_) {
        if (j != 0) {
          // we use contiguity to collapse dimension, if size == 1, it is
          // always collapsible
          // computeStrideProps also default to contiguous when stride == 1
          if (t_sizes[sorted_index] != 1 && t_strides[sorted_index] != 1) {
            TORCH_INTERNAL_ASSERT(
                stride_properties[j - 1]->stride_index_.has_value(),
                "Counknown index is meaningless");
            // TODO: merge this check up
            if (t_strides[sorted_index] !=
                t_strides[inner_dim] * t_sizes[inner_dim]) {
              return false;
            }
          }
        } else {
          // TODO: merge this check up
          if (t_strides[sorted_index] != 1) {
            return false;
          }
        }
      }

      // update inner_dim to be current dim. Note that we try to skip update
      // when current `t_size[sorted_index] == 1`, because:
      //   1. stride comparison on a size-1 dimension is meaningless
      //      [check b.1]
      //   2. contiguity on a size-1 dimension is misleading. For collapsing,
      //      we should actually look at the next non-size-1 dimension
      //      [check b.2]
      if (inner_dim == -1 || t_sizes[sorted_index] != 1) {
        inner_dim = sorted_index;
      }
    }

    // check c.1, we go along semantic ordered dimensions
    // check broadcast / size-1:
    bool guard_bcast = sizes[j].has_value() && sizes[j].value() == 1;
    if (guard_bcast != (t_sizes[j] == 1)) {
      return false;
    }

    // check c.2, check for size-0
    bool guard_size_0 = sizes[j].has_value() && sizes[j].value() == 0;
    if (guard_size_0 != (t_sizes[j] == 0)) {
      return false;
    }
  }

  return true;
}

} // namespace nvfuser

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators size_eq_guard({
    torch::jit::Operator(
        //"prim::CudaFusionSizeEq(int[] size, int[] ref) -> bool",
        "prim::CudaFusionSizeEq(...) -> bool",
        // prim::CudaFusionGuard returns a fresh Boolean type without aliasing.
        // if we would ever return refined tensor, which would change aliasing
        // analysis, we should update aliasdb pass.
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, 2);
            torch::jit::drop(stack, 2);

            if (!torch::jit::fuser::cuda::getCudaFusionGuardMode()) {
              torch::jit::push(stack, c10::IValue(true));
              return;
            }

            TORCH_INTERNAL_ASSERT(
                inputs[1].isIntList(), "reference needs to be of int list");
            auto ref = inputs[1].toIntList();

            auto ret = true;
            if (ref.empty()) {
              ret = inputs[0].isNone();
            } else {
              if (inputs[0].isIntList()) {
                auto inp = inputs[0].toIntList();
                if (inp.size() != ref.size()) {
                  torch::jit::push(stack, c10::IValue(false));
                  return;
                }

                for (const auto i : c10::irange(inp.size())) {
                  if (((inp[i] == 1) != (ref[i] == 1))) {
                    ret = false;
                    break;
                  }
                }
              } else {
                ret = false;
              }
            }

            torch::jit::push(stack, c10::IValue(ret));
            return;
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_fusion({
    torch::jit::Operator(
        at::prim::CudaFusionGroup,
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [node](torch::jit::Stack& stack) {
            torch::jit::fuser::cuda::runFusionGroup(node, stack);
          };
        },
        torch::jit::aliasAnalysisSpecialCase()),
});

torch::jit::RegisterOperators reg_guard({
    torch::jit::Operator(
        "prim::CudaFusionGuard(...) -> bool",
        // prim::CudaFusionGuard returns a fresh Boolean type without aliasing.
        // if we would ever return refined tensor, which would change aliasing
        // analysis, we should update aliasdb pass.
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [node](torch::jit::Stack& stack) {
            // TODO: check latency here!!!!
            std::vector<torch::jit::TypePtr> types = node->tys(at::attr::types);
            const auto num_inputs = types.size();
            at::ArrayRef<c10::IValue> inputs =
                torch::jit::last(stack, num_inputs);
            torch::jit::drop(stack, num_inputs);

            if (!torch::jit::fuser::cuda::getCudaFusionGuardMode()) {
              torch::jit::push(stack, c10::IValue(true));
              return;
            }

            for (const auto i : c10::irange(num_inputs)) {
              const c10::TensorTypePtr& guard_tensor_type =
                  types[i]->cast<at::TensorType>();

              // TODO: maybe we should just push false and fallback
              TORCH_INTERNAL_ASSERT(inputs[i].isTensor());
              const at::Tensor& tensor = inputs[i].toTensor();

              if (!nvfuser::complyWith(tensor, guard_tensor_type)) {
                torch::jit::push(stack, c10::IValue(false));
                return;
              }
            }

            // TODO: check type and return the right flag
            // naively return true;
            torch::jit::push(stack, c10::IValue(true));
            return;
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// Infer dynamic axis (-1) in view_sizes given tensor_sizes
bool inferViewShape(
    c10::List<int64_t> tensor_sizes,
    c10::List<int64_t> view_sizes) {
  int64_t dynamic_index = -1;
  size_t view_size_num_elements = 1;
  for (auto idx : c10::irange(view_sizes.size())) {
    if (view_sizes[idx] == -1) {
      TORCH_INTERNAL_ASSERT(
          dynamic_index == -1, "Only one dimension can by inferred.")
      dynamic_index = (int64_t)idx;
    } else {
      TORCH_INTERNAL_ASSERT(view_sizes[idx] > 0);
      view_size_num_elements *= view_sizes[idx];
    }
  }
  const auto kNumElements = std::accumulate(
      tensor_sizes.begin(),
      tensor_sizes.end(),
      int64_t(1),
      std::multiplies<>());

  if (kNumElements % view_size_num_elements != 0) {
    return false;
  }

  if (dynamic_index != -1) {
    view_sizes[dynamic_index] = kNumElements / (int64_t)view_size_num_elements;
  }

  return true;
}

//!
//! CudaFusionViewGuard Example Graph:
//!
//! graph(%self : __torch__.BiasViewRelu,
//!       %inputs.1 : Tensor):
//!   %2 : int = prim::Constant[value=-1]() # dynamic_bvg.py:50:40
//!   %3 : int = prim::Constant[value=1]() # dynamic_bvg.py:50:25
//!   %4 : NoneType = prim::Constant()
//!   %5 : int[] = prim::Constant[value=[2, 3]]()
//!   %6 : int[] = aten::size(%inputs.1) # dynamic_bvg.py:50:25
//!   %7 : int[] = aten::slice(%6, %4, %2, %3) # dynamic_bvg.py:50:25
//!   %view_shape.1 : int[] = aten::add(%7, %5) # dynamic_bvg.py:50:25
//!   %bias : Tensor = prim::GetAttr[name="bias"](%self)
//!   %10 : int[] = aten::size(%bias)
//!   %11 : int[] = prim::BroadcastSizes(%6, %10)
//!   %12 : bool = prim::CudaFusionGuard[types=[...]](%inputs.1, %bias)
//!   %13 : int[] = prim::Constant[value=[-1, -1, -1, 6]]()
//!   %14 : int[] = prim::Constant[value=[-1, -1, -1, 2, 3]]()
//!   %15 : bool = prim::CudaFusionViewGuard(%11, %view_shape.1, %13, %14)
//!   %16 : bool[] = prim::ListConstruct(%15, %12)
//!   %17 : bool = aten::all(%16)
//!   %18 : Tensor = prim::If(%17)
//!     block0():
//!       %19 : Tensor = prim::CudaFusionGroup_0[cache_id=0](%inputs.1,
//!       %bias)
//!       -> (%19)
//!     block1():
//!       %20 : Function = prim::Constant[name="fallback_fn", fallback=1]()
//!       %21 : (...) = prim::CallFunction(%20, %inputs.1, %bias,
//!       %view_shape.1) %22 : Float(...) = prim::TupleUnpack(%21)
//!       -> (%22)
//!   return (%18)
//! with prim::CudaFusionGroup_0 = graph(%0 : Float(...),
//!       %1 : Float(...)):
//!   %2 : int[] = prim::Constant[value=[2, 3, 4, 2, 3]]()
//!   %3 : int = prim::Constant[value=1]() # dynamic_bvg.py:50:25
//!   %o.1 : Float(...) = aten::add(%0, %1, %3) # dynamic_bvg.py:51:16
//!   %5 : Float(...) = aten::view_copy(%o.1, %2)
//!   %6 : Float(...) = aten::relu(%5) # dynamic_bvg.py:53:19
//!   return (%6)
//!
torch::jit::RegisterOperators view_guard({
    torch::jit::Operator(
        "prim::CudaFusionViewGuard(...) -> bool",
        // prim::CudaFusionViewGuard returns a fresh Boolean type without
        // aliasing. if we would ever return refined tensor, which would change
        // aliasing analysis, we should update aliasdb pass.
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            // view_sizes_constraint - Constant List[Int]
            at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, 3);

            // tensor_sizes is the runtime size for the self tensor
            // tensor_sizes - dynamic size List[Int]
            TORCH_INTERNAL_ASSERT(
                inputs[0].isIntList(), "tensor_sizes needs to be Int List");
            auto tensor_sizes = inputs[0].toIntList();

            // profiled_view_sizes is the runtime view size
            // profiled_view_sizes - profile_ivalue List[Int]
            TORCH_INTERNAL_ASSERT(
                inputs[1].isIntList(),
                "profiled_view_sizes needs to be Int list");
            auto profiled_view_sizes = inputs[1].toIntList();

            // tensor_constraints is a constant List[Int]
            // used to guard tensor_sizes
            TORCH_INTERNAL_ASSERT(
                inputs[2].isIntList(),
                "tensor constraint needs to be Int List");
            auto tensor_constraints = inputs[2].toIntList();

            // Drop after gather all input arguments
            // If an argument is moved, it is destroyed when dropped from stack
            torch::jit::drop(stack, 3);

            auto status = inferViewShape(tensor_sizes, profiled_view_sizes);
            if (!status) {
              torch::jit::push(stack, c10::IValue(false));
              return;
            }

            if (!torch::jit::fuser::cuda::getCudaFusionGuardMode()) {
              torch::jit::push(stack, c10::IValue(true));
              return;
            }
            std::vector<int64_t> tensor_sizes_int_vec = tensor_sizes.vec();
            std::vector<int64_t> view_sizes_int_vec = tensor_sizes.vec();
            std::vector<int64_t> previous_constraints =
                tensor_constraints.vec();
            auto new_constraints = nvfuser::analyzeViewConstraint(
                tensor_sizes_int_vec, view_sizes_int_vec);
            bool guard_status =
                (new_constraints.conglomerateString() == previous_constraints);
            torch::jit::push(stack, c10::IValue(guard_status));
            return;
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

torch::jit::RegisterOperators ivalue_guard({
    torch::jit::Operator(
        "prim::CudaFusionIvalGuard(...) -> bool",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, 2);
            torch::jit::drop(stack, 2);
            if (!torch::jit::fuser::cuda::getCudaFusionGuardMode()) {
              torch::jit::push(stack, c10::IValue(true));
              return;
            }
            torch::jit::push(stack, inputs[0].equals(inputs[1]));
            return;
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_add_optional({
    torch::jit::Operator(
        "prim::add_optional(Tensor(a) input, Tensor? bias) -> Tensor(a)",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            c10::IValue input, bias;
            torch::jit::pop(stack, input, bias);
            if (bias.isNone()) {
              torch::jit::push(stack, std::move(input));
            } else {
              torch::jit::push(
                  stack, at::add(input.toTensor(), bias.toTensor(), 1.0));
            }
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_transpose_copy({
    torch::jit::Operator(
        "prim::transpose_copy.int(Tensor(a) self, int dim0, int dim1) -> Tensor",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [node](torch::jit::Stack& stack) {
            TORCH_CHECK(
                node->s(at::attr::name) == "CudaFusionGroup",
                "transpose_copy is only used by nvfuser to identify non-mutating ",
                "alias ops, should be restored after fusion pass!");
            c10::IValue self, dim0, dim1;
            torch::jit::pop(stack, self, dim0, dim1);
            torch::jit::push(
                stack,
                at::transpose(self.toTensor(), dim0.toInt(), dim1.toInt()));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_flatten_copy({
    torch::jit::Operator(
        "prim::flatten_copy(Tensor self, int start_dim, int end_dim) -> Tensor",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [node](torch::jit::Stack& stack) {
            TORCH_CHECK(
                node->s(at::attr::name) == "CudaFusionGroup",
                "flatten_copy is only used by nvfuser to identify non-mutating ",
                "alias ops, should be restored after fusion pass!");
            c10::IValue self, start_dim, end_dim;
            torch::jit::pop(stack, self, start_dim, end_dim);
            torch::jit::push(
                stack,
                at::native::flatten(
                    self.toTensor(), start_dim.toInt(), end_dim.toInt()));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_infer_unsqueeze_size({
    torch::jit::Operator(
        "prim::infer_unsqueeze_size(int[] a, int dim) -> int[]",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            auto dim = torch::jit::pop(stack).toInt();
            auto size = torch::jit::pop(stack).toIntVector();
            if (dim < 0) {
              dim = dim + 1 + (int64_t)size.size();
            }
            auto it = size.begin() + dim;
            size.insert(it, 1);
            torch::jit::push(stack, c10::IValue(size));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_infer_squeeze_dim_size({
    torch::jit::Operator(
        "prim::infer_squeeze_size.dim(int[] a, int dim) -> int[]",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            auto dim = torch::jit::pop(stack).toInt();
            auto size = torch::jit::pop(stack).toIntVector();
            if (dim < 0) {
              dim = dim + (int64_t)size.size();
            }
            auto it = size.begin() + dim;
            if (*it == 1) {
              size.erase(it);
            }
            torch::jit::push(stack, c10::IValue(size));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_infer_squeeze_size({
    torch::jit::Operator(
        "prim::infer_squeeze_size(int[] a) -> int[]",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            auto size = torch::jit::pop(stack).toIntVector();

            for (auto it = size.begin(); it != size.end(); it++) {
              if (*it == 1) {
                auto pre = it - 1;
                size.erase(it);
                it = pre;
              }
            }
            torch::jit::push(stack, c10::IValue(size));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_expand_as_copy({
    torch::jit::Operator(
        "prim::expand_as_copy(Tensor self, Tensor other) -> Tensor",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [node](torch::jit::Stack& stack) {
            TORCH_CHECK(
                node->s(at::attr::name) == "CudaFusionGroup",
                "expand_as_copy is only used by nvfuser to identify non-mutating ",
                "alias ops, should be restored after fusion pass!");
            c10::IValue self, other;
            torch::jit::pop(stack, self, other);
            torch::jit::push(
                stack,
                at::native::expand_as(self.toTensor(), other.toTensor()));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
torch::jit::RegisterOperators reg_infer_index_select({
    torch::jit::Operator(
        "prim::infer_index_select_size(int[] inp_size, int[] idx_size, int selected_dim) -> int[]",
        [](const torch::jit::Node* node) -> torch::jit::Operation {
          return [](torch::jit::Stack& stack) {
            auto selected_dim = torch::jit::pop(stack).toInt();
            auto idx_size = torch::jit::pop(stack).toIntVector();
            auto size = torch::jit::pop(stack).toIntVector();
            size[selected_dim] = idx_size[0];
            torch::jit::push(stack, c10::IValue(size));
          };
        },
        torch::jit::aliasAnalysisFromSchema()),
});

} // namespace
