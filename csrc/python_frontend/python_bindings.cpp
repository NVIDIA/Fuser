// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/python_bindings.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_record.h>
#include <python_frontend/python_bindings.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <complex>
#include <iostream>
#include <optional>
#include <tuple>

#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace nvfuser::python_frontend {

// Set of local functions that are used to compose python FusionDefinition
// bindings. Ideally, these would be templated lambda functions but those
// are not available without C++20.
namespace {
Vector define_vector_base_fn(FusionDefinition& fd, std::vector<Scalar>& args) {
  FUSER_PERF_SCOPE("python_frontend::define_vector_base_fn");
  NVF_CHECK(!fd.completed(), "Attempting to add to a completed definition!");
  std::vector<State> inputs;
  inputs.reserve(args.size());
  for (const auto& arg : args) {
    inputs.push_back(fd.recordingState(arg()));
  }
  Vector out = fd.defineVector(inputs.size());
  fd.defineRecord(
      new VectorRecord(inputs, {fd.recordingState(out())}, DataType::Int));
  return out;
}

template <class ITERABLE>
Vector define_vector_fn(
    FusionDefinition& self,
    ITERABLE& values,
    PrimDataType dtype = DataType::Int) {
  FUSER_PERF_SCOPE("python_frontend::define_vector_fn");
  std::vector<Scalar> args;
  size_t idx = 0;
  for (const auto& item : values) {
    NVF_CHECK(
        idx < 8,
        "The specified vector size exceeds the max tensor size for nvfuser.");
    if (py::isinstance<py::int_>(item)) {
      auto int_value = py::cast<int64_t>(item);
      NVF_CHECK(
          int_value >= -1,
          "The value ",
          int_value,
          " at index ",
          idx,
          " was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1).");
      Scalar out = self.defineScalar();
      self.defineRecord(new ScalarRecord(
          {self.recordingState(out())}, py::cast<int64_t>(item), dtype));
      args.emplace_back(out);
    } else if (py::isinstance<Scalar>(item)) {
      args.emplace_back(py::cast<Scalar>(item));
    } else {
      NVF_CHECK(
          false,
          "Unsupported iterable object type for define_vector! Index:",
          idx);
    }
    ++idx;
  }
  return define_vector_base_fn(self, args);
}

template <class ShapeType>
Vector ShapeAsVector(ShapeType shape, FusionDefinition& fd) {
  static_assert(
      std::is_same_v<ShapeType, Vector> ||
      std::is_same_v<ShapeType, py::list> ||
      std::is_same_v<ShapeType, py::tuple>);
  if constexpr (std::is_same_v<ShapeType, Vector>) {
    return shape;
  } else {
    // It's important to call define_vector_fn in the if-else branch.
    //
    // ```
    // if constexpr (std::is_same_v<ShapeType, Vector>) {
    //   return shape;
    // }
    // return define_vector_fn<ShapeType>(fd, shape);
    // ```
    // would not work because the compiler would try to instantiate
    // define_vector_fn<Vector> and fail.
    return define_vector_fn<ShapeType>(fd, shape);
  }
}

template <class ShapeType>
Tensor broadcast_in_dim_fn(
    FusionDefinition::Operators& op,
    Tensor arg,
    ShapeType generic_output_shape,
    std::vector<int64_t>& broadcast_dims) {
  FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
  FusionDefinition* fd = op.fusion_definition;
  NVF_CHECK(!fd->completed(), "Attempting to add to a completed definition!");

  NVF_CHECK(op.validUse(), "Attempting to add to a completed definition!");
  Vector output_shape = ShapeAsVector(generic_output_shape, *fd);
  NVF_CHECK(
      output_shape.size >= broadcast_dims.size(),
      "broadcast_dims vector size is too big for output shape!");

  Tensor output = fd->defineTensor(output_shape.size);
  fd->defineRecord(new BroadcastInDimOpRecord(
      {fd->recordingState(arg()), fd->recordingState(output_shape())},
      {fd->recordingState(output())},
      output_shape.size,
      broadcast_dims));
  return output;
}

template <class ShapeType>
Tensor reshape_fn(
    FusionDefinition::Operators& self,
    Tensor arg,
    ShapeType generic_new_shape) {
  NVF_CHECK(self.validUse(), "Attempting to add to a completed definition!");

  FusionDefinition* fd = self.fusion_definition;
  Vector new_shape = ShapeAsVector(generic_new_shape, *fd);

  Tensor output = fd->defineTensor(new_shape.size);
  fd->defineRecord(new ReshapeOpRecord(
      {fd->recordingState(arg()), fd->recordingState(new_shape())},
      {fd->recordingState(output())},
      new_shape.size));
  return output;
}

struct DimInfo {
  int64_t index;
  int64_t size;
  int64_t stride;
  int64_t stride_order;
  std::optional<bool> contiguity = std::nullopt;

  bool isBroadcast() {
    return stride == 0 || size == 1;
  }
};

} // namespace

std::vector<std::optional<bool>> computeContiguity(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  NVF_CHECK(
      sizes.size() == strides.size(),
      "compute_contiguity: Sizes and strides must have the same number of dimensions");
  // Not a broadcast means neither the stride == 0 (size can be non-zero)
  // or the size == 1 that each can indicate a broadcast
  auto not_broadcast = [&](auto i) { return strides[i] != 0 && sizes[i] != 1; };
  // Contiguity defaults to vector of all None's
  std::vector<std::optional<bool>> contiguity(sizes.size(), std::nullopt);
  if (contiguity.empty()) { // zero-dim tensor
    return contiguity;
  }
  int64_t last = (int64_t)sizes.size() - 1; // inner most dimension
  // Contiguity normallly is determined by the current dimension and one
  // dimension to the right.  The innermost dimension, that is not broadcasted,
  // does not have any dimension to it's right and needs to be specially marked
  // contiguous.
  for (; last >= 0; --last) {
    if (not_broadcast(last)) {
      contiguity[last] = (strides.at(last) == 1);
      break;
    }
  }
  // Dimensions are marked contiguous by inspecting the current dimension and
  // one to the right towards the inner dimension while skipping over broadcast
  // dimensions.
  for (int64_t i = 0; i < last;) {
    if (not_broadcast(i)) {
      auto l = i++;
      for (; i <= last; i++) {
        if (not_broadcast(i)) {
          break;
        }
      }
      contiguity[l] = (strides[l] == strides[i] * sizes[i]);
    } else {
      i++;
    }
  }
  return contiguity;
}

// [ Note stride order and contiguity vector ]
//
// `stride order` vector corresponds to the order for each logical domain in
//     physical memory;
// `contiguity` vector to whether or not indexing could be collaped
//     corresponding to each physical domain;
//
// e.g. Given size and stride as follow:
//   sizes   = [2, 1, 3, 1, 4, 3]
//   strides = [12, 4, 4, 4, 1, 0]
// we would compute stride order as: [5, 4, 3, 2, 1, 0]. Since the original
// stride is in descending order. Note that there's more than one way to define
// a stride order when we have equal strides. In the context of index
// collapsing, how we resolve that shouldn't matter, hence we just go with
// preserving their original order. Similarly, we compute contiguity as: [True,
// None, True, None, True, None], Since the physical order is the same as the
// logical order, this one is trivial to compute.
//
// e.g. Given size and stride as follow:
//   sizes   = [2, 3, 1, 5, 4]
//   strides = [28, 4, 14, 0, 1]
// stride_order would be: [4, 2, 3, 0, 1], marking the order of strides in the
// vector. Meanwhile, contiguity would be computed on the physical domain, i.e.
// on sorted sizes & strides.
//   sorted_size    = [2, 1, 3, 4, 5]
//   sorted_strides = [28, 14, 4, 1, 0]
//   contiguity would be: [False, None, True, True, None]
//
// This function returns a pair of <contiguity, stride_order>
std::pair<std::vector<std::optional<bool>>, std::vector<int64_t>>
computeTensorDescriptor(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  NVF_CHECK(
      sizes.size() == strides.size(),
      "compute_contiguity: Sizes and strides must have the same number of dimensions");
  std::vector<DimInfo> dim_info_vec;
  for (auto i : c10::irange(sizes.size())) {
    // NOTE: not supporting negative stride yet.
    NVF_CHECK(strides[i] >= 0, "negative stride on tensor is not supported");
    dim_info_vec.emplace_back(DimInfo{(int64_t)i, sizes[i], strides[i]});
  }
  // sort by stride
  std::sort(
      dim_info_vec.begin(),
      dim_info_vec.end(),
      [](const auto& l, const auto& r) { return l.stride > r.stride; });

  // Dimensions are marked contiguous by inspecting the current dimension and
  // one to the right towards the inner dimension while skipping over broadcast
  // dimensions.
  // The innermost dimension, that is not broadcasted, does not have any
  // dimension to it's right and needs to have stride equal to 1 in order to be
  // marked contiguous.
  for (int64_t i = 0; i < (int64_t)sizes.size();) {
    dim_info_vec[i].stride_order = (int64_t)sizes.size() - 1 - i;
    if (!dim_info_vec[i].isBroadcast()) {
      auto l = i++;
      int64_t expected = 1;
      for (; i < (int64_t)sizes.size(); i++) {
        dim_info_vec[i].stride_order = (int64_t)sizes.size() - 1 - i;
        if (!dim_info_vec[i].isBroadcast()) {
          expected = dim_info_vec[i].stride * dim_info_vec[i].size;
          break;
        }
      }
      dim_info_vec[l].contiguity = (dim_info_vec[l].stride == expected);
    } else {
      i++;
    }
  }

  std::vector<int64_t> stride_order_vec(sizes.size(), -1);
  for (const auto& dim_info : dim_info_vec) {
    stride_order_vec[dim_info.index] = dim_info.stride_order;
  }
  std::vector<std::optional<bool>> contiguity_vec;
  std::transform(
      dim_info_vec.begin(),
      dim_info_vec.end(),
      std::back_inserter(contiguity_vec),
      [](const DimInfo& val) { return val.contiguity; });

  return std::make_pair(contiguity_vec, stride_order_vec);
}

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();

  //! DataTypes supported by nvFuser in the FusionDefinition
  py::enum_<PrimDataType>(nvfuser, "DataType")
      .value("Double", DataType::Double)
      .value("Float", DataType::Float)
      .value("Half", DataType::Half)
      .value("Int", DataType::Int)
      .value("Int32", DataType::Int32)
      .value("Bool", DataType::Bool)
      .value("BFloat16", DataType::BFloat16)
      .value("ComplexFloat", DataType::ComplexFloat)
      .value("ComplexDouble", DataType::ComplexDouble)
      .value("Null", DataType::Null);

  nvfuser.def("compute_contiguity", computeContiguity);
  nvfuser.def("compute_tensor_descriptor", computeTensorDescriptor);

  //! Binding the FusionCache that holds a cache of Fusions
  //! This is only bound to provide an interface to get the number of fusions
  //! that are cached.
  py::class_<FusionCache> fusion_cache(nvfuser, "FusionCache");
  fusion_cache
      .def_static(
          "get",
          &FusionCache::get,
          py::arg("max_fusions") = int(8192),
          py::return_value_policy::reference)
      .def("num_fusions", &FusionCache::numFusions)
      .def_static(
          "reset", &FusionCache::reset, py::return_value_policy::reference)
      .def(
          "serialize",
          [](FusionCache& self, std::string filename) {
            FUSER_PERF_SCOPE("FusionCache.serialize (string)");
            self.serialize(filename);
          },
          py::arg("filename"))
      .def(
          "deserialize",
          [](FusionCache& self, std::string filename) {
            FUSER_PERF_SCOPE("FusionCache.serialize (string)");
            self.deserialize(filename);
          },
          py::arg("filename"))
      .def(
          "__repr__",
          [](FusionCache& self) {
            std::stringstream ss;
            self.print(ss);
            return ss.str();
          })
      .def("stats", [](FusionCache& self) {
        std::stringstream ss;
        self.stats(ss);
        return ss.str();
      });

  //! These are the FusionDefinition supported object types that are either
  //! defined as inputs or the output of an operation.
  py::class_<Tensor> tensor_class(nvfuser, "Tensor");
  tensor_class.def("__repr__", [](Tensor& self) {
    std::stringstream ss;
    ss << "Tensor(index=" << self.index << ", ndim=" << self.dims << ")";
    return ss.str();
  });
  tensor_class.def_property_readonly(
      "ndim", [](Tensor& self) { return self.dims; });
  tensor_class.def("_get_fusion_definition", [](Tensor& self) {
    return self.fusion_definition;
  });

  py::class_<Scalar> scalar_class(nvfuser, "Scalar");
  scalar_class.def("__repr__", [](Scalar& self) {
    std::stringstream ss;
    ss << "Scalar(index=" << self.index << ")";
    return ss.str();
  });

  py::class_<Vector> vector_class(nvfuser, "Vector");
  vector_class.def("__repr__", [](Vector& self) {
    std::stringstream ss;
    ss << "Vector(index=" << self.index << ", size=" << self.size << ")";
    return ss.str();
  });
  vector_class.def_property_readonly(
      "size", [](Vector& self) { return self.size; });

  //! The FusionDefinition is a context manager in Python where the user will
  //! define the set the operations and connections between operations for
  //! nvFuser to create.
  py::class_<FusionDefinition> fusion_def(nvfuser, "_FusionDefinition");
  fusion_def
      .def(
          py::init<std::optional<size_t>, size_t>(),
          py::arg("id") = py::none(),
          py::arg("max_length") = int(1024))
      .def_readwrite("ops", &FusionDefinition::ops)
      .def_readwrite("sched", &FusionDefinition::sched)
      .def(
          "_setup_definition",
          [](FusionDefinition& self) -> FusionDefinition* {
            // Instrumentation to mark the beginning of a FusionDefinition
            inst::Trace::instance()->beginEvent("FusionDefinition Definition");
            return self.setupDefinition();
          })
      .def(
          "_finalize_definition",
          [](FusionDefinition& self) {
            self.finalizeDefinition();
            // Mark the end of a definition
            inst::Trace::instance()->endEvent(nullptr);
          })
      .def(
          "_setup_schedule",
          [](FusionDefinition& self, const py::iterable& iter) {
            // Instrumentation to mark the beginning of a schedule
            inst::Trace::instance()->beginEvent("FusionDefinition Schedule");
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            self.setupSchedule(inputs);
          })
      .def(
          "_finalize_schedule",
          [](FusionDefinition& self, const py::iterable& iter) {
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            self.finalizeSchedule(inputs);
            // Mark the end of a schedule
            inst::Trace::instance()->endEvent(nullptr);
          })
      .def(
          "__repr__",
          [](FusionDefinition& self) {
            std::stringstream ss;
            self.print(ss);
            return ss.str();
          })
      .def(
          "_execute",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool override_user_schedule,
             std::optional<int64_t> device,
             bool capture_debug_output) {
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              // Allows for a Vector of Sizes to be inputed as a list
              if (py::isinstance<py::list>(obj)) {
                for (py::handle item : obj) {
                  inputs.push_back(
                      torch::jit::toIValue(item, c10::AnyType::get()));
                }
              } else {
                inputs.push_back(
                    torch::jit::toIValue(obj, c10::AnyType::get()));
              }
            }
            std::optional<int8_t> int8_device = std::nullopt;
            if (device.has_value()) {
              NVF_CHECK(device.value() < 256, "Maximum device index is 255");
              int8_device = (int8_t)device.value();
            }
            return self.execute(
                inputs,
                override_user_schedule,
                capture_debug_output,
                int8_device);
          },
          py::arg("inputs"),
          py::arg("override_user_schedule") = false,
          py::kw_only(),
          py::arg("device") = py::none(),
          py::arg("capture_debug_output") = false,
          py::return_value_policy::reference)
      .def(
          "_debug_output",
          [](FusionDefinition& self) { return self.getDebugOutput(); },
          py::return_value_policy::reference)
      .def(
          "_fusion_ir",
          [](FusionDefinition& self) { return self.fusionIr(); },
          py::return_value_policy::reference)
      .def(
          "_last_cuda_code",
          [](FusionDefinition& self,
             bool intrinsic_code,
             bool override_user_schedule) {
            return self.lastCudaCode(intrinsic_code, override_user_schedule);
          },
          py::arg("intrinsic_code") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "_cuda_code_for",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool intrinsic_code,
             bool override_user_schedule) {
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            return self.cudaCodeFor(
                inputs, intrinsic_code, override_user_schedule);
          },
          py::arg("inputs"),
          py::arg("intrinsic_code") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "_last_scheduled_fusion_ir",
          [](FusionDefinition& self,
             bool tensor_transforms,
             bool override_user_schedule) {
            return self.lastScheduledFusionIr(
                tensor_transforms, override_user_schedule);
          },
          py::arg("tensor_transforms") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "_scheduled_fusion_ir_for",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool tensor_transforms,
             bool override_user_schedule) {
            std::vector<c10::IValue> inputs;
            for (py::handle obj : iter) {
              inputs.push_back(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            return self.scheduledFusionIrFor(
                inputs, tensor_transforms, override_user_schedule);
          },
          py::arg("inputs"),
          py::arg("tensor_transforms") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "id",
          [](FusionDefinition& self) -> std::optional<size_t> {
            return self.id();
          })
      .def(
          "add_output",
          [](FusionDefinition& self, Scalar output) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (scalar)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            self.defineRecord(new OutputRecord<Val>(
                {self.recordingState(output())}, serde::RecordType_OutputVal));
          },
          py::arg("output"))
      .def(
          "add_output",
          [](FusionDefinition& self,
             Tensor output,
             std::optional<Tensor> alias_input = std::nullopt) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            if (alias_input.has_value()) {
              self.defineRecord(new OutputRecord<TensorView>(
                  {self.recordingState(output()),
                   self.recordingState(alias_input.value()())},
                  serde::RecordType_OutputTv));
            } else {
              self.defineRecord(new OutputRecord<TensorView>(
                  {self.recordingState(output())}, serde::RecordType_OutputTv));
            }
          },
          py::arg("output"),
          py::arg("alias_input") = py::none())
      .def(
          "add_output",
          [](FusionDefinition& self,
             Tensor output,
             std::vector<int64_t> stride_order) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            NVF_CHECK(
                stride_order.empty() || output.dims == stride_order.size(),
                "stride_order needs to be either empty or the same length of Tensor `output`");
            int64_t duplicate_check = 0;
            for (const auto& v : stride_order) {
              NVF_CHECK(
                  v >= 0 && v < (int64_t)stride_order.size(),
                  "stride_order elements need to be within [0, stride_order.size())");
              duplicate_check |= 1 << v;
            }
            NVF_CHECK(
                duplicate_check == (1 << stride_order.size()) - 1,
                "duplicated elements in stride_order detected!");
            self.defineRecord(new OutputRecord<TensorView>(
                {self.recordingState(output())},
                serde::RecordType_OutputTv,
                stride_order));
          },
          py::arg("output"),
          py::arg("stride_order"))
      // This version of define_tensor is the canonical version
      // that displays the values as they are passed to the IR's
      // TensorViewBuilder.
      // Each dimension can be of value:
      // -1 : Symbolic for Dynamic usage
      //  0 : Zero-element
      //  1 : Broadcast
      // >1 : Static size
      // NOTE: A Tensor defined for dynamic shape usage should only
      // contain either symbolic(-1) or broadcast(1) defined dimensions.
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             std::vector<int64_t>& shape,
             std::vector<std::optional<bool>>& contiguity,
             PrimDataType dtype = DataType::Float,
             bool is_cpu = false) -> Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (default)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");

            for (size_t i = 0; i < shape.size(); ++i) {
              NVF_CHECK(
                  shape[i] >= -1,
                  "The value ",
                  shape[i],
                  " at index ",
                  i,
                  " was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1).");
            }

            Tensor out = self.defineTensor(shape.size());
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                shape,
                contiguity,
                dtype,
                is_cpu));

            return out;
          },
          py::arg("shape"),
          py::arg("contiguity"),
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             std::vector<int64_t>& sizes,
             std::vector<int64_t>& strides,
             PrimDataType dtype = DataType::Float,
             bool static_sizes = false,
             bool is_cpu = false) -> Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (integration)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            NVF_CHECK(
                sizes.size() == strides.size(),
                "The number of sizes does not match the number of strides.",
                sizes.size(),
                strides.size());

            // TensorViewBuilder assumes any dim with a compile time constant
            // size == 1 is a "maybe broadcast" axis, symbolic sizes are
            // identified by -1, and size == 0 is not supported.

            // Translate to TensorViewBuilder's view of the world.
            std::vector<int64_t> dim_sizes;
            dim_sizes.reserve(sizes.size());
            for (const auto i : c10::irange(sizes.size())) {
              NVF_ERROR(
                  sizes[i] >= 0,
                  "Size of ",
                  sizes[i],
                  " is not supported in nvFuser. Expected size >= 0.");
              if (static_sizes) {
                dim_sizes.push_back(sizes[i]);
              } else { // Symbolic defined tensor for dynamic shape usage
                if (sizes[i] == 1) {
                  dim_sizes.push_back(1);
                } else {
                  dim_sizes.push_back(-1);
                }
              }
            }

            Tensor out = self.defineTensor(sizes.size());
            // TODO: replace computeContiguity with computeTensorDescriptor
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                std::move(dim_sizes),
                computeContiguity(sizes, strides),
                dtype,
                is_cpu));

            return out;
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = DataType::Float,
          py::arg("static_sizes") = false,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_scalar",
          [](FusionDefinition& self,
             PrimDataType dtype = DataType::Double) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_scalar (input_specific)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(new ScalarRecord(
                {self.recordingState(out())}, std::monostate{}, dtype));
            return out;
          },
          py::arg("dtype") = DataType::Double,
          py::return_value_policy::reference);
  fusion_def.def(
      "define_scalar",
      [](FusionDefinition& self,
         PolymorphicValue::VariantType value,
         std::optional<PrimDataType> dtype) -> Scalar {
        FUSER_PERF_SCOPE("FusionDefinition.define_scalar");
        Scalar out = self.defineScalar();
        self.defineRecord(
            new ScalarRecord({self.recordingState(out())}, value, dtype));
        return out;
      },
      py::arg("value"),
      py::arg("dtype") = std::nullopt,
      py::return_value_policy::reference);
  fusion_def.def(
      "define_constant",
      [](FusionDefinition& self,
         PolymorphicValue::VariantType value,
         std::optional<PrimDataType> dtype) -> Scalar {
        FUSER_PERF_SCOPE("FusionDefinition.define_contant");
        TORCH_WARN_ONCE(
            "Deprecating define_constant functions in favor of define_scalar for constants.");
        Scalar out = self.defineScalar();
        self.defineRecord(
            new ScalarRecord({self.recordingState(out())}, value, dtype));
        return out;
      },
      py::arg("value"),
      py::arg("dtype") = std::nullopt,
      py::return_value_policy::reference);

  // This is the input version of define_vector
  fusion_def.def(
      "define_vector",
      [](FusionDefinition& self, size_t size) -> Vector {
        NVF_CHECK(
            size < 8,
            "The specified vector size exceeds the max tensor size for nvfuser.");
        std::vector<Scalar> args;
        args.reserve(size);
        for (size_t i = 0; i < size; ++i) {
          Scalar out = self.defineScalar();
          self.defineRecord(new ScalarRecord(
              {self.recordingState(out())}, std::monostate{}, DataType::Int));
          args.emplace_back(out);
        }
        return define_vector_base_fn(self, args);
      },
      py::arg("size"),
      py::return_value_policy::reference);
  // This is the constant version of define_vector when given a vector
  // of constant values.
  fusion_def.def(
      "define_vector",
      define_vector_fn<py::list>,
      py::arg("values"),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);
  fusion_def.def(
      "define_vector",
      define_vector_fn<py::tuple>,
      py::arg("values"),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);

  //! The Operators class is a nested class of FusionDefinition to allow the
  //! user to query the class for the list of operators.
  //!
  //! Example:
  //!   help(FusionDefinition.Operators)
  //!
  //! Additional operators are expected to be defined below as needed.  They
  //! may require defining a new RecordFunctor child class if they are unique.
  py::class_<FusionDefinition::Operators> nvf_ops(fusion_def, "Operators");
  nvf_ops.def(py::init<FusionDefinition*>());

  // ******************** INSERT OP BINDINGS BELOW HERE ********************
#define OP_PREFIX "Operators."
#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)                      \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self, Tensor input) -> Tensor {         \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(input.dims);                         \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(              \
            {fd->recordingState(input())},                                    \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Unary_TV,                                       \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));             \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self, Scalar input) -> Scalar {         \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new OpRecord<Val*, Val*>(                            \
            {fd->recordingState(input())},                                    \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Unary_VAL,                                      \
            static_cast<Val* (*)(Val*)>(op_name)));                           \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_UNARY_OP("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acos", acos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acosh", acosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asin", asin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asinh", asinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atan", atan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atanh", atanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("ceil", ceil)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cos", cos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cosh", cosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp", exp)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp2", exp2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("expm1", expm1)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erf", erf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfc", erfc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfinv", erfinv)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfcinv", erfcinv)
  NVFUSER_PYTHON_BINDING_UNARY_OP("floor", floor)
  NVFUSER_PYTHON_BINDING_UNARY_OP("frac", frac)
  NVFUSER_PYTHON_BINDING_UNARY_OP("lgamma", lgamma)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log", log)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log10", log10)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log1p", log1p)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log2", log2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("neg", neg)
  NVFUSER_PYTHON_BINDING_UNARY_OP("logical_not", logical_not)
  NVFUSER_PYTHON_BINDING_UNARY_OP("bitwise_not", bitwise_not)
  NVFUSER_PYTHON_BINDING_UNARY_OP("relu", relu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rand_like", rand_like)
  NVFUSER_PYTHON_BINDING_UNARY_OP("randn_like", randn_like)
  NVFUSER_PYTHON_BINDING_UNARY_OP("reciprocal", reciprocal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("round", round)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rsqrt", rsqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("set", set)
  NVFUSER_PYTHON_BINDING_UNARY_OP("segment_set", segment_set)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sign", sign)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sigmoid", sigmoid)
  NVFUSER_PYTHON_BINDING_UNARY_OP("signbit", signbit)
  NVFUSER_PYTHON_BINDING_UNARY_OP("silu", silu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sin", sin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sinh", sinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sqrt", sqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tan", tan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tanh", tanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("trunc", trunc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isfinite", isfinite)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isinf", isinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isnan", isnan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isneginf", isneginf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isposinf", isposinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isreal", isreal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("real", real)
  NVFUSER_PYTHON_BINDING_UNARY_OP("imag", imag)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP

  // overload to
  nvf_ops.def(
      "stride_order",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& stride_order) -> Tensor {
        FUSER_PERF_SCOPE("Operators.stride_order");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new DimsOpRecord<serde::RecordType_StrideOrderOp>(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            std::move(stride_order),
            "ops.stride_order"));
        return output;
      },
      py::arg("arg"),
      py::arg("stride_order"),
      py::return_value_policy::reference);

// rand_like and randn_like are normally used with a single TensorView argument,
// like a UnaryOp. However, they also take an optional pair (rng_seed,
// rng_offset) which converts them to deterministic ops. When those args are
// provided, and they must both be provided if either is, then the op behaves
// like a ternary op. We handle the UnaryOp case above and the TernaryOp case
// here.
#define NVFUSER_PYTHON_BINDING_TERNARY_RANDOM_OP(op_str, op_name)             \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor input,                                                        \
         Scalar rng_seed,                                                     \
         Scalar rng_offset) -> Tensor {                                       \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(input.dims);                         \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(              \
            {fd->recordingState(input()),                                     \
             fd->recordingState(rng_seed()),                                  \
             fd->recordingState(rng_offset())},                               \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_TV_VAL_VAL,                             \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));             \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::kw_only(),                                                          \
      py::arg("rng_seed"),                                                    \
      py::arg("rng_offset"),                                                  \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_RANDOM_OP("rand_like", rand_like)
  NVFUSER_PYTHON_BINDING_TERNARY_RANDOM_OP("randn_like", randn_like)

#undef NVFUSER_PYTHON_BINDING_UNARY_RANDOM_OP

#define NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL(op_str, op_name)               \
  tensor_class.def(                                                            \
      "__" op_str "__",                                                        \
      [](Tensor input) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        NVF_CHECK(                                                             \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(input.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(               \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Unary_TV,                                        \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));              \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      "__" op_str "__",                                                        \
      [](Scalar input) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        NVF_CHECK(                                                             \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*>(                             \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Unary_VAL,                                       \
            static_cast<Val* (*)(Val*)>(op_name)));                            \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);
  NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL("neg", neg)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL

#define NVFUSER_PYTHON_BINDING_BINARY_OP_TENSORS_ONLY(op_str, op_name)         \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_TV,                                       \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP_TENSORS_ONLY("_matmul_nn", _matmul_nn)
  NVFUSER_PYTHON_BINDING_BINARY_OP_TENSORS_ONLY("_matmul_nt", _matmul_nt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_TENSORS_ONLY("_matmul_tn", _matmul_tn)
  NVFUSER_PYTHON_BINDING_BINARY_OP_TENSORS_ONLY("_matmul_tt", _matmul_tt)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP_TENSORS_ONLY

#define NVFUSER_PYTHON_BINDING_BINARY_OP(op_str, op_name)                      \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_TV,                                       \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_TV_VAL,                                   \
            static_cast<TensorView* (*)(TensorView*, Val*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_VAL_TV,                                   \
            static_cast<TensorView* (*)(Val*, TensorView*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*>(                       \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_VAL,                                      \
            static_cast<Val* (*)(Val*, Val*)>(op_name)));                      \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP("add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP("atan2", atan2)
  NVFUSER_PYTHON_BINDING_BINARY_OP("div", div)
  NVFUSER_PYTHON_BINDING_BINARY_OP("truediv", truediv)
  NVFUSER_PYTHON_BINDING_BINARY_OP("fmod", fmod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP("nextafter", nextafter)
  NVFUSER_PYTHON_BINDING_BINARY_OP("pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP("remainder", remainder)
  NVFUSER_PYTHON_BINDING_BINARY_OP("sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP("lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP("logical_and", logical_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP("logical_or", logical_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_right_shift", bitwise_right_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("logical_right_shift", logical_right_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gcd", gcd)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP

#define NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(py_op, op_str, op_name)       \
  tensor_class.def(                                                            \
      py_op,                                                                   \
      [](Tensor arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_TV,                                       \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  tensor_class.def(                                                            \
      py_op,                                                                   \
      [](Tensor arg1, Scalar arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_TV_VAL,                                   \
            static_cast<TensorView* (*)(TensorView*, Val*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      py_op,                                                                   \
      [](Scalar arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_VAL_TV,                                   \
            static_cast<TensorView* (*)(Val*, TensorView*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      py_op,                                                                   \
      [](Scalar arg1, Scalar arg2) -> Scalar {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg2.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*>(                       \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Binary_VAL,                                      \
            static_cast<Val* (*)(Val*, Val*)>(op_name)));                      \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__add__", "add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__mul__", "mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__pow__", "pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__sub__", "sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__mod__", "mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__eq__", "eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__ge__", "ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__gt__", "gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__le__", "le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__lt__", "lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__ne__", "ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__and__", "bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__or__", "bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__xor__", "bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__lshift__", "bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__rshift__", "bitwise_right_shift", bitwise_right_shift)
  // In PyTorch, __div__ (//) and __truediv__ (/) are different.
  // When applied to integer-dtype arguments, they do as expected, returning
  // integer and float outputs, respectively. When applied to two floating-type
  // arguments, they return the floor of division for // and plain division for
  // /. When applied to mixed types, the types are promoted, so the
  // floating-point behavior is returned.
  // Our div operator matches the __truediv__ behavior, so we do not implement
  // __div__.
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__truediv__", "div", div)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL

#define NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP(op_str, op_name)          \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType_Ternary_TV_TV_VAL,                          \
                static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_TV_VAL_VAL,                             \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg2.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_VAL_TV_VAL,                             \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_VAL,                                    \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("sub_alpha", sub_alpha)
#undef NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_OP(op_str, op_name)                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_VAL,                                    \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Tensor arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>( \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType_Ternary_TV,                                 \
                static_cast<                                                  \
                    TensorView* (*)(TensorView*, TensorView*, TensorView*)>(  \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType_Ternary_TV_TV_VAL,                          \
                static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Scalar arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType_Ternary_TV_VAL_TV,                          \
                static_cast<TensorView* (*)(TensorView*, Val*, TensorView*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Tensor arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg2.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType_Ternary_VAL_TV_TV,                          \
                static_cast<TensorView* (*)(Val*, TensorView*, TensorView*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Scalar arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg3.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, Val*, Val*, TensorView*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_VAL_VAL_TV,                             \
            static_cast<TensorView* (*)(Val*, Val*, TensorView*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_TV_VAL_VAL,                             \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg2.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_Ternary_VAL_TV_VAL,                             \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_OP("lerp", lerp)
  NVFUSER_PYTHON_BINDING_TERNARY_OP("where", where)
#undef NVFUSER_PYTHON_BINDING_TERNARY_OP

#define NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP(op_str, op_name)              \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            !self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Ternary_VAL,                                     \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            !self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Ternary_TV_VAL_VAL,                              \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("clamp", clamp)
  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("threshold", threshold)
#undef NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP(op_str, op_name)          \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*, Val*>(           \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Ternary_Alpha_VAL,                               \
            static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<                                         \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         Val*>(                                                \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType_Ternary_Alpha_TV,                                \
            static_cast<                                                       \
                TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>( \
                op_name)));                                                    \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType_Ternary_Alpha_TV_TV_VAL,                     \
                static_cast<                                                   \
                    TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType_Ternary_Alpha_TV_VAL_TV,                     \
                static_cast<                                                   \
                    TensorView* (*)(TensorView*, Val*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType_Ternary_Alpha_VAL_TV_TV,                     \
                static_cast<                                                   \
                    TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg3.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, Val*, TensorView*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType_Ternary_Alpha_VAL_VAL_TV,                    \
                static_cast<TensorView* (*)(Val*, Val*, TensorView*, Val*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType_Ternary_Alpha_TV_VAL_VAL,                    \
                static_cast<TensorView* (*)(TensorView*, Val*, Val*, Val*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType_Ternary_Alpha_VAL_TV_VAL,                    \
                static_cast<TensorView* (*)(Val*, TensorView*, Val*, Val*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP("addcmul", addcmul)
#undef NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_REDUCTION_OP(op_str, op_name, record_type)     \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        size_t ndims = 0;                                                     \
        std::vector<int> axes(arg.dims);                                      \
        std::iota(axes.begin(), axes.end(), 0);                               \
        Tensor output = fd->defineTensor(ndims);                              \
        fd->defineRecord(new ReductionOpRecord(                               \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            record_type,                                                      \
            static_cast<TensorView* (*)(TensorView*,                          \
                                        const std::vector<int>&,              \
                                        bool,                                 \
                                        DataType)>(op_name),                  \
            axes,                                                             \
            false,                                                            \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dtype") = DataType::Null,                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         int axis,                                                            \
         bool keepdim,                                                        \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        size_t ndims = keepdim ? arg.dims : (arg.dims - 1);                   \
        Tensor output = fd->defineTensor(ndims);                              \
        fd->defineRecord(new ReductionOpRecord(                               \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            record_type,                                                      \
            static_cast<TensorView* (*)(TensorView*,                          \
                                        const std::vector<int>&,              \
                                        bool,                                 \
                                        DataType)>(op_name),                  \
            {axis},                                                           \
            keepdim,                                                          \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("axis"),                                                        \
      py::arg("keepdim") = false,                                             \
      py::arg("dtype") = DataType::Null,                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         const std::vector<int>& axes,                                        \
         bool keepdim,                                                        \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());         \
        Tensor output = fd->defineTensor(ndims);                              \
        fd->defineRecord(new ReductionOpRecord(                               \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            record_type,                                                      \
            static_cast<TensorView* (*)(TensorView*,                          \
                                        const std::vector<int>&,              \
                                        bool,                                 \
                                        DataType)>(op_name),                  \
            axes,                                                             \
            keepdim,                                                          \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("axes"),                                                        \
      py::arg("keepdim") = false,                                             \
      py::arg("dtype") = DataType::Null,                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "max", max, serde::RecordType::RecordType_ReductionMax)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "min", min, serde::RecordType::RecordType_ReductionMin)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "prod", prod, serde::RecordType::RecordType_ReductionProd)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "sum", sum, serde::RecordType::RecordType_ReductionSum)
#undef NVFUSER_PYTHON_BINDING_REDUCTION_OP

#define NVFUSER_PYTHON_BINDING_CAST_OP(op_str, op_name)                       \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg.dims);                           \
        fd->defineRecord(new CastOpRecord<TensorView*, TensorView*>(          \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_CastTv,                                         \
            static_cast<TensorView* (*)(DataType, TensorView*)>(op_name),     \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dtype"),                                                       \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg,                                                          \
         PrimDataType dtype) -> Scalar {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new CastOpRecord<Val*, Val*>(                        \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType_CastVal,                                        \
            static_cast<Val* (*)(DataType, Val*)>(op_name),                   \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dtype"),                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_CAST_OP("cast", castOp)
#undef NVFUSER_PYTHON_BINDING_CAST_OP

  nvf_ops.def(
      "batch_norm",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::optional<Tensor> weight,
         std::optional<Tensor> bias,
         std::optional<Tensor> running_mean,
         std::optional<Tensor> running_var,
         Scalar momentum,
         Scalar eps,
         bool training,
         bool channels_last) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.batch_norm");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        Tensor mean = fd->defineTensor(1);
        Tensor invstd = fd->defineTensor(1);
        auto weight_state = weight.has_value()
            ? fd->recordingState(weight.value()())
            : State(0, serde::StateType::StateType_None);
        auto bias_state = bias.has_value()
            ? fd->recordingState(bias.value()())
            : State(0, serde::StateType::StateType_None);
        auto running_mean_state = running_mean.has_value()
            ? fd->recordingState(running_mean.value()())
            : State(0, serde::StateType::StateType_None);
        auto running_var_state = running_var.has_value()
            ? fd->recordingState(running_var.value()())
            : State(0, serde::StateType::StateType_None);
        fd->defineRecord(new BatchNormOpRecord(
            {fd->recordingState(arg()),
             weight_state,
             bias_state,
             running_mean_state,
             running_var_state,
             fd->recordingState(momentum()),
             fd->recordingState(eps())},
            {fd->recordingState(output()),
             fd->recordingState(mean()),
             fd->recordingState(invstd())},
            training,
            channels_last));
        return std::make_tuple(output, mean, invstd);
      },
      py::arg("arg"),
      py::arg("weight").none(true),
      py::arg("bias").none(true),
      py::arg("running_mean").none(true),
      py::arg("running_var").none(true),
      py::arg("momentum"),
      py::arg("eps"),
      py::arg("training"),
      py::arg("channels_last") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<Vector>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<py::list>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  // NOTE: Tuple support was added to facilitate the direct usage of Pytorch's
  // Tensor.size() function that returns a child class of a Tuple.
  nvf_ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<py::tuple>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<bool>& is_broadcast_dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new BroadcastOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast",
            std::move(is_broadcast_dim)));
        return output;
      },
      py::arg("arg"),
      py::arg("is_broadcast_dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "cat",
      [](FusionDefinition::Operators& self,
         std::vector<Tensor> tensors,
         int64_t dim) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        NVF_CHECK(
            !tensors.empty(), "Attempting to concatenate empty list of tensors")
        Tensor output = fd->defineTensor(tensors[0].dims);
        std::vector<State> tensor_states;
        tensor_states.reserve(tensors.size());
        for (auto& t : tensors) {
          tensor_states.push_back(fd->recordingState(t()));
        }
        self.fusion_definition->defineRecord(new CatOpRecord(
            tensor_states, {fd->recordingState(output())}, dim));
        return output;
      },
      py::arg("tensors"),
      py::arg("dim") = 0,
      py::return_value_policy::reference);
  nvf_ops.def(
      "index_select",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.index_select");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new IndexSelectOpRecord(
            {
                fd->recordingState(arg()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "gather",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.gather");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg1.dims == index.dims,
            "Tensor arguments have different dimensions ",
            arg1.dims,
            " and ",
            index.dims);
        auto num_dims = (int64_t)arg1.dims;
        NVF_CHECK(
            dim >= -num_dims && dim < num_dims,
            "Tensor arguments have dimension ",
            num_dims,
            " so dim argument must satisfy ",
            -num_dims,
            " <= dim < ",
            num_dims,
            ", but received ",
            dim);
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg1.dims);
        fd->defineRecord(new TorchGatherOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      R"pbdoc(
        Index arg1 in dim at positions given by index.

        The dimension of arg1 and index must match. For all axes other than dim
        the extent of index in that axis need not be equal to its counterpart
        in arg1 but must not be greater than it.

        Args:
            arg1 (Tensor): Tensor of shape `(Ni...,M,Nk...)` where `M` is the
                extent of `arg1` in the dimension `dim`.
            index (Tensor): Tensor of dtype `DataType::Int` of shape
                `(Mi...,J,Mk...)` where all the extents other than `J` are less
                than or equal to their counterparts in `arg1`; for example `Mk
                <= Nk`.
            dim (int): Which position to index along.

        Returns:
            (Tensor): Tensor of same dtype as `arg1` and of shape
                `(Mi...,J,Mk...)` where the element at position `(i...,j,k...)`
                is equal to `arg1[i,...,index[i,...,j,k,...],k,...]`.
      )pbdoc",
      py::arg("arg1"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "pad",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& pad_widths,
         std::optional<Scalar> value) -> Tensor {
        FUSER_PERF_SCOPE("Operators.pad");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            pad_widths.size() <= 2 * arg.dims,
            "Number of pad widths must be at most twice the input dimension");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        auto value_state = value.has_value()
            ? fd->recordingState(value.value()())
            : State(0, serde::StateType_None);
        fd->defineRecord(new PadOpRecord(
            {fd->recordingState(arg()), value_state},
            {fd->recordingState(output())},
            std::move(pad_widths)));
        return output;
      },
      py::arg("arg"),
      py::arg("pad_widths"),
      py::arg("value") = py::none(),
      py::return_value_policy::reference);
  nvf_ops.def(
      "take_along_axis",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.take_along_axis");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg1.dims == index.dims,
            "Tensor arguments have different dimensions ",
            arg1.dims,
            " and ",
            index.dims);
        auto num_dims = (int64_t)arg1.dims;
        NVF_CHECK(
            dim >= -num_dims && dim < num_dims,
            "Tensor arguments have dimension ",
            num_dims,
            " so dim argument must satisfy ",
            -num_dims,
            " <= dim < ",
            num_dims,
            ", but received ",
            dim);
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg1.dims);
        fd->defineRecord(new TakeAlongAxisOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      R"pbdoc(
        Index arg1 in dim at positions given by index.

        This operation is very similar to :meth:'gather' but enforces that all
        dimensions other than dim must be equal between arg1 and index.

        Args:
            arg1 (Tensor): Tensor of shape `(Ni...,M,Nk...)` where `M` is the
                extent of `arg1` in the dimension `dim`.
            index (Tensor): Tensor of dtype `DataType::Int` of shape
                `(Ni...,J,Nk...)`.
            dim (int): Which position to index along.

        Returns:
            (Tensor): Tensor of same dtype as `arg1` and of shape
                `(Ni...,J,Nk...)` where the element at position `(i...,j,k...)`
                is equal to `arg1[i,...,index[i,...,j,k,...],k,...]`.
      )pbdoc",
      py::arg("arg1"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "permute",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& dims) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        self.fusion_definition->defineRecord(
            new DimsOpRecord<serde::RecordType_PermuteOp>(
                {fd->recordingState(arg())},
                {fd->recordingState(output())},
                std::move(dims),
                "ops.permute"));
        return output;
      },
      py::arg("arg"),
      py::arg("dims"),
      py::return_value_policy::reference);

  auto shape_def = [](Tensor arg) -> Vector {
    FUSER_PERF_SCOPE("Operators.shape");
    auto fd = arg.fusion_definition;
    NVF_CHECK(
        fd->ops.validUse(), "Attempting to add to a completed definition!");
    Vector output = fd->defineVector(arg.dims);
    fd->defineRecord(new ShapeOpRecord(
        {fd->recordingState(arg())}, {fd->recordingState(output())}));
    return output;
  };

  tensor_class.def(
      "shape",
      [&shape_def](Tensor arg) -> Vector { return shape_def(arg); },
      py::return_value_policy::reference);
  nvf_ops.def(
      "shape",
      [&shape_def](FusionDefinition::Operators& self, Tensor arg) -> Vector {
        return shape_def(arg);
      },
      py::arg("arg"),
      py::return_value_policy::reference);

  auto size_def = [](Tensor arg, int64_t dim) -> Scalar {
    FUSER_PERF_SCOPE("Operators.size");
    auto fd = arg.fusion_definition;
    NVF_CHECK(
        fd->ops.validUse(), "Attempting to add to a completed definition!");
    Scalar output = fd->defineScalar();
    fd->defineRecord(new SizeOpRecord(
        {fd->recordingState(arg())}, {fd->recordingState(output())}, dim));
    return output;
  };

  tensor_class.def(
      "size",
      [&size_def](Tensor arg, int64_t dim) -> Scalar {
        return size_def(arg, dim);
      },
      py::return_value_policy::reference);
  nvf_ops.def(
      "size",
      [&size_def](FusionDefinition::Operators& self, Tensor arg, int64_t dim)
          -> Scalar { return size_def(arg, dim); },
      py::arg("arg"),
      py::arg("dim"),
      py::return_value_policy::reference);

  auto at_def = [](Vector arg, int64_t index) -> Scalar {
    FUSER_PERF_SCOPE("Operators.at");
    auto fd = arg.fusion_definition;
    NVF_CHECK(
        fd->ops.validUse(), "Attempting to add to a completed definition!");
    Scalar output = fd->defineScalar();
    fd->defineRecord(new AtOpRecord(
        {fd->recordingState(arg())}, {fd->recordingState(output())}, index));
    return output;
  };

  vector_class.def(
      "at",
      [&at_def](Vector arg, int64_t index) -> Scalar {
        return at_def(arg, index);
      },
      py::return_value_policy::reference);
  vector_class.def(
      "__getitem__",
      [&at_def](Vector arg, int64_t index) -> Scalar {
        return at_def(arg, index);
      },
      py::return_value_policy::reference);
  nvf_ops.def(
      "at",
      [&at_def](FusionDefinition::Operators& self, Vector arg, int64_t index)
          -> Scalar { return at_def(arg, index); },
      py::arg("arg"),
      py::arg("index"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "slice",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& start_indices,
         std::vector<int64_t>& end_indices,
         // NOTE: Tried to use std::reference_wrapper to a vector and during
         // testing, I was not getting the proper value back.  It was like
         // like the code was referencing the strides vector that holds the
         // default value.
         std::optional<std::vector<int64_t>> opt_strides =
             std::nullopt) -> Tensor {
        FUSER_PERF_SCOPE("Operators.slice");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");

        std::vector<int64_t> strides(start_indices.size(), int64_t(1));
        if (opt_strides.has_value()) {
          NVF_CHECK(
              start_indices.size() == opt_strides.value().size(),
              "Slice start_indices and strides don't match! Start Indices: ",
              start_indices.size(),
              " Strides: ",
              opt_strides.value().size());
          strides.assign(
              opt_strides.value().begin(), opt_strides.value().end());
        }
        NVF_CHECK(
            arg.dims == start_indices.size(),
            "Number of tensor dimensions does not match slice dimensions! Tensor-dims: ",
            arg.dims,
            " Slice-dims: ",
            start_indices.size());
        NVF_CHECK(
            start_indices.size() == end_indices.size(),
            "Slice indexing attribute dimensions don't match! Start Indices: ",
            start_indices.size(),
            " End Indices: ",
            end_indices.size(),
            " Strides: ",
            strides.size());
        for (const auto i : c10::irange(arg.dims)) {
          auto start_idx = start_indices[i];
          auto end_idx = end_indices[i];
          auto stride = strides[i];
          NVF_CHECK(
              start_idx >= 0,
              "Slice operation start_indices must be greater-than-or-equal-to 0. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
          NVF_CHECK(
              end_idx >= start_idx,
              "Slice operation end_indices must be greater-than-or-equal-to start_indices. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
          NVF_CHECK(
              stride == 1,
              "nvFuser Limitation: All slice operation strides must be of size 1. Start Indices: ",
              start_indices,
              " End Indices: ",
              end_indices,
              " Strides: ",
              strides);
        }
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new SliceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            start_indices,
            end_indices,
            strides));
        return output;
      },
      py::arg("arg"),
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides") = py::none(),
      py::return_value_policy::reference);
  nvf_ops.def(
      "squeeze",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& original_shape,
         std::vector<int64_t>& dims) -> Tensor {
        FUSER_PERF_SCOPE("Operators.squeeze");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims - 1);
        fd->defineRecord(new SqueezeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            std::move(original_shape),
            std::move(dims)));
        return output;
      },
      py::arg("arg"),
      py::arg("original_shape"),
      py::arg("dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "tensor_sizes",
      [](FusionDefinition::Operators& self, Tensor arg) -> std::vector<Scalar> {
        FUSER_PERF_SCOPE("Operators.tensor_sizes");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        std::vector<Scalar> outputs;
        std::vector<State> output_state;
        for (const auto idx : c10::irange(arg.dims)) {
          outputs.push_back(fd->defineScalar());
          output_state.push_back(fd->recordingState(outputs[idx]()));
        }
        fd->defineRecord(
            new TensorSizesRecord({fd->recordingState(arg())}, output_state));
        return outputs;
      },
      py::arg("arg"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      reshape_fn<Vector>,
      py::arg("arg"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      reshape_fn<py::list>,
      py::arg("arg"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      reshape_fn<py::tuple>,
      py::arg("arg"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "full",
      [](FusionDefinition::Operators& self,
         std::vector<int64_t>& shape,
         Scalar fill_value,
         PrimDataType dtype) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(shape.size());
        fd->defineRecord(new FullOpRecord(
            {fd->recordingState(fill_value())},
            {fd->recordingState(output())},
            std::move(shape),
            dtype));
        return output;
      },
      py::arg("shape"),
      py::arg("fill_value"),
      py::arg("dtype"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "iota",
      [](FusionDefinition::Operators& self,
         Scalar length,
         std::optional<Scalar> start,
         std::optional<Scalar> step,
         PrimDataType dtype) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(1);
        auto start_state = start.has_value()
            ? fd->recordingState(start.value()())
            : State(0, serde::StateType_None);
        auto step_state = step.has_value() ? fd->recordingState(step.value()())
                                           : State(0, serde::StateType_None);
        fd->defineRecord(new IotaOpRecord(
            {fd->recordingState(length()), start_state, step_state},
            {fd->recordingState(output())},
            dtype));
        return output;
      },
      py::arg("length"),
      py::arg("start").none(true),
      py::arg("step").none(true),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.var");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());
        Tensor output = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            std::move(axes),
            correction,
            keepdim));
        return output;
      },
      py::arg("arg"),
      py::arg("axes"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var_mean",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int>& axes,
         int64_t correction,
         bool keepdim) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.var_mean");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - axes.size());
        Tensor var = fd->defineTensor(ndims);
        Tensor mean = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceMeanOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(var()), fd->recordingState(mean())},
            std::move(axes),
            correction,
            keepdim));
        return std::make_tuple(var, mean);
      },
      py::arg("arg"),
      py::arg("axes"),
      py::arg("correction") = 1,
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "uniform",
      [](FusionDefinition::Operators& self,
         Scalar minval,
         Scalar maxval,
         std::vector<Scalar>& shape,
         PrimDataType dtype,
         std::optional<Scalar> rng_seed,
         std::optional<Scalar> rng_offset) -> Tensor {
        FUSER_PERF_SCOPE("Operators.uniform");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(shape.size());
        std::vector<State> output_shape_states(
            shape.size(), State(0, serde::StateType_Scalar));
        std::transform(
            shape.begin(),
            shape.end(),
            output_shape_states.begin(),
            [&fd](const Scalar& s) { return fd->recordingState(s()); });
        std::vector<State> arg_states = {
            fd->recordingState(minval()),
            fd->recordingState(maxval()),
        };
        if (rng_seed.has_value()) {
          NVF_CHECK(
              rng_offset.has_value(),
              "When providing rng_seed, rng_offset must also be provided");
          arg_states.push_back(fd->recordingState(rng_seed.value()()));
          arg_states.push_back(fd->recordingState(rng_offset.value()()));
        }
        fd->defineRecord(new RandomOpRecord(
            arg_states,
            {fd->recordingState(output())},
            output_shape_states,
            "ops.uniform",
            dtype));
        return output;
      },
      py::arg("minval"),
      py::arg("maxval"),
      py::arg("shape"),
      py::arg("dtype") = DataType::Float,
      py::kw_only(),
      py::arg("rng_seed") = py::none(),
      py::arg("rng_offset") = py::none(),
      py::return_value_policy::reference);
  nvf_ops.def(
      "normal",
      [](FusionDefinition::Operators& self,
         Scalar mean,
         Scalar std,
         std::vector<Scalar>& shape,
         PrimDataType dtype,
         std::optional<Scalar> rng_seed,
         std::optional<Scalar> rng_offset) -> Tensor {
        FUSER_PERF_SCOPE("Operators.normal");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(shape.size());
        std::vector<State> output_shape_states(
            shape.size(), State(0, serde::StateType_Scalar));
        std::transform(
            shape.begin(),
            shape.end(),
            output_shape_states.begin(),
            [&fd](const Scalar& s) { return fd->recordingState(s()); });
        std::vector<State> arg_states = {
            fd->recordingState(mean()),
            fd->recordingState(std()),
        };
        if (rng_seed.has_value()) {
          NVF_CHECK(
              rng_offset.has_value(),
              "When providing rng_seed, rng_offset must also be provided");
          arg_states.push_back(fd->recordingState(rng_seed.value()()));
          arg_states.push_back(fd->recordingState(rng_offset.value()()));
        }
        fd->defineRecord(new RandomOpRecord(
            arg_states,
            {fd->recordingState(output())},
            output_shape_states,
            "ops.normal",
            dtype));
        return output;
      },
      py::arg("mean"),
      py::arg("std"),
      py::arg("shape"),
      py::arg("dtype") = DataType::Float,
      py::kw_only(),
      py::arg("rng_seed") = py::none(),
      py::arg("rng_offset") = py::none(),
      py::return_value_policy::reference);
  //! The ScedOperators class is a nested class of FusionDefinition to allow the
  //! user to query the class for the list of schedule operators.
  //!
  //! Example:
  //!   help(FusionDefinition.SchedOperators)
  //!
  //! Additional operators are expected to be defined below as needed.
  py::class_<FusionDefinition::SchedOperators> nvf_sched(
      fusion_def, "SchedOperators");
  nvf_sched.def(py::init<FusionDefinition*>());
  nvf_sched.def(
      "merge",
      [](FusionDefinition::SchedOperators& self, Tensor arg, int dim) {
        FUSER_PERF_SCOPE("SchedOperators.merge");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->merge(dim);
      },
      py::arg("arg"),
      py::arg("dim"));
  auto reduction_factor_func = [](FusionDefinition::SchedOperators& self,
                                  Tensor arg,
                                  const std::vector<int>& dims) -> Tensor {
    FUSER_PERF_SCOPE("SchedOperators.reduction_factor");
    NVF_CHECK(
        self.validUse(),
        "Attempting to use a SchedOperators Op prior to definition!");
    FusionDefinition* fd = self.fusion_definition;
    auto input_tv = fd->getFusionState(arg.index)->template as<TensorView>();
    auto output_tv = input_tv->rFactor(dims);
    Tensor output = fd->defineTensor(arg.dims);
    NVF_CHECK(
        output.index == fd->numFusionStates(),
        "Fusion State index does not match the size!");
    fd->addFusionState(output_tv);
    return output;
  };
  nvf_sched.def(
      "reduction_factor",
      reduction_factor_func,
      py::arg("arg"),
      py::arg("dims"));
  nvf_sched.def(
      "rfactor", reduction_factor_func, py::arg("arg"), py::arg("dims"));
  nvf_sched.def(
      "reorder",
      [](FusionDefinition::SchedOperators& self,
         Tensor arg,
         const std::unordered_map<int, int>& old2new) {
        FUSER_PERF_SCOPE("SchedOperators.reorder");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->reorder(old2new);
      },
      py::arg("arg"),
      py::arg("old2new"));
  nvf_sched.def(
      "split",
      [](FusionDefinition::SchedOperators& self,
         Tensor arg,
         int dim,
         unsigned int factor,
         bool inner_split,
         bool trim_out_of_bounds) {
        FUSER_PERF_SCOPE("SchedOperators.split");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->split(dim, factor, inner_split, trim_out_of_bounds);
      },
      py::arg("arg"),
      py::arg("dim"),
      py::arg("factor"),
      py::arg("inner_split") = true,
      py::arg("trim_out_of_bounds") = false);
}

} // namespace nvfuser::python_frontend
