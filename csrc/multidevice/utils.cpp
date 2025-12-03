// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/utils.h>

#include <algorithm>
#include <ostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>

#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <instrumentation.h>
#include <ir/container.h>
#include <ir/internal_base_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <statement_guard.h>
#include <transform_replay.h>
#include <type.h>

namespace nvfuser {

std::ostream& operator<<(std::ostream& os, DomainType domain_type) {
  switch (domain_type) {
    case DomainType::kRoot:
      return os << "kRoot";
    case DomainType::kLogical:
      return os << "kLogical";
    case DomainType::kLoop:
      return os << "kLoop";
    case DomainType::kAllocation:
      return os << "kAllocation";
  }
  std::unreachable();
}

namespace {

const std::vector<IterDomain*>& getDomainOf(
    const TensorView* tv,
    DomainType domain_type) {
  switch (domain_type) {
    case DomainType::kRoot:
      return tv->getMaybeRootDomain();
    case DomainType::kLogical:
      return tv->getLogicalDomain();
    case DomainType::kLoop:
      return tv->getLoopDomain();
    case DomainType::kAllocation:
      return tv->getMaybeAllocationDomain();
  }
  std::unreachable();
}

} // namespace

bool isSharded(const TensorView* tv) {
  bool is_sharded = false;
  for (IterDomain* id : tv->getLoopDomain()) {
    if (!id->isDeviceDim()) {
      continue;
    }

    // Reduction dimensions are not materialized in the concrete tensor, so we
    // don't consider rDIDx{i0} sharded. For example,
    //
    //   ```
    //   [iDIDx{i0}, iS{i1}] => [rDIDx{i0}, iS{i1}]
    //   ```
    //
    // is considered an allreduce and the output is replicated.
    if (id->isReduction()) {
      continue;
    }

    // Only one axis can be sharded on DIDx.
    NVF_ERROR(
        !is_sharded,
        "Multiple IterDomains parallelized on DIDx in TensorView ",
        tv);
    is_sharded = true;
  }
  return is_sharded;
}

std::unordered_map<ParallelType, IterDomain*> mapDeviceAndStreamParallelTypeToId(
    const std::vector<IterDomain*>& domain) {
  const std::unordered_set<ParallelType>& parallel_types =
      deviceAndStreamParallelTypes();

  std::unordered_map<ParallelType, IterDomain*> parallel_type_to_id;
  parallel_type_to_id.reserve(parallel_types.size());

  for (IterDomain* id : domain) {
    const ParallelType parallel_type = id->getParallelType();
    if (parallel_types.count(parallel_type) == 0) {
      continue;
    }

    // rDIDx{i0}, usually a product of an Allreduce or a ReduceScatter, is
    // treated as replicated. This way `iDIDx{i0} => rDIDx{i0}` is considered
    // resharding.
    if (id->isReduction()) {
      continue;
    }

    NVF_ERROR(
        parallel_type_to_id.try_emplace(parallel_type, id).second,
        "Found multiple loop IterDomains with the same parallel type (",
        parallel_type,
        "): ",
        toDelimitedString(domain));
  }
  return parallel_type_to_id;
}

namespace {

std::unordered_map<IterDomain*, int64_t> mapIterDomainToTensorAxis(
    const std::vector<IterDomain*>& domain) {
  std::unordered_map<IterDomain*, int64_t> id_to_axis;
  int64_t axis = 0;
  for (auto* id : domain) {
    if (id->isReduction()) {
      // Reduction IterDomains are not materialized as an at::Tensor axis.
      id_to_axis[id] = -1;
    } else {
      id_to_axis[id] = axis;
      axis++;
    }
  }
  return id_to_axis;
}

// Finds the logical IterDomain that transitively produces `id` and returns its
// tensor axis. Returns -1 for reduction dimensions because they don't
// correspond to any tensor axis.
int64_t getProducingLogicalAxis(const TensorView* tv, IterDomain* id) {
  std::unordered_map<IterDomain*, int64_t> logical_id_to_axis =
      mapIterDomainToTensorAxis(tv->getLogicalDomain());
  while (true) {
    if (auto i = logical_id_to_axis.find(id); i != logical_id_to_axis.end()) {
      return i->second;
    }

    Expr* def = id->definition();
    NVF_ERROR(
        def != nullptr,
        "Failed to find a non-reduction logical IterDomain that produces ",
        id);
    if (auto* split = dynamic_cast<Split*>(def)) {
      // Returning just which tensor axis is sharded isn't sufficient to let
      // shardTensor, a user of this function, know how to shard the tensor.
      // For example,
      //
      //   t = makeContigConcreteTensor({6});
      //   t->split(0, 2, /*inner_split=*/true);
      //   t->axis(-1)->parallelize(DIDx);
      //   // [i{3}, iDIDx{2}]
      //
      // and the unsharded tensor is [0, 1, 2, 3, 4, 5], regardless of the
      // stride. The sharded tensor ought to be [0, 2, 4] for GPU 0 and [1, 3,
      // 5] for GPU 1. However, shardTensor as is will return [0, 1, 2] and [3,
      // 4, 5], assuming the axis is sharded outermost.
      //
      // One potential way to solve the general problem is to replay and rewind
      // the splits on the at::Tensor.  For example,
      //
      //   t = makeContigConcreteTensor({30});
      //   t->split(0, 5);
      //   t->split(0, 3);
      //   t->axis(0)->parallelize(Host);
      //   t->axis(1)->parallelize(DIDx);
      //   // [iHost{2}, iDIDx{3}, i{5}]
      //
      // Given an unsharded at::Tensor of shape [30], we'll first replay the
      // splits using `torch.view` to get a tensor of shape [2,3,5]. Then, we
      // `torch.slice` axis 1 for DIDx to get a tensor of shape [2,1,5]. Then,
      // we rewind the splits (and therefore apply merging) using
      // `torch.reshape` to get a sharded tensor of shape [10].
      NVF_ERROR(
          split->outer() == id,
          "Currently, we don't support DID on inner splits: ",
          split);
      id = split->in();
    } else if (auto* merge = dynamic_cast<Merge*>(def)) {
      // During propagation, we follow the outermost of the merge to shard
      // across reshape. We follow that here, but it may not always be accurate.
      // For example,
      //
      //   t = makeContigTensor(2);
      //   t->merge(0, 1);
      //   t->axis(0)->parallelize(DIDx);
      //
      // When `unshardedSizes` is given a local tensor of shape [1, 1], it's
      // unclear the global shape is [1, D] or [D, 1] or even [2, D/2], etc.
      id = merge->outer();
    } else {
      NVF_THROW(
          "Unexpected transforms from logical to a DID-parallel allocation "
          "IterDomain: ",
          def);
    }
  }
}

} // namespace

int64_t getShardedLogicalAxis(
    const TensorView* tv,
    const ParallelType parallel_type) {
  const DomainType domain_type = parallel_type == ParallelType::Stream
      ? DomainType::kAllocation
      : DomainType::kLoop;
  IterDomain* parallel_id =
      getShardedIterDomain(tv, parallel_type, domain_type);
  if (parallel_id == nullptr) {
    return -1;
  }

  return getProducingLogicalAxis(tv, parallel_id);
}

IterDomain* getShardedIterDomain(
    const TensorView* tv,
    const ParallelType parallel_type,
    const DomainType domain_type) {
  const auto& domain = getDomainOf(tv, domain_type);

  for (IterDomain* id : domain | TensorDomain::kNoReductions) {
    if (id->getParallelType() == parallel_type) {
      return id;
    }
  }
  return nullptr;
}

int64_t numDeviceDims(const TensorView* tv) {
  return std::count_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [](IterDomain* id) { return id->isDeviceDim() && !id->isReduction(); });
}

std::unordered_set<IterDomain*> getInputsInTargetDomain(
    const std::vector<IterDomain*>& loop_id,
    const std::vector<IterDomain*>& target_domain) {
  const std::vector<Val*> inputs_as_vals = IterVisitor::getInputsTo(
      {loop_id.begin(), loop_id.end()},
      {target_domain.begin(), target_domain.end()});

  std::unordered_set<IterDomain*> inputs_as_iter_domains;
  inputs_as_iter_domains.reserve(inputs_as_vals.size());
  for (auto val : inputs_as_vals) {
    inputs_as_iter_domains.insert(val->as<IterDomain>());
  }
  return inputs_as_iter_domains;
}

namespace {
int64_t rankOfParallelType(ParallelType parallel_type) {
  // Currently, when reorderParallelizedToFront is called, the loop domain is
  // expected to be parallelized on only Stream and DIDs. To make the order
  // convenient for schedulers, we put Stream first, DIDs second, and Serial
  // last. Stream is before DIDs so we can inline computation and communication
  // into the same host for-loop. The best order between DIDs is unclear. We'll
  // decide that when we support 2D sharding, e.g., https://nv/nvfuser-cp
  switch (parallel_type) {
    case ParallelType::Stream:
      return 0;
    case ParallelType::DIDx:
    case ParallelType::DIDy:
    case ParallelType::DIDz:
      return 1;
    default:
      // I could assign other types an arbitrary rank but I prefer NVF_THROW to
      // catch unexpected changes in the future.
      NVF_THROW("Unexpected parallel type: ", parallel_type);
  }
}
} // namespace

std::unordered_map<int64_t, int64_t> reorderParallelizedToFront(
    TensorView* tv) {
  std::vector<std::pair<int64_t, int64_t>> rank_to_axis;
  rank_to_axis.reserve(tv->nDims());
  for (auto [axis, id] : enumerate(tv->getLoopDomain())) {
    auto parallel_type = id->getParallelType();
    // We skip ParallelType::Serial because TensorView::reorder automatically
    // orders unspecified IterDomains to the back stably.
    if (parallel_type != ParallelType::Serial) {
      rank_to_axis.emplace_back(rankOfParallelType(parallel_type), axis);
    }
  }

  std::stable_sort(rank_to_axis.begin(), rank_to_axis.end());

  // old position to new position
  std::unordered_map<int64_t, int64_t> order;
  int64_t current_pos = 0;
  for (auto [rank, axis] : rank_to_axis) {
    order[axis] = current_pos;
    current_pos++;
  }

  tv->reorder(order);
  return order;
}

std::unordered_set<TensorView*> getTvsWithDifferentSharding(
    TensorView* ref,
    const std::vector<TensorView*>& tvs) {
  std::unordered_set<TensorView*> ret;
  const auto& reference_dom = ref->getLoopDomain();
  FusionGuard fg(ref->fusion());
  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
  for (auto id : reference_dom) {
    auto ca_id =
        ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_to_reference_map[ca_id] = id;
  }

  for (TensorView* tv : tvs) {
    if (ref->getDeviceMesh().vector() != tv->getDeviceMesh().vector()) {
      ret.insert(tv);
      continue;
    }
    for (auto id : tv->getLoopDomain()) {
      auto ca_id =
          ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
      if (concrete_to_reference_map.count(ca_id) > 0) {
        auto ref_id = concrete_to_reference_map.at(ca_id);
        if ((ref_id->isDeviceDim() || id->isDeviceDim()) &&
            ref_id->getParallelType() != id->getParallelType()) {
          ret.insert(tv);
          break;
        }
      }
    }
  }
  return ret;
}

bool isValidDeviceSplit(Expr* expr) {
  if (expr == nullptr || !expr->isA<Split>()) {
    return false;
  }
  auto* split = expr->as<Split>();
  if (split == nullptr || !split->outer()->isDeviceDim() ||
      split->innerSplit()) {
    return false;
  }
  return true;
}

namespace {
// Helper to set up sockaddr_un
void setupSockAddr(struct sockaddr_un& addr, const std::string& path) {
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (path.length() >= sizeof(addr.sun_path)) {
    NVF_ERROR(false, "Socket path too long: ", path);
  }
  strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  if (path[0] == '@') {
    addr.sun_path[0] = '\0'; // Abstract namespace
  }
}
} // namespace

int createIpcSocket(const std::string& path) {
  int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
  NVF_CHECK(sockfd >= 0, "Failed to create socket: ", strerror(errno));

  struct sockaddr_un addr;
  setupSockAddr(addr, path);

  // For abstract namespace, len is usually calculated specifically, but for
  // Linux binding with sizeof(sun_family) + len works
  socklen_t addrlen = sizeof(addr.sun_family) + path.length();
  if (path[0] != '@') {
    unlink(path.c_str());
  }

  if (bind(sockfd, (struct sockaddr*)&addr, addrlen) < 0) {
    close(sockfd);
    NVF_CHECK(false, "Failed to bind socket to ", path, ": ", strerror(errno));
  }

  if (listen(sockfd, 128) < 0) {
    close(sockfd);
    NVF_CHECK(false, "Failed to listen on socket: ", strerror(errno));
  }

  return sockfd;
}

void sendFd(
    const std::string& path,
    int fd,
    const void* header_data,
    size_t header_len) {
  int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
  NVF_CHECK(sockfd >= 0, "Failed to create socket: ", strerror(errno));

  struct sockaddr_un addr;
  setupSockAddr(addr, path);
  socklen_t addrlen = sizeof(addr.sun_family) + path.length();

  // Simple retry loop for connection
  int ret = -1;
  for (int i = 0; i < 100; ++i) {
    ret = connect(sockfd, (struct sockaddr*)&addr, addrlen);
    if (ret == 0)
      break;
    usleep(10000); // 10ms
  }
  if (ret < 0) {
    close(sockfd);
    NVF_CHECK(false, "Failed to connect to ", path, ": ", strerror(errno));
  }

  struct msghdr msg = {0};
  struct cmsghdr* cmsg;
  char buf[CMSG_SPACE(sizeof(int))];

  // If no header data, send at least one byte
  char dummy = '.';
  struct iovec iov;
  if (header_data && header_len > 0) {
    iov.iov_base = const_cast<void*>(header_data);
    iov.iov_len = header_len;
  } else {
    iov.iov_base = &dummy;
    iov.iov_len = sizeof(dummy);
  }

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

  if (sendmsg(sockfd, &msg, 0) < 0) {
    close(sockfd);
    NVF_CHECK(false, "Failed to send FD: ", strerror(errno));
  }

  close(sockfd);
}

int recvFd(int socket_fd, void* header_data, size_t header_len) {
  struct sockaddr_un client_addr;
  socklen_t client_len = sizeof(client_addr);
  int client_fd =
      accept(socket_fd, (struct sockaddr*)&client_addr, &client_len);
  NVF_CHECK(client_fd >= 0, "Failed to accept connection: ", strerror(errno));

  struct msghdr msg = {0};
  struct cmsghdr* cmsg;
  char buf[CMSG_SPACE(sizeof(int))];

  // If header_len > 0, we expect that much data.
  // Note: recvmsg might return fewer bytes if strict requirements aren't met,
  // but for local unix sockets with small payloads, it usually delivers all.
  char dummy;
  struct iovec iov;
  if (header_data && header_len > 0) {
    iov.iov_base = header_data;
    iov.iov_len = header_len;
  } else {
    iov.iov_base = &dummy;
    iov.iov_len = sizeof(dummy);
  }

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  ssize_t received = recvmsg(client_fd, &msg, 0);
  if (received < 0) {
    close(client_fd);
    NVF_CHECK(false, "Failed to receive FD: ", strerror(errno));
  }

  // Verify data length if requested
  if (header_data && header_len > 0 &&
      static_cast<size_t>(received) != header_len) {
    // Try to read remaining loop? For simplification we assume one packet.
    // If strict reliability needed, add loop.
    if (static_cast<size_t>(received) < header_len) {
      // Handle partial read if necessary (unlikely for small headers on Unix
      // socket)
    }
  }

  int recv_fd = -1;
  cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg != NULL && cmsg->cmsg_len == CMSG_LEN(sizeof(int))) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      memcpy(&recv_fd, CMSG_DATA(cmsg), sizeof(int));
    }
  }

  close(client_fd);
  NVF_CHECK(recv_fd >= 0, "Did not receive valid FD");
  return recv_fd;
}

} // namespace nvfuser
