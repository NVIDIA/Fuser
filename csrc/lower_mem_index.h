#pragma once
#include <c10/macros/Export.h>

#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

struct AddressRecordKey;

//! [Note on data tensor and reference tensor]:
//!
//! In the indexing compute passes we have today we tend to have two tensors
//!  involved when generating the memory or register indices. In the analysis
//!  defined here they are referred to as "data tensor" and "index reference
//!  tensor", or "reference tensor".
//!
//! The "data tensor" is the tensor that represents the actual memory space
//! where
//!  we are generating the address math for while "reference tensor" refers to
//!  the tensorview whose tensor domain maps to the actual loop nest and index
//!  on the idgraph.
//!
//! The only two examples we have today are:
//!  when indexing producer tensors, the data tensor is the producer tv,
//!                                  the reference tensor is the consumer tv.
//!
//!  when indexing consumer tensors, the data tensor is the consumer tv,
//!                                  the reference tensor is the consumer tv as
//!                                  well.
//! These info are just here to keep the interface unified when defining the
//! index lifting logic
//!  in the following analyses and transforms.

//! A utility class to keep track of memory address pre-computation.
//!  See also [Notes on memory address lifting] in lower_mem_index.cpp.
//! Each address record corresponds to an allocated "address_tv" that holds
//!  the precomputed address for some "data_tv" which is the shared/global
//!  tv the "address_tv" holds the index for.
class AddressRecord {
 public:
  //! Utility class to note the read or write
  //!  direction of this address.
  enum class ReadWrite { READ, WRITE, PREDICATE };

  explicit AddressRecord(
      TensorView* data_tv,
      TensorView* address_tv,
      std::vector<IterDomain*> allocation_ids,
      TensorView* reference_tv,
      ReadWrite direction,
      IterDomain* serial_id);

  bool isRead() const {
    return access_direction_ == ReadWrite::READ;
  }

  bool isWrite() const {
    return access_direction_ == ReadWrite::WRITE;
  }

  bool isPredicate() const {
    return access_direction_ == ReadWrite::PREDICATE;
  }

  TensorView* dataTensor() const {
    return data_tv_;
  }

  TensorView* addressTensor() const {
    return address_tv_;
  }

  TensorView* indexReferenceTensor() const {
    return reference_tv_;
  }

  const auto& allocationIterDomains() const {
    return allocation_ids_;
  }

  AddressRecordKey key() const;

  IterDomain* getConcreteSerialLoopId() const {
    return loop_concrete_serial_id_;
  }

  //! Returns the serial loop that this address record requests
  //!  to lift the index math out of.
  c10::optional<kir::ForLoop*> getMaybeSerialLoop(
      std::vector<kir::ForLoop*> loops);

  //! (Predicate record only)
  //! Fill the contig id that this record is lifting the base
  //!  index for.
  //! TODO: supporting only one contig id for now,
  //!  need to re-enable supporting an array of contig ids which
  //!  requires some significant infrastructure heavy lifting.
  void setPredicateContigId(IterDomain* contig_id) {
    TORCH_INTERNAL_ASSERT(access_direction_ == ReadWrite::PREDICATE);
    TORCH_INTERNAL_ASSERT(
        predicate_contig_id_ == nullptr, "need multiple id support");
    predicate_contig_id_ = contig_id;
  }

  //! (Predicate record only)
  //! Returns the contig id that this record computes the base index for.
  IterDomain* getPredicateContigId() const {
    TORCH_INTERNAL_ASSERT(access_direction_ == ReadWrite::PREDICATE);
    return predicate_contig_id_;
  }

 private:
  //! The address tensor that will hold the
  //!  data address to the access_tv.
  TensorView* address_tv_;

  //! The tensorview that this address record
  //!  will save address for.
  TensorView* data_tv_;

  //! The tensorview that will be the consumer
  //!  if this is a record for the read access,
  //!  and would be ignored for write access since
  //!  the access tv would have all the info.
  TensorView* reference_tv_;

  //! Records if this is a read adddress or a write
  //!  address.
  ReadWrite access_direction_ = ReadWrite::WRITE;

  //! Loop id's that correspond to the allocation iterdomain
  //!  of this
  std::vector<IterDomain*> allocation_ids_;

  //! Loop id that this address record will be lifted
  //!  out of.
  IterDomain* loop_concrete_serial_id_;

  //! (predicate record only)
  IterDomain* predicate_contig_id_ = nullptr;
};

// Utility class to index address record
struct AddressRecordKey {
  const TensorView* reference_tv = nullptr;
  const TensorView* data_tv = nullptr;

  AddressRecordKey(const TensorView* reference_tv_, const TensorView* data_tv_)
      : reference_tv(reference_tv_), data_tv(data_tv_) {}

  bool operator==(const AddressRecordKey& other) const {
    return reference_tv == other.reference_tv && data_tv == other.data_tv;
  }
};

struct AddressRecordKeyHash {
  std::size_t operator()(const AddressRecordKey& key) const {
    auto h1 = std::hash<const TensorView*>{}(key.reference_tv);
    auto h2 = std::hash<const TensorView*>{}(key.data_tv);
    return h1 ^ h2;
  }
};

//! Data structure stored in GPULower to keep track of
//!  all the instances of indexing math with pre-computed
//!  components.
class AddressComputeInfo {
 public:
  void build(Fusion* fusion);

  //! Returns the address record corresponding to the
  //!  data_tv - reference tv pair, if any has been
  //!  found.
  //! Assumes reference_tv == data_tv, i.e. it is
  //!  a consumer indexing case when reference_tv is null.
  c10::optional<AddressRecord*> getMaybeLiftedAddress(
      const TensorView* data_tv,
      const TensorView* reference_tv = nullptr);

  //! Returns the corresponding address record for the given
  //!  address_tv if found.
  //! Each address_tv should be mapped to a unique address record.
  c10::optional<AddressRecord*> getMaybeRecordForAddressTv(
      const TensorView* tv);

  c10::optional<AddressRecord*> getMaybeLiftedPredicateIndex(
      const TensorView* reference_tv);

 private:
  // Utility to help allocate space for saving pre-computed address.
  TensorView* makeAddressTv(
      std::vector<IterDomain*> address_domains,
      bool is_global_address,
      bool is_predicate_index,
      bool is_cpasync_write = false);

  void makeAddressRecord(
      TensorView* data_tv,
      TensorView* reference_tv,
      bool is_predicate_record = false);

  void makePredicateRecord(TensorView* reference_tv) {
    makeAddressRecord(reference_tv, reference_tv, true);
  }

 private:
  using AddressRecordPtr = std::unique_ptr<AddressRecord>;

  // Collected records of all the indexing math that needs
  //  to be lifted, indexed with reference tv and data_tv.
  std::unordered_map<AddressRecordKey, AddressRecordPtr, AddressRecordKeyHash>
      index_lift_record_;

  // Collected records of all the predicate indexing math that needs
  //  to be lifted, indexed with reference tv and data_tv.
  std::unordered_map<AddressRecordKey, AddressRecordPtr, AddressRecordKeyHash>
      predicate_lift_record_;

  //! Short cut from the address tensorview to
  //!  the address record information.
  std::unordered_map<const TensorView*, AddressRecord*>
      address_tv_to_address_record_;
};

//! Kernel IR pass that inserts requested index pre-computations.
std::vector<Expr*> preComputeLiftedAddress(const std::vector<Expr*>& exprs);

} // namespace nvfuser
