  // Validate contiguity
  int64_t contiguous_stride = 1;
  auto contiguity_rev = contiguity.crbegin();
  for (int64_t i = (int64_t)sizes.size() - 1; i >= 0; i--) {
    if (alloc_dom_no_reductions.at(i)->isBroadcast()) {
      continue;
    }
    while (!contiguity_rev->has_value()) {
      contiguity_rev++;
    }
    auto size = sizes.at(i);
    auto stride = strides.at(i);
    NVF_ERROR(!contiguity.empty());
    auto last_contiguity = *contiguity_rev;
    NVF_ERROR(
        last_contiguity.has_value(),
        "I don't think this check makes sense, but unfortunately ",
        "clang-tidy is not smart enough to infer from the context that this is always true.");
    if (*last_contiguity) {
      NVF_CHECK(
          stride == contiguous_stride,
          "Stride mismatch with contiguity info. ",
          " allocation domain: ",
          ir_utils::toString(alloc_dom_no_reductions),
          " dim: ",
          i,
          " expected stride: ",
          contiguous_stride,
          " actual stride: ",
          stride);
    }
    contiguous_stride = stride * size;
    contiguity_rev++;
  }
  NVF_ERROR(
      contiguity_rev == contiguity.crend(),
      "The size of contiguity mismatch with the dimensionality of allocation domain");

  // Validate that for expanded broadcast, the stride must be zero.
  for (int64_t i : c10::irange((int64_t)strides.size())) {
    if (auto alloc_id = alloc_dom_no_reductions.at(i);
        alloc_id->hasExpandedExtent()) {
      auto stride = strides.at(i);
      NVF_CHECK(
          stride == 0,
          "Expecting an expanded dimension on dimension ",
          i,
          " but found stride ",
          stride);
    }
  }
