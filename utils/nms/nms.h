// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>
at::Tensor nms_cpu(const at::Tensor& dets, const at::Tensor& scores, const float threshold);
#ifdef WITH_CUDA
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("nms", &nms, "non-maximum suppression");
}
