// SPDX-License-Identifier: Apache-2.0

//
// Python bindings for the SYCL/XPU memory and CacheGen kernels.
// Exposed as `lmcache.xpu_ops`.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "../kv_transfer_plan_types.h"
#include "../kv_transfer_types.h"
#include "cachegen_kernels_sycl.h"
#include "mem_kernels_sycl.h"

namespace py = pybind11;

namespace {

struct ThreadLocalBlockIdUsm {
      int64_t* ptr = nullptr;
      size_t capacity = 0;
      sycl::context context;
      bool has_context = false;

      ~ThreadLocalBlockIdUsm() {
            if (ptr != nullptr && has_context) {
                  sycl::free(ptr, context);
            }
      }

      torch::Tensor copy_from(const torch::Tensor& source,
                              const sycl::queue& queue) {
            const size_t count = static_cast<size_t>(source.numel());
            const sycl::context& queue_context = queue.get_context();
            if (ptr == nullptr || capacity < count ||
                (has_context && context != queue_context)) {
                  if (ptr != nullptr && has_context) {
                        sycl::free(ptr, context);
                  }
                  ptr = sycl::malloc_host<int64_t>(count, queue_context);
                  TORCH_CHECK(ptr != nullptr, "failed to allocate USM host block-ID buffer");
                  capacity = count;
                  context = queue_context;
                  has_context = true;
            }
            std::memcpy(ptr, source.data_ptr<int64_t>(),
                        count * sizeof(int64_t));
            return torch::from_blob(
                ptr, {static_cast<int64_t>(count)},
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
      }
};

thread_local ThreadLocalBlockIdUsm g_block_id_usm;

}  // namespace

// TransferDirection / EngineKVFormat enums are owned and registered exclusively
// by the `lmcache_native` module (csrc/lmcache_native/pybind.cpp).
// Re-registering them here under a different module triggers pybind11's "type
// already registered" error, so xpu_ops accepts them as plain ints and casts.
// Same convention as csrc/cuda/pybind.cpp (cuda_ops).
PYBIND11_MODULE(xpu_ops, m) {
      m.def(
                  "multi_layer_block_kv_transfer",
                  [](const torch::Tensor& paged_buffer_ptrs,
                         const py::list& lmcache_objects_obj, const torch::Tensor& block_ids,
                         const torch::Device& device, const int direction,
                         const py::object& shape_desc_obj, const int lmcache_chunk_size,
                         const int engine_kv_format, const int skip_prefix_n_blocks) {
                        auto get_int = [&](const char* name) {
                              return shape_desc_obj.attr(name).cast<int>();
                        };
                        PageBufferShapeDesc shape_desc{
                                    get_int("kv_size"), get_int("nl"), get_int("nb"),
                                    get_int("bs"), get_int("nh"), get_int("hs"),
                                    get_int("element_size"), get_int("block_stride_elems")};
                        TORCH_CHECK(lmcache_chunk_size > 0 && shape_desc.bs > 0 &&
                                                                        lmcache_chunk_size % shape_desc.bs == 0,
                                                            "invalid LMCache chunk/block size");
                        const int blocks_per_object = lmcache_chunk_size / shape_desc.bs;

                        auto block_ids_cpu = block_ids.to(torch::kCPU)
                                                                                                 .to(torch::kInt64)
                                                                                                 .contiguous();
                        const int64_t total_blocks = block_ids_cpu.numel();

                        std::vector<torch::Tensor> object_tensors;
                        object_tensors.reserve(lmcache_objects_obj.size());
                        for (const auto& item : lmcache_objects_obj) {
                              const auto ptr = item.cast<uintptr_t>();
                              auto options = torch::TensorOptions().dtype(
                                          shape_desc.element_size == 4 ? torch::kFloat32
                                                                                                                               : torch::kFloat16);
                              object_tensors.push_back(torch::from_blob(
                                          reinterpret_cast<void*>(ptr),
                                          {shape_desc.kv_size, shape_desc.nl, lmcache_chunk_size,
                                           shape_desc.nh * shape_desc.hs},
                                          options));
                        }
                        TORCH_CHECK(!object_tensors.empty(),
                                                            "lmcache_objects_ptrs must contain objects");
                        TORCH_CHECK(total_blocks ==
                                                                        static_cast<int64_t>(object_tensors.size()) *
                                                                                    blocks_per_object,
                                                            "block_ids length must equal objects * blocks_per_object");

                        auto queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
                        auto block_ids_usm = g_block_id_usm.copy_from(block_ids_cpu, queue);
                        for (size_t object_idx = 0; object_idx < object_tensors.size();
                                     ++object_idx) {
                              const int64_t block_base =
                                          static_cast<int64_t>(object_idx) * blocks_per_object;
                              auto object_block_ids = block_ids_usm.narrow(
                                          0, block_base, blocks_per_object);
                              auto& object = object_tensors[object_idx];
                              multi_layer_kv_transfer_block_ids(
                                          object, paged_buffer_ptrs, object_block_ids, device,
                                          shape_desc.nb, static_cast<TransferDirection>(direction),
                                          static_cast<EngineKVFormat>(engine_kv_format),
                                          shape_desc.bs, shape_desc.hs,
                                                              skip_prefix_n_blocks * shape_desc.bs);
                        }
                  },
                  py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
                  py::arg("block_ids"), py::arg("device"), py::arg("direction"),
                  py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
                  py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"));
  m.def(
      "multi_layer_kv_transfer",
      [](torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
         const torch::Tensor& slot_mapping,
         const torch::Device& paged_memory_device, const int page_buffer_size,
         const int direction, const int engine_kv_format, const int block_size,
         const int head_size, const int skip_prefix_n_tokens) {
        return multi_layer_kv_transfer(
            key_value, key_value_ptrs, slot_mapping, paged_memory_device,
            page_buffer_size, static_cast<TransferDirection>(direction),
            static_cast<EngineKVFormat>(engine_kv_format), block_size,
            head_size, skip_prefix_n_tokens);
      },
      py::arg("key_value"), py::arg("key_value_ptrs"), py::arg("slot_mapping"),
      py::arg("paged_memory_device"), py::arg("page_buffer_size"),
      py::arg("direction"), py::arg("engine_kv_format"),
      py::arg("block_size") = 0, py::arg("head_size") = 0,
      py::arg("skip_prefix_n_tokens") = 0,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "single_layer_kv_transfer",
      [](torch::Tensor& lmc_key_value_cache,
         torch::Tensor& vllm_key_value_cache, torch::Tensor& slot_mapping,
         const int direction, const int engine_kv_format,
         const bool token_major) {
        return single_layer_kv_transfer(
            lmc_key_value_cache, vllm_key_value_cache, slot_mapping,
            static_cast<TransferDirection>(direction),
            static_cast<EngineKVFormat>(engine_kv_format), token_major);
      },
      py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
      py::arg("slot_mapping"), py::arg("direction"),
      py::arg("engine_kv_format"), py::arg("token_major") = false,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "single_layer_kv_transfer_sgl",
      [](torch::Tensor& lmc_key_value_cache, torch::Tensor& sgl_key_cache,
         torch::Tensor& sgl_value_cache, torch::Tensor& slot_mapping,
         const int direction, const bool token_major) {
        return single_layer_kv_transfer_sgl(
            lmc_key_value_cache, sgl_key_cache, sgl_value_cache, slot_mapping,
            static_cast<TransferDirection>(direction), token_major);
      },
      py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
      py::arg("sgl_value_cache"), py::arg("slot_mapping"), py::arg("direction"),
      py::arg("token_major") = false, py::call_guard<py::gil_scoped_release>());
  m.def(
      "multi_layer_kv_transfer_unilateral",
      [](torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
         const torch::Tensor& slot_mapping,
         const torch::Device& paged_memory_device, const int page_buffer_size,
         const int direction, const int engine_kv_format) {
        return multi_layer_kv_transfer_unilateral(
            key_value, key_value_ptrs, slot_mapping, paged_memory_device,
            page_buffer_size, static_cast<TransferDirection>(direction),
            static_cast<EngineKVFormat>(engine_kv_format));
      },
      py::arg("key_value"), py::arg("key_value_ptrs"), py::arg("slot_mapping"),
      py::arg("paged_memory_device"), py::arg("page_buffer_size"),
      py::arg("direction"), py::arg("engine_kv_format"),
      py::call_guard<py::gil_scoped_release>());
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def(
      "lmcache_memcpy_async",
      [](uintptr_t dest, uintptr_t src, size_t nbytes, int direction,
         size_t host_buffer_offset, size_t host_buffer_alignment) {
        return lmcache_memcpy_async(dest, src, nbytes,
                                    static_cast<TransferDirection>(direction),
                                    host_buffer_offset, host_buffer_alignment);
      },
      py::call_guard<py::gil_scoped_release>());

  // Pinned (USM host) allocation -- SYCL analog of the CUDA alloc_pinned_ptr.
  // Bound under the same names as csrc/cuda/pybind.cpp / torch_ops so
  // lmcache._get_backend() overrides them by name on XPU.
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr, py::arg("size"),
        py::arg("flags") = 0);
  m.def("free_pinned_ptr", &free_pinned_ptr, py::arg("ptr"));

  // CacheGen / RoPE kernels (Intel XPU).  Names match the
  // lmcache.v1.platform.torch_ops baseline so the backend selection in
  // lmcache.v1.platform can transparently override.
  m.def("calculate_cdf", &calculate_cdf_xpu, py::arg("input"),
        py::arg("max_bins"));
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused_xpu,
        py::arg("old_positions"), py::arg("new_positions"), py::arg("key"),
        py::arg("head_size"), py::arg("cos_sin_cache"), py::arg("is_neox"));
  m.def("encode_fast_new", &encode_fast_new_xpu, py::arg("cdf"),
        py::arg("input_sym"), py::arg("output_buffer"),
        py::arg("output_lengths"));
  m.def("decode_fast_new", &decode_fast_new_xpu, py::arg("cdf"),
        py::arg("bytestreams"), py::arg("lengths"), py::arg("output"));
  m.def("decode_fast_prefsum", &decode_fast_prefsum_xpu, py::arg("cdf"),
        py::arg("bytestreams"), py::arg("lengths_prefsum"), py::arg("output"));
}
