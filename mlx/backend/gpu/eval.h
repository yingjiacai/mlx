// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>
#include <optional>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::gpu {

struct CommandBufferLimits {
  std::optional<int> max_ops_per_buffer;
  std::optional<int> max_mb_per_buffer;
};

void init();
void new_stream(Stream s);
void new_thread_unsafe_stream(Stream s);
void eval(array& arr, const CommandBufferLimits& limits);
void finalize(Stream s);
void synchronize(Stream s);
void clear_streams();

} // namespace mlx::core::gpu
