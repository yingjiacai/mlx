// Copyright © 2026 Apple Inc.

#include "mlx/mlx.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace {

struct Config {
  std::string mode{"combined"};
  int batch{1};
  std::optional<int> sequence;
  std::optional<int> hidden;
  std::optional<int> depth;
  int warmup_runs{2};
  int runs{5};
  mx::EvalOptions eval_options;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " [options]\n\n"
            << "Options:\n"
            << "  --mode operation|memory|combined\n"
            << "  --batch N\n"
            << "  --sequence N\n"
            << "  --hidden N\n"
            << "  --depth N\n"
            << "  --warmup-runs N\n"
            << "  --runs N\n"
            << "  --max-ops N\n"
            << "  --max-mb N\n"
            << "  --help\n";
}

int parse_int(std::string_view text, std::string_view option) {
  int value;
  auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(
        std::string(option) + " requires an integer value.");
  }
  return value;
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    std::string_view option{argv[i]};
    if (option == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 == argc) {
      throw std::invalid_argument(std::string(option) + " requires a value.");
    }
    std::string_view value{argv[++i]};
    if (option == "--mode") {
      config.mode = value;
    } else if (option == "--batch") {
      config.batch = parse_int(value, option);
    } else if (option == "--sequence") {
      config.sequence = parse_int(value, option);
    } else if (option == "--hidden") {
      config.hidden = parse_int(value, option);
    } else if (option == "--depth") {
      config.depth = parse_int(value, option);
    } else if (option == "--warmup-runs") {
      config.warmup_runs = parse_int(value, option);
    } else if (option == "--runs") {
      config.runs = parse_int(value, option);
    } else if (option == "--max-ops") {
      config.eval_options.max_ops_per_buffer = parse_int(value, option);
    } else if (option == "--max-mb") {
      config.eval_options.max_mb_per_buffer = parse_int(value, option);
    } else {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }

  if (config.mode == "operation") {
    config.sequence = config.sequence.value_or(256);
    config.hidden = config.hidden.value_or(256);
    config.depth = config.depth.value_or(48);
  } else if (config.mode == "memory") {
    config.sequence = config.sequence.value_or(6144);
    config.hidden = config.hidden.value_or(768);
    config.depth = config.depth.value_or(6);
  } else if (config.mode == "combined") {
    config.sequence = config.sequence.value_or(4096);
    config.hidden = config.hidden.value_or(768);
    config.depth = config.depth.value_or(24);
  } else {
    throw std::invalid_argument(
        "--mode must be operation, memory, or combined.");
  }

  if (config.batch <= 0 || *config.sequence <= 0 || *config.hidden <= 0 ||
      *config.depth <= 0 || config.warmup_runs < 0 || config.runs <= 0) {
    throw std::invalid_argument(
        "Shape, depth, and runs must be positive; warmup runs can be zero.");
  }
  if (config.eval_options.max_ops_per_buffer.value_or(0) < 0 ||
      config.eval_options.max_mb_per_buffer.value_or(0) < 0) {
    throw std::invalid_argument("Command-buffer limits must be non-negative.");
  }
  return config;
}

mx::array make_workload(
    const mx::array& input,
    const std::string& mode,
    int depth,
    mx::Device device) {
  auto x = input;
  for (int i = 0; i < depth; ++i) {
    auto branch_a = mx::sin(x, device);
    if (mode == "combined") {
      auto branch_b = mx::cos(x, device);
      auto product = mx::multiply(branch_a, branch_b, device);
      x = mx::add(x, product, device);
    } else {
      auto branch_b = mx::cos(input, device);
      x = mx::add(branch_a, branch_b, device);
    }
  }
  return x;
}

mx::array evaluate(
    const mx::array& input,
    const std::string& mode,
    int depth,
    const mx::EvalOptions& options,
    mx::Stream stream,
    double* milliseconds) {
  auto output = make_workload(input, mode, depth, stream.device);
  auto start = std::chrono::steady_clock::now();
  if (options.max_ops_per_buffer || options.max_mb_per_buffer) {
    mx::eval({output}, options);
  } else {
    mx::eval(output);
  }
  mx::synchronize(stream);
  auto end = std::chrono::steady_clock::now();
  if (milliseconds != nullptr) {
    *milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
  }
  return output;
}

double to_mib(size_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(1ULL << 20);
}

std::string limit_string(const std::optional<int>& value) {
  return value ? std::to_string(*value) : "default";
}

size_t input_size(const Config& config) {
  size_t size = config.batch;
  for (auto dimension : {*config.sequence, *config.hidden}) {
    if (size > std::numeric_limits<size_t>::max() / dimension) {
      throw std::invalid_argument("The input shape is too large.");
    }
    size *= dimension;
  }
  return size;
}

} // namespace

int main(int argc, char** argv) {
  try {
    auto config = parse_args(argc, argv);
    mx::Device device{mx::Device::gpu};
    if (!mx::is_available(device)) {
      throw std::runtime_error("A GPU backend is required.");
    }

    auto stream = mx::default_stream(device);
    mx::Shape shape{config.batch, *config.sequence, *config.hidden};
    auto num_elements = input_size(config);
    auto input = mx::arange(
        0.0f, static_cast<float>(num_elements), 1.0f, mx::float32, stream);
    input = mx::reshape(input, shape, stream);
    input = mx::multiply(
        input, mx::array(8.0f / static_cast<float>(num_elements)), stream);
    mx::eval(input);
    mx::synchronize(stream);

    for (int i = 0; i < config.warmup_runs; ++i) {
      evaluate(
          input,
          config.mode,
          *config.depth,
          config.eval_options,
          stream,
          nullptr);
    }

    mx::clear_cache();
    mx::reset_peak_memory();

    std::vector<double> times;
    times.reserve(config.runs);
    size_t peak_memory = 0;
    size_t active_with_output = 0;
    size_t cache_with_output = 0;
    std::optional<mx::array> last_output;
    for (int i = 0; i < config.runs; ++i) {
      double milliseconds;
      mx::reset_peak_memory();
      auto output = evaluate(
          input,
          config.mode,
          *config.depth,
          config.eval_options,
          stream,
          &milliseconds);
      times.push_back(milliseconds);
      peak_memory = std::max(peak_memory, mx::get_peak_memory());
      active_with_output = mx::get_active_memory();
      cache_with_output = mx::get_cache_memory();
      if (i + 1 == config.runs) {
        last_output = std::move(output);
      }
    }

    float checksum_mean;
    float checksum_max;
    {
      auto mean_array = mx::mean(*last_output, false, stream);
      auto max_array = mx::max(*last_output, false, stream);
      mx::eval(mean_array, max_array);
      checksum_mean = mean_array.item<float>();
      checksum_max = max_array.item<float>();
    }
    if (!std::isfinite(checksum_mean) || !std::isfinite(checksum_max)) {
      throw std::runtime_error("A checksum is not finite.");
    }
    last_output.reset();
    mx::synchronize(stream);
    auto active_after_release = mx::get_active_memory();
    auto cache_after_release = mx::get_cache_memory();
    mx::clear_cache();
    auto active_after_clear = mx::get_active_memory();
    auto cache_after_clear = mx::get_cache_memory();

    auto sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median;
    if (sorted_times.size() % 2 == 0) {
      auto upper = sorted_times.size() / 2;
      median = (sorted_times[upper - 1] + sorted_times[upper]) / 2.0;
    } else {
      median = sorted_times[sorted_times.size() / 2];
    }
    auto mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    auto p95_index = static_cast<size_t>(std::ceil(
                         0.95 * static_cast<double>(sorted_times.size()))) -
        1;

    std::cout << std::fixed << std::setprecision(6) << "RESULT"
              << " mode=" << config.mode << " batch=" << config.batch
              << " sequence=" << *config.sequence
              << " hidden=" << *config.hidden << " depth=" << *config.depth
              << " warmup_runs=" << config.warmup_runs
              << " runs=" << config.runs << " max_ops="
              << limit_string(config.eval_options.max_ops_per_buffer)
              << " max_mb="
              << limit_string(config.eval_options.max_mb_per_buffer)
              << " min_ms=" << sorted_times.front() << " median_ms=" << median
              << " mean_ms=" << mean << " p95_ms=" << sorted_times[p95_index]
              << " max_ms=" << sorted_times.back()
              << " active_with_output_mib=" << to_mib(active_with_output)
              << " cache_with_output_mib=" << to_mib(cache_with_output)
              << " peak_mib=" << to_mib(peak_memory)
              << " active_after_release_mib=" << to_mib(active_after_release)
              << " cache_after_release_mib=" << to_mib(cache_after_release)
              << " active_after_clear_mib=" << to_mib(active_after_clear)
              << " cache_after_clear_mib=" << to_mib(cache_after_clear)
              << " checksum_mean=" << checksum_mean
              << " checksum_max=" << checksum_max << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
