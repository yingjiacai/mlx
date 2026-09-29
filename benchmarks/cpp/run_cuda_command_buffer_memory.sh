#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

usage() {
  cat <<'EOF'
Usage: run_cuda_command_buffer_memory.sh [options]

Build the CUDA command-buffer benchmark and run the four limit cells in fresh
processes for the operation, memory, and combined workloads.

Options:
  --cuda-root PATH       CUDA 13.0 toolkit root (required)
  --reference-cache PATH Known-good MLX CMakeCache.txt (required)
  --ptx-cache PATH       MLX PTX cache directory (default: under build dir)
  --build-dir PATH       Build directory for this checkout
  --output PATH          New result directory (required)
  --warmup N             Warm-up evaluations per process (default: 2)
  --runs N               Measured evaluations per process (default: 7)
  --jobs N               Parallel build jobs (default: at most 8)
  --help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cuda_root=""
reference_cache=""
ptx_cache=""
build_dir="$repo_root/build/cuda-command-buffer-memory"
output_dir=""
warmup=2
runs=7
if command -v nproc >/dev/null 2>&1; then
  jobs="$(nproc)"
else
  jobs="$(getconf _NPROCESSORS_ONLN)"
fi
if ((jobs > 8)); then
  jobs=8
fi

while (($#)); do
  case "$1" in
    --cuda-root|--reference-cache|--ptx-cache|--build-dir|--output|--warmup|--runs|--jobs)
      if (($# < 2)); then
        echo "$1 requires a value." >&2
        exit 2
      fi
      ;;
  esac
  case "$1" in
    --cuda-root)
      cuda_root="$2"
      shift 2
      ;;
    --reference-cache)
      reference_cache="$2"
      shift 2
      ;;
    --ptx-cache)
      ptx_cache="$2"
      shift 2
      ;;
    --build-dir)
      build_dir="$2"
      shift 2
      ;;
    --output)
      output_dir="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    --runs)
      runs="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$cuda_root" || -z "$reference_cache" || -z "$output_dir" ]]; then
  echo "--cuda-root, --reference-cache, and --output are required." >&2
  usage >&2
  exit 2
fi
if [[ -z "$ptx_cache" ]]; then
  ptx_cache="$build_dir/ptx-cache"
fi

for value_name in warmup runs jobs; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "--${value_name} must be a non-negative integer." >&2
    exit 2
  fi
done
if ((runs == 0 || jobs == 0)); then
  echo "--runs and --jobs must be positive." >&2
  exit 2
fi

if [[ ! -f "$reference_cache" ]]; then
  echo "Reference cache not found: $reference_cache" >&2
  exit 1
fi
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "CUDA compiler not found: $cuda_root/bin/nvcc" >&2
  exit 1
fi
cuda_root="$(realpath "$cuda_root")"
nvcc_version="$("$cuda_root/bin/nvcc" --version)"
if ! grep -q 'release 13\.0' <<<"$nvcc_version"; then
  echo "The selected nvcc is not CUDA 13.0: $cuda_root/bin/nvcc" >&2
  exit 1
fi
if [[ -e "$output_dir" ]]; then
  echo "Output path already exists: $output_dir" >&2
  exit 1
fi

cache_value() {
  local cache_file="$1"
  local key="$2"
  awk -v key="$key" '
    index($0, key ":") == 1 {
      sub(/^[^=]*=/, "")
      print
      exit
    }
  ' "$cache_file"
}

reference_cuda="$(cache_value "$reference_cache" CMAKE_CUDA_COMPILER)"
reference_c="$(cache_value "$reference_cache" CMAKE_C_COMPILER)"
reference_cxx="$(cache_value "$reference_cache" CMAKE_CXX_COMPILER)"
if [[ -z "$reference_c" || -z "$reference_cxx" || -z "$reference_cuda" ]]; then
  echo "Reference cache does not contain all required compiler paths." >&2
  exit 1
fi
if [[ "$(realpath "$reference_cuda")" != "$(realpath "$cuda_root/bin/nvcc")" ]]; then
  echo "Reference cache uses a different CUDA compiler: $reference_cuda" >&2
  exit 1
fi

if [[ -f "$build_dir/CMakeCache.txt" ]]; then
  build_source="$(cache_value "$build_dir/CMakeCache.txt" CMAKE_HOME_DIRECTORY)"
  if [[ "$(realpath "$build_source")" != "$(realpath "$repo_root")" ]]; then
    echo "Build directory belongs to another source tree: $build_source" >&2
    exit 1
  fi
  build_cuda="$(cache_value "$build_dir/CMakeCache.txt" CMAKE_CUDA_COMPILER)"
  if [[ -n "$build_cuda" ]] && \
    [[ "$(realpath "$build_cuda")" != "$(realpath "$cuda_root/bin/nvcc")" ]]; then
    echo "Build directory uses a different CUDA compiler: $build_cuda" >&2
    exit 1
  fi
fi

mkdir -p "$output_dir/raw"
mkdir -p "$ptx_cache"

cmake_args=(
  -S "$repo_root"
  -B "$build_dir"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc"
  -DCUDAToolkit_ROOT="$cuda_root"
  -DCMAKE_CUDA_ARCHITECTURES=120
  -DMLX_CUDA_ARCHITECTURES=120
  -DMLX_BUILD_CUDA=ON
  -DMLX_BUILD_CPU=OFF
  -DMLX_BUILD_METAL=OFF
  -DMLX_BUILD_TESTS=OFF
  -DMLX_BUILD_EXAMPLES=OFF
  -DMLX_BUILD_BENCHMARKS=ON
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
  -DMLX_BUILD_GGUF=OFF
  -DMLX_BUILD_SAFETENSORS=OFF
)

cache_keys=(
  CMAKE_C_COMPILER
  CMAKE_CXX_COMPILER
  CMAKE_CUDA_HOST_COMPILER
  CMAKE_C_FLAGS
  CMAKE_C_FLAGS_RELEASE
  CMAKE_CXX_FLAGS
  CMAKE_CXX_FLAGS_RELEASE
  CMAKE_CUDA_FLAGS
  CMAKE_CUDA_FLAGS_RELEASE
  CMAKE_EXE_LINKER_FLAGS
  CMAKE_EXE_LINKER_FLAGS_RELEASE
  CUDNN_INCLUDE_DIR
  cudnn_LIBRARY
)
for key in "${cache_keys[@]}"; do
  value="$(cache_value "$reference_cache" "$key")"
  if [[ -n "$value" && "$value" != *-NOTFOUND ]]; then
    cmake_args+=("-D${key}=${value}")
  fi
done

cudnn_include="$(cache_value "$reference_cache" CUDNN_INCLUDE_DIR)"
cudnn_library="$(cache_value "$reference_cache" cudnn_LIBRARY)"
if [[ -n "$cudnn_include" && "$cudnn_include" != *-NOTFOUND ]]; then
  cmake_args+=("-DCUDNN_INCLUDE_PATH=${cudnn_include}")
fi
if [[ -n "$cudnn_library" && "$cudnn_library" != *-NOTFOUND ]]; then
  cmake_args+=("-DCUDNN_LIBRARY_PATH=$(dirname "$cudnn_library")")
fi

echo "Configuring benchmark build..."
cmake "${cmake_args[@]}" 2>&1 | tee "$output_dir/configure.log"

echo "Building cuda_command_buffer_memory..."
cmake --build "$build_dir" \
  --target cuda_command_buffer_memory \
  --parallel "$jobs" 2>&1 | tee "$output_dir/build.log"

binary="$build_dir/benchmarks/cpp/cuda_command_buffer_memory"
if [[ ! -x "$binary" ]]; then
  echo "Benchmark binary not found: $binary" >&2
  exit 1
fi

cccl_target="$build_dir/_deps/cccl-src/include"
cccl_link="$build_dir/benchmarks/include/cccl"
if [[ ! -d "$cccl_target" ]]; then
  echo "CCCL headers not found: $cccl_target" >&2
  exit 1
fi
mkdir -p "$(dirname "$cccl_link")"
if [[ -L "$cccl_link" ]]; then
  if [[ "$(realpath "$cccl_link")" != "$(realpath "$cccl_target")" ]]; then
    echo "Existing CCCL link has a different target: $cccl_link" >&2
    exit 1
  fi
elif [[ -e "$cccl_link" ]]; then
  echo "CCCL path exists and is not a symlink: $cccl_link" >&2
  exit 1
else
  ln -s "$cccl_target" "$cccl_link"
fi

runtime_library_path="$cuda_root/targets/x86_64-linux/lib:$cuda_root/lib64"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  runtime_library_path="$runtime_library_path:$LD_LIBRARY_PATH"
fi

env LD_LIBRARY_PATH="$runtime_library_path" ldd "$binary" \
  >"$output_dir/ldd.txt"
if grep -E 'lib(cudart|cublas|cublasLt|cufft|cusolver|nvrtc).*=> /usr/local/cuda' \
  "$output_dir/ldd.txt" >/dev/null; then
  echo "The benchmark resolved a CUDA Toolkit library from /usr/local/cuda." >&2
  exit 1
fi
while read -r library resolved_path; do
  if [[ "$resolved_path" != "$cuda_root/"* ]]; then
    echo "$library resolved outside CUDA 13.0: $resolved_path" >&2
    exit 1
  fi
done < <(
  awk '$1 ~ /^lib(cudart|cublas|cublasLt|cufft|cusolver|nvrtc)/ && $2 == "=>" {print $1, $3}' \
    "$output_dir/ldd.txt"
)

manifest="$output_dir/manifest.txt"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$repo_root"
  echo "git_head=$(git -C "$repo_root" rev-parse HEAD)"
  echo "git_status_begin"
  git -C "$repo_root" status --short
  echo "git_status_end"
  echo "cuda_root=$cuda_root"
  echo "reference_cache=$reference_cache"
  echo "build_dir=$build_dir"
  echo "ptx_cache=$ptx_cache"
  echo "warmup=$warmup"
  echo "runs=$runs"
  echo "jobs=$jobs"
  cmake --version | sed -n '1p'
  ninja --version | sed 's/^/ninja=/'
  "$cuda_root/bin/nvcc" --version
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader
  sha256sum "$repo_root/benchmarks/cpp/cuda_command_buffer_memory.cpp"
  sha256sum "$binary"
  echo "reference_cache_values_begin"
  for key in "${cache_keys[@]}"; do
    value="$(cache_value "$reference_cache" "$key")"
    echo "$key=$value"
  done
  echo "reference_cache_values_end"
  echo "actual_cache_values_begin"
  for key in "${cache_keys[@]}" CMAKE_CUDA_ARCHITECTURES MLX_CUDA_ARCHITECTURES; do
    value="$(cache_value "$build_dir/CMakeCache.txt" "$key")"
    echo "$key=$value"
  done
  echo "actual_cache_values_end"
} >"$manifest"

summary_tsv="$output_dir/summary.tsv"
comparisons_tsv="$output_dir/comparisons.tsv"
diagnostics_tsv="$output_dir/diagnostics.tsv"
diagnostic_comparisons_tsv="$output_dir/diagnostic_comparisons.tsv"
printf '%s\n' \
  $'case\tmode\trequested_ops\trequested_mb\teffective_ops\teffective_mb\tbatch\tsequence\thidden\tdepth\twarmup_runs\truns\tmin_ms\tmedian_ms\tmean_ms\tp95_ms\tmax_ms\tactive_with_output_mib\tcache_with_output_mib\tpeak_mib\tactive_after_release_mib\tcache_after_release_mib\tactive_after_clear_mib\tcache_after_clear_mib\tchecksum_mean\tchecksum_max' \
  >"$summary_tsv"
printf '%s\n' \
  $'case\tmode\teffective_ops\teffective_mb\tprocess_dot_files\tgraph_prefix' \
  >"$diagnostics_tsv"

first_compute_capability="$(nvidia-smi --query-gpu=compute_cap \
  --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
if [[ "$first_compute_capability" != "12.0" ]]; then
  echo "Expected compute capability 12.0, found $first_compute_capability." >&2
  exit 1
fi

append_result() {
  local cell="$1"
  local log_file="$2"
  local effective_ops effective_mb
  case "$cell" in
    default)
      effective_ops=100
      effective_mb=1000
      ;;
    ops20)
      effective_ops=20
      effective_mb=1000
      ;;
    mb100)
      effective_ops=100
      effective_mb=100
      ;;
    both)
      effective_ops=20
      effective_mb=100
      ;;
  esac
  local result_line
  result_line="$(awk '/^RESULT / {line=$0; count++} END {
    if (count == 1) print line
  }' "$log_file")"
  if [[ -z "$result_line" ]]; then
    echo "Expected exactly one RESULT line in $log_file" >&2
    exit 1
  fi

  declare -A result=()
  local token key value
  for token in ${result_line#RESULT }; do
    key="${token%%=*}"
    value="${token#*=}"
    result["$key"]="$value"
  done

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell" "${result[mode]}" "${result[max_ops]}" "${result[max_mb]}" \
    "$effective_ops" "$effective_mb" "${result[batch]}" \
    "${result[sequence]}" "${result[hidden]}" "${result[depth]}" \
    "${result[warmup_runs]}" "${result[runs]}" "${result[min_ms]}" \
    "${result[median_ms]}" "${result[mean_ms]}" "${result[p95_ms]}" \
    "${result[max_ms]}" "${result[active_with_output_mib]}" \
    "${result[cache_with_output_mib]}" "${result[peak_mib]}" \
    "${result[active_after_release_mib]}" \
    "${result[cache_after_release_mib]}" \
    "${result[active_after_clear_mib]}" "${result[cache_after_clear_mib]}" \
    "${result[checksum_mean]}" "${result[checksum_max]}" >>"$summary_tsv"
}

run_cell() {
  local cell="$1"
  local mode="$2"
  shift 2
  local cell_dir="$output_dir/raw/$cell"
  local log_file="$cell_dir/$mode.log"
  mkdir -p "$cell_dir"

  local shape_args=()
  case "$mode" in
    operation)
      shape_args=(--sequence 256 --hidden 256 --depth 48)
      ;;
    memory)
      shape_args=(--sequence 8192 --hidden 768 --depth 6)
      ;;
    combined)
      shape_args=(--sequence 8192 --hidden 768 --depth 64)
      ;;
  esac

  echo "Running cell=$cell mode=$mode..."
  env \
    -u MLX_MAX_OPS_PER_BUFFER \
    -u MLX_MAX_MB_PER_BUFFER \
    -u MLX_SAVE_CUDA_GRAPHS_DOT_FILE \
    CUDA_HOME="$cuda_root" \
    LD_LIBRARY_PATH="$runtime_library_path" \
    MLX_PTX_CACHE_DIR="$ptx_cache" \
    MLX_USE_CUDA_GRAPHS=1 \
    "$binary" \
    --mode "$mode" \
    "${shape_args[@]}" \
    --warmup-runs "$warmup" \
    --runs "$runs" \
    "$@" 2>&1 | tee "$log_file"
  append_result "$cell" "$log_file"

  local graph_dir="$cell_dir/${mode}_graphs"
  local graph_prefix="$graph_dir/graph"
  local diagnostic_log="$cell_dir/${mode}_graphs.log"
  local dot_files effective_ops effective_mb
  mkdir -p "$graph_dir"
  env \
    -u MLX_MAX_OPS_PER_BUFFER \
    -u MLX_MAX_MB_PER_BUFFER \
    CUDA_HOME="$cuda_root" \
    LD_LIBRARY_PATH="$runtime_library_path" \
    MLX_PTX_CACHE_DIR="$ptx_cache" \
    MLX_USE_CUDA_GRAPHS=1 \
    MLX_SAVE_CUDA_GRAPHS_DOT_FILE="$graph_prefix" \
    "$binary" \
    --mode "$mode" \
    "${shape_args[@]}" \
    --warmup-runs 0 \
    --runs 1 \
    "$@" >"$diagnostic_log" 2>&1
  dot_files="$(find "$graph_dir" -maxdepth 1 -type f -name 'graph_*.dot' \
    | wc -l | tr -d '[:space:]')"
  if ((dot_files == 0)); then
    echo "No CUDA graph DOT files were written for cell=$cell mode=$mode." >&2
    exit 1
  fi
  case "$cell" in
    default)
      effective_ops=100
      effective_mb=1000
      ;;
    ops20)
      effective_ops=20
      effective_mb=1000
      ;;
    mb100)
      effective_ops=100
      effective_mb=100
      ;;
    both)
      effective_ops=20
      effective_mb=100
      ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell" "$mode" "$effective_ops" "$effective_mb" "$dot_files" \
    "$graph_prefix" >>"$diagnostics_tsv"
}

for cell in default ops20 mb100 both; do
  for mode in operation memory combined; do
    case "$cell" in
      default)
        run_cell "$cell" "$mode"
        ;;
      ops20)
        run_cell "$cell" "$mode" --max-ops 20
        ;;
      mb100)
        run_cell "$cell" "$mode" --max-mb 100
        ;;
      both)
        run_cell "$cell" "$mode" --max-ops 20 --max-mb 100
        ;;
    esac
  done
done

if ! awk -F '\t' '
  BEGIN {
    OFS = "\t"
    print "mode", "reference_case", "case", \
      "reference_process_dot_files", "process_dot_files", \
      "expected_relation", "status"
  }
  NR > 1 { count[$2, $1] = $5 }
  END {
    check("operation", "default", "ops20", "greater")
    check("operation", "default", "mb100", "equal")
    check("memory", "default", "ops20", "equal")
    check("memory", "default", "mb100", "greater")
    check("combined", "default", "ops20", "greater")
    check("combined", "default", "mb100", "greater")
    check("operation", "ops20", "both", "equal")
    check("memory", "mb100", "both", "equal")
    check("combined", "default", "both", "greater")
    check("combined", "mb100", "both", "equal")
    exit failed
  }
  function check(mode, reference, cell, relation, base, candidate, ok) {
    base = count[mode, reference]
    candidate = count[mode, cell]
    if (relation == "greater") ok = candidate > base
    else ok = candidate == base
    print mode, reference, cell, base, candidate, relation, \
      ok ? "PASS" : "FAIL"
    if (!ok) failed = 1
  }
' "$diagnostics_tsv" >"$diagnostic_comparisons_tsv"; then
  echo "Submission diagnostic failed. See $diagnostic_comparisons_tsv" >&2
  exit 1
fi

if ! awk -F '\t' '
  BEGIN {
    OFS = "\t"
    print "mode", "case", "default_mean", "mean", "mean_difference", \
      "default_max", "max", "max_difference", "status", \
      "peak_ratio_vs_default", "median_ratio_vs_default"
  }
  NR == FNR {
    if (FNR == 1) {
      for (i = 1; i <= NF; ++i) column[$i] = i
    } else if ($(column["case"]) == "default") {
      mode = $(column["mode"])
      checksum_mean[mode] = $(column["checksum_mean"])
      checksum_max[mode] = $(column["checksum_max"])
      peak[mode] = $(column["peak_mib"])
      median[mode] = $(column["median_ms"])
    }
    next
  }
  FNR == 1 { next }
  {
    mode = $(column["mode"])
    mean_value = $(column["checksum_mean"])
    max_value = $(column["checksum_max"])
    mean_difference = mean_value - checksum_mean[mode]
    max_difference = max_value - checksum_max[mode]
    if (mean_difference < 0) mean_difference = -mean_difference
    if (max_difference < 0) max_difference = -max_difference
    mean_base_abs = checksum_mean[mode] < 0 ? \
      -checksum_mean[mode] : checksum_mean[mode]
    max_base_abs = checksum_max[mode] < 0 ? \
      -checksum_max[mode] : checksum_max[mode]
    mean_tolerance = 0.00001 + 0.00001 * mean_base_abs
    max_tolerance = 0.00001 + 0.00001 * max_base_abs
    status = mean_difference <= mean_tolerance && \
      max_difference <= max_tolerance ? "PASS" : "FAIL"
    print mode, $(column["case"]), checksum_mean[mode], mean_value, \
      mean_difference, checksum_max[mode], max_value, max_difference, status, \
      $(column["peak_mib"]) / peak[mode], \
      $(column["median_ms"]) / median[mode]
    if (status == "FAIL") failed = 1
  }
  END { exit failed }
' "$summary_tsv" "$summary_tsv" >"$comparisons_tsv"; then
  echo "Checksum comparison failed. See $comparisons_tsv" >&2
  exit 1
fi

echo
echo "Benchmark complete."
echo "Summary: $summary_tsv"
echo "Comparisons: $comparisons_tsv"
echo "Submission diagnostics: $diagnostics_tsv"
echo "Diagnostic comparisons: $diagnostic_comparisons_tsv"
