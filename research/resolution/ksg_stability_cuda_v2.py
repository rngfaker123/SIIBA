"""Isolated exact CUDA query backend for the frozen FP64 KSG-1 estimator.

SciPy constructs exact cKDTrees on CPU. CUDA traverses immutable exported leaves
and conservative bounds with FP64 Chebyshev distances. The frozen estimator's
validation, kth-boundary convention and CPU digamma/mean formula are preserved.
This module does not modify or monkeypatch the original estimator, launch jobs,
write scientific records, or fall back to CPU queries after a CUDA failure.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

BACKEND_VERSION = "cuda_exact_stability_k1_3_5_10_20_v1"
ORIGINAL_SHA256 = "bb6f7fbf704a3baff582a712fefbaa6bb6d13c9d962ff4fa108347db05a19b0d"
STACK_CAPACITY = 128
COMPILE_OPTIONS = ("--std=c++11", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """C-order byte digest; dtype and shape are separately recorded."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    view = memoryview(array).cast("B")
    for offset in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[offset:offset + 8 * 1024 * 1024])
    return digest.hexdigest()


@dataclass
class ExportedTree:
    data: np.ndarray
    indices: np.ndarray
    left: np.ndarray
    right: np.ndarray
    start: np.ndarray
    end: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_depth: int
    scipy_tree: Any

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in
                   ("data", "left", "right", "start", "end", "lower", "upper"))


def export_tree(values: np.ndarray) -> ExportedTree:
    """Export SciPy's exact partition, never quantizing coordinates or bounds."""
    data = np.ascontiguousarray(values, dtype=np.float64)
    if data.ndim != 2 or not len(data) or not data.shape[1] or not np.isfinite(data).all():
        raise ValueError("Tree data must be a nonempty finite FP64 matrix")
    tree = cKDTree(data)
    indices = np.ascontiguousarray(tree.indices, dtype=np.int64)
    ordered = np.ascontiguousarray(data[indices])
    nodes, depths, children = [], [], []
    pending = [(tree.tree, 0, -1, -1)]
    while pending:
        node, depth, parent, side = pending.pop()
        current = len(nodes)
        if current >= np.iinfo(np.int32).max:
            raise MemoryError("Tree exceeds the pinned int32 node-index representation")
        nodes.append(node)
        depths.append(depth)
        children.append([-1, -1])
        if parent >= 0:
            children[parent][side] = current
        if node.split_dim >= 0:
            pending.append((node.greater, depth + 1, current, 1))
            pending.append((node.lesser, depth + 1, current, 0))
    max_depth = max(depths)
    if max_depth + 2 >= STACK_CAPACITY:
        raise ValueError("Tree depth cannot safely fit the exact CUDA DFS stack")
    children_array = np.asarray(children, dtype=np.int32)
    start = np.asarray([node.start_idx for node in nodes], dtype=np.int64)
    end = np.asarray([node.end_idx for node in nodes], dtype=np.int64)
    lower = np.empty((len(nodes), data.shape[1]), dtype=np.float64)
    upper = np.empty_like(lower)
    for current in range(len(nodes) - 1, -1, -1):
        a, b = children_array[current]
        if a < 0:
            points = ordered[start[current]:end[current]]
            if not len(points):
                raise AssertionError("Export encountered an empty leaf")
            lower[current] = points.min(axis=0)
            upper[current] = points.max(axis=0)
        else:
            lower[current] = np.minimum(lower[a], lower[b])
            upper[current] = np.maximum(upper[a], upper[b])
            if start[current] != start[a] or end[current] != end[b] or end[a] != start[b]:
                raise AssertionError("SciPy partition spans are not contiguous")
    exported = ExportedTree(ordered, indices, np.ascontiguousarray(children_array[:, 0]),
                            np.ascontiguousarray(children_array[:, 1]), start, end,
                            lower, upper, max_depth, tree)
    for name in ("data", "indices", "left", "right", "start", "end", "lower", "upper"):
        getattr(exported, name).flags.writeable = False
    return exported


def _box_lower(query: np.ndarray, tree: ExportedTree, node: int) -> float:
    return float(np.maximum(np.maximum(tree.lower[node] - query, query - tree.upper[node]), 0).max())


def reference_export_query(tree: ExportedTree, queries: np.ndarray,
                           radii: Optional[np.ndarray] = None) -> np.ndarray:
    """Small-fixture CPU traversal of exported arrays for independent unit tests.

    This is never selected as a production fallback, and is deliberately not the
    estimator API. Its output lets tests exercise exported bounding/leaf logic.
    """
    queries = np.asarray(queries, dtype=np.float64)
    output = np.empty((len(queries), 22), dtype=np.float64) if radii is None else np.empty(len(queries), np.int64)
    for row, query in enumerate(queries):
        stack = [0]
        best = [math.inf] * 22
        count = 0
        while stack:
            node = stack.pop()
            bound = _box_lower(query, tree, node)
            if bound > (best[-1] if radii is None else radii[row]):
                continue
            a, b = int(tree.left[node]), int(tree.right[node])
            if radii is not None:
                upper = np.maximum(np.abs(query - tree.lower[node]), np.abs(query - tree.upper[node])).max()
                if upper <= radii[row]:
                    count += int(tree.end[node] - tree.start[node])
                    continue
            if a < 0:
                for point in tree.data[tree.start[node]:tree.end[node]]:
                    distance = float(np.abs(point - query).max())
                    if radii is None:
                        if distance < best[-1]:
                            best = sorted(best + [distance])[:22]
                    else:
                        count += int(distance <= radii[row])
            else:
                if radii is None and _box_lower(query, tree, a) > _box_lower(query, tree, b):
                    a, b = b, a
                stack.extend((b, a))
        output[row] = best if radii is None else count
    return output


def check_neighbor_distances(distances: np.ndarray, n: int, k: int) -> np.ndarray:
    ranks = min(n, 22)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != 22:
        raise ValueError("CUDA must return all seven rank slots")
    radius = distances[:, k]
    tied = (radius <= 0) | (radius == distances[:, k-1])
    if ranks > k+1:
        tied |= radius == distances[:, k+1]
    if np.any(tied):
        raise ValueError("Tied or zero kth-neighbor radius at %d observations; no silent jitter" % int(tied.sum()))
    if not np.isfinite(radius).all():
        raise ValueError("Non-finite neighbor radius")
    if np.any(distances[:, 0] != 0) or np.any(distances[:, 1:ranks] < distances[:, :ranks-1]):
        raise ValueError("CUDA neighbor ranks are not ordered including zero-distance self")
    return np.ascontiguousarray(radius)


def _cuda_runtime(device: int):
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CuPy with a matching CUDA runtime is required; no CPU fallback") from error
    if cp.cuda.runtime.getDeviceCount() <= device or device < 0:
        raise ValueError("Requested CUDA device does not exist")
    return cp


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


def compile_module_direct_nvrtc(cp: Any, kernel_source: str):
    """Compile explicit FP64 options without CuPy's implicit --ftz=true.

    V1 used RawModule(code, options), whose CuPy compiler wrapper adds another
    FTZ option. NVRTC rejects two definitions before compiling any kernel. V2
    invokes the same NVIDIA NVRTC binding directly, then loads its PTX through
    RawModule(path). Neither source, arithmetic nor query execution changes.
    """
    from cupy_backends.cuda.libs import nvrtc
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    # 编译计算能力 8.6 PTX
    # 兼容 Ampere 与 Ada
    # 固定运算选项
    device_capability = (int(properties["major"]), int(properties["minor"]))
    if device_capability < (8, 6):
        raise RuntimeError("This execution plan requires compute capability >=8.6")
    architecture = "compute_86"
    options = COMPILE_OPTIONS + ("--gpu-architecture=" + architecture,)
    program = nvrtc.createProgram(kernel_source, "ksg_stability_kernels_v1.cu", (), ())
    try:
        try:
            nvrtc.compileProgram(program, options)
        except Exception as error:
            log = nvrtc.getProgramLog(program)
            if isinstance(log, bytes):
                log = log.decode("utf8", "replace")
            raise RuntimeError("Direct NVRTC compile failed: " + str(log)) from error
        ptx = nvrtc.getPTX(program)
        log = nvrtc.getProgramLog(program)
        version = nvrtc.getVersion()
    finally:
        nvrtc.destroyProgram(program)
    if isinstance(ptx, str):
        ptx = ptx.encode("utf8")
    if not isinstance(ptx, bytes) or not ptx:
        raise RuntimeError("Direct NVRTC produced no PTX")
    # 载入预编译 PTX
    # 预载两个核函数
    with tempfile.TemporaryDirectory(prefix="ksg_exact_nvrtc_v2_") as directory:
        path = Path(directory) / "ksg_stability_kernels_v1.ptx"
        path.write_bytes(ptx)
        module = cp.RawModule(path=str(path))
        module.get_function("exact_knn7")
        module.get_function("exact_ball_counts")
    provenance = {"compiler": "direct_cupy_nvrtc_binding_then_RawModule_PTX_path",
                  "nvrtc_version": list(version), "options": list(options),
                  "architecture": architecture, "actual_device_capability": list(device_capability), "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
                  "compile_log": _json_value(log), "implicit_CuPy_source_options": False,
                  "flush_to_zero": False, "fma_contraction": False}
    return module, provenance


def _query_cuda(cp: Any, exported: ExportedTree, queries: np.ndarray,
                chunk_size: int, radii: Optional[np.ndarray], kernel: Any,
                reserve_bytes: int) -> np.ndarray:
    free, total = cp.cuda.runtime.memGetInfo()
    dimension = exported.data.shape[1]
    # 预留线程局部内存
    # 预留启动工作区
    chunk_bytes = chunk_size * (dimension * 8 + 22 * 8 + 8 + 128 * 4 + 22 * 8)
    required = exported.nbytes + chunk_bytes + reserve_bytes
    if required > free:
        raise MemoryError("Exact query needs %d bytes including reserve; CUDA free=%d total=%d" % (required, free, total))
    names = ("data", "left", "right", "start", "end", "lower", "upper")
    arrays = [cp.asarray(getattr(exported, name)) for name in names]
    data, left, right, start, end, lower, upper = arrays
    output = np.empty((len(queries), 22), dtype=np.float64) if radii is None else np.empty(len(queries), dtype=np.int64)
    for offset in range(0, len(queries), chunk_size):
        stop = min(offset + chunk_size, len(queries))
        query_chunk = cp.asarray(np.ascontiguousarray(queries[offset:stop]))
        error = cp.zeros(1, dtype=cp.int32)
        count = stop - offset
        if radii is None:
            result = cp.empty((count, 22), dtype=cp.float64)
            args = (data, np.int32(dimension), left, right, start, end, lower, upper,
                    query_chunk, np.int64(count), result, error)
        else:
            radius_chunk = cp.asarray(np.ascontiguousarray(radii[offset:stop]))
            result = cp.empty(count, dtype=cp.int64)
            args = (data, np.int32(dimension), left, right, start, end, lower, upper,
                    query_chunk, radius_chunk, np.int64(count), result, error)
        kernel(((count + 127) // 128,), (128,), args)
        error_code = int(cp.asnumpy(error)[0])
        if error_code:
            raise FloatingPointError("CUDA exact traversal failed closed, error flag %d" % error_code)
        output[offset:stop] = cp.asnumpy(result)
        del query_chunk, result, error, args
        if radii is not None:
            del radius_chunk
    del arrays, data, left, right, start, end, lower, upper
    cp.get_default_memory_pool().free_all_blocks()
    return output


def estimate_ksg_gpu(x: Any, y: Any, original_module: Any, k: int = 5, device: int = 0,
                     workers: int = 4, chunk_size: int = 65536,
                     validation_query_indices: Optional[Sequence[int]] = None,
                     return_arrays: bool = False,
                     reserve_bytes: int = 768 * 1024 * 1024) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute unchanged KSG1 k5 using exact CUDA queries, CPU final formula.

    Returns (result, diagnostics). Optional validation indices trigger SciPy
    comparisons against the same entire reference data before returning a result.
    Returned arrays (when explicitly requested) are inside diagnostics['arrays'];
    callers must remove them before JSON serialization. No scientific output is
    written by this function. Any runtime/query/validation discrepancy raises.
    """
    started = time.perf_counter()
    original_path = Path(original_module.__file__).resolve()
    original_sha = sha256_file(original_path)
    if original_sha != ORIGINAL_SHA256:
        raise ValueError("Frozen original estimator source SHA mismatch")
    workers = original_module._positive_integer(workers, "workers")
    chunk_size = original_module._positive_integer(chunk_size, "chunk_size")
    reserve_bytes = original_module._positive_integer(reserve_bytes, "reserve_bytes")
    x = original_module._matrix(x, "x")
    y = original_module._matrix(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have the same N")
    k = original_module._positive_integer(k, "k")
    if k not in (1,3,5,10,20):
        raise ValueError("Unfrozen k")
    if len(x) <= k:
        raise ValueError("Need N > k, got N=%d and k=5" % len(x))
    original_module._assert_no_duplicate_rows(x, "x")
    original_module._assert_no_duplicate_rows(y, "y")
    validation = None
    if validation_query_indices is not None:
        validation = np.asarray(validation_query_indices)
        if validation.ndim != 1 or validation.dtype.kind not in "iu" or not len(validation):
            raise ValueError("Validation indices must be a nonempty integer vector")
        validation = np.asarray(validation, dtype=np.int64)
        if validation.min() < 0 or validation.max() >= len(x) or len(np.unique(validation)) != len(validation):
            raise ValueError("Validation indices must be distinct and in range")
    cp = _cuda_runtime(int(device))
    kernel_path = Path(__file__).with_name("ksg_stability_kernels_v1.cu")
    kernel_source = kernel_path.read_text(encoding="utf-8")
    timings: Dict[str, float] = {"input_validation": time.perf_counter() - started}
    tree_diagnostics = []
    cpu_validation: Dict[str, Any] = {"requested": validation is not None, "passed": None}
    with cp.cuda.Device(int(device)):
        # 禁用融合与次正规数归零
        # 禁用快速数学
        module, compile_provenance = compile_module_direct_nvrtc(cp, kernel_source)
        knn_kernel = module.get_function("exact_knn7")
        count_kernel = module.get_function("exact_ball_counts")
        joint = np.concatenate([x, y], axis=1)
        phase = time.perf_counter()
        tree = export_tree(joint)
        timings["joint_tree_export"] = time.perf_counter() - phase
        tree_diagnostics.append({"kind": "joint", "nodes": len(tree.start), "max_depth": tree.max_depth, "gpu_bytes": tree.nbytes})
        phase = time.perf_counter()
        distances = _query_cuda(cp, tree, joint, chunk_size, None, knn_kernel, reserve_bytes)
        timings["joint_cuda_query"] = time.perf_counter() - phase
        radii = check_neighbor_distances(distances, len(x), k)
        if validation is not None:
            expected = tree.scipy_tree.query(joint[validation], k=min(len(x), 22), p=np.inf, workers=workers)[0]
            actual = distances[validation, :min(len(x), 22)]
            if not np.array_equal(expected, actual):
                raise AssertionError("Exact CUDA kNN validation differs from SciPy")
            cpu_validation["knn_distances_exact"] = True
        del tree, joint
        if not return_arrays:
            del distances
        gc.collect()
        strict_radii = np.nextafter(radii, 0.0)
        counts = []
        for label, values in (("x", x), ("y", y)):
            phase = time.perf_counter()
            tree = export_tree(values)
            timings[label + "_tree_export"] = time.perf_counter() - phase
            tree_diagnostics.append({"kind": label, "nodes": len(tree.start), "max_depth": tree.max_depth, "gpu_bytes": tree.nbytes})
            phase = time.perf_counter()
            count = _query_cuda(cp, tree, values, chunk_size, strict_radii, count_kernel, reserve_bytes)
            timings[label + "_cuda_count"] = time.perf_counter() - phase
            if np.any(count < 1) or np.any(count > len(x)):
                raise FloatingPointError("Marginal count violates self-inclusive domain")
            if validation is not None:
                expected = np.asarray(tree.scipy_tree.query_ball_point(values[validation], strict_radii[validation],
                                      p=np.inf, workers=workers, return_length=True), dtype=np.int64)
                if not np.array_equal(expected, count[validation]):
                    raise AssertionError("Exact CUDA %s strict count differs from SciPy" % label)
                cpu_validation[label + "_counts_exact"] = True
            counts.append(count)
            del tree
            gc.collect()
        properties = cp.cuda.runtime.getDeviceProperties(int(device))
        runtime = {"cupy_version": cp.__version__, "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                   "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()), "device_index": int(device),
                   "device_pci_bus_id": cp.cuda.Device(int(device)).pci_bus_id,
                   "device_name": _json_value(properties.get("name", "")),
                   "device_properties": {str(key): _json_value(value) for key, value in properties.items()}}
    nx_plus_one, ny_plus_one = counts
    phase = time.perf_counter()
    # 沿用 CPU 双伽马公式
    estimate = digamma(k) + digamma(len(x)) - np.mean(digamma(nx_plus_one) + digamma(ny_plus_one))
    result = original_module._base_result(float(estimate), len(x), k, workers, "KSG_1")
    result.update({"dx": int(x.shape[1]), "dy": int(y.shape[1]),
                   "marginal_boundary": "strict_less_than_joint_kth_radius",
                   "self_in_marginal_digamma_count": True,
                   "radius_min": float(radii.min()), "radius_max": float(radii.max()),
                   "reference": "https://arxiv.org/abs/cond-mat/0305641",
                   "query_backend": BACKEND_VERSION,
                   "base_formula_implementation_version": original_module.ESTIMATOR_VERSION})
    timings["cpu_original_formula"] = time.perf_counter() - phase
    timings["total"] = time.perf_counter() - started
    if validation is not None:
        cpu_validation.update({"passed": True, "query_indices": validation.tolist(),
                               "reference_n": len(x), "full_reference_used": True})
    diagnostics = {"query_backend": BACKEND_VERSION, "original_estimator_source_sha256": original_sha,
                   "python_source_sha256": sha256_file(__file__), "cuda_source_sha256": sha256_file(kernel_path),
                   "compile_options": list(COMPILE_OPTIONS), "compile_provenance": compile_provenance,
                   "cuda_executed": True,
                   "dtype": "float64", "distance": "chebyshev_max_norm", "k": k,
                   "n": len(x), "dx": x.shape[1], "dy": y.shape[1], "chunk_size": chunk_size,
                   "query_count": len(x), "max_stack_capacity": STACK_CAPACITY,
                   "stack_overflow": False, "cuda_error_flags": 0,
                   "self_included": True, "strict_boundary": "np.nextafter(joint_kth_radius, 0.0)",
                   "cpu_final_formula": True, "gpu_float_reduction": False,
                   "approximate_search": False, "pairwise_matrix_allocated": False,
                   "silent_cpu_query_fallback": False, "trees": tree_diagnostics,
                   "runtime": runtime, "timings_seconds": timings, "cpu_validation": cpu_validation,
                   "arrays_sha256": {"x": array_sha256(x), "y": array_sha256(y),
                                      "radii": array_sha256(radii), "nx_plus_one": array_sha256(nx_plus_one),
                                      "ny_plus_one": array_sha256(ny_plus_one)}}
    if return_arrays:
        diagnostics["arrays"] = {"distances": distances, "radii": radii, "strict_radii": strict_radii,
                                  "nx_plus_one": nx_plus_one, "ny_plus_one": ny_plus_one}
    return result, diagnostics
