// SPDX-License-Identifier: Apache-2.0

/**
 * @file      GpuPluginSupport.hpp
 * @brief     Shared support for GPU execution plugins.
 *
 * @author    Caleb Buahin <caleb.buahin@gmail.com>
 * @copyright Copyright (c) 2026 Caleb Buahin. All rights reserved.
 * @license   Apache-2.0
 */

#pragma once
#include "../solver/GpuPluginAbi.h"
#include "ExplicitKokkosSurfaceSolver.hpp"
#include "../../hydraulics/fv/FvDeviceMarcher.hpp"
#include "../../hydraulics/fv/gpu/KokkosNetworkMarcher.hpp"
#include "../../hydraulics/dw/DwDeviceMarcher.hpp"
#include "../../hydraulics/dw/gpu/KokkosDwMarcher.hpp"
#include <atomic>
#include <cstdio>
#include <exception>

#ifndef OPENSWMM_GPU_BUILD_CONTRACT
#error "GPU plugins require the generated build contract"
#endif
namespace openswmm::twoD::gpu::plugin {
inline thread_local char error[512]{};
/// Objects the make helpers below created and the destroy entry points have not yet destroyed. A
/// backend may finalize its runtime at process exit only while this is zero.
inline std::atomic<long> live_objects{0};
/**
 * @brief Construct a plugin-owned Kokkos surface marcher with thread-local diagnostics.
 * @tparam Initialize Callable type invocable with no arguments.
 * @param[in] initialize Borrowed/copied synchronous initialization callable; not retained. May
 * throw to refuse construction.
 * @return Owned ISurfaceSolver pointer as void*, or NULL on any exception, with bounded text
 * copied to the calling thread's error buffer.
 * @note Clears the previous diagnostic first. Requires the plugin's controlling thread;
 * destruction uses the matching plugin entry point. No exception escapes, and process-runtime
 * finalization is outside this helper.
 */
template <class Initialize> void* make(Initialize initialize, bool metal = false) noexcept {
    error[0] = '\0';
    try {
        initialize();
        ISurfaceSolver* solver = new ExplicitKokkosSurfaceSolver(metal);
        ++live_objects;
        return solver;
    } catch (const std::exception& e) {
        std::snprintf(error, sizeof error, "%s", e.what());
    } catch (...) {
        std::snprintf(error, sizeof error, "Unknown plugin initialization failure");
    }
    return nullptr;
}

/**
 * @brief Create the 1D network substep marcher after the backend runtime is initialized.
 *
 * @param[in] initialize Callable that initializes the backend's Kokkos runtime; may throw.
 * @return Owning openswmm::fv::IFvDeviceMarcher* as void*, or nullptr with `error` set.
 */
template <class Initialize>
void* makeNetworkMarcher(Initialize initialize, bool metal = false) noexcept {
    error[0] = '\0';
    try {
        initialize();
        openswmm::fv::IFvDeviceMarcher* marcher =
            openswmm::fv::gpu::makeKokkosNetworkMarcher(metal);
        ++live_objects;
        return marcher;
    } catch (const std::exception& e) {
        std::snprintf(error, sizeof error, "%s", e.what());
    } catch (...) {
        std::snprintf(error, sizeof error, "Unknown plugin initialization failure");
    }
    return nullptr;
}

/**
 * @brief Create the dynamic wave Picard loop after the backend runtime is initialized.
 *
 * @param[in] initialize Callable that initializes the backend's Kokkos runtime; may throw.
 * @return Owning openswmm::dynwave::IDwDeviceMarcher* as void*, or nullptr with `error` set.
 */
template <class Initialize> void* makeDwMarcher(Initialize initialize) noexcept {
    error[0] = '\0';
    try {
        initialize();
        openswmm::dynwave::IDwDeviceMarcher* marcher =
            openswmm::dynwave::gpu::makeKokkosDwMarcher();
        ++live_objects;
        return marcher;
    } catch (const std::exception& e) {
        std::snprintf(error, sizeof error, "%s", e.what());
    } catch (...) {
        std::snprintf(error, sizeof error, "Unknown plugin initialization failure");
    }
    return nullptr;
}
}
extern "C" OPENSWMM_GPU_ABI const char* openswmm_gpu_build_contract() {
    return OPENSWMM_GPU_BUILD_CONTRACT;
}
extern "C" OPENSWMM_GPU_ABI const char* openswmm_gpu_last_error() {
    return openswmm::twoD::gpu::plugin::error;
}
extern "C" OPENSWMM_GPU_ABI void openswmm_destroy_gpu_explicit_solver(void* solver) {
    if (solver != nullptr)
        --openswmm::twoD::gpu::plugin::live_objects;
    delete static_cast<openswmm::twoD::ISurfaceSolver*>(solver);
}

extern "C" OPENSWMM_GPU_ABI void openswmm_destroy_gpu_network_marcher(void* marcher) {
    if (marcher != nullptr)
        --openswmm::twoD::gpu::plugin::live_objects;
    delete static_cast<openswmm::fv::IFvDeviceMarcher*>(marcher);
}

extern "C" OPENSWMM_GPU_ABI void openswmm_destroy_gpu_dw_marcher(void* marcher) {
    if (marcher != nullptr)
        --openswmm::twoD::gpu::plugin::live_objects;
    delete static_cast<openswmm::dynwave::IDwDeviceMarcher*>(marcher);
}
