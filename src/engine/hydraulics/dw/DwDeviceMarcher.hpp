// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026 Farzan Zeinali
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file      DwDeviceMarcher.hpp
 * @brief     Interface between DWSolver and a device Picard loop.
 *
 * @details The classic dynamic wave solver keeps one owner of the model state: DWSolver. A device
 *          backend does not replace that class. It replaces the Picard loop of one execute():
 *          the solver hands the marcher its arrays, the marcher iterates the routing step on the
 *          device and hands the arrays back. The work before and after the loop stays on the host.
 *
 *          The marcher lives in the GPU plugin (src/engine/hydraulics/dw/gpu), so this header
 *          carries no Kokkos and no engine types: plain structs of raw pointers (DwStepArrays.hpp)
 *          that cross the plugin boundary as-is.
 *
 *          Scope (DWSolver::deviceScope checks it, and the CPU solver runs otherwise):
 *          junctions and FREE, NORMAL or FIXED outfalls, conduits of CIRCULAR or RECT_CLOSED
 *          section with one barrel, SURCHARGE_METHOD EXTRAN or SLOT, EXPLICIT node continuity.
 *
 * @see DwStepArrays.hpp, DwKernels.hpp, gpu/KokkosDwMarcher.cpp
 * @ingroup engine_hydraulics
 *
 * @author    Farzan Zeinali
 * @license   Apache-2.0
 */

#ifndef OPENSWMM_ENGINE_DW_DEVICE_MARCHER_HPP
#define OPENSWMM_ENGINE_DW_DEVICE_MARCHER_HPP

#include "DwStepArrays.hpp"

namespace openswmm::dynwave {

/// Static description of one network, valid for the duration of bind(). Every pointer is a host
/// array the marcher copies; none is retained.
struct DwDeviceNetwork {
    /// Sizes, options and static arrays. Its state pointers are ignored here; the marcher points
    /// them at its own storage.
    step::Arrays host{};
};

/// One execute() worth of state, on the host. The marcher copies it to the device, iterates, and
/// copies the state back into the same arrays.
struct DwDeviceStep {
    double dt = 0.0;    ///< routing step, s
    int max_trials = 0; ///< Picard iteration limit

    /// State, per-step inputs and per-step options, all host arrays sized as in DwDeviceNetwork.
    /// The static pointers are ignored.
    step::Arrays host{};
};

/// What the Picard loop reports back, mirroring the CPU loop's locals.
struct DwDeviceResult {
    int iterations = 0;     ///< Picard iterations performed
    bool converged = false; ///< the last iteration converged
    bool ok = true;         ///< false: `error` says why; the host arrays are unspecified
    const char* error = nullptr;
};

/**
 * @brief Device Picard loop of the classic dynamic wave solver.
 *
 * Owned by DWSolver through a shared_ptr whose deleter is the plugin's destroy entry point; the
 * virtual destructor keeps deletion across the plugin boundary correct under the shared C++ ABI
 * the GPU plugins already require (GpuPluginAbi.h).
 */
class IDwDeviceMarcher {
public:
    virtual ~IDwDeviceMarcher() = default;

    /// Copy the static network to the device. Called once, after DWSolver::init.
    virtual void bind(const DwDeviceNetwork& network) = 0;

    /// Run the Picard loop of one routing step on the device.
    virtual DwDeviceResult execute(const DwDeviceStep& step) = 0;

    /// Short backend label for logs, e.g. "omp" or "cuda".
    virtual const char* label() const noexcept = 0;
};

} // namespace openswmm::dynwave

#endif // OPENSWMM_ENGINE_DW_DEVICE_MARCHER_HPP
