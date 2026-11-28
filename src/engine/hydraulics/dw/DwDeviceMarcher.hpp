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
 *          backend does not replace that class. It replaces the Picard loop of one execute() and
 *          the Courant step search, and it keeps the routing state on the device between calls:
 *
 *          - static data (geometry, lengths, inverts, tables, the node-to-conduit lists) and the
 *            whole state are uploaded once, by the first call after bind() or after the solver
 *            invalidates the device copy (DwDeviceStep::upload_all);
 *          - after that a call uploads only what the host changed: the lateral inflows, and every
 *            element whose host value differs from the device's, among the arrays the loop reads
 *            before it writes them (the outfall stages and types included);
 *          - execute() downloads, in one transfer, every array of the model state the host can read
 *            (the public NodeData, LinkData and ConduitData arrays the loop writes), so no host
 *            reader of those arrays sees an old value;
 *          - the solver-only arrays stay on the device; pull() brings them back when the host
 *            needs them (a checkpoint, a fallback to the CPU loop, [REPORT] LINK_STEPS).
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

/// @name Solver-only array groups that pull() brings back to the host
/// @{
inline constexpr unsigned kDwPullConverged = 1u; ///< node convergence flags (DWNodeArrays)
inline constexpr unsigned kDwPullLinkStep = 2u;  ///< conduit Froude numbers and midpoint areas
inline constexpr unsigned kDwPullRest = 4u;      ///< every other solver-only array
inline constexpr unsigned kDwPullAll = 7u;       ///< all of them
/// @}

/// Sizes of one network for bind(). Only the sizes are read: the marcher allocates its device
/// arrays and copies nothing until the first execute() or courantStep().
struct DwDeviceNetwork {
    step::Arrays host{}; ///< n_nodes, n_conduits, n_csr and the table sizes
};

/// One execute(). Every pointer is read during the call only; none is retained.
struct DwDeviceStep {
    double dt = 0.0;    ///< routing step, s
    int max_trials = 0; ///< Picard iteration limit

    /// True when the host holds the latest state: the static data and every array are uploaded
    /// first. The first call after bind() and the first call after the solver changed its own
    /// arrays on the host must set it.
    bool upload_all = false;

    /// Every array, on the host, sized as in DwDeviceNetwork (DWSolver::deviceArrays). On return
    /// the public state arrays (step::Arrays node depth to normal_flow_limited that NodeData,
    /// LinkData and ConduitData own) hold the step's result.
    step::Arrays host{};
};

/// What the Picard loop reports back, mirroring the CPU loop's locals.
struct DwDeviceResult {
    int iterations = 0;     ///< Picard iterations performed
    bool converged = false; ///< the last iteration converged
    unsigned pulled = 0;    ///< solver-only groups (kDwPull*) also brought back to the host
    bool ok = true;         ///< false: `error` says why; the host arrays are unspecified
    const char* error = nullptr;
};

/// What DWSolver::getRoutingStep's search for the critical element needs.
struct DwCourantStep {
    double fixed_step = 0.0;     ///< ROUTING_STEP, s; only a smaller step counts
    double courant_factor = 0.0; ///< the COURANT factor applied to each link's step
    bool upload_all = false;     ///< as DwDeviceStep::upload_all

    /// Every array, as in DwDeviceStep. The search reads the node depths and the link flows and
    /// volumes as the host holds them now; it changes no state.
    step::Arrays host{};
};

/// One pull(): solver-only arrays to bring back to the host.
struct DwDevicePull {
    unsigned groups = 0; ///< kDwPull* bits

    /// Host arrays to fill: the solver-only pointers of step::Arrays (DWSolver::solverArrays).
    step::Arrays host{};
};

/// The smallest Courant step below fixed_step and the element that sets it.
struct DwCourantResult {
    double dt = 0.0; ///< that step, s, before the minimum-step floor; fixed_step if none
    int link = -1;   ///< the link setting it, or -1
    int node = -1;   ///< the node setting it, or -1
    bool ok = true;  ///< false: `error` says why
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

    /// Allocate the device arrays for a network. Called once, after DWSolver::init.
    virtual void bind(const DwDeviceNetwork& network) = 0;

    /// Bring the device up to date with the host, run the Picard loop of one routing step there,
    /// and download the public state.
    virtual DwDeviceResult execute(const DwDeviceStep& step) = 0;

    /// Bring the device up to date with the host and search the links, then the nodes, for the
    /// smallest Courant step, as DWSolver::getRoutingStep does; a tie goes to the element searched
    /// first.
    virtual DwCourantResult courantStep(const DwCourantStep& step) = 0;

    /// Copy solver-only arrays from the device to the host. The device keeps its copy.
    virtual DwDeviceResult pull(const DwDevicePull& request) = 0;

    /// Short backend label for logs, e.g. "omp" or "cuda".
    virtual const char* label() const noexcept = 0;
};

} // namespace openswmm::dynwave

#endif // OPENSWMM_ENGINE_DW_DEVICE_MARCHER_HPP
