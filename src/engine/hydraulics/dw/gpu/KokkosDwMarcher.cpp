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
 * @file      KokkosDwMarcher.cpp
 * @brief     Kokkos Picard loop of the classic dynamic wave solver.
 *
 * @details One execute() is: copy the routing step's state to the device, run the Picard
 *          iterations there, copy the state back. The loop is DWSolver::execute's, and the
 *          per-element work is DwKernels.hpp compiled with OPENSWMM_KERNEL_FN =
 *          KOKKOS_INLINE_FUNCTION. Every kernel writes only the element it is called with, so the
 *          result does not depend on the thread count.
 *
 *          A stage the device does not run yet goes through the step's host callback: the state
 *          is copied back, the host runs the stage, and the state is copied in again.
 *
 * @see ../DwKernels.hpp, ../DwDeviceMarcher.hpp
 * @ingroup engine_hydraulics
 *
 * @author    Farzan Zeinali
 * @license   Apache-2.0
 */

#include <Kokkos_Core.hpp>

#define OPENSWMM_KERNEL_FN KOKKOS_INLINE_FUNCTION

#include "KokkosDwMarcher.hpp"

#include "../DwDeviceMarcher.hpp"
#include "../DwKernels.hpp"
#include "../../../2d/gpu/KokkosTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace openswmm::dynwave::gpu {

namespace {

using twoD::gpu::ExecSpace;
using twoD::gpu::MemSpace;

using Range = Kokkos::RangePolicy<ExecSpace>;
using Bytes = Kokkos::View<unsigned char*, MemSpace>;
using DevBytes = Kokkos::View<unsigned char*, MemSpace, Kokkos::MemoryUnmanaged>;
using HostBytes = Kokkos::View<unsigned char*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

#if defined(OPENSWMM_GPU_EXECSPACE_CUDA)
constexpr const char* kLabel = "cuda";
#elif defined(OPENSWMM_GPU_EXECSPACE_HIP)
constexpr const char* kLabel = "hip";
#elif defined(OPENSWMM_GPU_EXECSPACE_SYCL)
constexpr const char* kLabel = "sycl";
#else
constexpr const char* kLabel = "omp";
#endif

/// One array the device holds, and the host array it is copied from and back to.
struct Sync {
    unsigned char* dev = nullptr;
    unsigned char* host = nullptr;
    std::size_t bytes = 0;
};

/// Pair a device array with its host array; a null or empty host array is left out.
template <class T> void pairOf(std::vector<Sync>& list, T* dev, const T* host, int n) {
    if (dev == nullptr || host == nullptr || n <= 0)
        return;
    list.push_back({reinterpret_cast<unsigned char*>(dev),
                    reinterpret_cast<unsigned char*>(const_cast<T*>(host)),
                    static_cast<std::size_t>(n) * sizeof(T)});
}

} // namespace

/**
 * @brief The marcher. A named class outside any unnamed namespace, with every kernel-launching
 *        method public: CUDA refuses a device lambda whose enclosing function is private,
 *        protected or unnamed.
 */
class KokkosDwMarcher final : public IDwDeviceMarcher {
public:
    KokkosDwMarcher() = default;
    ~KokkosDwMarcher() override { report(); }
    KokkosDwMarcher(const KokkosDwMarcher&) = delete;
    KokkosDwMarcher& operator=(const KokkosDwMarcher&) = delete;

    void bind(const DwDeviceNetwork& network) override;
    DwDeviceResult execute(const DwDeviceStep& step) override;
    const char* label() const noexcept override { return kLabel; }

    // Public only for the reason above; not part of the interface.
    void linkGeometry();

private:
    template <class T> T* alloc(const char* name, int n);
    template <class T> T* upload(const char* name, const T* host, int n);
    std::vector<Sync> stateOf(const step::Arrays& h) const;
    static void push(const std::vector<Sync>& list);
    static void pull(const std::vector<Sync>& list);
    int onHost(const DwDeviceStep& step, const std::vector<Sync>& state, int stage, int iteration);
    void report() const;

    std::vector<Bytes> keep_; ///< owns every device array
    step::Arrays d_{};        ///< every pointer is a device array
    double* node_depth_ = nullptr;
    double* q1_ = nullptr;
    std::uint8_t* bypassed_ = nullptr;
    long steps_ = 0;
    long host_stages_ = 0;
    std::string error_;
};

template <class T> T* KokkosDwMarcher::alloc(const char* name, int n) {
    const std::size_t count = static_cast<std::size_t>(n > 0 ? n : 1);
    keep_.emplace_back(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing),
                       count * sizeof(T));
    return reinterpret_cast<T*>(keep_.back().data());
}

template <class T> T* KokkosDwMarcher::upload(const char* name, const T* host, int n) {
    T* dev = alloc<T>(name, n);
    std::vector<Sync> one;
    pairOf(one, dev, host, n);
    push(one);
    return dev;
}

void KokkosDwMarcher::push(const std::vector<Sync>& list) {
    for (const Sync& s : list)
        Kokkos::deep_copy(DevBytes(s.dev, s.bytes), HostBytes(s.host, s.bytes));
}

void KokkosDwMarcher::pull(const std::vector<Sync>& list) {
    for (const Sync& s : list)
        Kokkos::deep_copy(HostBytes(s.host, s.bytes), DevBytes(s.dev, s.bytes));
}

void KokkosDwMarcher::bind(const DwDeviceNetwork& network) {
    const step::Arrays& h = network.host;
    const int nn = h.n_nodes, nc = h.n_conduits;
    keep_.clear();
    d_ = step::Arrays{};
    d_.n_nodes = nn;
    d_.n_conduits = nc;
    d_.surcharge_method = h.surcharge_method;

    d_.node_invert = upload("dw_node_invert", h.node_invert, nn);
    d_.node_is_outfall = upload("dw_node_is_outfall", h.node_is_outfall, nn);

    d_.conduit_link = upload("dw_conduit_link", h.conduit_link, nc);
    d_.conduit_n1 = upload("dw_conduit_n1", h.conduit_n1, nc);
    d_.conduit_n2 = upload("dw_conduit_n2", h.conduit_n2, nc);
    d_.conduit_shape = upload("dw_conduit_shape", h.conduit_shape, nc);
    d_.conduit_has_offset = upload("dw_conduit_has_offset", h.conduit_has_offset, nc);
    d_.conduit_offset1 = upload("dw_conduit_offset1", h.conduit_offset1, nc);
    d_.conduit_offset2 = upload("dw_conduit_offset2", h.conduit_offset2, nc);
    d_.conduit_y_full = upload("dw_conduit_y_full", h.conduit_y_full, nc);
    d_.conduit_a_full = upload("dw_conduit_a_full", h.conduit_a_full, nc);
    d_.conduit_r_full = upload("dw_conduit_r_full", h.conduit_r_full, nc);
    d_.conduit_w_max = upload("dw_conduit_w_max", h.conduit_w_max, nc);
    d_.conduit_s_full = upload("dw_conduit_s_full", h.conduit_s_full, nc);
    d_.conduit_s_max = upload("dw_conduit_s_max", h.conduit_s_max, nc);
    d_.conduit_length = upload("dw_conduit_length", h.conduit_length, nc);
    d_.conduit_beta = upload("dw_conduit_beta", h.conduit_beta, nc);
    d_.conduit_q_max = upload("dw_conduit_q_max", h.conduit_q_max, nc);

    // The shared geometry tables, copied to the device once.
    const step::Tables& t = h.tables;
    d_.tables.A_Circ = upload("dw_A_Circ", t.A_Circ, t.N_A_Circ);
    d_.tables.R_Circ = upload("dw_R_Circ", t.R_Circ, t.N_R_Circ);
    d_.tables.W_Circ = upload("dw_W_Circ", t.W_Circ, t.N_W_Circ);
    d_.tables.S_Circ = upload("dw_S_Circ", t.S_Circ, t.N_S_Circ);
    d_.tables.Y_Circ = upload("dw_Y_Circ", t.Y_Circ, t.N_Y_Circ);
    d_.tables.Amax = upload("dw_Amax", t.Amax, static_cast<int>(std::size(xsect_tables::Amax)));
    d_.tables.N_A_Circ = t.N_A_Circ;
    d_.tables.N_R_Circ = t.N_R_Circ;
    d_.tables.N_W_Circ = t.N_W_Circ;
    d_.tables.N_S_Circ = t.N_S_Circ;
    d_.tables.N_Y_Circ = t.N_Y_Circ;

    // State: allocated here, filled from the host at every execute().
    node_depth_ = alloc<double>("dw_node_depth", nn);
    d_.node_depth = node_depth_;
    q1_ = alloc<double>("dw_q1", nc);
    d_.conduit_q1 = q1_;
    bypassed_ = alloc<std::uint8_t>("dw_bypassed", nc);
    d_.conduit_bypassed = bypassed_;
    double** geometry[] = {&d_.depth1,    &d_.depth2,     &d_.depth_mid,  &d_.h1,
                           &d_.h2,        &d_.fasnh,      &d_.width1,     &d_.width2,
                           &d_.width_mid, &d_.surf_area1, &d_.surf_area2, &d_.area1,
                           &d_.area2,     &d_.area_mid,   &d_.hrad1,      &d_.hrad_mid,
                           &d_.wcap_d1,   &d_.wcap_d2,    &d_.wcap_dm};
    for (double** field : geometry)
        *field = alloc<double>("dw_geometry", nc);
    d_.flow_class = alloc<std::int8_t>("dw_flow_class", nc);
}

std::vector<Sync> KokkosDwMarcher::stateOf(const step::Arrays& h) const {
    const int nn = d_.n_nodes, nc = d_.n_conduits;
    std::vector<Sync> list;
    pairOf(list, node_depth_, h.node_depth, nn);
    pairOf(list, q1_, h.conduit_q1, nc);
    pairOf(list, bypassed_, h.conduit_bypassed, nc);
    pairOf(list, d_.depth1, h.depth1, nc);
    pairOf(list, d_.depth2, h.depth2, nc);
    pairOf(list, d_.depth_mid, h.depth_mid, nc);
    pairOf(list, d_.h1, h.h1, nc);
    pairOf(list, d_.h2, h.h2, nc);
    pairOf(list, d_.fasnh, h.fasnh, nc);
    pairOf(list, d_.width1, h.width1, nc);
    pairOf(list, d_.width2, h.width2, nc);
    pairOf(list, d_.width_mid, h.width_mid, nc);
    pairOf(list, d_.surf_area1, h.surf_area1, nc);
    pairOf(list, d_.surf_area2, h.surf_area2, nc);
    pairOf(list, d_.area1, h.area1, nc);
    pairOf(list, d_.area2, h.area2, nc);
    pairOf(list, d_.area_mid, h.area_mid, nc);
    pairOf(list, d_.hrad1, h.hrad1, nc);
    pairOf(list, d_.hrad_mid, h.hrad_mid, nc);
    pairOf(list, d_.wcap_d1, h.wcap_d1, nc);
    pairOf(list, d_.wcap_d2, h.wcap_d2, nc);
    pairOf(list, d_.wcap_dm, h.wcap_dm, nc);
    pairOf(list, d_.flow_class, h.flow_class, nc);
    return list;
}

int KokkosDwMarcher::onHost(const DwDeviceStep& step, const std::vector<Sync>& state, int stage,
                            int iteration) {
    ++host_stages_;
    pull(state);
    const int r = step.host_stage(step.user, stage, iteration);
    push(state);
    return r;
}

void KokkosDwMarcher::linkGeometry() {
    const step::Arrays D = d_;
    Kokkos::parallel_for(
        "dw_link_geometry", Range(0, D.n_conduits),
        KOKKOS_LAMBDA(const int c) { step::linkGeometry(D, c); });
}

DwDeviceResult KokkosDwMarcher::execute(const DwDeviceStep& step) {
    DwDeviceResult r;
    try {
        if (step.host_stage == nullptr)
            throw std::runtime_error("the device Picard loop needs the host stage callback");
        const std::vector<Sync> state = stateOf(step.host);
        push(state);

        // DWSolver::execute's loop control (DynamicWave.cpp:1564-1648).
        int iterations = 0;
        bool converged = false;
        const long before = host_stages_;
        while (iterations < step.max_trials) {
            onHost(step, state, kDwInitNodes, iterations);
            linkGeometry();
            onHost(step, state, kDwMomentum, iterations);
            onHost(step, state, kDwGather, iterations);
            onHost(step, state, kDwBoundary, iterations);
            const int unconverged = onHost(step, state, kDwNodeDepths, iterations);
            converged = (unconverged == 0);
            ++iterations;
            if (iterations > 1) {
                if (converged)
                    break;
                onHost(step, state, kDwBypass, iterations);
            }
        }

        Kokkos::fence();
        pull(state);
        r.iterations = iterations;
        r.converged = converged;
        r.host_stages = host_stages_ - before;
        ++steps_;
    } catch (const std::exception& e) {
        error_ = e.what();
        r.ok = false;
        r.error = error_.c_str();
    } catch (...) {
        error_ = "unknown failure in the device Picard loop";
        r.ok = false;
        r.error = error_.c_str();
    }
    return r;
}

void KokkosDwMarcher::report() const {
    if (steps_ > 0)
        std::fprintf(stderr, "[openswmm DW] %s marcher: %ld routing steps, %ld host stages\n",
                     kLabel, steps_, host_stages_);
}

IDwDeviceMarcher* makeKokkosDwMarcher() {
    return new KokkosDwMarcher();
}

} // namespace openswmm::dynwave::gpu
