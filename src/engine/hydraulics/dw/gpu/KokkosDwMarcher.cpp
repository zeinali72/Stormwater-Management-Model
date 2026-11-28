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
 * @details The loop is DWSolver::execute's, and the per-element work is DwKernels.hpp compiled
 *          with OPENSWMM_KERNEL_FN = KOKKOS_INLINE_FUNCTION. Every kernel writes only the element
 *          it is called with, so the result does not depend on the thread count; the one
 *          reduction, the count of unconverged nodes, is an integer.
 *
 *          Every array lives in one device block, laid out by group: the public state first, then
 *          the solver-only groups pull() returns, then the per-step inputs, then the static data.
 *          A pinned host mirror of the block holds what the device holds. The first call after
 *          bind() (DwDeviceStep::upload_all) packs every array into the mirror and uploads it in
 *          one copy. Every later call compares the host's arrays bit for bit with the mirror and
 *          uploads, in one copy of a pinned staging buffer, the lateral inflows and the changed
 *          elements; a kernel scatters them into the block. execute() then downloads the public
 *          state in one copy.
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
#include <cstring>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace openswmm::dynwave::gpu {

namespace {

using twoD::gpu::ExecSpace;
using twoD::gpu::MemSpace;

using Range = Kokkos::RangePolicy<ExecSpace>;
using Bytes = Kokkos::View<unsigned char*, MemSpace>;
using Pinned = Kokkos::View<unsigned char*, Kokkos::SharedHostPinnedSpace>;
using DevBytes = Kokkos::View<unsigned char*, MemSpace, Kokkos::MemoryUnmanaged>;
using PinnedBytes =
    Kokkos::View<unsigned char*, Kokkos::SharedHostPinnedSpace, Kokkos::MemoryUnmanaged>;
using Ints = Kokkos::View<int*, MemSpace>;

#if defined(OPENSWMM_GPU_EXECSPACE_CUDA)
constexpr const char* kLabel = "cuda";
#elif defined(OPENSWMM_GPU_EXECSPACE_HIP)
constexpr const char* kLabel = "hip";
#elif defined(OPENSWMM_GPU_EXECSPACE_SYCL)
constexpr const char* kLabel = "sycl";
#else
constexpr const char* kLabel = "omp";
#endif

/// Which part of the block an array is in, in block order.
enum class Group : std::uint8_t {
    Public,    ///< public state the host reads: downloaded after every execute()
    Converged, ///< kDwPullConverged
    LinkStep,  ///< kDwPullLinkStep
    Rest,      ///< kDwPullRest
    Input,     ///< per-step inputs: uploaded, never downloaded
    Static,    ///< uploaded with upload_all only
};

/// How a call after the first brings an array up to date with the host.
enum class Sync : std::uint8_t {
    None,      ///< not uploaded: the device's copy is the latest, or the loop writes it first
    Compare,   ///< elements whose bits differ from the device's are uploaded
    Outfalls,  ///< as Compare, at outfall nodes only
    Dense,     ///< uploaded whole at every execute()
    Derived,   ///< the device copies Field::ref's value into it first; then as Compare
    NetInflow, ///< the device forms inflow - outflow into it first; then as Compare
};

/// One array of the block.
struct Field {
    Group group = Group::Static;
    Sync sync = Sync::None;
    std::size_t elem = 0;   ///< bytes per element
    std::size_t count = 0;  ///< elements
    std::size_t offset = 0; ///< byte offset in the block
    int ref = -1;           ///< Sync::Derived: index of the field copied
    const void* (*host)(const step::Arrays&) = nullptr;
    void (*point)(step::Arrays&, unsigned char*) = nullptr;
};

template <auto Member> Field field(Group group, Sync sync, std::size_t count) {
    using Ptr =
        std::remove_cv_t<std::remove_reference_t<decltype(std::declval<step::Arrays&>().*Member)>>;
    using T = std::remove_const_t<std::remove_pointer_t<Ptr>>;
    Field f;
    f.group = group;
    f.sync = sync;
    f.elem = sizeof(T);
    f.count = count;
    f.host = [](const step::Arrays& a) -> const void* { return a.*Member; };
    f.point = [](step::Arrays& a, unsigned char* p) { a.*Member = reinterpret_cast<T*>(p); };
    return f;
}

template <auto Member> Field table(std::size_t count) {
    using Ptr =
        std::remove_cv_t<std::remove_reference_t<decltype(std::declval<step::Tables&>().*Member)>>;
    using T = std::remove_const_t<std::remove_pointer_t<Ptr>>;
    Field f;
    f.elem = sizeof(T);
    f.count = count;
    f.host = [](const step::Arrays& a) -> const void* { return a.tables.*Member; };
    f.point = [](step::Arrays& a, unsigned char* p) { a.tables.*Member = reinterpret_cast<T*>(p); };
    return f;
}

/// One changed element: its byte offset in the block, with kByte set for a 1-byte element or
/// kWord for a 4-byte one, and its new bits.
struct Delta {
    std::uint64_t at;
    std::uint64_t bits;
};
constexpr std::uint64_t kByte = std::uint64_t{1} << 63;
constexpr std::uint64_t kWord = std::uint64_t{1} << 62;

/// Every segment starts on this boundary, so that every element is aligned.
constexpr std::size_t kAlign = 64;

std::size_t aligned(std::size_t n) {
    return (n + kAlign - 1) / kAlign * kAlign;
}

/// The kDwPull* bit of a solver-only group; 0 for the others.
unsigned pullBit(Group g) {
    switch (g) {
    case Group::Converged:
        return kDwPullConverged;
    case Group::LinkStep:
        return kDwPullLinkStep;
    case Group::Rest:
        return kDwPullRest;
    default:
        return 0;
    }
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
    DwCourantResult courantStep(const DwCourantStep& in) override;
    DwDeviceResult pull(const DwDevicePull& request) override;
    const char* label() const noexcept override { return kLabel; }

    // Public only for the reason above; not part of the interface.
    void linkGeometry();
    void momentum(double dt, int iteration);
    void initNodes();
    void gatherNodeFlows();
    int nodeDepths(double dt, int iteration);
    void outfallDepths();
    void bypass();
    void startStep(bool derive, std::size_t deltas);
    void scatter(std::size_t first, std::size_t count);

private:
    int add(Field f);
    void uploadAll(const step::Arrays& h);
    std::size_t stage(const step::Arrays& h, bool inputs);
    void stageField(const Field& f, const void* host, const unsigned char* ref, bool net);
    void download(std::size_t begin, std::size_t end);
    void report() const;

    std::vector<Field> fields_;
    std::size_t public_end_ = 0;    ///< end of the Public group
    std::size_t converged_end_ = 0; ///< end of the Converged group
    int inflow_ = -1, outflow_ = -1;

    Bytes block_;               ///< every device array
    Pinned mirror_;             ///< what the device holds, on the host
    step::Arrays d_{};          ///< every pointer is into block_
    Ints outfall_nodes_;        ///< nodes whose depth a conduit sets, ascending
    int n_outfall_nodes_ = 0;   ///< entries of outfall_nodes_
    std::vector<int> outfalls_; ///< every OUTFALL node, ascending (Sync::Outfalls)

    Pinned stage_;    ///< [node lateral inflows][Delta...], on the host
    Bytes stage_dev_; ///< the same, on the device
    std::vector<Delta> deltas_;

    long steps_ = 0;
    long iterations_ = 0;
    long uploaded_ = 0; ///< changed elements uploaded, all calls
    std::string error_;
};

int KokkosDwMarcher::add(Field f) {
    fields_.push_back(f);
    return static_cast<int>(fields_.size()) - 1;
}

void KokkosDwMarcher::bind(const DwDeviceNetwork& network) {
    using A = step::Arrays;
    const step::Arrays& h = network.host;
    const auto nn = static_cast<std::size_t>(h.n_nodes);
    const auto nc = static_cast<std::size_t>(h.n_conduits);
    const auto ncsr = static_cast<std::size_t>(h.n_csr);
    fields_.clear();
    d_ = step::Arrays{};
    d_.n_nodes = h.n_nodes;
    d_.n_conduits = h.n_conduits;
    d_.n_csr = h.n_csr;
    d_.surcharge_method = h.surcharge_method;

    constexpr Group P = Group::Public, C = Group::Converged, L = Group::LinkStep, R = Group::Rest,
                    I = Group::Input, S = Group::Static;
    constexpr Sync none = Sync::None, cmp = Sync::Compare;

    // Public state. A junction's overflow and every node's inflow and outflow are written by the
    // loop before it reads them (step::junctionDepth, step::initNodeState).
    const int depth = add(field<&A::node_depth>(P, cmp, nn));
    add(field<&A::node_head>(P, cmp, nn));
    const int volume = add(field<&A::node_volume>(P, cmp, nn));
    add(field<&A::node_overflow>(P, Sync::Outfalls, nn));
    add(field<&A::node_held_volume>(P, cmp, nn));
    inflow_ = add(field<&A::node_inflow>(P, none, nn));
    outflow_ = add(field<&A::node_outflow>(P, none, nn));
    const int flow = add(field<&A::link_flow>(P, cmp, nc));
    add(field<&A::link_dqdh>(P, cmp, nc));
    add(field<&A::link_depth>(P, cmp, nc));
    add(field<&A::link_volume>(P, cmp, nc));
    add(field<&A::flow_class>(P, cmp, nc));
    add(field<&A::full_state>(P, cmp, nc));
    add(field<&A::inlet_control>(P, cmp, nc));
    add(field<&A::normal_flow_limited>(P, cmp, nc));

    // Solver-only state: the device's copy is the latest.
    add(field<&A::node_converged>(C, none, nn));
    add(field<&A::froude>(L, none, nc));
    add(field<&A::area_mid>(L, none, nc));
    add(field<&A::node_sumdqdh>(R, none, nn));
    add(field<&A::node_new_surf_area>(R, none, nn));
    add(field<&A::node_old_surf_area>(R, none, nn));
    add(field<&A::node_dydt>(R, none, nn));
    add(field<&A::node_is_surcharged>(R, none, nn));
    add(field<&A::conduit_q1>(R, none, nc));
    add(field<&A::conduit_bypassed>(R, none, nc));
    add(field<&A::depth1>(R, none, nc));
    add(field<&A::depth2>(R, none, nc));
    add(field<&A::depth_mid>(R, none, nc));
    add(field<&A::h1>(R, none, nc));
    add(field<&A::h2>(R, none, nc));
    add(field<&A::fasnh>(R, none, nc));
    add(field<&A::width1>(R, none, nc));
    add(field<&A::width2>(R, none, nc));
    add(field<&A::width_mid>(R, none, nc));
    add(field<&A::surf_area1>(R, none, nc));
    add(field<&A::surf_area2>(R, none, nc));
    add(field<&A::area1>(R, none, nc));
    add(field<&A::area2>(R, none, nc));
    add(field<&A::hrad1>(R, none, nc));
    add(field<&A::hrad_mid>(R, none, nc));
    add(field<&A::wcap_d1>(R, none, nc));
    add(field<&A::wcap_d2>(R, none, nc));
    add(field<&A::wcap_dm>(R, none, nc));
    add(field<&A::new_flow>(R, none, nc));
    add(field<&A::dqdh>(R, none, nc));
    add(field<&A::velocity>(R, none, nc));
    add(field<&A::sigma>(R, none, nc));
    add(field<&A::category>(R, none, nc));
    add(field<&A::area_old>(R, none, nc));

    // Per-step inputs. The old values are the state at the step start (NodeData and LinkData
    // save_hyd_state), which the device forms from its own state (startStep).
    Field f = field<&A::node_old_depth>(I, Sync::Derived, nn);
    f.ref = depth;
    add(f);
    f = field<&A::node_old_volume>(I, Sync::Derived, nn);
    f.ref = volume;
    add(f);
    add(field<&A::node_old_net_inflow>(I, Sync::NetInflow, nn));
    add(field<&A::node_lat_flow>(I, Sync::Dense, nn));
    add(field<&A::node_losses>(I, cmp, nn));
    add(field<&A::conduit_setting>(I, cmp, nc));
    f = field<&A::conduit_old_flow>(I, Sync::Derived, nc);
    f.ref = flow;
    add(f);
    // swmm_node_set_head_boundary makes an outfall FIXED at a new stage during the run.
    add(field<&A::node_outfall_type>(I, Sync::Outfalls, nn));
    add(field<&A::node_outfall_stage>(I, Sync::Outfalls, nn));

    // Static data.
    add(field<&A::node_full_depth>(S, none, nn));
    add(field<&A::node_crown_elev>(S, none, nn));
    add(field<&A::node_ponded_area>(S, none, nn));
    add(field<&A::node_sur_depth>(S, none, nn));
    add(field<&A::node_rpt_full_volume>(S, none, nn));
    add(field<&A::node_degree>(S, none, nn));
    add(field<&A::node_invert>(S, none, nn));
    add(field<&A::node_is_outfall>(S, none, nn));
    add(field<&A::node_outfall_flap>(S, none, nn));
    add(field<&A::node_outfall_link>(S, none, nn));
    add(field<&A::node_outfall_offset>(S, none, nn));
    add(field<&A::csr_row>(S, none, nn + 1));
    add(field<&A::csr_link>(S, none, ncsr));
    add(field<&A::csr_is_n2>(S, none, ncsr));
    add(field<&A::conduit_link>(S, none, nc));
    add(field<&A::conduit_n1>(S, none, nc));
    add(field<&A::conduit_n2>(S, none, nc));
    add(field<&A::conduit_shape>(S, none, nc));
    add(field<&A::conduit_has_offset>(S, none, nc));
    add(field<&A::conduit_offset1>(S, none, nc));
    add(field<&A::conduit_offset2>(S, none, nc));
    add(field<&A::conduit_y_full>(S, none, nc));
    add(field<&A::conduit_a_full>(S, none, nc));
    add(field<&A::conduit_r_full>(S, none, nc));
    add(field<&A::conduit_w_max>(S, none, nc));
    add(field<&A::conduit_s_full>(S, none, nc));
    add(field<&A::conduit_s_max>(S, none, nc));
    add(field<&A::conduit_length>(S, none, nc));
    add(field<&A::conduit_beta>(S, none, nc));
    add(field<&A::conduit_q_max>(S, none, nc));
    add(field<&A::conduit_links_length>(S, none, nc));
    add(field<&A::conduit_mod_length>(S, none, nc));
    add(field<&A::conduit_rough_factor>(S, none, nc));
    add(field<&A::conduit_barrels>(S, none, nc));
    add(field<&A::conduit_is_closed>(S, none, nc));
    add(field<&A::conduit_has_losses>(S, none, nc));
    add(field<&A::conduit_loss_inlet>(S, none, nc));
    add(field<&A::conduit_loss_outlet>(S, none, nc));
    add(field<&A::conduit_loss_avg>(S, none, nc));
    add(field<&A::conduit_q_limit>(S, none, nc));
    add(field<&A::conduit_has_flap_gate>(S, none, nc));
    add(field<&A::conduit_direction>(S, none, nc));

    // The shared geometry tables.
    const step::Tables& t = h.tables;
    using T = step::Tables;
    add(table<&T::A_Circ>(static_cast<std::size_t>(t.N_A_Circ)));
    add(table<&T::R_Circ>(static_cast<std::size_t>(t.N_R_Circ)));
    add(table<&T::W_Circ>(static_cast<std::size_t>(t.N_W_Circ)));
    add(table<&T::S_Circ>(static_cast<std::size_t>(t.N_S_Circ)));
    add(table<&T::Y_Circ>(static_cast<std::size_t>(t.N_Y_Circ)));
    add(table<&T::Amax>(std::size(xsect_tables::Amax)));
    d_.tables.N_A_Circ = t.N_A_Circ;
    d_.tables.N_R_Circ = t.N_R_Circ;
    d_.tables.N_W_Circ = t.N_W_Circ;
    d_.tables.N_S_Circ = t.N_S_Circ;
    d_.tables.N_Y_Circ = t.N_Y_Circ;

    // Lay the fields out in order; the groups are declared in block order.
    std::size_t at = 0;
    for (Field& g : fields_) {
        if (g.sync != Sync::None && g.sync != Sync::Dense && g.elem != 1 && g.elem != 4 &&
            g.elem != 8)
            throw std::logic_error("dynamic wave device: a compared array is not 1, 4 or 8 bytes");
        g.offset = at;
        at = aligned(at + g.elem * g.count);
        if (g.group == Group::Public)
            public_end_ = at;
        if (g.group == Group::Converged)
            converged_end_ = at;
    }
    block_ = Bytes(Kokkos::view_alloc(std::string("dw_block"), Kokkos::WithoutInitializing),
                   at > 0 ? at : 1);
    mirror_ = Pinned(Kokkos::view_alloc(std::string("dw_mirror"), Kokkos::WithoutInitializing),
                     at > 0 ? at : 1);
    for (const Field& g : fields_)
        g.point(d_, block_.data() + g.offset);
}

void KokkosDwMarcher::uploadAll(const step::Arrays& h) {
    unsigned char* m = mirror_.data();
    for (const Field& f : fields_) {
        const void* src = f.host(h);
        const std::size_t bytes = f.elem * f.count;
        if (bytes == 0)
            continue;
        if (src == nullptr)
            throw std::runtime_error("dynamic wave device: a host array is missing");
        std::memcpy(m + f.offset, src, bytes);
    }
    Kokkos::deep_copy(DevBytes(block_.data(), block_.extent(0)),
                      PinnedBytes(mirror_.data(), mirror_.extent(0)));

    // Node lists: the outfalls a conduit sets (the outfall boundary kernel), and every outfall.
    const int nn = d_.n_nodes;
    std::vector<int> linked;
    outfalls_.clear();
    for (int i = 0; i < nn; ++i) {
        if (h.node_outfall_link[i] >= 0)
            linked.push_back(i);
        if (h.node_is_outfall[i])
            outfalls_.push_back(i);
    }
    n_outfall_nodes_ = static_cast<int>(linked.size());
    outfall_nodes_ =
        Ints(Kokkos::view_alloc(std::string("dw_outfall_nodes"), Kokkos::WithoutInitializing),
             linked.empty() ? 1 : linked.size());
    if (!linked.empty())
        Kokkos::deep_copy(Kokkos::View<int*, MemSpace, Kokkos::MemoryUnmanaged>(
                              outfall_nodes_.data(), linked.size()),
                          Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                              linked.data(), linked.size()));
}

void KokkosDwMarcher::stageField(const Field& f, const void* host, const unsigned char* ref,
                                 bool net) {
    unsigned char* m = mirror_.data() + f.offset;
    const auto* src = static_cast<const unsigned char*>(host);
    const std::size_t e = f.elem;
    const auto visit = [&](std::size_t i) {
        const unsigned char* now = src + i * e;
        unsigned char was[8];
        if (net) {
            // The device's old_net_inflow, NodeData::save_hyd_state's expression.
            double in, out;
            std::memcpy(&in, mirror_.data() + fields_[inflow_].offset + i * 8, 8);
            std::memcpy(&out, mirror_.data() + fields_[outflow_].offset + i * 8, 8);
            const double v = in - out;
            std::memcpy(was, &v, 8);
        } else {
            std::memcpy(was, ref + i * e, e);
        }
        if (std::memcmp(now, was, e) == 0)
            return;
        Delta d{(f.offset + i * e) | (e == 1 ? kByte : (e == 4 ? kWord : 0)), 0};
        std::memcpy(&d.bits, now, e);
        deltas_.push_back(d);
        if (!net && ref == m)
            std::memcpy(m + i * e, now, e);
    };
    if (f.sync == Sync::Outfalls) {
        for (int i : outfalls_)
            visit(static_cast<std::size_t>(i));
        return;
    }
    // Whole runs that match are skipped with one memcmp.
    constexpr std::size_t kRun = 256;
    for (std::size_t i0 = 0; i0 < f.count; i0 += kRun) {
        const std::size_t n = (f.count - i0 < kRun) ? f.count - i0 : kRun;
        if (!net && std::memcmp(src + i0 * e, ref + i0 * e, n * e) == 0)
            continue;
        for (std::size_t i = i0; i < i0 + n; ++i)
            visit(i);
    }
}

std::size_t KokkosDwMarcher::stage(const step::Arrays& h, bool inputs) {
    deltas_.clear();
    const unsigned char* m = mirror_.data();
    // The old values first: they compare with the device's state before this call changes it.
    if (inputs)
        for (const Field& f : fields_) {
            if (f.sync == Sync::Derived)
                stageField(f, f.host(h), m + fields_[f.ref].offset, false);
            else if (f.sync == Sync::NetInflow)
                stageField(f, f.host(h), nullptr, true);
        }
    for (const Field& f : fields_) {
        if (!inputs && f.group == Group::Input)
            continue;
        if (f.sync == Sync::Compare || f.sync == Sync::Outfalls)
            stageField(f, f.host(h), m + f.offset, false);
    }

    const std::size_t lat = inputs ? static_cast<std::size_t>(d_.n_nodes) * sizeof(double) : 0;
    const std::size_t bytes = lat + deltas_.size() * sizeof(Delta);
    if (bytes == 0)
        return 0;
    if (stage_.extent(0) < bytes) {
        const std::size_t cap = bytes + bytes / 2;
        stage_ =
            Pinned(Kokkos::view_alloc(std::string("dw_stage"), Kokkos::WithoutInitializing), cap);
        stage_dev_ = Bytes(
            Kokkos::view_alloc(std::string("dw_stage_dev"), Kokkos::WithoutInitializing), cap);
    }
    if (lat > 0)
        std::memcpy(stage_.data(), h.node_lat_flow, lat);
    if (!deltas_.empty())
        std::memcpy(stage_.data() + lat, deltas_.data(), deltas_.size() * sizeof(Delta));
    Kokkos::deep_copy(DevBytes(stage_dev_.data(), bytes), PinnedBytes(stage_.data(), bytes));
    uploaded_ += static_cast<long>(deltas_.size());
    return deltas_.size();
}

void KokkosDwMarcher::startStep(bool derive, std::size_t deltas) {
    const step::Arrays D = d_;
    // DWSolver::execute's step start (area_old, bypassed) and, after the first call, the old
    // values of NodeData and LinkData save_hyd_state and the lateral inflows.
    if (derive) {
        const auto* lat = reinterpret_cast<const double*>(stage_dev_.data());
        double* old_depth = const_cast<double*>(D.node_old_depth);
        double* old_volume = const_cast<double*>(D.node_old_volume);
        double* old_net = const_cast<double*>(D.node_old_net_inflow);
        double* lat_flow = const_cast<double*>(D.node_lat_flow);
        Kokkos::parallel_for(
            "dw_start_nodes", Range(0, D.n_nodes), KOKKOS_LAMBDA(const int i) {
                old_depth[i] = D.node_depth[i];
                old_volume[i] = D.node_volume[i];
                old_net[i] = D.node_inflow[i] - D.node_outflow[i];
                lat_flow[i] = lat[i];
            });
    }
    double* area_old = const_cast<double*>(D.area_old);
    double* old_flow = const_cast<double*>(D.conduit_old_flow);
    Kokkos::parallel_for(
        "dw_start_conduits", Range(0, D.n_conduits), KOKKOS_LAMBDA(const int c) {
            area_old[c] = D.area_mid[c];
            D.conduit_bypassed[c] = 0;
            if (derive)
                old_flow[c] = D.link_flow[c];
        });
    if (deltas > 0)
        scatter(derive ? static_cast<std::size_t>(D.n_nodes) * sizeof(double) : 0, deltas);
}

void KokkosDwMarcher::scatter(std::size_t first, std::size_t count) {
    unsigned char* const block = block_.data();
    const auto* list = reinterpret_cast<const Delta*>(stage_dev_.data() + first);
    Kokkos::parallel_for(
        "dw_scatter", Range(0, static_cast<int>(count)), KOKKOS_LAMBDA(const int k) {
            const Delta d = list[k];
            unsigned char* p = block + (d.at & ~(kByte | kWord));
            if (d.at & kByte)
                *p = static_cast<unsigned char>(d.bits);
            else if (d.at & kWord)
                *reinterpret_cast<std::uint32_t*>(p) = static_cast<std::uint32_t>(d.bits);
            else
                *reinterpret_cast<std::uint64_t*>(p) = d.bits;
        });
}

void KokkosDwMarcher::download(std::size_t begin, std::size_t end) {
    if (end > begin)
        Kokkos::deep_copy(PinnedBytes(mirror_.data() + begin, end - begin),
                          DevBytes(block_.data() + begin, end - begin));
}

void KokkosDwMarcher::linkGeometry() {
    const step::Arrays D = d_;
    Kokkos::parallel_for(
        "dw_link_geometry", Range(0, D.n_conduits),
        KOKKOS_LAMBDA(const int c) { step::linkGeometry(D, c); });
}

void KokkosDwMarcher::initNodes() {
    const step::Arrays D = d_;
    Kokkos::parallel_for(
        "dw_init_nodes", Range(0, D.n_nodes),
        KOKKOS_LAMBDA(const int i) { step::initNodeState(D, i); });
}

void KokkosDwMarcher::gatherNodeFlows() {
    const step::Arrays D = d_;
    Kokkos::parallel_for(
        "dw_gather_node_flows", Range(0, D.n_nodes),
        KOKKOS_LAMBDA(const int n) { step::gatherNodeFlows(D, n); });
}

int KokkosDwMarcher::nodeDepths(double dt, int iteration) {
    const step::Arrays D = d_;
    int unconverged = 0;
    // An integer count: the reduction order does not change it.
    Kokkos::parallel_reduce(
        "dw_node_depths", Range(0, D.n_nodes),
        KOKKOS_LAMBDA(const int i, int& sum) { sum += step::nodeDepth(D, i, dt, iteration); },
        unconverged);
    return unconverged;
}

void KokkosDwMarcher::outfallDepths() {
    const step::Arrays D = d_;
    const int* const nodes = outfall_nodes_.data();
    Kokkos::parallel_for(
        "dw_outfall_depths", Range(0, n_outfall_nodes_),
        KOKKOS_LAMBDA(const int k) { step::outfallDepth(D, nodes[k]); });
}

void KokkosDwMarcher::bypass() {
    const step::Arrays D = d_;
    Kokkos::parallel_for(
        "dw_bypass", Range(0, D.n_conduits),
        KOKKOS_LAMBDA(const int c) { step::bypassLink(D, c); });
}

void KokkosDwMarcher::momentum(double dt, int iteration) {
    const step::Arrays D = d_;
    Kokkos::parallel_for(
        "dw_momentum", Range(0, D.n_conduits),
        KOKKOS_LAMBDA(const int c) { step::conduitMomentum(D, c, dt, iteration); });
}

DwDeviceResult KokkosDwMarcher::execute(const DwDeviceStep& step) {
    DwDeviceResult r;
    try {
        const step::Arrays& h = step.host;
        if (step.upload_all) {
            uploadAll(h);
            startStep(false, 0);
        } else {
            startStep(true, stage(h, true));
        }
        d_.omega = h.omega;
        d_.normal_flow_ltd = h.normal_flow_ltd;
        d_.inertial_damping = h.inertial_damping;
        d_.head_tol = h.head_tol;
        d_.min_surf_area = h.min_surf_area;
        d_.allow_ponding = h.allow_ponding;

        // DWSolver::execute's loop control (DynamicWave.cpp:1567-1687).
        int iterations = 0;
        bool converged = false;
        while (iterations < step.max_trials) {
            initNodes();
            linkGeometry();
            momentum(step.dt, iterations);
            gatherNodeFlows();
            // The structure callback and DWSolver::limitConduitLosses do nothing in scope.
            outfallDepths();
            const int unconverged = nodeDepths(step.dt, iterations);
            converged = (unconverged == 0);
            ++iterations;
            if (iterations > 1) {
                if (converged)
                    break;
                bypass();
            }
        }

        // The public state, and the convergence flags the host counts when the step did not
        // converge, in one copy.
        download(0, converged ? public_end_ : converged_end_);
        const unsigned char* m = mirror_.data();
        for (const Field& f : fields_) {
            if (f.count == 0 ||
                (f.group != Group::Public && (converged || f.group != Group::Converged)))
                continue;
            std::memcpy(const_cast<void*>(f.host(h)), m + f.offset, f.elem * f.count);
        }
        r.pulled = converged ? 0u : kDwPullConverged;
        r.iterations = iterations;
        r.converged = converged;
        ++steps_;
        iterations_ += iterations;
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

DwDeviceResult KokkosDwMarcher::pull(const DwDevicePull& request) {
    DwDeviceResult r;
    try {
        // The solver-only groups are contiguous; copy the span from the first wanted to the last.
        std::size_t begin = 0, end = 0;
        bool any = false;
        for (const Field& f : fields_) {
            if ((pullBit(f.group) & request.groups) == 0)
                continue;
            if (!any)
                begin = f.offset;
            any = true;
            end = f.offset + f.elem * f.count;
        }
        download(begin, end);
        const unsigned char* m = mirror_.data();
        for (const Field& f : fields_) {
            const unsigned bit = pullBit(f.group);
            if (bit == 0 || f.offset < begin || f.offset >= end || f.count == 0)
                continue;
            std::memcpy(const_cast<void*>(f.host(request.host)), m + f.offset, f.elem * f.count);
            r.pulled |= bit;
        }
    } catch (const std::exception& e) {
        error_ = e.what();
        r.ok = false;
        r.error = error_.c_str();
    } catch (...) {
        error_ = "unknown failure in the device state download";
        r.ok = false;
        r.error = error_.c_str();
    }
    return r;
}

DwCourantResult KokkosDwMarcher::courantStep(const DwCourantStep& in) {
    DwCourantResult r;
    try {
        // The host may have changed the state since the last execute().
        if (in.upload_all)
            uploadAll(in.host);
        else if (const std::size_t n = stage(in.host, false); n > 0)
            scatter(0, n);
        const step::Arrays D = d_;
        const double fixed = in.fixed_step;
        const double cf = in.courant_factor;
        const int n = D.n_conduits + D.n_nodes;

        double dt = fixed;
        Kokkos::parallel_reduce(
            "dw_courant_step", Range(0, n),
            KOKKOS_LAMBDA(const int k, double& lmin) {
                const double v = step::courantCandidate(D, k, fixed, cf);
                if (v < lmin)
                    lmin = v;
            },
            Kokkos::Min<double>(dt));

        int first = n;
        if (dt < fixed) {
            Kokkos::parallel_reduce(
                "dw_courant_element", Range(0, n),
                KOKKOS_LAMBDA(const int k, int& lfirst) {
                    if (k < lfirst && step::courantCandidate(D, k, fixed, cf) == dt)
                        lfirst = k;
                },
                Kokkos::Min<int>(first));
        }
        r.dt = dt;
        if (first < D.n_conduits)
            r.link = first;
        else if (first < n)
            r.node = first - D.n_conduits;
    } catch (const std::exception& e) {
        error_ = e.what();
        r.ok = false;
        r.error = error_.c_str();
    } catch (...) {
        error_ = "unknown failure in the device Courant step";
        r.ok = false;
        r.error = error_.c_str();
    }
    return r;
}

void KokkosDwMarcher::report() const {
    if (steps_ > 0)
        std::fprintf(stderr,
                     "[openswmm DW] %s marcher: %ld routing steps, %ld Picard iterations, %ld "
                     "changed elements uploaded\n",
                     kLabel, steps_, iterations_, uploaded_);
}

IDwDeviceMarcher* makeKokkosDwMarcher() {
    return new KokkosDwMarcher();
}

} // namespace openswmm::dynwave::gpu
