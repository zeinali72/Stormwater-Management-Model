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
 * @file      test_dw_kernels_geometry.cpp
 * @brief     The dynamic-wave link geometry kernels (hydraulics/dw/DwKernels.hpp) reproduce
 *            DWSolver::computeLinkGeometry bit for bit, at every Picard iteration of real runs.
 *
 * @details The engine runs each deck at one thread with DWSolver::stage_probe set. On entry to
 *          computeLinkGeometry the probe copies everything the stage reads (node depths, the last
 *          per-barrel flows, the bypass flags) and everything it may hold (the solver's current
 *          geometry arrays) into a step::Arrays, fills the output slots of every live conduit
 *          with a sentinel, and runs step::linkGeometry for every conduit on the host. On return
 *          it compares all twenty outputs of every conduit with the solver's arrays as 64-bit
 *          patterns. The sentinel makes a kernel that forgets a write fail even where the CPU
 *          value did not change; it goes only where the CPU writes (the capped depths
 *          wcap_d1/d2/dm are written under EXTRAN alone, so under SLOT they are copied in).
 *
 *          Decks: a few of HydroCouple's own models (kRepositoryDecks) and the small networks in
 *          data/dw_kernels_geometry/, written for this test for what those models do not reach
 *          (a FIXED outfall, reverse flow over a drop, both shapes, US and SI). Every deck is
 *          inside the device path's scope (DYNWAVE, junctions and FREE/NORMAL/FIXED outfalls,
 *          conduits of CIRCULAR or RECT_CLOSED section with one barrel, none of the excluded
 *          features), which the test checks. Each runs once under EXTRAN and once under SLOT.
 *          The test reports the comparison count and per-condition counts (flow classes, drops,
 *          FIXED-outfall backwater, full pipes, surcharge, flooding) and asserts that every
 *          condition, and every flow class but the unreachable DN_DRY, is reached under both
 *          methods. Every file a run writes lands in data/dw_kernels_geometry_out/.
 *
 * @see hydraulics/dw/DwKernels.hpp, hydraulics/DynamicWave.cpp
 * @ingroup engine_hydraulics
 *
 * @author    Farzan Zeinali
 * @license   Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <openswmm/engine/openswmm_engine.h>

#include "core/SWMMEngine.hpp"
#include "core/SimulationContext.hpp"
#include "hydraulics/DynamicWave.hpp"
#include "hydraulics/dw/DwKernels.hpp"

namespace fs = std::filesystem;

namespace {

namespace step = openswmm::dynwave::step;
using openswmm::FlowClass;
using openswmm::SimulationContext;

const fs::path kRepoRoot = OPENSWMM_DWK_REPO_ROOT;
const fs::path kDeckDir = OPENSWMM_DWK_DECK_DIR;
const fs::path kOutDir = OPENSWMM_DWK_OUT_DIR;

/// HydroCouple's own in-scope models the test runs beside its decks, from the repository root.
const char* const kRepositoryDecks[] = {
    "tests/regression/data/slot_surcharge.inp",    // surcharge, flooding, the slot (US)
    "python/tests/legacy/output/b8_aq_probe.inp",  // legacy_small's network for a day: drops,
                                                   // critical and dry ends (US)
    "tests/parity/corpus/decks/si_dw_quality.inp", // free-surface flow in LPS (SI)
};

constexpr int kFields = 20;
/// First of the crown-capped depths in kFieldNames, which the CPU writes only under EXTRAN.
constexpr int kFirstWcap = 16;
const char* const kFieldNames[kFields] = {
    "depth1", "depth2",    "depth_mid",  "h1",         "h2",      "fasnh",     "width1",
    "width2", "width_mid", "surf_area1", "surf_area2", "area1",   "area2",     "area_mid",
    "hrad1",  "hrad_mid",  "wcap_d1",    "wcap_d2",    "wcap_dm", "flow_class"};

/// The bit pattern written into every live output slot before the kernel runs (a quiet NaN).
constexpr std::uint64_t kSentinelBits = 0x7FF8DEADBEEFCAFEull;

std::uint64_t bitsOf(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

double sentinel() {
    double v;
    std::memcpy(&v, &kSentinelBits, sizeof v);
    return v;
}

std::string hexFloat(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a (%016llx)", v, static_cast<unsigned long long>(bitsOf(v)));
    return buf;
}

/// Conditions a run passed through, counted over live (not bypassed) conduit evaluations.
struct Tally {
    std::uint64_t comparisons = 0; ///< individual 64-bit field comparisons
    std::uint64_t evaluations = 0; ///< live conduit evaluations
    std::uint64_t held = 0;        ///< bypassed conduit evaluations (values held)
    std::uint64_t reslotted = 0;   ///< held widths the SLOT override changed
    std::uint64_t classes[7] = {}; ///< by FlowClass code
    std::uint64_t drops = 0;       ///< offset conduit at a critical end or between normal/critical
    std::uint64_t backwater = 0;   ///< wet outlet end below a FIXED outfall's stage
    std::uint64_t backwater_upstream = 0; ///< the same, for a conduit not joined to that outfall
    std::uint64_t full = 0;               ///< both ends at or above full depth
    std::uint64_t above_crown = 0;        ///< a node head above the crown at a conduit end
    std::uint64_t slot_active = 0;        ///< SLOT: midpoint depth above full depth
    std::uint64_t flooding = 0;           ///< junction-iterations with overflow > 0
    long steps = 0;
    long iterations = 0;

    void add(const Tally& o) {
        comparisons += o.comparisons;
        evaluations += o.evaluations;
        held += o.held;
        reslotted += o.reslotted;
        for (int i = 0; i < 7; ++i)
            classes[i] += o.classes[i];
        drops += o.drops;
        backwater += o.backwater;
        backwater_upstream += o.backwater_upstream;
        full += o.full;
        above_crown += o.above_crown;
        slot_active += o.slot_active;
        flooding += o.flooding;
        steps += o.steps;
        iterations += o.iterations;
    }
};

/// State of one engine run under the probe.
struct ProbedRun {
    std::string deck;
    int method = 0; ///< expected surcharge method
    Tally tally;
    std::string error; ///< a harness precondition that failed
    std::string first_mismatch;
    std::uint64_t mismatches = 0;

    // ---- static arrays, built at the first probe call ----
    bool built = false;
    int n_nodes = 0;
    int n_conduits = 0;
    std::vector<double> node_invert;
    std::vector<std::uint8_t> node_is_outfall;
    std::vector<std::uint8_t> node_is_junction;
    std::vector<int> link, n1, n2, shape;
    std::vector<std::uint8_t> has_offset;
    std::vector<double> off1, off2, y_full, a_full, r_full, w_max, s_full, s_max, length, beta,
        q_max;
    std::vector<double> fixed_stage;       ///< per conduit: highest FIXED stage elevation (or -inf)
    std::vector<std::uint8_t> joins_fixed; ///< conduit's downstream node is a FIXED outfall
    std::vector<std::string> conduit_name;

    // ---- per-iteration arrays ----
    std::vector<double> node_depth, q1;
    std::vector<std::uint8_t> bypassed;
    std::array<std::vector<double>, kFields - 1> out;
    std::vector<std::int8_t> flow_class;
    step::Arrays A;
};

ProbedRun* g_run = nullptr;

} // namespace

namespace openswmm::dynwave {

/// Reads DWSolver's private state for the harness (declared a friend in DynamicWave.hpp).
struct DwStageProbe {
    /// The solver arrays the geometry stage writes, in kFieldNames order (flow_class apart).
    static std::array<const std::vector<double>*, kFields - 1> solverFields(const DWSolver& s) {
        return {&s.depth1_,     &s.depth2_,  &s.depth_mid_, &s.h1_,        &s.h2_,
                &s.fasnh_,      &s.width1_,  &s.width2_,    &s.width_mid_, &s.surf_area1_,
                &s.surf_area2_, &s.area1_,   &s.area2_,     &s.area_mid_,  &s.hrad1_,
                &s.hrad_mid_,   &s.wcap_d1_, &s.wcap_d2_,   &s.wcap_dm_};
    }

    static void build(ProbedRun& r, const DWSolver& s, const SimulationContext& ctx) {
        const auto& nodes = ctx.nodes;
        const auto& links = ctx.links;
        const auto& CD = ctx.link_subtypes.conduits;
        r.n_nodes = ctx.n_nodes();
        r.n_conduits = s.n_conduits_;
        const auto nn = static_cast<std::size_t>(r.n_nodes);
        const auto nc = static_cast<std::size_t>(r.n_conduits);

        double max_fixed = -HUGE_VAL;
        r.node_invert.assign(nn, 0.0);
        r.node_is_outfall.assign(nn, 0);
        r.node_is_junction.assign(nn, 0);
        std::vector<std::uint8_t> node_is_fixed(nn, 0);
        for (std::size_t i = 0; i < nn; ++i) {
            r.node_invert[i] = nodes.invert_elev[i];
            r.node_is_outfall[i] = nodes.type[i] == NodeType::OUTFALL ? 1 : 0;
            r.node_is_junction[i] = nodes.type[i] == NodeType::JUNCTION ? 1 : 0;
            const int row = ctx.node_subtypes.outfall_row(static_cast<int>(i));
            if (row >= 0 && ctx.node_subtypes.outfalls.bc_type[static_cast<std::size_t>(row)] ==
                                OutfallType::FIXED) {
                node_is_fixed[i] = 1;
                max_fixed = std::max(
                    max_fixed, ctx.node_subtypes.outfalls.param[static_cast<std::size_t>(row)]);
            }
        }

        for (auto* v : {&r.n1, &r.n2, &r.link, &r.shape})
            v->assign(nc, 0);
        for (auto* v : {&r.off1, &r.off2, &r.y_full, &r.a_full, &r.r_full, &r.w_max, &r.s_full,
                        &r.s_max, &r.length, &r.beta, &r.q_max, &r.fixed_stage})
            v->assign(nc, 0.0);
        r.has_offset.assign(nc, 0);
        r.joins_fixed.assign(nc, 0);
        r.conduit_name.assign(nc, "");
        for (std::size_t c = 0; c < nc; ++c) {
            const int j = s.conduit_idx_[c];
            const auto uj = static_cast<std::size_t>(j);
            const auto ucr = static_cast<std::size_t>(ctx.link_subtypes.conduit_row(j));
            r.link[c] = j;
            r.n1[c] = links.node1[uj];
            r.n2[c] = links.node2[uj];
            r.shape[c] = links.xsect_batch_shape[uj];
            r.off1[c] = links.offset1[uj];
            r.off2[c] = links.offset2[uj];
            r.has_offset[c] = (links.offset1[uj] > 0.0 || links.offset2[uj] > 0.0) ? 1 : 0;
            r.y_full[c] = links.xsect_y_full[uj];
            r.a_full[c] = links.xsect_a_full[uj];
            r.r_full[c] = links.xsect_r_full[uj];
            r.w_max[c] = links.xsect_w_max[uj];
            r.s_full[c] = links.xsect_s_full[uj];
            r.s_max[c] = links.xsect_s_max[uj];
            // DWSolver::init: modified length, else the authored length.
            double len = CD.mod_length[ucr];
            if (len <= 0.0)
                len = CD.length[ucr];
            r.length[c] = len;
            r.beta[c] = CD.beta[ucr];
            r.q_max[c] = CD.q_max[ucr];
            r.fixed_stage[c] = max_fixed;
            r.joins_fixed[c] = node_is_fixed[static_cast<std::size_t>(r.n2[c])];
            r.conduit_name[c] = ctx.link_names.name_of(j);
        }

        r.node_depth.assign(nn, 0.0);
        r.q1.assign(nc, 0.0);
        r.bypassed.assign(nc, 0);
        for (auto& v : r.out)
            v.assign(nc, 0.0);
        r.flow_class.assign(nc, 0);

        step::Arrays& A = r.A;
        A.n_nodes = r.n_nodes;
        A.n_conduits = r.n_conduits;
        A.surcharge_method = static_cast<int>(s.surcharge_method);
        A.node_invert = r.node_invert.data();
        A.node_is_outfall = r.node_is_outfall.data();
        A.conduit_link = r.link.data();
        A.conduit_n1 = r.n1.data();
        A.conduit_n2 = r.n2.data();
        A.conduit_shape = r.shape.data();
        A.conduit_has_offset = r.has_offset.data();
        A.conduit_offset1 = r.off1.data();
        A.conduit_offset2 = r.off2.data();
        A.conduit_y_full = r.y_full.data();
        A.conduit_a_full = r.a_full.data();
        A.conduit_r_full = r.r_full.data();
        A.conduit_w_max = r.w_max.data();
        A.conduit_s_full = r.s_full.data();
        A.conduit_s_max = r.s_max.data();
        A.conduit_length = r.length.data();
        A.conduit_beta = r.beta.data();
        A.conduit_q_max = r.q_max.data();
        A.node_depth = r.node_depth.data();
        A.conduit_q1 = r.q1.data();
        A.conduit_bypassed = r.bypassed.data();
        double** outs[kFields - 1] = {
            &A.depth1, &A.depth2,    &A.depth_mid,  &A.h1,         &A.h2,     &A.fasnh, &A.width1,
            &A.width2, &A.width_mid, &A.surf_area1, &A.surf_area2, &A.area1,  &A.area2, &A.area_mid,
            &A.hrad1,  &A.hrad_mid,  &A.wcap_d1,    &A.wcap_d2,    &A.wcap_dm};
        for (int f = 0; f < kFields - 1; ++f)
            *outs[f] = r.out[static_cast<std::size_t>(f)].data();
        A.flow_class = r.flow_class.data();
        A.tables = step::hostTables();
        r.built = true;
    }

    /// Point 0: snapshot what the stage reads and holds, then run the kernels.
    static void before(ProbedRun& r, const DWSolver& s, const SimulationContext& ctx,
                       int iteration) {
        if (iteration == 0)
            ++r.tally.steps;
        ++r.tally.iterations;
        const auto fields = solverFields(s);
        const auto nc = static_cast<std::size_t>(r.n_conduits);
        for (std::size_t i = 0; i < static_cast<std::size_t>(r.n_nodes); ++i) {
            r.node_depth[i] = ctx.nodes.depth[i];
            if (r.node_is_junction[i] && ctx.nodes.overflow[i] > 0.0)
                ++r.tally.flooding;
        }
        const double poison = sentinel();
        const bool extran = r.method == step::kSurchargeExtran;
        for (std::size_t c = 0; c < nc; ++c) {
            const auto j = static_cast<std::size_t>(r.link[c]);
            r.q1[c] = s.q1_[j];
            r.bypassed[c] = s.bypassed_[j];
            const bool live = r.bypassed[c] == 0;
            for (int f = 0; f < kFields - 1; ++f) {
                const bool written = live && (f < kFirstWcap || extran);
                r.out[static_cast<std::size_t>(f)][c] = written ? poison : (*fields[f])[j];
            }
            r.flow_class[c] =
                live ? std::int8_t{-1} : static_cast<std::int8_t>(ctx.links.flow_class[j]);
        }
        for (int c = 0; c < r.n_conduits; ++c) {
            const auto uc = static_cast<std::size_t>(c);
            const double w[3] = {r.out[6][uc], r.out[7][uc], r.out[8][uc]};
            step::linkGeometry(r.A, c);
            if (r.bypassed[uc] &&
                (bitsOf(w[0]) != bitsOf(r.out[6][uc]) || bitsOf(w[1]) != bitsOf(r.out[7][uc]) ||
                 bitsOf(w[2]) != bitsOf(r.out[8][uc])))
                ++r.tally.reslotted;
        }
    }

    /// Point 1: compare every output with the solver's, then count the conditions reached.
    static void after(ProbedRun& r, const DWSolver& s, const SimulationContext& ctx,
                      int iteration) {
        const auto fields = solverFields(s);
        const auto nc = static_cast<std::size_t>(r.n_conduits);
        for (std::size_t c = 0; c < nc; ++c) {
            const auto j = static_cast<std::size_t>(r.link[c]);
            for (int f = 0; f < kFields; ++f) {
                double cpu, ker;
                if (f < kFields - 1) {
                    cpu = (*fields[f])[j];
                    ker = r.out[static_cast<std::size_t>(f)][c];
                } else {
                    cpu = static_cast<double>(static_cast<int>(ctx.links.flow_class[j]));
                    ker = static_cast<double>(r.flow_class[c]);
                }
                ++r.tally.comparisons;
                if (bitsOf(cpu) == bitsOf(ker))
                    continue;
                if (r.mismatches++ == 0) {
                    std::ostringstream m;
                    m << "deck " << r.deck << " (" << (r.method ? "SLOT" : "EXTRAN")
                      << "), routing step " << r.tally.steps << ", Picard iteration " << iteration
                      << ", conduit " << r.conduit_name[c] << (r.bypassed[c] ? " (bypassed)" : "")
                      << ", field " << kFieldNames[f] << ": cpu " << hexFloat(cpu) << " kernel "
                      << hexFloat(ker);
                    r.first_mismatch = m.str();
                }
            }

            if (r.bypassed[c]) {
                ++r.tally.held;
                continue;
            }
            ++r.tally.evaluations;
            const int fc = static_cast<int>(ctx.links.flow_class[j]);
            if (fc >= 0 && fc < 7)
                ++r.tally.classes[fc];
            const double yf = r.y_full[c];
            const double d1 = s.depth1_[j], d2 = s.depth2_[j], dm = s.depth_mid_[j];
            const double z1 = r.node_invert[static_cast<std::size_t>(r.n1[c])] + r.off1[c];
            const double z2 = r.node_invert[static_cast<std::size_t>(r.n2[c])] + r.off2[c];
            if (r.has_offset[c] &&
                (fc == static_cast<int>(FlowClass::UP_CRITICAL) ||
                 fc == static_cast<int>(FlowClass::DN_CRITICAL) || s.fasnh_[j] < 1.0))
                ++r.tally.drops;
            if (z2 < r.fixed_stage[c] && s.h2_[j] > z2 + constants::FUDGE) {
                ++r.tally.backwater;
                if (!r.joins_fixed[c])
                    ++r.tally.backwater_upstream;
            }
            if (d1 >= yf && d2 >= yf)
                ++r.tally.full;
            if (s.h1_[j] - z1 > yf || s.h2_[j] - z2 > yf)
                ++r.tally.above_crown;
            if (r.method == step::kSurchargeSlot && dm > yf)
                ++r.tally.slot_active;
        }
    }

    static void probe(const DWSolver& s, const SimulationContext& ctx, int iteration, int point) {
        ProbedRun* r = g_run;
        if (!r || !r->error.empty())
            return;
        if (!r->built) {
            if (s.num_threads_ != 1) {
                r->error = "the solver runs " + std::to_string(s.num_threads_) + " threads";
                return;
            }
            if (static_cast<int>(s.surcharge_method) != r->method) {
                r->error = "the solver runs surcharge method " +
                           std::to_string(static_cast<int>(s.surcharge_method));
                return;
            }
            build(*r, s, ctx);
        }
        if (point == 0)
            before(*r, s, ctx, iteration);
        else
            after(*r, s, ctx, iteration);
    }
};

} // namespace openswmm::dynwave

namespace {

using openswmm::dynwave::DwStageProbe;

/// Clears the probe even when an assertion leaves the test early.
struct ProbeGuard {
    explicit ProbeGuard(ProbedRun* r) {
        g_run = r;
        openswmm::dynwave::DWSolver::stage_probe = &DwStageProbe::probe;
    }
    ~ProbeGuard() {
        openswmm::dynwave::DWSolver::stage_probe = nullptr;
        g_run = nullptr;
    }
    ProbeGuard(const ProbeGuard&) = delete;
    ProbeGuard& operator=(const ProbeGuard&) = delete;
};

/// An engine handle that is always ended, closed and destroyed.
struct Engine {
    SWMM_Engine e = swmm_engine_create();
    bool started = false;
    ~Engine() {
        if (!e)
            return;
        if (started)
            swmm_engine_end(e);
        swmm_engine_close(e);
        swmm_engine_destroy(e);
    }
    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    openswmm::SWMMEngine& eng() { return *static_cast<openswmm::SWMMEngine*>(e); }
};

bool open(Engine& en, const fs::path& deck, const std::string& tag) {
    if (!en.e)
        return false;
    const fs::path rpt = kOutDir / (tag + ".rpt");
    const fs::path out = kOutDir / (tag + ".out");
    return swmm_engine_open(en.e, deck.string().c_str(), rpt.string().c_str(), out.string().c_str(),
                            nullptr) == SWMM_OK;
}

/**
 * @brief The device path's scope, checked on an opened (and, for the 2D test, initialized) deck.
 *
 * @param[in] ctx Parsed context.
 * @return Empty when in scope, else the first reason it is not.
 */
std::string scopeReason(const SimulationContext& ctx) {
    using namespace openswmm;
    const auto& o = ctx.options;
    if (o.routing_model != RoutingModel::DYNWAVE)
        return "FLOW_ROUTING is not DYNWAVE";
    if (o.ignore_routing)
        return "IGNORE_ROUTING";
    if (o.surcharge_method != 0 && o.surcharge_method != 1)
        return "SURCHARGE_METHOD is not EXTRAN or SLOT";
    if (o.anderson_accel)
        return "Anderson acceleration";
    if (o.node_continuity != NodeContinuity::EXPLICIT)
        return "NODE_CONTINUITY is not EXPLICIT";
    if (o.unsteady_friction != 0)
        return "unsteady friction";
    if (ctx.files.has_any())
        return "[FILES] interface or hot start files";
    if (ctx.restart.has_any())
        return "[RESTART] source or record files";
    for (int i = 0; i < ctx.n_nodes(); ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const NodeType t = ctx.nodes.type[ui];
        if (t != NodeType::JUNCTION && t != NodeType::OUTFALL)
            return "node " + ctx.node_names.name_of(i) + " is not a junction or outfall";
        if (ui < ctx.nodes.is_virtual.size() && ctx.nodes.is_virtual[ui])
            return "virtual junction " + ctx.node_names.name_of(i);
        if (ctx.node_subtypes.storage_row(i) >= 0)
            return "node " + ctx.node_names.name_of(i) + " has a storage row (LID node)";
        const int row = ctx.node_subtypes.outfall_row(i);
        if (t == NodeType::OUTFALL && row >= 0) {
            const OutfallType bc =
                ctx.node_subtypes.outfalls.bc_type[static_cast<std::size_t>(row)];
            if (bc != OutfallType::FREE && bc != OutfallType::NORMAL && bc != OutfallType::FIXED)
                return "outfall " + ctx.node_names.name_of(i) + " is TIDAL or TIMESERIES";
        }
        if (ui < ctx.coupled_node.size() && ctx.coupled_node[ui])
            return "2D-coupled node " + ctx.node_names.name_of(i);
    }
    if (ctx.inlet_usages.count() > 0)
        return "street inlets";
    if (!ctx.surface_outfall_link_limit.empty())
        return "2D surface outfall coupling";
    int conduits = 0;
    const auto& CD = ctx.link_subtypes.conduits;
    for (int j = 0; j < ctx.n_links(); ++j) {
        const auto uj = static_cast<std::size_t>(j);
        if (ctx.links.type[uj] != LinkType::CONDUIT)
            return "link " + ctx.link_names.name_of(j) + " is not a conduit";
        const XsectShape sh = ctx.links.xsect_shape[uj];
        if (sh != XsectShape::CIRCULAR && sh != XsectShape::RECT_CLOSED)
            return "conduit " + ctx.link_names.name_of(j) + " is not CIRCULAR or RECT_CLOSED";
        const int row = ctx.link_subtypes.conduit_row(j);
        if (row < 0)
            return "conduit " + ctx.link_names.name_of(j) + " has no conduit row";
        const auto ur = static_cast<std::size_t>(row);
        if (CD.barrels[ur] != 1)
            return "conduit " + ctx.link_names.name_of(j) + " has more than one barrel";
        if (CD.culvert_code[ur] != 0)
            return "conduit " + ctx.link_names.name_of(j) + " is a culvert";
        if (CD.seep_rate[ur] != 0.0)
            return "conduit " + ctx.link_names.name_of(j) + " seeps";
        if (ur < CD.gw_coupled.size() && CD.gw_coupled[ur])
            return "conduit " + ctx.link_names.name_of(j) + " is aquifer-coupled";
        ++conduits;
    }
    if (conduits == 0)
        return "no conduits";
    return {};
}

/**
 * @brief Open and initialize a deck to decide whether it is in scope.
 *
 * @param[in] deck Deck path.
 * @param[out] si True when the deck's flow units are SI.
 * @return Empty when in scope; otherwise "open failed: ...", "initialize failed: ..." or the scope
 *     reason.
 */
std::string checkScope(const fs::path& deck, bool& si) {
    Engine en;
    const std::string tag = "scope__" + deck.stem().string();
    if (!open(en, deck, tag))
        return "open failed";
    SimulationContext& ctx = en.eng().context();
    si = static_cast<int>(ctx.options.flow_units) >= static_cast<int>(openswmm::FlowUnits::CMS);
    std::string why = scopeReason(ctx);
    if (!why.empty())
        return why;
    ctx.options.num_threads = 1;
    if (swmm_engine_initialize(en.e) != SWMM_OK)
        return "initialize failed";
    return scopeReason(en.eng().context()); // 2D coupling is resolved at initialize
}

/**
 * @brief Run one deck under the probe and the given surcharge method.
 *
 * @param[in] deck Deck path.
 * @param[in] method 0 EXTRAN, 1 SLOT; overrides the deck's own SURCHARGE_METHOD.
 * @param[out] r Run record.
 */
void runDeck(const fs::path& deck, int method, ProbedRun& r) {
    r.deck = deck.lexically_relative(kRepoRoot).generic_string();
    r.method = method;
    Engine en;
    const std::string tag = deck.stem().string() + (method ? "__slot" : "__extran");
    if (!open(en, deck, tag)) {
        r.error = "open failed";
        return;
    }
    SimulationContext& ctx = en.eng().context();
    ctx.options.num_threads = 1;
    ctx.options.surcharge_method = method;
    if (swmm_engine_initialize(en.e) != SWMM_OK) {
        r.error = "initialize failed";
        return;
    }
    ProbeGuard guard(&r);
    if (swmm_engine_start(en.e, 0) != SWMM_OK) {
        r.error = "start failed";
        return;
    }
    en.started = true;
    double elapsed = 0.0;
    int rc = SWMM_OK;
    // A run stops at its first mismatch or harness error; the rest of it would only repeat it.
    while (r.mismatches == 0 && r.error.empty() &&
           (rc = swmm_engine_step(en.e, &elapsed)) == SWMM_OK && elapsed > 0.0) {
    }
    if (rc != SWMM_OK && r.error.empty())
        r.error = "step failed with code " + std::to_string(rc);
}

std::string tallyLine(const ProbedRun& r) {
    const Tally& t = r.tally;
    std::ostringstream s;
    s << (r.method ? "SLOT  " : "EXTRAN") << " " << r.deck << ": steps " << t.steps
      << ", iterations " << t.iterations << ", comparisons " << t.comparisons << ", live "
      << t.evaluations << ", held " << t.held << " (re-slotted " << t.reslotted << ")"
      << ", classes DRY " << t.classes[0] << " UP_DRY " << t.classes[1] << " DN_DRY "
      << t.classes[2] << " SUB " << t.classes[3] << " UP_CRIT " << t.classes[5] << " DN_CRIT "
      << t.classes[6] << ", drops " << t.drops << ", backwater " << t.backwater << " (upstream "
      << t.backwater_upstream << "), full " << t.full << ", above crown " << t.above_crown
      << ", slot active " << t.slot_active << ", flooding " << t.flooding;
    return s.str();
}

} // namespace

TEST(DwKernelsGeometry, MatchesTheSolverBitwise) {
#ifdef SWMM_XSECT_FAST_LOOKUP
    GTEST_SKIP() << "the kernels mirror the bit-exact geometry build, not SWMM_XSECT_FAST_LOOKUP";
#endif
    if (std::getenv("SWMM_DISABLE_BYPASS_MASK"))
        GTEST_SKIP() << "SWMM_DISABLE_BYPASS_MASK changes what a bypassed conduit holds";
    if (std::getenv("SWMM_DW_THREADS"))
        GTEST_SKIP() << "SWMM_DW_THREADS overrides the one-thread run this test needs";

    fs::create_directories(kOutDir);

    struct Deck {
        fs::path path;
        bool own = false; ///< written for this test
        bool si = false;
    };
    std::vector<Deck> decks;
    for (const char* rel : kRepositoryDecks)
        decks.push_back({kRepoRoot / rel, false});
    std::vector<fs::path> own;
    for (const auto& e : fs::directory_iterator(kDeckDir))
        if (e.path().extension() == ".inp")
            own.push_back(e.path());
    ASSERT_FALSE(own.empty()) << "no deck in " << kDeckDir;
    std::sort(own.begin(), own.end());
    for (const fs::path& p : own)
        decks.push_back({p, true});
    for (Deck& d : decks) {
        const std::string why = checkScope(d.path, d.si);
        ASSERT_TRUE(why.empty()) << d.path << " is out of scope: " << why;
    }

    std::vector<ProbedRun> runs;
    runs.reserve(decks.size() * 2);
    for (const Deck& d : decks)
        for (int method : {step::kSurchargeExtran, step::kSurchargeSlot}) {
            runs.emplace_back();
            runDeck(d.path, method, runs.back());
        }

    // ---- report ----
    Tally per_method[2];
    Tally by_source[2][2]; // [written for this test][method]
    bool us = false, si = false;
    std::ostringstream report;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const ProbedRun& r = runs[i];
        const Deck& d = decks[i / 2];
        report << tallyLine(r) << (r.error.empty() ? "" : "  ERROR: " + r.error) << "\n";
        per_method[r.method].add(r.tally);
        by_source[d.own ? 1 : 0][r.method].add(r.tally);
        if (r.tally.comparisons > 0) {
            us = us || !d.si;
            si = si || d.si;
        }
    }
    for (int m = 0; m < 2; ++m) {
        ProbedRun all;
        all.method = m;
        all.deck = "REPOSITORY DECKS";
        all.tally = by_source[0][m];
        report << tallyLine(all) << "\n";
        all.deck = "NEW DECKS";
        all.tally = by_source[1][m];
        report << tallyLine(all) << "\n";
        all.deck = "ALL DECKS";
        all.tally = per_method[m];
        report << tallyLine(all) << "\n";
    }
    std::ofstream(kOutDir / "summary.txt") << report.str();
    std::printf("%s", report.str().c_str());

    // ---- verdict ----
    std::uint64_t comparisons = 0;
    for (const ProbedRun& r : runs) {
        comparisons += r.tally.comparisons;
        EXPECT_TRUE(r.error.empty()) << r.deck << (r.method ? " SLOT: " : " EXTRAN: ") << r.error;
        EXPECT_GT(r.tally.comparisons, 0u) << r.deck << (r.method ? " SLOT" : " EXTRAN");
        EXPECT_EQ(r.mismatches, 0u) << r.mismatches << " mismatches; first: " << r.first_mismatch;
    }
    std::printf("Total bitwise comparisons: %llu\n", static_cast<unsigned long long>(comparisons));
    EXPECT_TRUE(us) << "no deck in US units ran";
    EXPECT_TRUE(si) << "no deck in SI units ran";
    for (int m = 0; m < 2; ++m) {
        const Tally& t = per_method[m];
        const char* name = m ? "SLOT" : "EXTRAN";
        EXPECT_GT(t.classes[static_cast<int>(FlowClass::DRY)], 0u) << name << ": no dry pipe";
        EXPECT_GT(t.drops, 0u) << name << ": no drop at an invert offset";
        // DN_DRY cannot occur: under DYNWAVE an adverse conduit is reversed when the deck is read,
        // so node1 is the higher end and h1 >= inv1 >= inv2 for a conduit without offsets.
        EXPECT_GT(t.classes[static_cast<int>(FlowClass::UP_DRY)], 0u) << name << ": no UP_DRY";
        EXPECT_GT(t.classes[static_cast<int>(FlowClass::UP_CRITICAL)], 0u)
            << name << ": no UP_CRITICAL (reverse flow over a drop)";
        EXPECT_GT(t.classes[static_cast<int>(FlowClass::DN_CRITICAL)], 0u)
            << name << ": no DN_CRITICAL";
        EXPECT_GT(t.classes[static_cast<int>(FlowClass::SUBCRITICAL)], 0u)
            << name << ": no free-surface flow";
        EXPECT_GT(t.backwater_upstream, 0u) << name << ": no backwater from a FIXED outfall";
        EXPECT_GT(t.above_crown, 0u) << name << ": no surcharge";
        EXPECT_GT(t.flooding, 0u) << name << ": no flooding";
        EXPECT_GT(t.held, 0u) << name << ": no bypassed conduit";
        if (m == 1)
            EXPECT_GT(t.slot_active, 0u) << "SLOT: the slot never engaged";
    }
}
