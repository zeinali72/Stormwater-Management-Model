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
 * @file      DwStepArrays.hpp
 * @brief     The arrays one dynamic-wave routing step reads and writes, as raw pointers.
 *
 * @details Declarations only, in the style of fv/FvSubstepArrays.hpp: the host solver, a future
 *          device marcher and the tests can all name the same struct without pulling in the
 *          kernel bodies of DwKernels.hpp.
 *
 *          The struct describes the classic dynamic wave solver (DWSolver) for the network the
 *          device path supports: junctions and outfalls, conduits of CIRCULAR or RECT_CLOSED
 *          section with one barrel, SURCHARGE_METHOD EXTRAN or SLOT. Every conduit array is
 *          indexed by the conduit's position c in DWSolver's dense conduit tile (0..n_conduits-1),
 *          every node array by the node index. It holds what the link geometry stage
 *          (DWSolver::computeLinkGeometry) reads and writes; the later stages (momentum, node
 *          update, outfall boundaries, time step) add their own fields to the same struct.
 *
 * @see DwKernels.hpp
 * @ingroup engine_hydraulics
 *
 * @author    Farzan Zeinali
 * @license   Apache-2.0
 */

#ifndef OPENSWMM_ENGINE_DW_STEP_ARRAYS_HPP
#define OPENSWMM_ENGINE_DW_STEP_ARRAYS_HPP

#include <cstdint>

namespace openswmm::dynwave::step {

/// SURCHARGE_METHOD EXTRAN, numbered as dynwave::SurchargeMethod::EXTRAN.
inline constexpr int kSurchargeExtran = 0;
/// SURCHARGE_METHOD SLOT, numbered as dynwave::SurchargeMethod::SLOT.
inline constexpr int kSurchargeSlot = 1;

/// The shared circular geometry tables (xsect_tables.hpp) and the Amax ratios, in whichever memory
/// space the kernels run. Plain pointers, trivially copyable into a device kernel by value.
struct Tables {
    const double* A_Circ = nullptr; ///< area over full area, by depth over full depth
    const double* R_Circ = nullptr; ///< hydraulic radius over full, by depth over full depth
    const double* W_Circ = nullptr; ///< top width over maximum width, by depth over full depth
    const double* S_Circ = nullptr; ///< section factor over full, by area over full area
    const double* Y_Circ = nullptr; ///< depth over full depth, by area over full area
    const double* Amax = nullptr;   ///< 26 ratios indexed by the kernel XSectShape code
    int N_A_Circ = 0;               ///< entries in A_Circ
    int N_R_Circ = 0;               ///< entries in R_Circ
    int N_W_Circ = 0;               ///< entries in W_Circ
    int N_S_Circ = 0;               ///< entries in S_Circ
    int N_Y_Circ = 0;               ///< entries in Y_Circ
};

/// Every array one routing step reads or writes. Plain pointers so the same struct describes host
/// vectors and device views.
struct Arrays {
    // ---- sizes ------------------------------------------------------------
    int n_nodes = 0;    ///< nodes in the network
    int n_conduits = 0; ///< conduits in DWSolver's dense conduit tile

    // ---- options (static) -------------------------------------------------
    int surcharge_method = kSurchargeExtran; ///< kSurchargeExtran or kSurchargeSlot

    // ---- nodes (static) ---------------------------------------------------
    const double* node_invert = nullptr;           ///< NodeData::invert_elev, ft
    const std::uint8_t* node_is_outfall = nullptr; ///< 1 for an OUTFALL node

    // ---- conduits (static) ------------------------------------------------
    const int* conduit_link = nullptr;  ///< global link index of conduit c
    const int* conduit_n1 = nullptr;    ///< upstream node
    const int* conduit_n2 = nullptr;    ///< downstream node
    const int* conduit_shape = nullptr; ///< kernel XSectShape code (LinkData::xsect_batch_shape)
    const std::uint8_t* conduit_has_offset = nullptr; ///< offset1 > 0 || offset2 > 0
    const double* conduit_offset1 = nullptr;          ///< LinkData::offset1, ft
    const double* conduit_offset2 = nullptr;          ///< LinkData::offset2, ft
    const double* conduit_y_full = nullptr;           ///< LinkData::xsect_y_full, ft
    const double* conduit_a_full = nullptr;           ///< LinkData::xsect_a_full, ft2
    const double* conduit_r_full = nullptr;           ///< LinkData::xsect_r_full, ft
    const double* conduit_w_max = nullptr;            ///< LinkData::xsect_w_max, ft
    const double* conduit_s_full = nullptr;           ///< LinkData::xsect_s_full, ft^(8/3)
    const double* conduit_s_max = nullptr;            ///< LinkData::xsect_s_max, ft^(8/3)
    const double* conduit_length = nullptr; ///< modified length (mod_length, else length), ft
    const double* conduit_beta = nullptr;   ///< ConduitData::beta
    const double* conduit_q_max = nullptr;  ///< ConduitData::q_max, per barrel, cfs

    // ---- node state (one Picard iteration) --------------------------------
    const double* node_depth = nullptr; ///< NodeData::depth, ft

    // ---- conduit state read by the geometry stage --------------------------
    const double* conduit_q1 = nullptr;             ///< last per-barrel flow (DWSolver::q1_), cfs
    const std::uint8_t* conduit_bypassed = nullptr; ///< both end nodes converged (held this pass)

    // ---- link geometry (written for live conduits, held for bypassed ones) --
    double* depth1 = nullptr;          ///< upstream flow depth, ft
    double* depth2 = nullptr;          ///< downstream flow depth, ft
    double* depth_mid = nullptr;       ///< midpoint flow depth, ft
    double* h1 = nullptr;              ///< upstream head, ft
    double* h2 = nullptr;              ///< downstream head, ft
    double* fasnh = nullptr;           ///< fraction between normal and critical depth
    double* width1 = nullptr;          ///< top width at depth1, ft
    double* width2 = nullptr;          ///< top width at depth2, ft
    double* width_mid = nullptr;       ///< top width at depth_mid, ft
    double* surf_area1 = nullptr;      ///< surface area given to the upstream node, ft2
    double* surf_area2 = nullptr;      ///< surface area given to the downstream node, ft2
    double* area1 = nullptr;           ///< flow area at depth1, ft2
    double* area2 = nullptr;           ///< flow area at depth2, ft2
    double* area_mid = nullptr;        ///< flow area at depth_mid, ft2
    double* hrad1 = nullptr;           ///< hydraulic radius at depth1, ft
    double* hrad_mid = nullptr;        ///< hydraulic radius at depth_mid, ft
    std::int8_t* flow_class = nullptr; ///< FlowClass code (LinkData::flow_class)

    // ---- crown-capped depths (DWSolver::wcap_d1_/d2_/dm_) -------------------
    // Written for live conduits under EXTRAN only; SLOT never writes them, so they keep their
    // initial 0.0. The momentum stage reads wcap_dm (DWSolver::processManningLink).
    double* wcap_d1 = nullptr; ///< upstream depth STEP B evaluates the width at, ft
    double* wcap_d2 = nullptr; ///< downstream depth STEP B evaluates the width at, ft
    double* wcap_dm = nullptr; ///< midpoint depth STEP B evaluates the width at, ft

    // ---- shared tables ----------------------------------------------------
    Tables tables;
};

} // namespace openswmm::dynwave::step

#endif // OPENSWMM_ENGINE_DW_STEP_ARRAYS_HPP
