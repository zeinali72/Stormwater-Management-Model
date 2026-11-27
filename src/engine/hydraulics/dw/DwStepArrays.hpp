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
 *          (DWSolver::computeLinkGeometry), the momentum stage (DWSolver::momentumKernels) and
 *          the node stages (initNodeStates, gatherConduitNodeFlows, updateNodeDepthsTeam,
 *          findBypassedLinks), the outfall boundaries (outfall::setAllOutfallDepths) and the
 *          Courant step (DWSolver::getRoutingStep) read and write.
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

/// FREE outfall, numbered as OutfallType::FREE.
inline constexpr int kOutfallFree = 0;
/// NORMAL outfall, numbered as OutfallType::NORMAL.
inline constexpr int kOutfallNormal = 1;
/// FIXED outfall, numbered as OutfallType::FIXED.
inline constexpr int kOutfallFixed = 2;

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
    int n_csr = 0;      ///< entries of the node-to-conduit adjacency (csr_link, csr_is_n2)

    // ---- options (static) -------------------------------------------------
    int surcharge_method = kSurchargeExtran; ///< kSurchargeExtran or kSurchargeSlot

    // ---- options (read at every routing step) -------------------------------
    double omega = 0.0;       ///< DWSolver::omega, the Picard under-relaxation weight
    int normal_flow_ltd = 0;  ///< SimulationOptions::normal_flow_ltd
    int inertial_damping = 0; ///< SimulationOptions::inertial_damping
    double head_tol = 0.0;    ///< DWSolver::head_tol, the node convergence tolerance, ft
    double min_surf_area = 0.0; ///< DWSolver::min_surf_area_, ft2
    int allow_ponding = 0;      ///< SimulationOptions::allow_ponding

    // ---- nodes (static) ---------------------------------------------------
    const double* node_invert = nullptr;           ///< NodeData::invert_elev, ft
    const std::uint8_t* node_is_outfall = nullptr; ///< 1 for an OUTFALL node
    const std::uint8_t* node_outfall_flap = nullptr; ///< 1 for an OUTFALL node with a flap gate
    const int* node_outfall_link = nullptr;      ///< link setting an outfall's depth; -1 for none
    const int* node_outfall_type = nullptr;      ///< kOutfallFree, kOutfallNormal or kOutfallFixed
    const double* node_outfall_offset = nullptr; ///< that link's offset at the outfall, ft
    const double* node_outfall_stage = nullptr;  ///< FIXED outfall stage, ft

    // ---- node-to-conduit adjacency (static; DWSolver::buildConduitNodeCSR) --
    const int* csr_row = nullptr;            ///< n_nodes + 1 offsets into csr_link
    const int* csr_link = nullptr;           ///< incident conduit, ascending per node
    const std::uint8_t* csr_is_n2 = nullptr; ///< 1 where the node is the conduit's node2 end

    // ---- node inputs of one routing step (read only by the loop) ------------
    const double* node_old_depth = nullptr;       ///< NodeData::old_depth, ft
    const double* node_old_volume = nullptr;      ///< NodeData::old_volume, ft3
    const double* node_old_net_inflow = nullptr;  ///< NodeData::old_net_inflow, cfs
    const double* node_lat_flow = nullptr;        ///< NodeData::lat_flow, cfs
    const double* node_losses = nullptr;          ///< NodeData::losses, cfs
    const double* node_full_depth = nullptr;      ///< NodeData::full_depth, ft
    const double* node_crown_elev = nullptr;      ///< NodeData::crown_elev, ft
    const double* node_ponded_area = nullptr;     ///< NodeData::ponded_area, ft2
    const double* node_sur_depth = nullptr;       ///< NodeData::sur_depth, ft
    const double* node_rpt_full_volume = nullptr; ///< NodeData::rpt_full_volume, ft3
    const int* node_degree = nullptr;             ///< NodeData::degree; negative upstream end

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
    const double* conduit_links_length = nullptr; ///< routing length (true_length, else length), ft
    const double* conduit_mod_length = nullptr;   ///< ConduitData::mod_length, ft
    const double* conduit_rough_factor = nullptr; ///< ConduitData::rough_factor
    const double* conduit_barrels = nullptr;      ///< barrel count as a double
    const std::uint8_t* conduit_is_closed = nullptr;     ///< LinkData::is_closed at initialization
    const std::uint8_t* conduit_has_losses = nullptr;    ///< any of the three loss coefficients set
    const double* conduit_loss_inlet = nullptr;          ///< ConduitData::loss_inlet
    const double* conduit_loss_outlet = nullptr;         ///< ConduitData::loss_outlet
    const double* conduit_loss_avg = nullptr;            ///< ConduitData::loss_avg
    const double* conduit_q_limit = nullptr;             ///< LinkData::q_limit, cfs; 0 for none
    const std::uint8_t* conduit_has_flap_gate = nullptr; ///< LinkData::has_flap_gate
    const std::int8_t* conduit_direction = nullptr;      ///< LinkData::direction, +1 or -1

    // ---- conduit inputs of one routing step (read only by the loop) ---------
    const double* conduit_setting = nullptr;  ///< LinkData::setting; 0 when a rule closed it
    const double* conduit_old_flow = nullptr; ///< LinkData::old_flow, cfs
    const double* area_old = nullptr;         ///< midpoint area at the step start, ft2

    // ---- node state (one Picard iteration) --------------------------------
    double* node_depth = nullptr;               ///< NodeData::depth, ft
    double* node_head = nullptr;                ///< NodeData::head, ft
    double* node_volume = nullptr;              ///< NodeData::volume, ft3
    double* node_overflow = nullptr;            ///< NodeData::overflow, cfs
    double* node_held_volume = nullptr;         ///< NodeData::held_volume, ft3
    double* node_inflow = nullptr;              ///< NodeData::inflow, cfs
    double* node_outflow = nullptr;             ///< NodeData::outflow, cfs
    std::uint8_t* node_converged = nullptr;     ///< DWNodeArrays::converged
    double* node_sumdqdh = nullptr;             ///< DWNodeArrays::sumdqdh, ft2/s
    double* node_new_surf_area = nullptr;       ///< DWNodeArrays::new_surf_area, ft2
    double* node_old_surf_area = nullptr;       ///< DWNodeArrays::old_surf_area, ft2
    std::uint8_t* node_is_surcharged = nullptr; ///< DWNodeArrays::is_surcharged
    double* node_dydt = nullptr;                ///< DWNodeArrays::dYdT, ft/s

    // ---- conduit state read by the geometry stage --------------------------
    double* conduit_q1 = nullptr;                   ///< last per-barrel flow (DWSolver::q1_), cfs
    std::uint8_t* conduit_bypassed = nullptr;       ///< both end nodes converged (held this pass)

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

    // ---- momentum (DWSolver::momentumKernels) --------------------------------
    // Link arrays: in scope, conduit c is link c and ConduitData row c.
    double* link_flow = nullptr;                 ///< LinkData::flow, cfs
    double* link_dqdh = nullptr;                 ///< LinkData::dqdh, ft2/s
    double* link_depth = nullptr;                ///< LinkData::depth, ft
    double* link_volume = nullptr;               ///< LinkData::volume, ft3
    double* new_flow = nullptr;                  ///< DWSolver::new_flow_, cfs
    double* dqdh = nullptr;                      ///< DWSolver::dqdh_, ft2/s
    double* velocity = nullptr;                  ///< DWSolver::velocity_, ft/s
    double* froude = nullptr;                    ///< DWSolver::froude_
    double* sigma = nullptr;                     ///< DWSolver::sigma_, the inertial damping factor
    std::uint8_t* category = nullptr;            ///< DWSolver::category_ (MomentumCategory)
    std::int8_t* full_state = nullptr;           ///< ConduitData::full_state bits
    std::uint8_t* inlet_control = nullptr;       ///< ConduitData::inlet_control
    std::uint8_t* normal_flow_limited = nullptr; ///< ConduitData::normal_flow_limited

    // ---- shared tables ----------------------------------------------------
    Tables tables;
};

} // namespace openswmm::dynwave::step

#endif // OPENSWMM_ENGINE_DW_STEP_ARRAYS_HPP
