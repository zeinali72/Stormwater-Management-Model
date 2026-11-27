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
 * @file      DwKernels.hpp
 * @brief     Per-conduit bodies of the classic dynamic wave solver, written against raw arrays so a
 *            device backend can run them.
 *
 * @details These are the bodies of DWSolver (DynamicWave.cpp) for the network the device path
 *          supports: junctions and outfalls (FREE, NORMAL, FIXED), conduits of CIRCULAR or
 *          RECT_CLOSED section with one barrel, SURCHARGE_METHOD EXTRAN or SLOT. This version
 *          holds the link geometry stage, DWSolver::computeLinkGeometry, and the momentum stage,
 *          DWSolver::momentumKernels.
 *
 *          Each function keeps the floating-point expressions of the code it mirrors, in the same
 *          order, so the two give the same bits; the mirrored code is named in each comment, with
 *          line numbers at commit 25404228. A change to one must be made in the other, and
 *          tests/unit/engine/test_dw_kernels_geometry is what catches a miss.
 *
 *          The CPU solver evaluates its batch geometry (STEP B and STEP D) through XSectBatch.cpp,
 *          not through XsectEval, and its STEP C through the scalar XsectEval accessors. Both are
 *          mirrored here separately, because the two paths do not always spell the same lookup the
 *          same way. The table lookups of XSectLookup.hpp are plain `inline` host functions, so
 *          they are mirrored too rather than called; the analytic leaves of `xsect::shape` are
 *          already `OPENSWMM_KERNEL_FN` and are called directly.
 *
 *          Every function writes only the conduit it is called with, so a kernel may run the
 *          conduits in any order or in parallel. The arithmetic assumes the default (bit-exact)
 *          geometry build: under SWMM_XSECT_FAST_LOOKUP the CPU kernels use another
 *          normalization and these functions do not mirror it. A device build must pin
 *          `-ffp-contract=off`, as for XSectKernels.hpp.
 *
 * @see DwStepArrays.hpp, DynamicWave.cpp, XSectBatch.cpp, XSectKernels.hpp
 * @ingroup engine_hydraulics
 *
 * @author    Farzan Zeinali
 * @license   Apache-2.0
 */

#ifndef OPENSWMM_ENGINE_DW_KERNELS_HPP
#define OPENSWMM_ENGINE_DW_KERNELS_HPP

#include <cmath>
#include <cstdint>

#include "../../core/Constants.hpp"
#include "../XSectBatch.hpp"
#include "../XSectKernels.hpp"
#include "../xsect_tables.hpp"
#include "DwStepArrays.hpp"

namespace openswmm::dynwave::step {

/// Kernel XSectShape code of a CIRCULAR section.
inline constexpr int kShapeCircular = static_cast<int>(XSectShape::CIRCULAR);
/// Kernel XSectShape code of a RECT_CLOSED section.
inline constexpr int kShapeRectClosed = static_cast<int>(XSectShape::RECT_CLOSED);

// FlowClass codes (LinkData.hpp), as stored in Arrays::flow_class.
/// Both ends dry.
inline constexpr std::int8_t kDry = static_cast<std::int8_t>(FlowClass::DRY);
/// Upstream end dry.
inline constexpr std::int8_t kUpDry = static_cast<std::int8_t>(FlowClass::UP_DRY);
/// Downstream end dry.
inline constexpr std::int8_t kDnDry = static_cast<std::int8_t>(FlowClass::DN_DRY);
/// Free-surface or full flow with no critical end.
inline constexpr std::int8_t kSubcritical = static_cast<std::int8_t>(FlowClass::SUBCRITICAL);
/// Critical depth at the upstream end (reverse flow over a drop).
inline constexpr std::int8_t kUpCritical = static_cast<std::int8_t>(FlowClass::UP_CRITICAL);
/// Critical depth at the downstream end.
inline constexpr std::int8_t kDnCritical = static_cast<std::int8_t>(FlowClass::DN_CRITICAL);
/// Subcritical flow the momentum stage found to have a Froude number above one.
inline constexpr std::int8_t kSupercritical = static_cast<std::int8_t>(FlowClass::SUPERCRITICAL);

// MomentumCategory codes (DynamicWave.hpp), as stored in Arrays::category. DynamicWave.cpp
// checks they agree.
/// Dry, empty or closed: zero flow.
inline constexpr std::uint8_t kSkipDry = 0;
/// Manning friction, open channel.
inline constexpr std::uint8_t kManningOpen = 1;
/// Manning friction, closed conduit with a free surface.
inline constexpr std::uint8_t kManningClosedFs = 2;
/// Manning friction, closed conduit full at both ends.
inline constexpr std::uint8_t kManningClosedFull = 3;

/**
 * @brief Bind the shared geometry tables in host memory.
 *
 * Host only: a device backend binds Tables to its own copies of the same arrays.
 *
 * @return Tables pointing at the xsect_tables.hpp arrays.
 */
inline Tables hostTables() {
    Tables t;
    t.A_Circ = xsect_tables::A_Circ;
    t.R_Circ = xsect_tables::R_Circ;
    t.W_Circ = xsect_tables::W_Circ;
    t.S_Circ = xsect_tables::S_Circ;
    t.Y_Circ = xsect_tables::Y_Circ;
    t.Amax = xsect_tables::Amax;
    t.N_A_Circ = xsect_tables::N_A_Circ;
    t.N_R_Circ = xsect_tables::N_R_Circ;
    t.N_W_Circ = xsect_tables::N_W_Circ;
    t.N_S_Circ = xsect_tables::N_S_Circ;
    t.N_Y_Circ = xsect_tables::N_Y_Circ;
    return t;
}

// ===========================================================================
// Small helpers
// ===========================================================================

/// std::max for doubles (returns `a` unless `a < b`), spelled so it is callable in device code.
OPENSWMM_KERNEL_FN double maxOf(double a, double b) noexcept {
    return (a < b) ? b : a;
}

/// std::min for doubles (returns `a` unless `b < a`), spelled so it is callable in device code.
OPENSWMM_KERNEL_FN double minOf(double a, double b) noexcept {
    return (b < a) ? b : a;
}

/// The scalar section view DynamicWave.cpp buildXSP (2075) builds, for the fields the in-scope
/// shapes read.
struct Section {
    int shape = 0;       ///< kernel XSectShape code
    double y_full = 0.0; ///< full depth, ft
    double a_full = 0.0; ///< full area, ft2
    double w_max = 0.0;  ///< maximum width, ft
    double r_full = 0.0; ///< full hydraulic radius, ft
    double s_full = 0.0; ///< section factor when full, ft^(8/3)
    double s_max = 0.0;  ///< maximum section factor, ft^(8/3)
};

/**
 * @brief Gather the scalar section view of one conduit.
 *
 * @param[in] A Step arrays.
 * @param[in] c Conduit index.
 * @return The conduit's Section.
 */
OPENSWMM_KERNEL_FN Section sectionOf(const Arrays& A, int c) noexcept {
    Section s;
    s.shape = A.conduit_shape[c];
    s.y_full = A.conduit_y_full[c];
    s.a_full = A.conduit_a_full[c];
    s.w_max = A.conduit_w_max[c];
    s.r_full = A.conduit_r_full[c];
    s.s_full = A.conduit_s_full[c];
    s.s_max = A.conduit_s_max[c];
    return s;
}

/**
 * @brief XsectEval::isOpen (XSectKernels.hpp:2851), on the kernel shape code.
 *
 * Also stands for DynamicWave.cpp isOpenShape (151), which tests the same shapes on the storage
 * shape code.
 *
 * @param[in] shape Kernel XSectShape code.
 * @return True for the open shapes.
 */
OPENSWMM_KERNEL_FN bool shapeIsOpen(int shape) noexcept {
    switch (static_cast<XSectShape>(shape)) {
    case XSectShape::DUMMY:
    case XSectShape::RECT_OPEN:
    case XSectShape::TRAPEZOIDAL:
    case XSectShape::TRIANGULAR:
    case XSectShape::PARABOLIC:
    case XSectShape::POWERFUNC:
    case XSectShape::IRREGULAR:
    case XSectShape::STREET_XSECT:
        return true;
    default:
        return false;
    }
}

// ===========================================================================
// Table lookups (XSectLookup.hpp, XsectEval)
// ===========================================================================

/**
 * @brief xsect::lookup_exact (XSectLookup.hpp:76).
 *
 * @param[in] x Normalized coordinate, finite and nonnegative.
 * @param[in] t Table of n equally spaced entries on [0, 1].
 * @param[in] n Entry count, at least 4.
 * @return Interpolated table value.
 */
OPENSWMM_KERNEL_FN double lookupExact(double x, const double* t, int n) noexcept {
    const double delta = 1.0 / static_cast<double>(n - 1);
    int i = static_cast<int>(x / delta);
    if (i >= n - 1)
        return t[n - 1];

    double x0 = i * delta;
    double y = t[i] + (x - x0) * (t[i + 1] - t[i]) / delta;

    if (i < 2) {
        double x1 = (static_cast<double>(i) + 1.0) * delta;
        double y2 =
            y + (x - x0) * (x - x1) / (delta * delta) * (t[i] / 2.0 - t[i + 1] + t[i + 2] / 2.0);
        if (y2 > 0.0)
            y = y2;
    }
    if (y < 0.0)
        y = 0.0;
    return y;
}

/**
 * @brief xsect::norm_x (XSectLookup.hpp:166), default (bit-exact) build.
 *
 * @param[in] depth Depth, ft.
 * @param[in] param Full depth, ft.
 * @return depth / param when both are positive, else 0.
 */
OPENSWMM_KERNEL_FN double normX(double depth, double param) noexcept {
    return (param > 0.0 && depth > 0.0) ? (depth / param) : 0.0;
}

/**
 * @brief XsectEval::lookup (XSectKernels.hpp:631): lookupExact with guarded ends.
 *
 * @param[in] x Normalized coordinate.
 * @param[in] table Table of n_items equally spaced entries on [0, 1].
 * @param[in] n_items Entry count, at least 4.
 * @return table[0] for x <= 0 or NaN, the last entry for x > 1, else lookupExact.
 */
OPENSWMM_KERNEL_FN double lookupGuarded(double x, const double* table, int n_items) noexcept {
    if (!(x > 0.0))
        return table[0];
    if (x > 1.0)
        return table[n_items - 1];
    return lookupExact(x, table, n_items);
}

/**
 * @brief xsect::locate_bisect (XSectLookup.hpp:274).
 *
 * The CPU's XsectEval::invLookup narrows the search with a LocateLut when one is bound; the LUT
 * returns the same index as this bisection for every input (XSectLookup.hpp, "Index identity"),
 * so only the index search differs, never the arithmetic.
 *
 * @param[in] y Query value.
 * @param[in] table Nondecreasing table with indices 0..jLast.
 * @param[in] jLast Last index searched.
 * @return Lower bracketing index.
 */
OPENSWMM_KERNEL_FN int locateBisect(double y, const double* table, int jLast) noexcept {
    int j1 = 0;
    int j2 = jLast;
    if (y <= table[0])
        return 0;
    if (y >= table[jLast])
        return jLast;
    while (j2 - j1 > 1) {
        int j = (j1 + j2) >> 1;
        if (y >= table[j])
            j1 = j;
        else
            j2 = j;
    }
    return j1;
}

/**
 * @brief XsectEval::invLookup (XSectKernels.hpp:672).
 *
 * @param[in] y Table value to invert.
 * @param[in] table Table of n_items equally spaced entries on [0, 1].
 * @param[in] n_items Entry count, at least 4.
 * @return Normalized coordinate in [0, 1].
 */
OPENSWMM_KERNEL_FN double invLookup(double y, const double* table, int n_items) noexcept {
    double dx = 1.0 / static_cast<double>(n_items - 1);
    int n = n_items;

    // Truncate item count if last 2 entries are decreasing (section-factor tables)
    if (table[n - 3] > table[n - 1])
        n = n - 2;

    int i;
    if (n < n_items && y > table[n_items - 1]) {
        if (y >= table[n_items - 3])
            return static_cast<double>(n - 1) * dx;
        if (y <= table[n_items - 2])
            i = n_items - 2;
        else
            i = n_items - 3;
    } else {
        i = locateBisect(y, table, n - 1);
    }
    if (i >= n - 1)
        return static_cast<double>(n - 1) * dx;

    double x0 = i * dx;
    double dy = table[i + 1] - table[i];
    double x;
    if (dy == 0.0)
        x = x0;
    else
        x = x0 + (y - table[i]) * dx / dy;
    if (x < 0.0)
        x = 0.0;
    if (x > 1.0)
        x = 1.0;
    return x;
}

// ===========================================================================
// Circular section (XsectEval)
// ===========================================================================

/**
 * @brief XsectEval::getThetaOfAlpha (XSectKernels.hpp:724).
 *
 * @param[in] alpha Area over full area, positive.
 * @return Central angle, radians.
 */
OPENSWMM_KERNEL_FN double thetaOfAlpha(double alpha) noexcept {
    double theta, theta1, ap, d;
    if (alpha > 0.04)
        theta = 1.2 + 5.08 * (alpha - 0.04) / 0.96;
    else
        theta = 0.031715 - 12.79384 * alpha + 8.28479 * std::sqrt(alpha);
    theta1 = theta;
    ap = (2.0 * constants::PI) * alpha;
    for (int k = 1; k <= 40; ++k) {
        d = -(ap - theta + std::sin(theta)) / (1.0 - std::cos(theta));
        if (d > 1.0)
            d = std::copysign(1.0, d);
        theta = theta - d;
        if (std::fabs(d) <= 0.0001)
            return theta;
    }
    return theta1;
}

/**
 * @brief XsectEval::getThetaOfPsi (XSectKernels.hpp:752).
 *
 * @param[in] psi Section factor over full section factor, positive.
 * @return Central angle, radians.
 */
OPENSWMM_KERNEL_FN double thetaOfPsi(double psi) noexcept {
    double theta, theta1, ap, tt, tt23, t3, d;
    if (psi > 0.90)
        theta = 4.17 + 1.12 * (psi - 0.90) / 0.176;
    else if (psi > 0.5)
        theta = 3.14 + 1.03 * (psi - 0.5) / 0.4;
    else if (psi > 0.015)
        theta = 1.2 + 1.94 * (psi - 0.015) / 0.485;
    else
        theta = 0.12103 - 55.5075 * psi + 15.62254 * std::sqrt(psi);
    theta1 = theta;
    ap = (2.0 * constants::PI) * psi;
    for (int k = 1; k <= 40; ++k) {
        theta = std::fabs(theta);
        tt = theta - std::sin(theta);
        tt23 = std::pow(tt, 2.0 / 3.0);
        t3 = std::pow(theta, 1.0 / 3.0);
        d = ap * theta / t3 - tt * tt23;
        d = d / (ap * (2.0 / 3.0) / t3 - (5.0 / 3.0) * tt23 * (1.0 - std::cos(theta)));
        theta = theta - d;
        if (std::fabs(d) <= 0.0001)
            return theta;
    }
    return theta1;
}

/**
 * @brief XsectEval::getYcircular (XSectKernels.hpp:788).
 *
 * @param[in] alpha Area over full area.
 * @return Depth over full depth.
 */
OPENSWMM_KERNEL_FN double yCircular(double alpha) noexcept {
    double theta;
    if (alpha >= 1.0)
        return 1.0;
    if (alpha <= 0.0)
        return 0.0;
    if (alpha <= 1.0e-5) {
        theta = std::pow(37.6911 * alpha, 1.0 / 3.0);
        return theta * theta / 16.0;
    }
    theta = thetaOfAlpha(alpha);
    return (1.0 - std::cos(theta / 2.0)) / 2.0;
}

/**
 * @brief XsectEval::getAcircular (XSectKernels.hpp:836).
 *
 * @param[in] psi Section factor over full section factor.
 * @return Area over full area.
 */
OPENSWMM_KERNEL_FN double aCircular(double psi) noexcept {
    double theta;
    if (psi >= 1.0)
        return 1.0;
    if (psi <= 0.0)
        return 0.0;
    if (psi <= 1.0e-6) {
        theta = std::pow(124.4797 * psi, 3.0 / 13.0);
        return theta * theta * theta / 37.6911;
    }
    theta = thetaOfPsi(psi);
    return (theta - std::sin(theta)) / (2.0 * constants::PI);
}

/**
 * @brief XsectEval::circ_getYofA (XSectKernels.hpp:943).
 *
 * @param[in] T Shared tables.
 * @param[in] xs Circular section.
 * @param[in] a Flow area, ft2.
 * @return Depth, ft.
 */
OPENSWMM_KERNEL_FN double circYofA(const Tables& T, const Section& xs, double a) noexcept {
    double alpha = a / xs.a_full;
    if (alpha < 0.04)
        return xs.y_full * yCircular(alpha);
    return xs.y_full * lookupGuarded(alpha, T.Y_Circ, T.N_Y_Circ);
}

/**
 * @brief XsectEval::circ_getAofS (XSectKernels.hpp:983).
 *
 * @param[in] T Shared tables.
 * @param[in] xs Circular section.
 * @param[in] s Section factor, ft^(8/3).
 * @return Flow area, ft2.
 */
OPENSWMM_KERNEL_FN double circAofS(const Tables& T, const Section& xs, double s) noexcept {
    double psi = s / xs.s_full;
    if (psi == 0.0)
        return 0.0;
    if (psi >= 1.0)
        return xs.a_full;
    if (psi <= 0.015)
        return xs.a_full * aCircular(psi);
    return xs.a_full * invLookup(psi, T.S_Circ, T.N_S_Circ);
}

// ===========================================================================
// Closed rectangular section (XsectEval)
// ===========================================================================

/**
 * @brief XsectEval::rect_closed_getSofA (XSectKernels.hpp:1130).
 *
 * @param[in] xs Closed rectangular section.
 * @param[in] a Flow area, ft2.
 * @return Section factor, ft^(8/3).
 */
OPENSWMM_KERNEL_FN double rectClosedSofA(const Section& xs, double a) noexcept {
    if (a / xs.a_full > xsect::RECT_ALFMAX)
        return xs.s_max + (xs.s_full - xs.s_max) * (a / xs.a_full - xsect::RECT_ALFMAX) /
                              (1.0 - xsect::RECT_ALFMAX);
    return a * std::pow(xsect::shape::rectClosedRofA(a, xs.w_max, xs.a_full), 2.0 / 3.0);
}

/**
 * @brief XsectEval::generic_getdSdA (XSectKernels.hpp:865) for a closed rectangle, whose getSofA
 *        is rect_closed_getSofA.
 *
 * @param[in] xs Closed rectangular section.
 * @param[in] a Flow area, ft2.
 * @return dS/dA, ft^(2/3).
 */
OPENSWMM_KERNEL_FN double rectClosedGenericDsDa(const Section& xs, double a) noexcept {
    double alpha = a / xs.a_full;
    double alpha1 = alpha - 0.001;
    double alpha2 = alpha + 0.001;
    if (alpha1 < 0.0)
        alpha1 = 0.0;
    double a1 = alpha1 * xs.a_full;
    double a2 = alpha2 * xs.a_full;
    return (rectClosedSofA(xs, a2) - rectClosedSofA(xs, a1)) / (a2 - a1);
}

/**
 * @brief XsectEval::rect_closed_getdSdA (XSectKernels.hpp:1150).
 *
 * @param[in] xs Closed rectangular section.
 * @param[in] a Flow area, ft2.
 * @return dS/dA, ft^(2/3).
 */
OPENSWMM_KERNEL_FN double rectClosedDsDa(const Section& xs, double a) noexcept {
    double alpha = a / xs.a_full;
    if (alpha > xsect::RECT_ALFMAX)
        return (xs.s_full - xs.s_max) / ((1.0 - xsect::RECT_ALFMAX) * xs.a_full);
    if (alpha <= 1.0e-30)
        return rectClosedGenericDsDa(xs, a);
    double r = xsect::shape::rectClosedRofA(a, xs.w_max, xs.a_full);
    return (5.0 / 3.0 - (2.0 / 3.0) * (2.0 / xs.w_max) * r) * std::pow(r, 2.0 / 3.0);
}

/**
 * @brief XsectEval::findroot_Newton (XSectKernels.hpp:493) on S(a) - s for a closed rectangle,
 *        the residual of XsectEval::getAofS's default branch (XSectKernels.hpp:2692).
 *
 * @param[in] xs Closed rectangular section.
 * @param[in] x1 Lower bracket end, ft2.
 * @param[in] x2 Upper bracket end, ft2.
 * @param[in,out] rts Initial guess in, root out, ft2.
 * @param[in] xacc Absolute tolerance, ft2.
 * @param[in] s Target section factor, ft^(8/3).
 * @return Function evaluation count when at most 60, else 0.
 */
OPENSWMM_KERNEL_FN int rectClosedNewtonAofS(const Section& xs, double x1, double x2, double* rts,
                                            double xacc, double s) noexcept {
    constexpr int MAXIT = 60;
    int n = 0;
    double df, dx, dxold, f, x, temp, xhi, xlo;

    x = *rts;
    xlo = x1;
    xhi = x2;
    dxold = std::fabs(x2 - x1);
    dx = dxold;
    f = rectClosedSofA(xs, x) - s;
    df = rectClosedDsDa(xs, x);
    n++;
    for (int j = 1; j <= MAXIT; ++j) {
        if (((x - xhi) * df - f) * ((x - xlo) * df - f) >= 0.0 ||
            (std::fabs(2.0 * f) > std::fabs(dxold * df))) {
            dxold = dx;
            dx = 0.5 * (xhi - xlo);
            x = xlo + dx;
            if (xlo == x)
                break;
        } else {
            dxold = dx;
            dx = f / df;
            temp = x;
            x -= dx;
            if (temp == x)
                break;
        }
        if (std::fabs(dx) < xacc)
            break;
        f = rectClosedSofA(xs, x) - s;
        df = rectClosedDsDa(xs, x);
        n++;
        if (f < 0.0)
            xlo = x;
        else
            xhi = x;
    }
    *rts = x;
    return (n <= MAXIT) ? n : 0;
}

// ===========================================================================
// Scalar accessors used by STEP C (XsectEval dispatchers, in-scope shapes)
// ===========================================================================

/**
 * @brief XsectEval::getAofY (XSectKernels.hpp:2159) for CIRCULAR and RECT_CLOSED.
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] y Depth, ft.
 * @return Flow area, ft2.
 */
OPENSWMM_KERNEL_FN double scalarAofY(const Tables& T, const Section& xs, double y) noexcept {
    if (y <= 0.0)
        return 0.0;
    if (xs.y_full <= 0.0)
        return 0.0;
    double y_norm = y / xs.y_full;
    if (xs.shape == kShapeCircular)
        return xs.a_full * lookupGuarded(y_norm, T.A_Circ, T.N_A_Circ);
    if (xs.shape == kShapeRectClosed)
        return xsect::shape::rectAofY(y, xs.w_max);
    return 0.0;
}

/**
 * @brief XsectEval::getWofY (XSectKernels.hpp:2246) for CIRCULAR and RECT_CLOSED.
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] y Depth, ft.
 * @return Top width, ft.
 */
OPENSWMM_KERNEL_FN double scalarWofY(const Tables& T, const Section& xs, double y) noexcept {
    double y_norm = y / xs.y_full;
    if (xs.shape == kShapeCircular)
        return xs.w_max * lookupGuarded(y_norm, T.W_Circ, T.N_W_Circ);
    if (xs.shape == kShapeRectClosed)
        return xsect::shape::rectClosedWofY(y_norm, xs.w_max);
    return 0.0;
}

/**
 * @brief XsectEval::getYofA (XSectKernels.hpp:2391) for CIRCULAR and RECT_CLOSED.
 *
 * The CPU also forms alpha = a / a_full at entry; only tabular shapes read it.
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] a Flow area, ft2.
 * @return Depth, ft.
 */
OPENSWMM_KERNEL_FN double scalarYofA(const Tables& T, const Section& xs, double a) noexcept {
    if (a <= 0.0)
        return 0.0;
    if (xs.shape == kShapeCircular)
        return circYofA(T, xs, a);
    if (xs.shape == kShapeRectClosed)
        return a / xs.w_max;
    return 0.0;
}

/**
 * @brief XsectEval::getAofS (XSectKernels.hpp:2638) for CIRCULAR and RECT_CLOSED.
 *
 * The CPU also forms psi = s / s_full before the s_max clamp; only tabular shapes read it.
 * RECT_CLOSED takes the default branch: Newton on S(a) - s, bracketed by getAmax.
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] s Section factor, ft^(8/3).
 * @return Flow area, ft2.
 */
OPENSWMM_KERNEL_FN double scalarAofS(const Tables& T, const Section& xs, double s) noexcept {
    if (s <= 0.0)
        return 0.0;
    if (s > xs.s_max)
        s = xs.s_max;
    if (xs.shape == kShapeCircular)
        return circAofS(T, xs, s);
    if (xs.shape != kShapeRectClosed)
        return 0.0;
    // default branch (XSectKernels.hpp:2671); getAmax (2715) for a non-transect shape
    double a1, a2;
    const double amax_ratio = (xs.shape >= 0 && xs.shape <= 25) ? T.Amax[xs.shape] : 1.0;
    double a_max = xs.a_full * amax_ratio;
    if ((s <= xs.s_max && s >= xs.s_full) && xs.s_max != xs.s_full) {
        a1 = xs.a_full;
        a2 = a_max;
    } else {
        a1 = 0.0;
        a2 = a_max;
    }
    double a = 0.5 * (a1 + a2);
    double tol = 0.0001 * xs.a_full;
    rectClosedNewtonAofS(xs, a1, a2, &a, tol, s);
    return a;
}

/**
 * @brief Critical-flow residual of XsectEval::getYcrit (XSectKernels.hpp:2763, legacy
 *        getQcritical).
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] yc Trial depth, ft.
 * @param[in] qTarget Flow subtracted from the critical flow, cfs.
 * @return Critical flow at yc minus qTarget, cfs.
 */
OPENSWMM_KERNEL_FN double qCritical(const Tables& T, const Section& xs, double yc,
                                    double qTarget) noexcept {
    double a = scalarAofY(T, xs, yc);
    double w = scalarWofY(T, xs, yc);
    if (w > 0.0)
        return a * std::sqrt(constants::GRAVITY * a / w) - qTarget;
    return -qTarget;
}

/**
 * @brief XsectEval::getYcrit (XSectKernels.hpp:2739) for CIRCULAR and RECT_CLOSED.
 *
 * RECT_CLOSED is the closed-form branch. CIRCULAR takes the default branch, whose conduit area is
 * exactly the equivalent circle's (setParams writes a_full = PI / 4.0 * d * d with d == y_full),
 * so its area ratio r is 1 and the interval enumeration runs; the Ridder branch the CPU takes for
 * r outside [0.5, 2] is unreachable for the in-scope shapes and is not mirrored.
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] q Per-barrel flow, cfs.
 * @return Critical depth, ft, at most y_full.
 */
OPENSWMM_KERNEL_FN double scalarYcrit(const Tables& T, const Section& xs, double q) noexcept {
    if (q <= 0.0)
        return 0.0;
    double q2g = q * q / constants::GRAVITY;
    if (q2g == 0.0)
        return 0.0;

    double y;
    if (xs.shape == kShapeRectClosed) {
        y = std::pow(q2g / (xs.w_max * xs.w_max), 1.0 / 3.0);
    } else {
        // Initial estimate from equivalent circular conduit.
        double y0 = 1.01 * std::pow(q2g / xs.y_full, 0.25);
        if (!(y0 < xs.y_full)) // also NaN
            y0 = 0.97 * xs.y_full;

        // --- interval enumeration (legacy getYcritEnum), 25 increments
        constexpr int N_INC = 25;
        double dy = xs.y_full / N_INC;
        int i1 = static_cast<int>(y0 / dy);
        double q0 = qCritical(T, xs, i1 * dy, 0.0);
        if (q0 < q) {
            y = xs.y_full;
            for (int i = i1 + 1; i <= N_INC; ++i) {
                double qc = qCritical(T, xs, i * dy, 0.0);
                if (qc >= q) {
                    y = ((q - q0) / (qc - q0) + static_cast<double>(i - 1)) * dy;
                    break;
                }
                q0 = qc;
            }
        } else {
            y = 0.0;
            for (int i = i1 - 1; i >= 0; --i) {
                double qc = qCritical(T, xs, i * dy, 0.0);
                if (qc < q) {
                    y = ((q - qc) / (q0 - qc) + static_cast<double>(i)) * dy;
                    break;
                }
                q0 = qc;
            }
        }
    }
    return minOf(y, xs.y_full);
}

/**
 * @brief DynamicWave.cpp computeYnorm (2018): normal depth from flow.
 *
 * @param[in] T Shared tables.
 * @param[in] xs In-scope section.
 * @param[in] beta Manning conveyance coefficient.
 * @param[in] q_max Per-barrel capacity cap, cfs; <= 0 disables it.
 * @param[in] q Per-barrel flow, cfs.
 * @return Normal depth, ft.
 */
OPENSWMM_KERNEL_FN double normalDepth(const Tables& T, const Section& xs, double beta, double q_max,
                                      double q) noexcept {
    if (beta <= 0.0 || q <= 0.0)
        return 0.0;
    if (q_max > 0.0 && q > q_max)
        q = q_max;
    double s = q / beta;
    double a = scalarAofS(T, xs, s);
    return scalarYofA(T, xs, a);
}

// ===========================================================================
// Preissmann slot (DWSolver members)
// ===========================================================================

/**
 * @brief DWSolver::getCrownCutoff (DynamicWave.cpp:138) for EXTRAN and SLOT.
 *
 * @param[in] surcharge_method kSurchargeExtran or kSurchargeSlot.
 * @return Normalized crown cutoff depth.
 */
OPENSWMM_KERNEL_FN double crownCutoff(int surcharge_method) noexcept {
    if (surcharge_method == kSurchargeSlot)
        return constants::SLOT_CROWN_CUTOFF;
    return constants::EXTRAN_CROWN_CUTOFF;
}

/**
 * @brief DWSolver::getSlotWidth (DynamicWave.cpp:182).
 *
 * @param[in] surcharge_method kSurchargeExtran or kSurchargeSlot.
 * @param[in] y Depth, ft.
 * @param[in] y_full Full depth, ft.
 * @param[in] w_max Maximum width, ft.
 * @param[in] shape Kernel XSectShape code.
 * @return Slot width, ft; 0 unless SLOT and y / y_full reaches the slot crown cutoff.
 */
OPENSWMM_KERNEL_FN double slotWidth(int surcharge_method, double y, double y_full, double w_max,
                                    int shape) noexcept {
    if (surcharge_method != kSurchargeSlot)
        return 0.0;
    if (y_full <= 0.0 || shapeIsOpen(shape))
        return 0.0;
    const double yNorm = y / y_full;
    if (yNorm < constants::SLOT_CROWN_CUTOFF)
        return 0.0;
    if (yNorm > 1.78)
        return 0.01 * w_max;
    return w_max * 0.5423 * std::exp(-std::pow(yNorm, 2.4));
}

// ===========================================================================
// Batch geometry (XSectBatch.cpp kernels, in-scope shapes)
// ===========================================================================

/**
 * @brief Top width as the STEP B batch computes it: width_circular (XSectBatch.cpp:726) or
 *        width_rect_closed (703), dispatched by apply_width_kernel (1235).
 *
 * @param[in] T Shared tables.
 * @param[in] shape Kernel XSectShape code.
 * @param[in] d Depth, ft.
 * @param[in] y_full Full depth, ft.
 * @param[in] w_max Maximum width, ft.
 * @return Top width, ft.
 */
OPENSWMM_KERNEL_FN double batchWidth(const Tables& T, int shape, double d, double y_full,
                                     double w_max) noexcept {
    if (shape == kShapeCircular)
        return w_max * lookupExact(normX(d, y_full), T.W_Circ, T.N_W_Circ);
    if (shape == kShapeRectClosed)
        return xsect::shape::rectClosedWofY(normX(d, y_full), w_max);
    return 0.0;
}

/**
 * @brief Flow area as the STEP D batch computes it for the downstream end: area_circular
 *        (XSectBatch.cpp:296) or area_rect (309, an element-wise depth * w_max), dispatched by
 *        apply_area_kernel (1063).
 *
 * @param[in] T Shared tables.
 * @param[in] shape Kernel XSectShape code.
 * @param[in] d Depth, ft.
 * @param[in] y_full Full depth, ft.
 * @param[in] a_full Full area, ft2.
 * @param[in] w_max Maximum width, ft.
 * @return Flow area, ft2.
 */
OPENSWMM_KERNEL_FN double batchArea(const Tables& T, int shape, double d, double y_full,
                                    double a_full, double w_max) noexcept {
    if (shape == kShapeCircular)
        return a_full * lookupExact(normX(d, y_full), T.A_Circ, T.N_A_Circ);
    if (shape == kShapeRectClosed)
        return d * w_max;
    return 0.0;
}

/**
 * @brief Flow area and hydraulic radius as the STEP D batch computes them for the upstream end
 *        and the midpoint, dispatched by apply_area_hydrad_kernel (XSectBatch.cpp:1321):
 *        area_hydrad_circular (499) for CIRCULAR, area_rect (309) then hydrad_rect_closed (648)
 *        for RECT_CLOSED.
 *
 * @param[in] T Shared tables.
 * @param[in] shape Kernel XSectShape code.
 * @param[in] d Depth, ft.
 * @param[in] y_full Full depth, ft.
 * @param[in] a_full Full area, ft2.
 * @param[in] r_full Full hydraulic radius, ft.
 * @param[in] w_max Maximum width, ft.
 * @param[out] area Flow area, ft2.
 * @param[out] hydrad Hydraulic radius, ft.
 */
OPENSWMM_KERNEL_FN void batchAreaHydrad(const Tables& T, int shape, double d, double y_full,
                                        double a_full, double r_full, double w_max, double& area,
                                        double& hydrad) noexcept {
    if (shape == kShapeRectClosed) {
        area = d * w_max;
        hydrad = xsect::shape::rectClosedRofA(xsect::shape::rectAofY(d, w_max), w_max, a_full);
        return;
    }
    if (shape != kShapeCircular) {
        area = 0.0;
        hydrad = 0.0;
        return;
    }
    const double* At = T.A_Circ;
    const double* Rt = T.R_Circ;
    const int n = T.N_A_Circ; // == N_R_Circ
    const double delta = 1.0 / static_cast<double>(n - 1);
    const double dd = delta * delta;

    double yn = normX(d, y_full);
    int i = static_cast<int>(yn / delta);
    if (i >= n - 1) {
        area = a_full * At[n - 1];
        hydrad = r_full * Rt[n - 1];
        return;
    }
    double x0 = i * delta;
    double a = At[i] + (yn - x0) * (At[i + 1] - At[i]) / delta;
    double r = Rt[i] + (yn - x0) * (Rt[i + 1] - Rt[i]) / delta;
    if (i < 2) { // quadratic refinement (legacy lookup)
        double x1 = (static_cast<double>(i) + 1.0) * delta;
        double a2 = a + (yn - x0) * (yn - x1) / dd * (At[i] / 2.0 - At[i + 1] + At[i + 2] / 2.0);
        double r2 = r + (yn - x0) * (yn - x1) / dd * (Rt[i] / 2.0 - Rt[i + 1] + Rt[i + 2] / 2.0);
        if (a2 > 0.0)
            a = a2;
        if (r2 > 0.0)
            r = r2;
    }
    if (a < 0.0)
        a = 0.0;
    if (r < 0.0)
        r = 0.0;
    area = a_full * a;
    hydrad = r_full * r;
}

// ===========================================================================
// The link geometry stage (DWSolver::computeLinkGeometry, DynamicWave.cpp:2028)
// ===========================================================================

/**
 * @brief STEP A (DynamicWave.cpp:2063-2134): end heads, end and midpoint depths, fasnh = 1, and
 *        under EXTRAN the crown-capped depths STEP B evaluates widths at.
 *
 * lidnode::portDepth is the node depth itself for a node with no storage row, which every
 * in-scope node is.
 *
 * @param[in] A Step arrays; depth1/2/mid, h1/h2 and fasnh of conduit c are written, and under
 *     EXTRAN its capped depths wcap_d1/d2/dm.
 * @param[in] c Live (not bypassed) conduit.
 */
OPENSWMM_KERNEL_FN void geometryDepths(const Arrays& A, int c) noexcept {
    const int n1 = A.conduit_n1[c];
    const int n2 = A.conduit_n2[c];
    const bool slot_mode = A.surcharge_method == kSurchargeSlot;

    const double inv1 = A.node_invert[n1];
    const double inv2 = A.node_invert[n2];
    const double z1 = inv1 + A.conduit_offset1[c];
    const double z2 = inv2 + A.conduit_offset2[c];
    const double h1 = maxOf(A.node_depth[n1] + inv1, z1);
    const double h2 = maxOf(A.node_depth[n2] + inv2, z2);

    double y1 = maxOf(h1 - z1, constants::FUDGE);
    double y2 = maxOf(h2 - z2, constants::FUDGE);

    const double yf = A.conduit_y_full[c];
    if (!slot_mode) {
        y1 = minOf(y1, yf);
        y2 = minOf(y2, yf);
    }
    const double yMid = 0.5 * (y1 + y2);

    A.depth1[c] = y1;
    A.depth2[c] = y2;
    A.depth_mid[c] = yMid;
    A.h1[c] = h1;
    A.h2[c] = h2;
    A.fasnh[c] = 1.0;

    if (!slot_mode) {
        const bool is_open = shapeIsOpen(A.conduit_shape[c]);
        const double yCap = (!is_open && yf > 0.0) ? constants::EXTRAN_CROWN_CUTOFF * yf : 1e30;
        A.wcap_d1[c] = minOf(y1, yCap);
        A.wcap_d2[c] = minOf(y2, yCap);
        A.wcap_dm[c] = minOf(yMid, yCap);
    }
}

/**
 * @brief STEP B (DynamicWave.cpp:2136-2161): end and midpoint top widths through the batch width
 *        kernel, at the crown-capped depths under EXTRAN and at the depths themselves under SLOT.
 *
 * @param[in] A Step arrays; width1/2/mid of conduit c are written. Reads the capped depths
 *     geometryDepths wrote (EXTRAN) or the depths themselves (SLOT).
 * @param[in] c Live conduit.
 */
OPENSWMM_KERNEL_FN void geometryWidths(const Arrays& A, int c) noexcept {
    const int shape = A.conduit_shape[c];
    const double yf = A.conduit_y_full[c];
    const double wm = A.conduit_w_max[c];
    const bool slot_mode = A.surcharge_method == kSurchargeSlot;
    const double d1 = slot_mode ? A.depth1[c] : A.wcap_d1[c];
    const double d2 = slot_mode ? A.depth2[c] : A.wcap_d2[c];
    const double dm = slot_mode ? A.depth_mid[c] : A.wcap_dm[c];
    A.width1[c] = batchWidth(A.tables, shape, d1, yf, wm);
    A.width2[c] = batchWidth(A.tables, shape, d2, yf, wm);
    A.width_mid[c] = batchWidth(A.tables, shape, dm, yf, wm);
}

/**
 * @brief SLOT width override (DynamicWave.cpp:2163-2206): the slot width replaces a top width
 *        wherever it is nonzero.
 *
 * The CPU loop has no bypass test, so it runs for bypassed conduits too, on their held depths and
 * widths; linkGeometry does the same. No-op under EXTRAN.
 *
 * @param[in] A Step arrays; width1/2/mid of conduit c may be overwritten.
 * @param[in] c Conduit, live or bypassed.
 */
OPENSWMM_KERNEL_FN void slotWidthOverride(const Arrays& A, int c) noexcept {
    if (A.surcharge_method != kSurchargeSlot)
        return;
    const double yf = A.conduit_y_full[c];
    const double wm = A.conduit_w_max[c];
    const int shape = A.conduit_shape[c];
    double ws = slotWidth(A.surcharge_method, A.depth1[c], yf, wm, shape);
    if (ws > 0.0)
        A.width1[c] = ws;
    ws = slotWidth(A.surcharge_method, A.depth2[c], yf, wm, shape);
    if (ws > 0.0)
        A.width2[c] = ws;
    ws = slotWidth(A.surcharge_method, A.depth_mid[c], yf, wm, shape);
    if (ws > 0.0)
        A.width_mid[c] = ws;
}

/**
 * @brief STEP C (DynamicWave.cpp:2208-2507): flow classification, end surface areas, and the
 *        depths, heads and midpoint width it patches for critical and dry ends.
 *
 * @param[in] A Step arrays; surf_area1/2, fasnh and flow_class of conduit c are written, and
 *     depth1/2/mid, h1/h2 and width_mid where the class patches them.
 * @param[in] c Live conduit.
 */
OPENSWMM_KERNEL_FN void flowClassSurfArea(const Arrays& A, int c) noexcept {
    using constants::FUDGE;
    const Tables& T = A.tables;
    const int n1 = A.conduit_n1[c];
    const int n2 = A.conduit_n2[c];
    const int sm = A.surcharge_method;
    const int shape = A.conduit_shape[c];

    const double yf = A.conduit_y_full[c];
    const double z1_off = A.conduit_offset1[c];
    const double z2_off = A.conduit_offset2[c];
    const double beta_j = A.conduit_beta[c];
    const double qmax_j = A.conduit_q_max[c];
    const double inv1 = A.node_invert[n1];
    const double inv2 = A.node_invert[n2];
    double y1 = A.depth1[c];
    double y2 = A.depth2[c];
    double h1 = A.h1[c];
    double h2 = A.h2[c];
    double length = A.conduit_length[c];

    double surfArea1 = 0.0, surfArea2 = 0.0;
    std::int8_t fc = kSubcritical;
    double fasnh = 1.0;

    double cached_yN = 0.0, cached_yC = 0.0;
    const Section xs = sectionOf(A, c);

    const bool both_full = (y1 >= yf) & (y2 >= yf);
    const bool both_wet = (y1 > FUDGE) & (y2 > FUDGE);
    const bool both_dry = (y1 <= FUDGE) & (y2 <= FUDGE);
    const bool up_dry_only = (y1 <= FUDGE) & (y2 > FUDGE);

    if (both_full) {
        fc = kSubcritical;
    } else if (both_dry) {
        fc = kDry;
    } else if (!A.conduit_has_offset[c]) {
        if (up_dry_only) {
            if (h2 < inv1)
                fc = kUpDry;
        } else if (!both_wet) { // dn_dry_only
            if (h1 < inv2)
                fc = kDnDry;
        }
    } else {
        double z1_off_eff = z1_off;
        double z2_off_eff = z2_off;
        if (A.node_is_outfall[n1])
            z1_off_eff = maxOf(0.0, z1_off_eff - A.node_depth[n1]);
        if (A.node_is_outfall[n2])
            z2_off_eff = maxOf(0.0, z2_off_eff - A.node_depth[n2]);

        const double q = std::fabs(A.conduit_q1[c]);

        if (both_wet) {
            if (A.conduit_q1[c] < 0.0) {
                if (z1_off_eff > 0.0) {
                    cached_yN = normalDepth(T, xs, beta_j, qmax_j, q);
                    cached_yC = scalarYcrit(T, xs, q);
                    double ycMin = minOf(cached_yN, cached_yC);
                    if (y1 < ycMin)
                        fc = kUpCritical;
                }
            } else {
                if (z2_off_eff > 0.0) {
                    cached_yN = normalDepth(T, xs, beta_j, qmax_j, q);
                    cached_yC = scalarYcrit(T, xs, q);
                    double ycMin = minOf(cached_yN, cached_yC);
                    double ycMax = maxOf(cached_yN, cached_yC);
                    if (y2 < ycMin) {
                        fc = kDnCritical;
                    } else if (y2 < ycMax) {
                        if (ycMax - ycMin < FUDGE)
                            fasnh = 0.0;
                        else
                            fasnh = (ycMax - y2) / (ycMax - ycMin);
                    }
                }
            }
        } else if (up_dry_only) {
            const double z1_elev = inv1 + z1_off;
            if (h2 < z1_elev) {
                fc = kUpDry;
            } else if (z1_off_eff > 0.0) {
                cached_yN = normalDepth(T, xs, beta_j, qmax_j, q);
                cached_yC = scalarYcrit(T, xs, q);
                fc = kUpCritical;
            }
        } else { // dn_dry_only
            const double z2_elev = inv2 + z2_off;
            if (h1 < z2_elev) {
                fc = kDnDry;
            } else if (z2_off_eff > 0.0) {
                cached_yN = normalDepth(T, xs, beta_j, qmax_j, q);
                cached_yC = scalarYcrit(T, xs, q);
                fc = kDnCritical;
            }
        }
    }

    switch (fc) {
    case kSubcritical: {
        double w1 = A.width1[c];
        double wM = A.width_mid[c];
        double w2 = A.width2[c];
        surfArea1 = (w1 + wM) * length / 4.0;
        surfArea2 = (wM + w2) * length / 4.0 * fasnh;
        break;
    }
    case kUpCritical: {
        y1 = cached_yC;
        if (cached_yN < cached_yC)
            y1 = cached_yN;
        y1 = maxOf(y1, FUDGE);
        const double z1_elev = inv1 + z1_off;
        h1 = z1_elev + y1;
        double yMid = maxOf(0.5 * (y1 + y2), FUDGE);
        double w2 = A.width2[c];
        double wM = scalarWofY(T, xs, yMid);
        double wMsa;
        double wSlotM = slotWidth(sm, yMid, yf, A.conduit_w_max[c], shape);
        if (wSlotM > 0.0)
            wMsa = wSlotM;
        else if (yMid / yf >= crownCutoff(sm) && !shapeIsOpen(shape))
            wMsa = scalarWofY(T, xs, crownCutoff(sm) * yf);
        else
            wMsa = wM;
        surfArea2 = (wMsa + w2) * length * 0.5;
        A.depth1[c] = y1;
        A.depth_mid[c] = yMid;
        A.width_mid[c] = wM;
        A.h1[c] = h1;
        break;
    }
    case kDnCritical: {
        y2 = cached_yC;
        if (cached_yN < cached_yC)
            y2 = cached_yN;
        y2 = maxOf(y2, FUDGE);
        const double z2_elev = inv2 + z2_off;
        h2 = z2_elev + y2;
        double yMid = maxOf(0.5 * (y1 + y2), FUDGE);
        double w1 = A.width1[c];
        double wM = scalarWofY(T, xs, yMid);
        double wMsa;
        double wSlotM = slotWidth(sm, yMid, yf, A.conduit_w_max[c], shape);
        if (wSlotM > 0.0)
            wMsa = wSlotM;
        else if (yMid / yf >= crownCutoff(sm) && !shapeIsOpen(shape))
            wMsa = scalarWofY(T, xs, crownCutoff(sm) * yf);
        else
            wMsa = wM;
        surfArea1 = (w1 + wMsa) * length * 0.5;
        A.depth2[c] = y2;
        A.depth_mid[c] = yMid;
        A.width_mid[c] = wM;
        A.h2[c] = h2;
        break;
    }
    case kUpDry: {
        double yMid = maxOf(0.5 * (FUDGE + y2), FUDGE);
        double w1 = A.width1[c];
        double w2 = A.width2[c];
        double wM = A.width_mid[c];
        surfArea2 = (wM + w2) * length / 4.0;
        if (z1_off <= 0.0)
            surfArea1 = (w1 + wM) * length / 4.0;
        A.depth1[c] = FUDGE;
        A.depth_mid[c] = yMid;
        A.width_mid[c] = scalarWofY(T, xs, yMid);
        break;
    }
    case kDnDry: {
        double yMid = maxOf(0.5 * (y1 + FUDGE), FUDGE);
        double w1 = A.width1[c];
        double w2 = A.width2[c];
        double wM = A.width_mid[c];
        surfArea1 = (wM + w1) * length / 4.0;
        if (z2_off <= 0.0)
            surfArea2 = (w2 + wM) * length / 4.0;
        A.depth2[c] = FUDGE;
        A.depth_mid[c] = yMid;
        A.width_mid[c] = scalarWofY(T, xs, yMid);
        break;
    }
    case kDry:
        surfArea1 = FUDGE * length / 2.0;
        surfArea2 = surfArea1;
        break;
    default:
        break;
    }

    A.surf_area1[c] = surfArea1;
    A.surf_area2[c] = surfArea2;
    A.fasnh[c] = fasnh;
    A.flow_class[c] = fc;
}

/**
 * @brief STEP D (DynamicWave.cpp:2509-2542): areas and hydraulic radii through the batch kernels
 *        at the (patched) depths: depth1 to (area1, hrad1), depth2 to area2, depth_mid to
 *        (area_mid, hrad_mid), as XSectGroups::computeAreaHydRadTripleTeam (XSectBatch.cpp:1474)
 *        orders them.
 *
 * @param[in] A Step arrays; area1/2/mid, hrad1 and hrad_mid of conduit c are written.
 * @param[in] c Live conduit.
 */
OPENSWMM_KERNEL_FN void geometryAreas(const Arrays& A, int c) noexcept {
    const int shape = A.conduit_shape[c];
    const double yf = A.conduit_y_full[c];
    const double af = A.conduit_a_full[c];
    const double rf = A.conduit_r_full[c];
    const double wm = A.conduit_w_max[c];
    batchAreaHydrad(A.tables, shape, A.depth1[c], yf, af, rf, wm, A.area1[c], A.hrad1[c]);
    A.area2[c] = batchArea(A.tables, shape, A.depth2[c], yf, af, wm);
    batchAreaHydrad(A.tables, shape, A.depth_mid[c], yf, af, rf, wm, A.area_mid[c], A.hrad_mid[c]);
}

/**
 * @brief Everything DWSolver::computeLinkGeometry produces for one conduit, in its order.
 *
 * A bypassed conduit keeps every value it holds, as the CPU's bypass-masked batch kernels and its
 * STEP A and STEP C skips leave it, except that the SLOT width override still runs for it. The
 * DPS and TPA overrides of STEP E (2544) are out of scope; the EXTRAN and static-slot STEP E
 * overrides belong to DWSolver::momentumKernels and are not part of this stage.
 *
 * @param[in] A Step arrays.
 * @param[in] c Conduit index in [0, n_conduits).
 */
OPENSWMM_KERNEL_FN void linkGeometry(const Arrays& A, int c) noexcept {
    if (A.conduit_bypassed[c]) {
        slotWidthOverride(A, c);
        return;
    }
    geometryDepths(A, c);
    geometryWidths(A, c);
    slotWidthOverride(A, c);
    flowClassSurfArea(A, c);
    geometryAreas(A, c);
}

// ===========================================================================
// Momentum (DWSolver::momentumKernels)
// ===========================================================================

/**
 * @brief DynamicWave.cpp conveyArea (205): a momentum area without the Preissmann slot.
 *
 * @param[in] a Flow area, ft2.
 * @param[in] a_full Full area, ft2.
 * @return min(a, a_full).
 */
OPENSWMM_KERNEL_FN double conveyArea(double a, double a_full) noexcept {
    return (a > a_full) ? a_full : a;
}

/**
 * @brief DynamicWave.cpp slotArea (217): area of the static slot above the crown at depth y.
 *
 * @param[in] y Depth, ft, above y_full.
 * @param[in] y_full Full depth, ft.
 * @param[in] w_max Maximum width, ft.
 * @return Slot area, ft2.
 */
OPENSWMM_KERNEL_FN double slotArea(double y, double y_full, double w_max) noexcept {
    const int n = 8;
    const double eta = minOf(y / y_full, 1.78);
    const double dx = (eta - 1.0) / n;
    double sum = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double f = std::exp(-std::pow(1.0 + i * dx, 2.4));
        if (i == 0 || i == n)
            sum += f;
        else if (i % 2 == 1)
            sum += 4.0 * f;
        else
            sum += 2.0 * f;
    }
    double a = y_full * w_max * 0.5423 * sum * dx / 3.0;
    if (y > 1.78 * y_full)
        a += (y - 1.78 * y_full) * 0.01 * w_max;
    return a;
}

/**
 * @brief hydkernels::froudeNumber (HydClosureKernels.hpp:73), spelled for device code.
 *
 * @param[in] velocity Flow velocity, ft/s; the sign is dropped.
 * @param[in] hyd_depth Hydraulic depth A/W, ft, positive.
 * @return |v| / sqrt(GRAVITY * hyd_depth), in that order.
 */
OPENSWMM_KERNEL_FN double froudeNumber(double velocity, double hyd_depth) noexcept {
    const double v_abs = (velocity < 0.0) ? -velocity : velocity;
    return v_abs / std::sqrt(constants::GRAVITY * hyd_depth);
}

/**
 * @brief STEP E of DWSolver::momentumKernels (DynamicWave.cpp:2778-2847) for EXTRAN and SLOT:
 *        full area and hydraulic radius at and above the crown, plus the slot's area and width.
 *
 * The TPA slot line of that block is out of scope.
 *
 * @param[in] A Step arrays; area1/2/mid, hrad1, hrad_mid and width_mid of conduit c may be
 *     overwritten.
 * @param[in] c Live conduit.
 */
OPENSWMM_KERNEL_FN void staticSlotAreas(const Arrays& A, int c) noexcept {
    const double yf = A.conduit_y_full[c];
    const double af = A.conduit_a_full[c];
    const double rf = A.conduit_r_full[c];
    const double wm = A.conduit_w_max[c];
    const int shape = A.conduit_shape[c];
    const int sm = A.surcharge_method;

    if (A.depth1[c] >= yf) {
        A.area1[c] = af;
        A.hrad1[c] = rf;
    }
    if (A.depth2[c] >= yf) {
        A.area2[c] = af;
    }
    if (A.depth_mid[c] >= yf) {
        A.area_mid[c] = af;
        A.hrad_mid[c] = rf;
    }

    if (A.depth1[c] > yf) {
        const double wSlot = slotWidth(sm, A.depth1[c], yf, wm, shape);
        A.area1[c] = af + (wSlot > 0.0 ? slotArea(A.depth1[c], yf, wm) : 0.0);
        A.hrad1[c] = rf;
    }
    if (A.depth2[c] > yf) {
        const double wSlot = slotWidth(sm, A.depth2[c], yf, wm, shape);
        A.area2[c] = af + (wSlot > 0.0 ? slotArea(A.depth2[c], yf, wm) : 0.0);
    }
    const double yMid = A.depth_mid[c];
    if (yMid > yf) {
        const double wSlot = slotWidth(sm, yMid, yf, wm, shape);
        A.area_mid[c] = af + (wSlot > 0.0 ? slotArea(yMid, yf, wm) : 0.0);
        if (wSlot > 0.0)
            A.width_mid[c] = wSlot;
        A.hrad_mid[c] = rf;
    }
}

/**
 * @brief DWSolver::processDryLink (DynamicWave.cpp:2942): zero flow on a dry, empty or closed
 *        conduit.
 *
 * @param[in] A Step arrays.
 * @param[in] c Live conduit.
 * @param[in] dt_g Routing step times GRAVITY (DWSolver::dt_gravity_).
 */
OPENSWMM_KERNEL_FN void dryLink(const Arrays& A, int c, double dt_g) noexcept {
    const double aMid = A.area_mid[c];
    const double barrels_d = A.conduit_barrels[c];
    A.area_mid[c] = 0.5 * (A.area1[c] + A.area2[c]);
    A.dqdh[c] = dt_g * aMid / A.conduit_length[c] * barrels_d;
    A.froude[c] = 0.0;
    A.new_flow[c] = 0.0;
    A.conduit_q1[c] = 0.0;
    const double yf = A.conduit_y_full[c];
    A.link_depth[c] = minOf(A.depth_mid[c], yf);
    A.link_volume[c] = A.area_mid[c] * A.conduit_links_length[c] * barrels_d;
}

/**
 * @brief DWSolver::applyFlowLimits (DynamicWave.cpp:2979): normal-flow limit, under-relaxation,
 *        flow limit, flap gates and dry ends, then the new flow, depth, volume and full state.
 *
 * The culvert inlet-control branch is out of scope. An in-scope node has no storage row, so
 * lidnode::portDepth is its depth.
 *
 * @param[in] A Step arrays.
 * @param[in] c Live conduit.
 * @param[in] step Zero-based Picard iteration.
 * @param[in] q Per-barrel flow from the momentum equation, cfs.
 * @param[in] qLast Per-barrel flow of the previous iteration, cfs.
 * @param[in] barrels_d Barrel count.
 */
OPENSWMM_KERNEL_FN void flowLimits(const Arrays& A, int c, int step, double q, double qLast,
                                   double barrels_d) noexcept {
    using constants::FUDGE;
    const double yf = A.conduit_y_full[c];
    const int n1 = A.conduit_n1[c];
    const int n2 = A.conduit_n2[c];

    A.inlet_control[c] = 0;
    A.normal_flow_limited[c] = 0;
    if (q > 0.0) {
        const std::int8_t fc2 = A.flow_class[c];
        const int nfl = A.normal_flow_ltd;
        if (nfl != 3 && A.depth1[c] < yf && (fc2 == kSubcritical || fc2 == kSupercritical)) {
            bool hasOutfall = false;
            if (n1 >= 0 && n2 >= 0)
                hasOutfall = (A.node_is_outfall[n1] != 0 || A.node_is_outfall[n2] != 0);
            const bool slope_check =
                (nfl == 0 || nfl == 2 || hasOutfall) && (A.depth1[c] < A.depth2[c]);
            bool froude_check = false;
            if (!slope_check && (nfl == 1 || nfl == 2) && !hasOutfall) {
                if (A.depth1[c] > FUDGE && A.depth2[c] > FUDGE) {
                    const double y1d = A.depth1[c];
                    const bool closed_nearfull =
                        !shapeIsOpen(A.conduit_shape[c]) && (yf - y1d <= FUDGE);
                    if (!closed_nearfull) {
                        const double v1 = q / A.area1[c];
                        const double w1 = scalarWofY(A.tables, sectionOf(A, c), y1d);
                        const double dh1 = A.area1[c] / w1;
                        const double f1 = std::fabs(v1) / std::sqrt(constants::GRAVITY * dh1);
                        froude_check = (f1 >= 1.0);
                    }
                }
            }
            if (slope_check || froude_check) {
                const double qNorm =
                    A.conduit_beta[c] * A.area1[c] * std::pow(A.hrad1[c], 2.0 / 3.0);
                if (qNorm < q) {
                    q = qNorm;
                    A.normal_flow_limited[c] = 1;
                }
            }
        }
    }

    if (step > 0) {
        q = (1.0 - A.omega) * qLast + A.omega * q;
        if (q * qLast < 0.0)
            q = 0.001 * ((q > 0.0) ? 1.0 : -1.0);
    }

    const double q_limit = A.conduit_q_limit[c];
    if (q_limit > 0.0) {
        if (std::fabs(q) > q_limit)
            q = ((q > 0.0) ? 1.0 : -1.0) * q_limit;
    }

    if (A.conduit_has_flap_gate[c]) {
        if (q * static_cast<double>(A.conduit_direction[c]) < 0.0)
            q = 0.0;
    }
    if (q < 0.0 && n2 >= 0 && A.node_outfall_flap[n2])
        q = 0.0;
    if (q > 0.0 && n1 >= 0 && A.node_outfall_flap[n1])
        q = 0.0;

    if (q > FUDGE && A.node_depth[n1] <= FUDGE)
        q = FUDGE;
    if (q < -FUDGE && A.node_depth[n2] <= FUDGE)
        q = -FUDGE;

    A.new_flow[c] = q * barrels_d;
    A.conduit_q1[c] = q;

    A.link_depth[c] = minOf(A.depth_mid[c], yf);
    const double aMidAvg = (A.area1[c] + A.area2[c]) * 0.5;
    A.link_volume[c] = aMidAvg * A.conduit_links_length[c] * barrels_d;

    const double a_full = A.conduit_a_full[c];
    const bool up_full = A.area1[c] >= a_full;
    const bool dn_full = A.area2[c] >= a_full;
    const bool mid_full = A.area_mid[c] >= a_full;
    A.full_state[c] =
        static_cast<std::int8_t>((up_full ? 1 : 0) | (dn_full ? 2 : 0) | (mid_full ? 4 : 0));
}

/**
 * @brief DWSolver::processManningLink (DynamicWave.cpp:3135) for the Manning categories.
 *
 * Out of scope and left out: the virtual-junction coupling (its zero term is still added, for
 * the sign of zero), unsteady friction, and evaporation and seepage, whose rates are zero for a
 * closed conduit with no seepage (DWSolver::recomputeConduitLossOne). An in-scope conduit is
 * closed, so the Froude branch never sees a midpoint depth above full, where the CPU
 * re-evaluates the area.
 *
 * @param[in] A Step arrays.
 * @param[in] c Live conduit.
 * @param[in] dt Routing step, s.
 * @param[in] dt_g dt times GRAVITY (DWSolver::dt_gravity_).
 * @param[in] step Zero-based Picard iteration.
 * @param[in] cat kManningOpen, kManningClosedFs or kManningClosedFull.
 */
OPENSWMM_KERNEL_FN void manningLink(const Arrays& A, int c, double dt, double dt_g, int step,
                                    std::uint8_t cat) noexcept {
    using constants::FUDGE;
    using constants::MAX_VELOCITY;
    const bool is_closed_full = (cat == kManningClosedFull);
    const bool is_open_cat = (cat == kManningOpen);

    const double barrels_d = A.conduit_barrels[c];
    const double length = A.conduit_length[c];
    const double aMid = A.area_mid[c];
    const double rMid = A.hrad_mid[c];
    const double qLast = A.conduit_q1[c];
    const double yf = A.conduit_y_full[c];
    const bool isFull = (A.depth1[c] >= yf && A.depth2[c] >= yf);

    const double aFull = A.conduit_a_full[c];
    const double a1Conv = conveyArea(A.area1[c], aFull);
    const double a2Conv = conveyArea(A.area2[c], aFull);
    const double aMidConv = conveyArea(aMid, aFull);

    double v = qLast / aMidConv;
    double absv = std::fabs(v);
    if (absv > MAX_VELOCITY) {
        v = (v > 0.0) ? MAX_VELOCITY : -MAX_VELOCITY;
        absv = MAX_VELOCITY;
    }
    A.velocity[c] = v;

    double fr = 0.0;
    double sig = 0.0;
    if (!is_closed_full) {
        double wMid = A.width_mid[c];
        if (A.flow_class[c] == kSubcritical && A.wcap_dm[c] < A.depth_mid[c])
            wMid = scalarWofY(A.tables, sectionOf(A, c), A.depth_mid[c]);
        const bool closed_nearfull =
            !shapeIsOpen(A.conduit_shape[c]) && (yf - A.depth_mid[c] <= FUDGE);
        if (A.depth_mid[c] > FUDGE && !closed_nearfull) {
            const double aFr = aMid;
            const double dh = (wMid > 0.0) ? aFr / wMid : 0.0;
            fr = (dh > 0.0) ? froudeNumber(absv, dh) : 0.0;
        }
        if (A.flow_class[c] == kSubcritical && fr > 1.0)
            A.flow_class[c] = kSupercritical;
        sig = maxOf(0.0, minOf(1.0, 2.0 * (1.0 - fr)));
    }
    A.froude[c] = fr;

    const double h1 = A.h1[c];
    const double h2 = A.h2[c];

    const double r1_val = A.hrad1[c];
    double rho = 1.0;
    if (!is_closed_full && !isFull && qLast > 0.0 && h1 >= h2)
        rho = sig;
    const double aWtd = a1Conv + (aMidConv - a1Conv) * rho;
    const double rWtd = r1_val + (rMid - r1_val) * rho;

    if (!is_closed_full) {
        if (A.inertial_damping == 0)
            sig = 1.0;
        else if (A.inertial_damping == 2)
            sig = 0.0;
        if (!is_open_cat && isFull)
            sig = 0.0;
    }
    A.sigma[c] = sig;

    const double r43 = std::pow(rWtd, 1.33333);
    const double dq1 = dt * A.conduit_rough_factor[c] / r43 * absv;
    const double dq2 = dt_g * aWtd * (h2 - h1) / length;

    const double aOld = maxOf(conveyArea(A.area_old[c], aFull), FUDGE);
    double dq3 = 0.0, dq4 = 0.0;
    if (sig > 0.0) {
        dq3 = 2.0 * v * (aMidConv - aOld) * sig;
        if (length > 0.0)
            dq4 = dt * v * v * (a2Conv - a1Conv) / length * sig;
        dq4 += 0.0; // the CPU adds the virtual-junction term, zero here
    }

    double dq5 = 0.0;
    if (A.conduit_has_losses[c]) {
        const double absq = std::fabs(qLast);
        double losses = 0.0;
        if (a1Conv > FUDGE)
            losses += A.conduit_loss_inlet[c] * (absq / a1Conv);
        if (a2Conv > FUDGE)
            losses += A.conduit_loss_outlet[c] * (absq / a2Conv);
        if (aMidConv > FUDGE)
            losses += A.conduit_loss_avg[c] * (absq / aMidConv);
        dq5 = losses / 2.0 / length * dt;
    }

    const double dq6 = 0.0;

    const double qOld = A.conduit_old_flow[c] / barrels_d;
    const double denom = 1.0 + dq1 + dq5;
    const double q = (qOld - dq2 + dq3 + dq4 + dq6) / denom;
    A.dqdh[c] = 1.0 / denom * constants::GRAVITY * dt * aWtd / length * barrels_d;

    flowLimits(A, c, step, q, qLast, barrels_d);
}

/**
 * @brief One conduit of DWSolver::momentumKernels (DynamicWave.cpp:2735-2919, solve_conduit).
 *
 * A bypassed conduit keeps its flow and dQ/dH. Out of scope and left out: the loss-rate recompute
 * (zero for a closed conduit with no seepage), the force-main categories, and the chamber volume
 * of a node with a storage area.
 *
 * @param[in] A Step arrays.
 * @param[in] c Conduit index in [0, n_conduits).
 * @param[in] dt Routing step, s.
 * @param[in] step Zero-based Picard iteration.
 */
OPENSWMM_KERNEL_FN void conduitMomentum(const Arrays& A, int c, double dt, int step) noexcept {
    using constants::FUDGE;
    A.new_flow[c] = A.link_flow[c];
    if (A.conduit_bypassed[c])
        return;

    A.dqdh[c] = 0.0;
    staticSlotAreas(A, c);

    const std::int8_t fc = A.flow_class[c];
    const double aMid = A.area_mid[c];
    const double yf_c = A.conduit_y_full[c];
    const bool isFull = (A.depth1[c] >= yf_c && A.depth2[c] >= yf_c);
    const bool is_open = shapeIsOpen(A.conduit_shape[c]);

    std::uint8_t cat;
    if (fc == kDry || fc == kUpDry || fc == kDnDry || aMid <= FUDGE || A.conduit_is_closed[c] ||
        A.conduit_setting[c] == 0.0) {
        cat = kSkipDry;
    } else if (!is_open && isFull) {
        cat = kManningClosedFull;
    } else if (is_open) {
        cat = kManningOpen;
    } else {
        cat = kManningClosedFs;
    }
    A.category[c] = cat;

    // DWSolver::execute sets dt_gravity_ the same way.
    const double dt_g = dt * constants::GRAVITY;
    if (cat == kSkipDry)
        dryLink(A, c, dt_g);
    else
        manningLink(A, c, dt, dt_g, step, cat);

    A.link_flow[c] = A.new_flow[c];
    A.link_dqdh[c] = A.dqdh[c];
}

} // namespace openswmm::dynwave::step

#endif // OPENSWMM_ENGINE_DW_KERNELS_HPP
