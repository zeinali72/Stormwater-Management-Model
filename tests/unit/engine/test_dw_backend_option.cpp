// SPDX-License-Identifier: Apache-2.0
/**
 * @file      test_dw_backend_option.cpp
 * @brief     Gates for the `[OPTIONS] DW_BACKEND` key of FLOW_ROUTING DYNWAVE.
 *
 * @details `DW_BACKEND` selects where the classic dynamic wave solver runs, the way `FV_BACKEND`
 *          does for the finite-volume router. In this version the key is only accepted and
 *          carried: device tokens run the CPU solver and say so with one warning. The gates are
 *
 *            1. tokens `CPU AUTO OMP CUDA HIP SYCL` in any letter case; any other token is
 *               rejected the way an unknown `FV_BACKEND` token is (including `METAL`, which only
 *               the FV key accepts);
 *            2. the default is `AUTO`;
 *            3. the key is readable/writable through the string-keyed options API and is listed
 *               by the options key enumeration, values read back upper-case;
 *            4. `.inp` (and GeoPackage) round trip, default written as `FV_BACKEND` is;
 *            5. the `OPENSWMM_DW_BACKEND` environment override, an unrecognised value acting as
 *               `CPU`;
 *            6. the run-time effect: exactly one `DW_BACKEND ... CPU solver ...` warning for a
 *               device token, none for `CPU`/`AUTO`, byte-identical binary output and end-of-run
 *               hot start against `DW_BACKEND CPU`, and no effect under another FLOW_ROUTING;
 *            7. a model that does not use the key is not changed by it.
 *
 *          Every artefact lands in tests/output/dw_backend_option/<case>/ (CLAUDE.md §4.1).
 *
 * @ingroup engine_hydraulics
 *
 * @author    Farzan Zeinali
 * @license   Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "openswmm/engine/openswmm_engine.h"
#include "openswmm/engine/openswmm_model.h"
#include "platform_test_support.hpp"

namespace fs = std::filesystem;

namespace {

/// Shared regression corpus (Example1) and the unit-test data tree (structure decks).
const char* kDataDir = OPENSWMM_DW_TEST_DATA_DIR;
const char* kUnitDataDir = OPENSWMM_DW_UNIT_DATA_DIR;
const char* kEnv = "OPENSWMM_DW_BACKEND";

#ifdef OPENSWMM_HAS_GEOPACKAGE
const char* kGeoPackagePlugin = "org.hydrocouple.openswmm.plugins.geopackage";
#endif

/** @brief Upper-case an ASCII string. */
std::string upper(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

/** @brief Whole file as bytes; empty when the file cannot be read. */
std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

/** @brief Run directory for one case, created (and emptied) under tests/output. */
fs::path caseDir(const std::string& name) {
    const fs::path d = fs::path(OPENSWMM_DW_TEST_OUT_DIR) / "dw_backend_option" / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d, ec);
    return d;
}

/** @brief An existing DYNWAVE test model the cases are run on. */
struct Model {
    const char* name;  ///< short name used in artefact directories
    std::string path;  ///< deck in the repository
    bool cut_to_storm; ///< shorten the run window (the full deck runs for 30 h)
};

/// Example1: subcatchment runoff into a conduit network (conduits only).
Model example1() {
    return {"example1", std::string(kDataDir) + "/Example1.inp", true};
}

/// fv_structures: pump, orifice, weir, outlet, storage node, control rules, ponding and flap gates.
Model structures() {
    return {"structures", std::string(kUnitDataDir) + "/fv_structures.inp", false};
}

/// ideal_pump: ideal and curve pumps drawing from storage nodes.
Model pumps() {
    return {"pumps", std::string(kUnitDataDir) + "/pumps/ideal_pump.inp", false};
}

/**
 * @brief Write a copy of an existing model into @p dir/deck.inp with extra [OPTIONS] lines.
 *
 * @p routing, when not empty, replaces FLOW_ROUTING; @p extra lines follow it. For Example1 the
 * window is cut to the first 13.5 h (as the FV integration tests do). When @p hotstart is true,
 * `[FILES] SAVE HOTSTART dir/run.hsf` is appended so the end-of-run hot start lands next to the
 * other artefacts. The deck file name and every sibling file name are the same for every case, so
 * differing bytes cannot come from differing names.
 *
 * @return Path of the deck.
 */
fs::path writeDeck(const fs::path& dir, const Model& model, const std::string& routing,
                   const std::vector<std::string>& extra, bool hotstart = true) {
    std::ifstream in(model.path);
    EXPECT_TRUE(in.good()) << "cannot open " << model.path;
    const fs::path deck = dir / "deck.inp";
    std::ofstream os(deck);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("FLOW_ROUTING", 0) == 0) {
            os << "FLOW_ROUTING         " << (routing.empty() ? std::string("DYNWAVE") : routing)
               << "\n";
            for (const auto& e : extra)
                os << e << "\n";
        } else if (model.cut_to_storm && line.rfind("END_DATE", 0) == 0) {
            os << "END_DATE             01/01/1998\n";
        } else if (model.cut_to_storm && line.rfind("END_TIME", 0) == 0) {
            os << "END_TIME             13:30:00\n";
        } else {
            os << line << "\n";
        }
    }
    if (hotstart)
        os << "\n[FILES]\nSAVE HOTSTART \"" << (dir / "run.hsf").string() << "\"\n";
    return deck;
}

/** @brief Everything one run leaves behind that a test looks at. */
struct RunResult {
    int rc = -1;                       ///< first non-zero lifecycle code, else 0
    std::vector<std::string> warnings; ///< the warnings API after the run
    std::vector<std::string> errors;   ///< the errors API after the run
    std::string rpt;                   ///< report file text
    std::string out;                   ///< binary output bytes
    std::string hsf;                   ///< end-of-run hot start bytes
    std::string dw_option;             ///< DW_BACKEND read back through the API ("" if unreadable)
};

/**
 * @brief Open, initialize, run to the end and close the deck in @p dir with the step API.
 *
 * Collects the warnings and errors lists before the engine is closed and the three files the run
 * writes (`run.rpt`, `run.out`, `run.hsf`, all in the deck's directory).
 */
RunResult runDeck(const fs::path& deck) {
    const fs::path dir = deck.parent_path();
    const std::string rpt = (dir / "run.rpt").string();
    const std::string out = (dir / "run.out").string();
    std::error_code ec;
    fs::remove(dir / "run.hsf", ec);

    RunResult r;
    SWMM_Engine e = swmm_engine_create();
    EXPECT_NE(e, nullptr);
    if (!e)
        return r;
    r.rc = swmm_engine_open(e, deck.string().c_str(), rpt.c_str(), out.c_str(), nullptr);
    if (r.rc == 0) {
        char buf[64] = {};
        if (swmm_options_get(e, "DW_BACKEND", buf, sizeof buf) == 0)
            r.dw_option = buf;
        r.rc = swmm_engine_initialize(e);
        if (r.rc == 0)
            r.rc = swmm_engine_start(e, 1);
    }
    if (r.rc == 0) {
        double elapsed = 0.0;
        int rc = 0;
        while ((rc = swmm_engine_step(e, &elapsed)) == 0 && elapsed > 0.0) {
        }
        r.rc = rc;
        const int rc_end = swmm_engine_end(e);
        if (r.rc == 0)
            r.rc = rc_end;
        const int rc_rpt = swmm_engine_report(e);
        if (r.rc == 0)
            r.rc = rc_rpt;
    }
    for (int i = 0, n = swmm_get_warning_count(e); i < n; ++i)
        r.warnings.emplace_back(swmm_get_warning_at(e, i));
    for (int i = 0, n = swmm_get_error_count(e); i < n; ++i)
        r.errors.emplace_back(swmm_get_error_at(e, i));
    swmm_engine_close(e);
    swmm_engine_destroy(e);
    r.rpt = slurp(dir / "run.rpt");
    r.out = slurp(dir / "run.out");
    r.hsf = slurp(dir / "run.hsf");
    return r;
}

/** @brief Warnings that mention DW_BACKEND. */
std::vector<std::string> dwWarnings(const RunResult& r) {
    std::vector<std::string> v;
    for (const auto& w : r.warnings)
        if (w.find("DW_BACKEND") != std::string::npos)
            v.push_back(w);
    return v;
}

/** @brief Warnings that carry the CPU-fallback message: mention DW_BACKEND and "CPU solver". */
std::vector<std::string> fallbackWarnings(const RunResult& r) {
    std::vector<std::string> v;
    for (const auto& w : r.warnings)
        if (w.find("DW_BACKEND") != std::string::npos && w.find("CPU solver") != std::string::npos)
            v.push_back(w);
    return v;
}

/** @brief Report-file lines that carry the CPU-fallback message. */
std::vector<std::string> fallbackLines(const std::string& rpt) {
    std::istringstream ss(rpt);
    std::vector<std::string> v;
    std::string line;
    while (std::getline(ss, line))
        if (line.find("DW_BACKEND") != std::string::npos &&
            line.find("CPU solver") != std::string::npos)
            v.push_back(line);
    return v;
}

/** @brief Number of report-file lines that mention @p word. */
int reportMentions(const std::string& rpt, const std::string& word) {
    std::istringstream ss(rpt);
    std::string line;
    int n = 0;
    while (std::getline(ss, line))
        if (line.find(word) != std::string::npos)
            ++n;
    return n;
}

/**
 * @brief Whether @p text names backend @p token as a whole word, ignoring ASCII case.
 *
 * A bare substring would find `omp` inside other words; the warning may print the backend in
 * either case, as the FV warnings print theirs in lower case.
 */
bool namesBackend(const std::string& text, const std::string& token) {
    const std::string hay = upper(text), want = upper(token);
    auto word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    for (std::size_t at = hay.find(want); at != std::string::npos; at = hay.find(want, at + 1)) {
        const bool before = at == 0 || !word(hay[at - 1]);
        const bool after = at + want.size() >= hay.size() || !word(hay[at + want.size()]);
        if (before && after)
            return true;
    }
    return false;
}

/** @brief Read an option as text; "<error N>" when the API refuses the key. */
std::string getOption(SWMM_Engine e, const char* key) {
    char buf[128] = {};
    const int rc = swmm_options_get(e, key, buf, sizeof buf);
    return rc == 0 ? std::string(buf) : "<error " + std::to_string(rc) + ">";
}

/** @brief Open @p deck on a fresh engine, with no report or output file. */
SWMM_Engine openDeck(const fs::path& deck, int* rc = nullptr, const char* plugin = nullptr) {
    SWMM_Engine e = swmm_engine_create();
    const int r = swmm_engine_open(e, deck.string().c_str(), "", "", plugin);
    if (rc)
        *rc = r;
    return e;
}

/** @brief Close and destroy an engine. */
void drop(SWMM_Engine e) {
    swmm_engine_close(e);
    swmm_engine_destroy(e);
}

/**
 * @brief Key/value pairs of the [OPTIONS] section of a written model file.
 *
 * Only the first token after the key is kept, as the FV round-trip tests do.
 */
std::vector<std::pair<std::string, std::string>> optionsSection(const fs::path& inp) {
    std::ifstream in(inp);
    std::vector<std::pair<std::string, std::string>> rows;
    std::string line;
    bool inside = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '[') {
            inside = upper(line).rfind("[OPTIONS]", 0) == 0;
            continue;
        }
        if (!inside)
            continue;
        std::istringstream ss(line);
        std::string k, v;
        if (ss >> k >> v)
            rows.emplace_back(upper(k), v);
    }
    return rows;
}

/** @brief Value of @p key in @p rows, or "<absent>". */
std::string rowValue(const std::vector<std::pair<std::string, std::string>>& rows,
                     const std::string& key) {
    for (const auto& [k, v] : rows)
        if (k == key)
            return v;
    return "<absent>";
}

/** The device tokens: the run carries the CPU-solver warning. */
const char* const kDeviceTokens[] = {"OMP", "CUDA", "HIP", "SYCL"};

/**
 * Fixture: OPENSWMM_DW_BACKEND is cleared before every test and restored after, so a variable in
 * the caller's environment cannot change a test and a test cannot leak into the next.
 */
class DwBackendOption : public ::testing::Test {
protected:
    void SetUp() override {
        const char* prev = std::getenv(kEnv);
        had_ = prev != nullptr;
        saved_ = prev ? prev : "";
        plattest::unsetEnvVar(kEnv);
    }
    void TearDown() override {
        if (had_)
            plattest::setEnvVar(kEnv, saved_.c_str());
        else
            plattest::unsetEnvVar(kEnv);
    }

private:
    bool had_ = false;
    std::string saved_;
};

} // namespace

// ---------------------------------------------------------------------------
// 1. Key and tokens
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, DeckTokensAreAcceptedInAnyLetterCase) {
    const struct {
        const char* token;
        const char* want;
    } cases[] = {{"cpu", "CPU"},   {"Auto", "AUTO"}, {"omp", "OMP"},
                 {"CUDA", "CUDA"}, {"hIp", "HIP"},   {"SyCl", "SYCL"}};
    for (const auto& c : cases) {
        SCOPED_TRACE(c.token);
        const fs::path dir = caseDir(std::string("tok_") + c.want);
        const fs::path deck = writeDeck(dir, example1(), "DYNWAVE",
                                        {std::string("DW_BACKEND          ") + c.token}, false);
        int rc = -1;
        SWMM_Engine e = openDeck(deck, &rc);
        ASSERT_EQ(rc, 0);
        EXPECT_EQ(getOption(e, "DW_BACKEND"), c.want);
        drop(e);
    }
}

TEST_F(DwBackendOption, ApiAcceptsTokensInAnyLetterCaseAndReadsBackUpperCase) {
    const fs::path dir = caseDir("api_tokens");
    SWMM_Engine e = openDeck(writeDeck(dir, example1(), "DYNWAVE", {}, false));
    const char* tokens[] = {"cpu", "Auto", "omp", "Cuda", "HIP", "sycl", "OMP"};
    for (const char* t : tokens) {
        SCOPED_TRACE(t);
        ASSERT_EQ(swmm_options_set(e, "DW_BACKEND", t), 0);
        EXPECT_EQ(getOption(e, "DW_BACKEND"), upper(t));
    }
    drop(e);
}

TEST_F(DwBackendOption, ApiRejectsABadTokenLikeABadFvBackendToken) {
    const fs::path dir = caseDir("api_bad_token");
    SWMM_Engine e = openDeck(writeDeck(dir, example1(), "DYNWAVE", {}, false));

    // The FV key's rejection is the reference: same status code for the same kind of mistake.
    const int fv_rc = swmm_options_set(e, "FV_BACKEND", "OPENCL");
    ASSERT_NE(fv_rc, 0) << "FV_BACKEND must still reject an unknown token";

    ASSERT_EQ(swmm_options_set(e, "DW_BACKEND", "OMP"), 0);
    for (const char* bad : {"OPENCL", "", "GPU", "OMP2", "METAL"}) {
        SCOPED_TRACE(bad);
        EXPECT_EQ(swmm_options_set(e, "DW_BACKEND", bad), fv_rc);
        // A rejected token does not clobber the previous value.
        EXPECT_EQ(getOption(e, "DW_BACKEND"), "OMP");
    }
    drop(e);
}

TEST_F(DwBackendOption, DeckBadTokenIsHandledAsAFvBackendBadTokenIs) {
    // A bad token in the deck goes through the [OPTIONS] parser. Whatever it does for FV_BACKEND
    // (today: the line is skipped, the value stays at its default, the model still opens) it
    // must do for DW_BACKEND. The comparison is the contract; the final check makes sure the
    // key is known at all.
    const fs::path dfv = caseDir("deck_bad_fv");
    const fs::path ddw = caseDir("deck_bad_dw");
    int rc_fv = -1, rc_dw = -1;
    SWMM_Engine fv = openDeck(
        writeDeck(dfv, example1(), "DYNWAVE", {"FV_BACKEND          OPENCL"}, false), &rc_fv);
    SWMM_Engine dw = openDeck(
        writeDeck(ddw, example1(), "DYNWAVE", {"DW_BACKEND          OPENCL"}, false), &rc_dw);
    EXPECT_EQ(rc_dw, rc_fv);
    EXPECT_EQ(swmm_get_error_count(dw), swmm_get_error_count(fv));
    EXPECT_EQ(swmm_get_warning_count(dw), swmm_get_warning_count(fv));
    EXPECT_EQ(getOption(dw, "DW_BACKEND"), "AUTO") << "a rejected token leaves the default";
    drop(fv);
    drop(dw);

    // METAL is an FV-only token: it is not a DW_BACKEND token and is rejected like any other.
    const fs::path dm = caseDir("deck_metal_dw");
    int rc = -1;
    SWMM_Engine m =
        openDeck(writeDeck(dm, example1(), "DYNWAVE", {"DW_BACKEND          METAL"}, false), &rc);
    EXPECT_EQ(rc, rc_fv);
    EXPECT_EQ(getOption(m, "DW_BACKEND"), "AUTO");
    drop(m);
}

// ---------------------------------------------------------------------------
// 2. Default
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, DefaultIsAuto) {
    const fs::path dir = caseDir("default_auto");
    SWMM_Engine e = openDeck(writeDeck(dir, example1(), "DYNWAVE", {}, false));
    EXPECT_EQ(getOption(e, "DW_BACKEND"), "AUTO");
    // The FV key keeps its own default and the two do not alias.
    EXPECT_EQ(getOption(e, "FV_BACKEND"), "AUTO");
    ASSERT_EQ(swmm_options_set(e, "DW_BACKEND", "OMP"), 0);
    EXPECT_EQ(getOption(e, "FV_BACKEND"), "AUTO");
    ASSERT_EQ(swmm_options_set(e, "FV_BACKEND", "CPU"), 0);
    EXPECT_EQ(getOption(e, "DW_BACKEND"), "OMP");
    drop(e);
}

// ---------------------------------------------------------------------------
// 3. Programmatic access and key listing
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, KeyIsListedWithTheOtherOptionKeys) {
    const int n = swmm_options_key_count();
    ASSERT_GT(n, 0);
    int fv_at = -1, dw_at = -1;
    for (int i = 0; i < n; ++i) {
        char buf[128] = {};
        ASSERT_EQ(swmm_options_key_at(i, buf, sizeof buf), 0);
        if (std::string(buf) == "FV_BACKEND")
            fv_at = i;
        if (std::string(buf) == "DW_BACKEND") {
            EXPECT_EQ(dw_at, -1) << "DW_BACKEND listed twice";
            dw_at = i;
        }
    }
    EXPECT_GE(fv_at, 0) << "control: FV_BACKEND is listed";
    EXPECT_GE(dw_at, 0) << "DW_BACKEND is not listed by swmm_options_key_at";
}

// ---------------------------------------------------------------------------
// 4. Round trip
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, InpRoundTripKeepsEveryToken) {
    for (const char* token : {"CPU", "AUTO", "OMP", "CUDA", "HIP", "SYCL"}) {
        SCOPED_TRACE(token);
        const fs::path dir = caseDir(std::string("rt_inp_") + token);
        // Written lower-case in the deck, read as upper-case, written back upper-case.
        std::string deck_token = token;
        std::transform(deck_token.begin(), deck_token.end(), deck_token.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        SWMM_Engine e = openDeck(
            writeDeck(dir, example1(), "DYNWAVE", {"DW_BACKEND          " + deck_token}, false));
        ASSERT_EQ(getOption(e, "DW_BACKEND"), token);
        const fs::path written = dir / "written.inp";
        ASSERT_EQ(swmm_model_write(e, written.string().c_str()), 0);
        drop(e);

        EXPECT_EQ(rowValue(optionsSection(written), "DW_BACKEND"), token);

        int rc = -1;
        SWMM_Engine again = openDeck(written, &rc);
        ASSERT_EQ(rc, 0);
        EXPECT_EQ(getOption(again, "DW_BACKEND"), token);
        // A second generation is stable.
        const fs::path written2 = dir / "written2.inp";
        ASSERT_EQ(swmm_model_write(again, written2.string().c_str()), 0);
        drop(again);
        EXPECT_EQ(rowValue(optionsSection(written2), "DW_BACKEND"), token);
    }
}

TEST_F(DwBackendOption, InpRoundTripKeepsAValueSetThroughTheApi) {
    const fs::path dir = caseDir("rt_inp_api");
    SWMM_Engine e = openDeck(writeDeck(dir, example1(), "DYNWAVE", {}, false));
    ASSERT_EQ(swmm_options_set(e, "DW_BACKEND", "hip"), 0);
    const fs::path written = dir / "written.inp";
    ASSERT_EQ(swmm_model_write(e, written.string().c_str()), 0);
    drop(e);
    int rc = -1;
    SWMM_Engine again = openDeck(written, &rc);
    ASSERT_EQ(rc, 0);
    EXPECT_EQ(getOption(again, "DW_BACKEND"), "HIP");
    drop(again);
}

TEST_F(DwBackendOption, InpWriterWritesTheDefaultAsItDoesForFvBackend) {
    // FV_BACKEND is written even at its default; DW_BACKEND follows the same rule.
    const fs::path dir = caseDir("rt_inp_default");
    SWMM_Engine e = openDeck(writeDeck(dir, example1(), "DYNWAVE", {}, false));
    const fs::path written = dir / "written.inp";
    ASSERT_EQ(swmm_model_write(e, written.string().c_str()), 0);
    drop(e);
    const auto rows = optionsSection(written);
    ASSERT_EQ(rowValue(rows, "FV_BACKEND"), "AUTO") << "control: FV_BACKEND default is written";
    EXPECT_EQ(rowValue(rows, "DW_BACKEND"), "AUTO");
}

#ifdef OPENSWMM_HAS_GEOPACKAGE
TEST_F(DwBackendOption, GeoPackageRoundTripKeepsEveryToken) {
    // The GeoPackage stores the option rows FV_BACKEND is stored in; this key must be there too.
    for (const char* token : {"AUTO", "CPU", "OMP", "CUDA", "HIP", "SYCL"}) {
        SCOPED_TRACE(token);
        const fs::path dir = caseDir(std::string("rt_gpkg_") + token);
        SWMM_Engine e = openDeck(writeDeck(dir, example1(), "DYNWAVE", {}, false));
        ASSERT_EQ(swmm_options_set(e, "DW_BACKEND", token), 0);
        ASSERT_EQ(swmm_options_set(e, "FV_BACKEND", "CPU"), 0);
        const fs::path written = dir / "saved.gpkg";
        ASSERT_EQ(swmm_model_write_with_plugin(e, written.string().c_str(), kGeoPackagePlugin), 0);
        drop(e);

        int rc = -1;
        SWMM_Engine again = openDeck(written, &rc, kGeoPackagePlugin);
        ASSERT_EQ(rc, 0);
        EXPECT_EQ(getOption(again, "FV_BACKEND"), "CPU") << "control: FV_BACKEND round-trips";
        EXPECT_EQ(getOption(again, "DW_BACKEND"), token);
        drop(again);
    }
}
#endif

// ---------------------------------------------------------------------------
// 5. Environment override
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, EnvironmentDeviceTokenOverridesACpuOption) {
    for (const char* env : {"omp", "Cuda", "HIP", "sycl"}) {
        SCOPED_TRACE(env);
        const fs::path dir = caseDir(std::string("env_over_") + upper(env));
        const fs::path deck = writeDeck(dir, example1(), "DYNWAVE", {"DW_BACKEND          CPU"});
        plattest::setEnvVar(kEnv, env);
        const RunResult r = runDeck(deck);
        plattest::unsetEnvVar(kEnv);
        ASSERT_EQ(r.rc, 0);
        EXPECT_EQ(r.dw_option, "CPU") << "the environment overrides at run time, not in the model";
        const auto w = fallbackWarnings(r);
        ASSERT_EQ(w.size(), 1u) << "env " << env << " over option CPU";
        EXPECT_TRUE(namesBackend(w[0], env)) << w[0];
    }
}

TEST_F(DwBackendOption, EnvironmentCpuOrAutoSilencesADeviceOption) {
    const fs::path dir = caseDir("env_silences");
    const fs::path deck = writeDeck(dir, example1(), "DYNWAVE", {"DW_BACKEND          OMP"});

    // Control: with the variable unset the deck's own OMP request warns once.
    const RunResult base = runDeck(deck);
    ASSERT_EQ(base.rc, 0);
    ASSERT_EQ(base.dw_option, "OMP");
    ASSERT_EQ(fallbackWarnings(base).size(), 1u);

    for (const char* env : {"cpu", "CPU", "auto", "Auto"}) {
        SCOPED_TRACE(env);
        plattest::setEnvVar(kEnv, env);
        const RunResult r = runDeck(deck);
        plattest::unsetEnvVar(kEnv);
        ASSERT_EQ(r.rc, 0);
        EXPECT_EQ(r.dw_option, "OMP");
        EXPECT_TRUE(dwWarnings(r).empty()) << "env " << env;
        EXPECT_EQ(r.out, base.out);
        EXPECT_EQ(r.hsf, base.hsf);
    }
}

TEST_F(DwBackendOption, UnrecognisedEnvironmentValueBehavesAsCpu) {
    const fs::path dir = caseDir("env_unrecognised");
    const fs::path deck = writeDeck(dir, example1(), "DYNWAVE", {"DW_BACKEND          CUDA"});

    const RunResult base = runDeck(deck);
    ASSERT_EQ(base.rc, 0);
    ASSERT_EQ(base.dw_option, "CUDA");
    ASSERT_EQ(fallbackWarnings(base).size(), 1u) << "control: the deck's CUDA request warns once";

    for (const char* env : {"bogus", "METAL", "opencl"}) {
        SCOPED_TRACE(env);
        plattest::setEnvVar(kEnv, env);
        const RunResult r = runDeck(deck);
        plattest::unsetEnvVar(kEnv);
        ASSERT_EQ(r.rc, 0);
        EXPECT_EQ(r.dw_option, "CUDA");
        EXPECT_TRUE(dwWarnings(r).empty()) << "env " << env << " must act as CPU";
        EXPECT_EQ(r.out, base.out);
        EXPECT_EQ(r.hsf, base.hsf);
    }
}

TEST_F(DwBackendOption, EnvironmentOverrideAppliesToADeckWithNoKey) {
    const fs::path dir = caseDir("env_no_key");
    const fs::path deck = writeDeck(dir, example1(), "DYNWAVE", {});
    plattest::setEnvVar(kEnv, "sycl");
    const RunResult r = runDeck(deck);
    plattest::unsetEnvVar(kEnv);
    ASSERT_EQ(r.rc, 0);
    EXPECT_EQ(r.dw_option, "AUTO");
    const auto w = fallbackWarnings(r);
    ASSERT_EQ(w.size(), 1u);
    EXPECT_TRUE(namesBackend(w[0], "SYCL")) << w[0];
}

// ---------------------------------------------------------------------------
// 6. Run-time effect
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, DeviceTokenRunsTheCpuSolverWithExactlyOneWarning) {
    // Three existing DYNWAVE models: conduits only, then two with non-conduit links and storage.
    for (const Model& m : {example1(), structures(), pumps()}) {
        SCOPED_TRACE(m.name);
        const RunResult cpu = runDeck(writeDeck(caseDir(std::string("run_") + m.name + "_CPU"), m,
                                                "DYNWAVE", {"DW_BACKEND          CPU"}));
        ASSERT_EQ(cpu.rc, 0) << (cpu.errors.empty() ? "" : cpu.errors.front());
        ASSERT_EQ(cpu.dw_option, "CPU");
        ASSERT_FALSE(cpu.out.empty());
        ASSERT_FALSE(cpu.hsf.empty()) << "the deck's [FILES] SAVE HOTSTART wrote nothing";

        for (const char* token : kDeviceTokens) {
            SCOPED_TRACE(token);
            const RunResult r =
                runDeck(writeDeck(caseDir(std::string("run_") + m.name + "_" + token), m, "DYNWAVE",
                                  {std::string("DW_BACKEND          ") + token}));
            ASSERT_EQ(r.rc, 0) << (r.errors.empty() ? "" : r.errors.front());
            EXPECT_EQ(r.dw_option, token);

            // Warnings API: exactly one, naming the key, the fallback and the backend.
            const auto w = fallbackWarnings(r);
            ASSERT_EQ(w.size(), 1u) << "warnings naming DW_BACKEND and the CPU solver";
            EXPECT_TRUE(namesBackend(w[0], token)) << w[0];

            // Report file: the same single warning is visible there.
            const auto lines = fallbackLines(r.rpt);
            ASSERT_EQ(lines.size(), 1u) << "report file";
            EXPECT_TRUE(namesBackend(lines[0], token)) << lines[0];

            // Results are those of DW_BACKEND CPU, byte for byte.
            EXPECT_EQ(r.out, cpu.out) << "binary output differs from DW_BACKEND CPU";
            EXPECT_EQ(r.hsf, cpu.hsf) << "end-of-run hot start differs from DW_BACKEND CPU";
        }
    }
}

TEST_F(DwBackendOption, DeviceTokenGivenInAnyCaseIsNamedInTheWarning) {
    const fs::path dir = caseDir("run_mixed_case");
    const RunResult r = runDeck(writeDeck(dir, example1(), "DYNWAVE", {"DW_BACKEND          oMp"}));
    ASSERT_EQ(r.rc, 0);
    EXPECT_EQ(r.dw_option, "OMP");
    const auto w = fallbackWarnings(r);
    ASSERT_EQ(w.size(), 1u);
    EXPECT_TRUE(namesBackend(w[0], "OMP")) << w[0];
}

TEST_F(DwBackendOption, CpuAndAutoRunWithoutAWarningAndWithUnchangedResults) {
    for (const Model& m : {example1(), structures(), pumps()}) {
        SCOPED_TRACE(m.name);
        const RunResult none =
            runDeck(writeDeck(caseDir(std::string("run_") + m.name + "_nokey"), m, "DYNWAVE", {}));
        ASSERT_EQ(none.rc, 0);
        ASSERT_FALSE(none.out.empty());
        ASSERT_FALSE(none.hsf.empty());

        for (const char* token : {"CPU", "AUTO"}) {
            SCOPED_TRACE(token);
            const RunResult r =
                runDeck(writeDeck(caseDir(std::string("run_") + m.name + "_" + token), m, "DYNWAVE",
                                  {std::string("DW_BACKEND          ") + token}));
            ASSERT_EQ(r.rc, 0);
            EXPECT_EQ(r.dw_option, token) << "the key must be accepted";
            EXPECT_TRUE(dwWarnings(r).empty());
            EXPECT_TRUE(fallbackLines(r.rpt).empty());
            EXPECT_EQ(r.out, none.out);
            EXPECT_EQ(r.hsf, none.hsf);
        }
    }
}

TEST_F(DwBackendOption, KeyHasNoEffectUnderAnotherFlowRouting) {
    const struct {
        Model model;
        const char* routing;
        std::vector<std::string> extra;
    } cases[] = {{example1(), "KINWAVE", {}},
                 {example1(), "FV", {"FV_CELL_LENGTH       0"}},
                 {structures(), "FV", {"FV_CELL_LENGTH       0"}}};
    for (const auto& c : cases) {
        SCOPED_TRACE(std::string(c.model.name) + " under " + c.routing);
        auto with = [&](const std::string& token, const std::string& tag) {
            std::vector<std::string> extra = c.extra;
            if (!token.empty())
                extra.push_back("DW_BACKEND          " + token);
            return runDeck(writeDeck(
                caseDir(std::string("other_") + c.model.name + "_" + c.routing + "_" + tag),
                c.model, c.routing, extra));
        };
        const RunResult none = with("", "nokey");
        ASSERT_EQ(none.rc, 0) << (none.errors.empty() ? "" : none.errors.front());
        ASSERT_FALSE(none.out.empty());
        ASSERT_FALSE(none.hsf.empty());
        for (const char* token : {"CPU", "OMP", "CUDA"}) {
            SCOPED_TRACE(token);
            const RunResult r = with(token, token);
            ASSERT_EQ(r.rc, 0);
            EXPECT_EQ(r.dw_option, token) << "the key is accepted under any FLOW_ROUTING";
            EXPECT_TRUE(fallbackWarnings(r).empty());
            EXPECT_TRUE(fallbackLines(r.rpt).empty());
            EXPECT_EQ(r.out, none.out);
            EXPECT_EQ(r.hsf, none.hsf);
        }
        // The environment variable is equally inert.
        plattest::setEnvVar(kEnv, "omp");
        const RunResult env = with("", "env");
        plattest::unsetEnvVar(kEnv);
        ASSERT_EQ(env.rc, 0);
        EXPECT_TRUE(fallbackWarnings(env).empty());
        EXPECT_EQ(env.out, none.out);
        EXPECT_EQ(env.hsf, none.hsf);
    }
}

// ---------------------------------------------------------------------------
// 7. No change for existing models
// ---------------------------------------------------------------------------

TEST_F(DwBackendOption, ModelWithoutTheKeyRunsExactlyAsBefore) {
    // Two runs of the same key-free deck, environment unset: no warning mentions the key and the
    // outputs are reproducible byte for byte. Byte-identity against the pre-change engine is
    // recorded in reports/0001-tests.md (hashes taken on untouched upstream).
    for (const Model& m : {example1(), structures(), pumps()}) {
        SCOPED_TRACE(m.name);
        const RunResult a =
            runDeck(writeDeck(caseDir(std::string("nochange_") + m.name + "_a"), m, "DYNWAVE", {}));
        const RunResult b =
            runDeck(writeDeck(caseDir(std::string("nochange_") + m.name + "_b"), m, "DYNWAVE", {}));
        ASSERT_EQ(a.rc, 0);
        ASSERT_EQ(b.rc, 0);
        EXPECT_TRUE(dwWarnings(a).empty());
        EXPECT_EQ(reportMentions(a.rpt, "DW_BACKEND"), 0);
        EXPECT_EQ(a.out, b.out);
        EXPECT_EQ(a.hsf, b.hsf);
        EXPECT_EQ(a.warnings, b.warnings);
    }
}
