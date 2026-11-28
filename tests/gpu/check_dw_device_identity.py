"""Require the dynamic wave device backend to reproduce the CPU solver bit for bit.

The deck is run twice: once with OPENSWMM_DW_BACKEND=cpu, once with the Picard loop on the plugin.
The two runs must write the same binary output, the same end-of-run hot start file, and the same
report apart from wall-clock lines. The hot start is saved in the NATIVE format (SAVE HOTSTART ...
NATIVE), whose node depths, heads and volumes and link flows, depths and volumes are doubles; it is
compared byte for byte apart from the save's wall-clock time and the checksum over it.

With --hotstart-restart the first run also saves a NATIVE hot start at the given time, which must
match too, and a second pair of runs starts at that time from it (USE HOTSTART, each backend from
its own file) and must match in the same three files. With --append a file's text is added to the
deck, for example a [CONTROLS] section.

With --expect-refusal the plugin run must instead name the unsupported item and fall back to the
CPU solver.
"""

import argparse
import filecmp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

# Report lines that legitimately differ: wall clock.
VOLATILE = re.compile(r"Analysis begun|Analysis ended|Total elapsed")
# Printed by the plugin's marcher when it is destroyed.
MARCHER = re.compile(r"\[openswmm DW\] \w+ marcher: (\d+) routing steps")


def forced_deck(text, forced, files=("SAVE HOTSTART end_state.hsf NATIVE",), report_all=False):
    """Return the deck with `forced` replacing those [OPTIONS] keys and the `files` rows added to
    [FILES]; with `report_all`, every node and link is reported."""
    keys = {k.upper() for k in forced}
    rows = ["%-22s %s" % item for item in forced.items()]
    out, in_options, written = [], False, False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if in_options and not written:
                out += rows
                written = True
            in_options = stripped.upper().startswith("[OPTIONS]")
            out.append(line)
            continue
        if in_options and stripped and not stripped.startswith(";"):
            if stripped.split()[0].upper() in keys:
                continue
        out.append(line)
    if in_options and not written:
        out += rows
        written = True
    if not written:
        raise AssertionError("deck has no [OPTIONS] section")
    at = [i for i, line in enumerate(out) if line.strip().upper().startswith("[FILES]")]
    if at:
        out[at[0] + 1:at[0] + 1] = list(files)
    else:
        out += ["", "[FILES]"] + list(files)
    if report_all:
        at = [i for i, line in enumerate(out) if line.strip().upper().startswith("[REPORT]")]
        rows = ["NODES ALL", "LINKS ALL"]
        if at:
            end = next((i for i in range(at[0] + 1, len(out)) if out[i].strip().startswith("[")),
                       len(out))
            kept = [line for line in out[at[0] + 1:end]
                    if line.strip().split()[:1] not in (["NODES"], ["LINKS"])]
            out[at[0] + 1:end] = kept + rows
        else:
            out += ["", "[REPORT]"] + rows
    return "\n".join(out) + "\n"


def same_hotstart(a, b):
    """Compare two NATIVE hot start files byte for byte, apart from the wall-clock save time.

    The NATIVE layout (HotStartManager::encode) is [magic 16B][version 4B][timestamp 8B]...
    [crc32 4B]; the timestamp is the save's Unix time, and the checksum covers it.
    """
    x, y = a.read_bytes(), b.read_bytes()
    return len(x) == len(y) and x[:20] == y[:20] and x[28:-4] == y[28:-4]


def stable_report(path):
    return [line for line in path.read_text(errors="replace").splitlines()
            if not VOLATILE.search(line)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--plugin-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="omp")
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="extra [OPTIONS] override applied to both runs")
    parser.add_argument("--expect-refusal", default=None, metavar="TEXT",
                        help="the plugin run must report this unsupported item and use the CPU")
    parser.add_argument("--append", action="append", default=[], type=Path, metavar="FILE",
                        help="text added to the end of the deck, e.g. a [CONTROLS] section")
    parser.add_argument("--hotstart-restart", default=None, metavar="MM/DD/YYYY HH:MM:SS",
                        help="also save a hot start at this time and restart a second run from it")
    parser.add_argument("--report-all", action="store_true",
                        help="report every node and link in the binary output")
    args = parser.parse_args()
    # The engine runs inside each run folder, so every path must be absolute.
    args.engine = args.engine.resolve()
    args.plugin_dir = args.plugin_dir.resolve()
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    forced = {"THREADS": str(args.threads)}
    for item in args.option:
        key, value = item.split("=", 1)
        forced[key] = value
    name = args.input.stem
    source = args.input.read_text(errors="replace")
    for extra in args.append:
        source += "\n" + extra.resolve().read_text(errors="replace") + "\n"
    saves = ["SAVE HOTSTART end_state.hsf NATIVE"]
    if args.hotstart_restart:
        saves.append(f"SAVE HOTSTART mid_state.hsf {args.hotstart_restart} NATIVE")
    text = forced_deck(source, forced, saves, args.report_all)

    # Side files the deck names by relative path (rain gage data and the like).
    side = [p for p in args.input.parent.iterdir()
            if p.is_file() and p.suffix.lower() not in (".inp", ".rpt", ".out", ".h5", ".hsf")
            and p.stat().st_size < 50e6]

    backends = (("cpu", {"OPENSWMM_DW_BACKEND": "cpu"}),
                ("device", {"OPENSWMM_DW_BACKEND": args.backend,
                            "OPENSWMM_GPU_PLUGIN_PATH": str(args.plugin_dir)}))

    def run_pair(suffix, deck_text):
        """Run the deck on the CPU and on the plugin, in <output>/<backend><suffix>."""
        logs = {}
        for tag, env in backends:
            folder = args.output / (tag + suffix)
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)
            for p in side:
                shutil.copy(p, folder)
            deck = folder / (name + ".inp")
            deck.write_text(deck_text(tag))
            environment = dict(os.environ)
            environment.pop("OPENSWMM_DW_BACKEND", None)
            environment.update(env)
            environment["OMP_NUM_THREADS"] = str(args.threads)
            done = subprocess.run(
                [str(args.engine), str(deck), str(folder / "out.rpt"), str(folder / "out.out")],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                cwd=folder)
            (folder / "console.txt").write_text(done.stdout)
            if done.returncode != 0:
                raise AssertionError(
                    f"{tag}{suffix}: engine exit {done.returncode}\n{done.stdout[-2000:]}")
            logs[tag] = done.stdout
        return logs

    logs = run_pair("", lambda tag: text)
    pairs = [("", ["out.out", "end_state.hsf"] + (["mid_state.hsf"] if args.hotstart_restart
                                                    else []), logs)]
    if args.hotstart_restart:
        day, clock = args.hotstart_restart.split()
        restart = {"START_DATE": day, "START_TIME": clock, "REPORT_START_DATE": day,
                   "REPORT_START_TIME": clock}

        def restart_text(tag):
            used = (args.output / tag / "mid_state.hsf").resolve()
            return forced_deck(source, dict(forced, **restart),
                               ["SAVE HOTSTART end_state.hsf NATIVE", f'USE HOTSTART "{used}"'],
                               args.report_all)

        pairs.append(("_restart", ["out.out", "end_state.hsf"],
                      run_pair("_restart", restart_text)))

    device_log = logs["device"]
    device_report = (args.output / "device" / "out.rpt").read_text(errors="replace")
    refused = "DW_BACKEND" in device_report and "is not used" in device_report

    if args.expect_refusal is not None:
        if not refused or args.expect_refusal not in device_report:
            raise AssertionError(
                f"expected a refusal naming '{args.expect_refusal}':\n{device_log[-2000:]}")
        if "[openswmm DW] using device backend" in device_log:
            raise AssertionError("the device backend was loaded for a refused model")
    else:
        if "[openswmm DW] using device backend" not in device_log:
            raise AssertionError("the device backend was not loaded:\n" + device_log[-2000:])
        if refused:
            raise AssertionError("the device backend stepped aside:\n" + device_report[-2000:])
        for suffix, _, pair_logs in pairs:
            marcher = MARCHER.search(pair_logs["device"])
            if not marcher or int(marcher.group(1)) == 0:
                raise AssertionError(f"the device marcher ran no routing step{suffix}:\n" +
                                     pair_logs["device"][-2000:])

    failures = []
    for suffix, items, _ in pairs:
        for item in items:
            a, b = args.output / ("cpu" + suffix) / item, args.output / ("device" + suffix) / item
            what = item + suffix
            if not a.exists() or not b.exists():
                failures.append(f"{what}: missing")
            elif a.stat().st_size == 0:
                failures.append(f"{what}: empty")
            elif item.endswith(".hsf") and not same_hotstart(a, b):
                failures.append(f"{what}: differs")
            elif not item.endswith(".hsf") and not filecmp.cmp(a, b, shallow=False):
                failures.append(f"{what}: differs")
        cpu_lines = stable_report(args.output / ("cpu" + suffix) / "out.rpt")
        dev_lines = stable_report(args.output / ("device" + suffix) / "out.rpt")
        if args.expect_refusal is not None:
            # The refusal itself is one extra warning line in the device report.
            dev_lines = [line for line in dev_lines if args.expect_refusal not in line]
        if cpu_lines != dev_lines:
            differing = [f"  cpu:    {a}\n  device: {b}"
                         for a, b in zip(cpu_lines, dev_lines) if a != b]
            failures.append("report%s differs (%d lines, lengths %d/%d)\n%s" % (
                suffix, len(differing), len(cpu_lines), len(dev_lines),
                "\n".join(differing[:10])))
    if failures:
        raise AssertionError(f"{name} [{args.backend}, {args.threads} threads]:\n" +
                             "\n".join(failures))
    what = "refuses and runs the CPU solver" if args.expect_refusal else "matches the CPU solver"
    files = "binary output, NATIVE hot start, report"
    if args.hotstart_restart:
        files += "; mid-run hot start and the run restarted from it"
    print(f"{name}: {args.backend} backend at {args.threads} thread(s) {what} ({files})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
