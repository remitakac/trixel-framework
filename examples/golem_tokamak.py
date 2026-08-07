"""
examples/golem_tokamak.py
=========================

TRIXEL applied to real tokamak data from GOLEM (CVUT Prague).
Shots 53620-53624, He plasma + Ar seeding, 8 July 2026.

This example demonstrates a case where V and D are genuinely
independent physical measurements — not derived from each other.
Independence is not just asserted in comments: it is checked
programmatically before the calibrators are trusted (see
check_independence() / check_functional_dependence() calls below,
per Independence First Rule, docs/mapping_guide.md).

TRIXEL mapping:
    S = time [ms]
    V = plasma current Ip(t) [A]  — measured by Rogowski coil
    D = loop voltage Uloop(t) [V] — measured independently

Physical meaning:
    VS = 1/|dIp/dt| — drops when current changes rapidly (instability)
    SD = |dUloop/dt| — rises when driving voltage changes rapidly
    VD = SD/n       — bridge between voltage dynamics and current sensitivity

Data source:
    Raw oscilloscope CSV files from GOLEM shot database.
    Publicly available at: golem.fjfi.cvut.cz/shots/<shot_number>/
    File: Devices/Oscilloscopes/TektrMSO56-a/TektrMSO56_ALL.csv

    Expected local filename pattern (shot number embedded, per project
    file-naming convention adopted 2026-08):
        golem_<shot_number>_<series_label>.csv
    e.g. golem_53620_He_rlp_scan.csv
    Older/alternate patterns (TektrMSO56_ALL_shot<shot>_.csv) are also
    accepted as a fallback — see find_shot_file().

Note on V and D independence:
    Ip(t) is measured by the Rogowski coil (CH3/MATH2).
    Uloop(t) is measured by a separate voltage loop (CH1).
    These are physically distinct diagnostic channels.
    D is NOT computed as dV/dS — it is an independent measurement.
    This is verified programmatically, not just asserted: see
    check_independence() (linear, rank test) and
    check_functional_dependence() (nonlinear, derivative-correlation test)
    calls in analyze_shot() below.
"""

import glob
import os
import sys

import numpy as np

# Add parent directory to path for calibrators import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibrators import (
    compute_all,
    dominant_calibrator,
    check_independence,
    check_functional_dependence,
)


def load_golem_shot(filepath):
    """
    Load raw GOLEM oscilloscope CSV file.

    Returns: t_ms, Ip_A, Uloop_V
    """
    t_arr, Ip_arr, Ul_arr = [], [], []
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines[17:]:
        p = line.strip().split(',')
        if len(p) < 13:
            continue
        try:
            t_arr.append(float(p[0]) * 1000)  # s -> ms
            Ul_arr.append(float(p[1]))          # CH1: U_loop [V]
            Ip_arr.append(abs(float(p[12])))    # MATH2: I_p [A]
        except (ValueError, IndexError):
            continue
    return np.array(t_arr), np.array(Ip_arr), np.array(Ul_arr)


def find_shot_file(shot_id, search_dir="."):
    """
    Locate a local CSV file for a given shot number, regardless of which
    naming convention was used when it was downloaded. Tries, in order:
        golem_<shot_id>_*.csv
        *shot<shot_id>_*.csv        (older TektrMSO56_ALL_shot<shot>_.csv style)
        shot_<shot_id>.csv          (legacy convention, kept as last resort)

    Returns the first match found, or None.
    """
    patterns = [
        f"golem_{shot_id}_*.csv",
        f"*shot{shot_id}_*.csv",
        f"shot_{shot_id}.csv",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            return matches[0]
    return None


def smooth(signal, window=100):
    """Simple moving average for noise reduction."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(signal, window)


def analyze_shot(filepath, shot_id, r_lp, plasma_fraction=0.05):
    """
    Apply TRIXEL calibrators to a single GOLEM shot.

    V and D are independent measurements:
        V = Ip(t)    [plasma current, Rogowski coil]
        D = Uloop(t) [loop voltage, separate measurement]
        S = time [ms]

    Independence is checked, not assumed:
        check_independence(V, D, S)            -> linear (rank) test
        check_functional_dependence(V, D, S)    -> nonlinear (derivative) test

    plasma_fraction:
        Fraction of peak Ip used as the "plasma present" threshold.
        Relative, not a fixed magic number in amps, so it scales
        across shots/sessions with very different Ip_max (this project
        has seen shots ranging from ~0.6 kA to >13 kA).
    """
    print(f"\nShot #{shot_id} (r_lp = {r_lp} mm)")
    print("-" * 40)

    t, Ip, Ul = load_golem_shot(filepath)
    Ip_s = smooth(Ip, 100)
    Ul_s = smooth(Ul, 100)

    # TRIXEL mapping — V and D are independent
    S = t
    V = Ip_s   # existence: plasma current
    D = Ul_s   # dynamics:  loop voltage (independent measurement)

    # --- Independence First Rule: verify before trusting the calibrators ---
    rank = check_independence(V, D, S)
    func_corr = check_functional_dependence(V, D, S)
    print(f"  Independence: rank={rank} (want 3), "
          f"functional corr(D, signed dV/dS)={func_corr:.3f} (want <0.95)")
    if rank < 3 or abs(func_corr) > 0.95:
        print("  WARNING: independence check failed for this shot — "
              "VS/SD/VD results below may not be physically meaningful.")

    c = compute_all(V, D, S)

    # Plasma window: relative threshold, not a fixed amp value (Mistake 2
    # in mapping_guide.md — thresholds should have physical/relative
    # justification, not be tuned to make results look clean).
    if Ip_s.max() <= 0:
        print("  No plasma detected.")
        return None
    threshold = plasma_fraction * Ip_s.max()
    mask = Ip_s > threshold
    idx = np.where(mask)[0]
    if len(idx) < 10:
        print("  No plasma detected.")
        return None

    t_start = t[idx[0]]
    t_end   = t[idx[-1]]
    ft_s = idx[0] + int(0.4 * (idx[-1] - idx[0]))
    ft_e = idx[0] + int(0.7 * (idx[-1] - idx[0]))

    VS_ft   = c['VS'][ft_s:ft_e].mean()
    VS_rise = c['VS'][idx[0]:ft_s].mean()
    SD_ft   = c['SD'][ft_s:ft_e].mean()

    dom = dominant_calibrator(c['SD'][ft_s:ft_e].mean(),
                               c['n'][ft_s:ft_e].mean())

    print(f"  Plasma: {t_start:.2f} to {t_end:.2f} ms "
          f"(threshold = {plasma_fraction:.0%} of Ip_max = {threshold:.0f} A)")
    print(f"  Ip peak: {Ip_s.max():.0f} A")
    print(f"  VS (stable phase): {VS_ft:.4f}")
    print(f"  VS (rise phase):   {VS_rise:.6f}")
    print(f"  VS ratio (stable/rise): {VS_ft/VS_rise:.1f}x")
    print(f"  SD (stable phase): {SD_ft:.6f}")
    print(f"  Dominant calibrator: {dom}")

    return {
        'shot': shot_id, 'r_lp': r_lp,
        'VS_ft': VS_ft, 'VS_rise': VS_rise,
        'SD_ft': SD_ft, 'dominant': dom,
        'independence_rank': rank, 'independence_func_corr': func_corr,
    }


if __name__ == "__main__":
    # Example with synthetic data if no CSV files available
    print("TRIXEL — GOLEM Tokamak Example")
    print("=" * 50)
    print("V = Ip(t) [plasma current — Rogowski coil]")
    print("D = Uloop(t) [loop voltage — independent measurement]")
    print("S = time [ms]")
    print()

    # Shot metadata (r_lp = Langmuir probe radial position, mm)
    shot_list = [
        (53620, 64),
        (53621, 60),
        (53622, 56),
        (53623, 52),
        (53624, 48),
    ]

    # Locate files using flexible naming (see find_shot_file docstring)
    real_files = []
    for shot_id, r_lp in shot_list:
        fpath = find_shot_file(shot_id)
        if fpath:
            real_files.append((fpath, shot_id, r_lp))

    if real_files:
        results = []
        for filepath, shot_id, r_lp in real_files:
            r = analyze_shot(filepath, shot_id, r_lp)
            if r:
                results.append(r)

        if len(results) > 1:
            print("\nSummary across shots:")
            print(f"{'Shot':>6} {'r_lp':>6} {'VS_ft':>10} {'rank':>5} "
                  f"{'func_corr':>10} {'dominant':>10}")
            for r in results:
                print(f"  {r['shot']:6d} {r['r_lp']:6d} "
                      f"{r['VS_ft']:10.4f} {r['independence_rank']:5d} "
                      f"{r['independence_func_corr']:10.3f} {r['dominant']:>10}")

    else:
        # Synthetic demonstration
        print("No CSV files found. Running synthetic demonstration.")
        print("To use real data, download CSV files from:")
        print("  golem.fjfi.cvut.cz/shots/53620/")
        print("  (Devices/Oscilloscopes/TektrMSO56-a/TektrMSO56_ALL.csv)")
        print("  Save locally as golem_53620_<series_label>.csv")
        print()

        # Synthetic plasma-like signals
        t = np.linspace(0, 20, 4000)
        # V: plasma current (rise, flat, decay)
        Ip = np.zeros_like(t)
        mask_r = (t >= 3) & (t < 9)
        mask_f = (t >= 9) & (t <= 12)
        mask_d = (t > 12) & (t <= 15)
        Ip[mask_r] = 2500 * np.sin(np.pi * (t[mask_r]-3) / 12)
        Ip[mask_f] = 2000 * (1 + 0.05 * np.random.randn(mask_f.sum()))
        Ip[mask_d] = 2000 * np.exp(-2*(t[mask_d]-12))

        # D: loop voltage — INDEPENDENT measurement
        Ul = np.zeros_like(t)
        Ul[mask_r] = 15 * np.ones(mask_r.sum())
        Ul[mask_f] = 5  * np.ones(mask_f.sum())
        Ul[mask_d] = 2  * np.ones(mask_d.sum())
        Ul += 0.3 * np.random.randn(len(t))

        S = t
        V = Ip
        D = Ul

        rank = check_independence(V, D, S)
        func_corr = check_functional_dependence(V, D, S)
        print(f"Independence: rank={rank} (want 3), "
              f"functional corr(D, signed dV/dS)={func_corr:.3f} (want <0.95)")

        c = compute_all(V, D, S)

        threshold = 0.05 * Ip.max()
        mask_plasma = Ip > threshold
        idx = np.where(mask_plasma)[0]
        ft_s = idx[0] + int(0.4*(idx[-1]-idx[0]))
        ft_e = idx[0] + int(0.7*(idx[-1]-idx[0]))

        VS_ft   = c['VS'][ft_s:ft_e].mean()
        VS_rise = c['VS'][idx[0]:ft_s].mean()
        dom = dominant_calibrator(c['SD'][ft_s:ft_e].mean(),
                                   c['n'][ft_s:ft_e].mean())

        print(f"\nSynthetic shot results:")
        print(f"  VS (stable phase): {VS_ft:.4f}")
        print(f"  VS (rise phase):   {VS_rise:.6f}")
        print(f"  VS ratio: {VS_ft/VS_rise:.1f}x")
        print(f"  Dominant calibrator: {dom}")
        print()
        print("Key point: V=Ip(t) and D=Uloop(t) are independent measurements.")
        print("D is NOT computed as dV/dS — it comes from a separate sensor.")
        print("This is verified above via check_independence() and")
        print("check_functional_dependence(), not just asserted in comments.")
