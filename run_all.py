"""
run_all.py — Climate-PDE Full Pipeline

Orchestrates the complete Climate-PDE workflow:
1. PDE discovery (PySINDy weak form) on anomaly data & threshold sensitivity
2. Robustness checks (noise sensitivity and WSINDy dominant term confirmation)
3. Single-variable PDE validation + spatial cross-validation (Table 3)
4. Multivariate PDE discovery for t2m and q (Section 3.2)
5. Forward integration and predictive skill assessment vs baselines (Section 3.3)
6. Resolution sensitivity analysis (0.05° vs 0.25°, Table 4)
7. Block-bootstrap confidence intervals (100 replicates, Section 3.1)
8. Generation of publication figures (fig1, fig3, fig4, fig5, fig6)

Usage:
    python run_all.py              # Run full pipeline
    python run_all.py --fast       # Quick sanity check (discovery + validation + multivar)

Output:
    Discovery results saved under output/
    Figures saved under figures/
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent  # project root
DISCOVERY = SCRIPTS_DIR / "src" / "discovery"
PREDICTION = SCRIPTS_DIR / "src" / "prediction"
OUTPUT = SCRIPTS_DIR / "output"
OUTPUT.mkdir(exist_ok=True)


def run(cmd, label, timeout=None):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[-500:]}")
    print(f"  → Completed in {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"  ⚠ WARNING: exit code {result.returncode}")
    return result


def main():
    fast = "--fast" in sys.argv

    if fast:
        print("FAST MODE: running discovery + validation only")
        # Single-variable discovery (PySINDy weak form) — one region
        run([sys.executable, str(DISCOVERY / "pde_discovery.py")], "Single-variable weak-form PDE discovery")

        # Single-variable validation + cross-validation
        run([sys.executable, str(DISCOVERY / "validate_pde.py")], "Single-variable validation + cross-validation")

        # Multivariate PDE discovery
        run([sys.executable, str(DISCOVERY / "pde_multivar.py")], "Multivariate PDE discovery")

        print(f"\n{'='*60}")
        print(f"Quick check complete. Results in output/")
        print(f"To run full pipeline (hours), use: python run_all.py")
        return

    print("CLIMATE-PDE: FULL PIPELINE")
    print("=" * 60)

    # 1. Single-variable PDE discovery (PySINDy weak form)
    run([sys.executable, str(DISCOVERY / "pde_discovery.py")], "Single-variable PDE discovery (weak form, PySINDy)", timeout=3600)
    # 1b. Sensitivity analysis (Table 2)
    run([sys.executable, str(DISCOVERY / "pde_discovery.py"), "--sensitivity"], "Sensitivity analysis (Table 2)", timeout=3600)

    # 1c. Robustness checks cited in Sections 2.2 / 3.1
    run([sys.executable, str(DISCOVERY / "robustness_checks.py")], "Weak-form confirmation + noise sensitivity", timeout=3600)

    # 2. Single-variable validation + cross-validation
    run([sys.executable, str(DISCOVERY / "validate_pde.py")], "Single-variable validation + cross-validation (Table 3)", timeout=3600)

    # 3. Multivariate PDE discovery
    run([sys.executable, str(DISCOVERY / "pde_multivar.py")], "Multivariate coupled PDE discovery", timeout=3600)

    # 4. Forward integration
    run([sys.executable, str(PREDICTION / "forward_integrate.py")], "Forward integration + predictive skill", timeout=3600)

    # 5. Resolution comparison
    run([sys.executable, str(DISCOVERY / "resolution_compare.py")], "Resolution sensitivity analysis (Table 4)", timeout=1800)

    # 6. Bootstrap confidence intervals (run inline if script exists)
    bootstrap_script = DISCOVERY / "bootstrap_ci.py"
    if bootstrap_script.exists():
        run([sys.executable, str(bootstrap_script)], "Bootstrap confidence intervals")

    # 7. Generate all figures
    figs = sorted((SCRIPTS_DIR / "figures" / "scripts").glob("fig*.py"))
    for fg in figs:
        run([sys.executable, str(fg)], f"Figure: {fg.name}", timeout=3600)

    # 8. Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Results:  {OUTPUT}/")
    print(f"  Figures:  {SCRIPTS_DIR / 'figures'}/")


if __name__ == "__main__":
    main()
