# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 5 — Final Feature Matrix Generation                       ║
║                                                                            ║
║  Pipeline position:                                                        ║
║    Stage 0 (Preselection) → Stage 1 (Univariate) → Stage 2 (SFFS)         ║
║      → Stage 3 (Interactions) → Stage 4 (Stability) →                     ║
║      [Stage 5: Final Feature Matrix]                                       ║
║                                                                            ║
║  Purpose:                                                                  ║
║    Generate and cache the final feature matrix for the validated           ║
║    feature set discovered in Stage 3 (Interaction Discovery).             ║
║                                                                            ║
║    This is the DECOUPLING STAGE between Phase 2 (Model Development)       ║
║    and Phase 3 (Backtesting). By paying the pandas_ta generation cost      ║
║    ONCE here, all Phase 3 backtesting scripts can simply load a parquet    ║
║    file and run simulations in seconds rather than waiting 2+ minutes      ║
║    for TA recomputation on every run.                                      ║
║                                                                            ║
║  What this script does:                                                    ║
║    1. Reads interaction_report.json to discover the validated feature set  ║
║       (no hardcoded feature names — fully config-driven).                  ║
║    2. Regenerates the 4 base TA features in parallel (loky, n_jobs=6).    ║
║    3. Computes the 5-day forward log return target per ticker.             ║
║    4. Constructs the 1 interaction column (element-wise product).          ║
║    5. Saves the final matrix: ticker + date + 5 features + target.        ║
║                                                                            ║
║  What this script does NOT do:                                             ║
║    ✗ No StandardScaler — raw values are saved; scaling happens in          ║
║      Phase 3 inside each CV/walk-forward fold to prevent leakage.         ║
║    ✗ No Ridge training — pure feature engineering only.                    ║
║    ✗ No hardcoded feature names — all driven by interaction_report.json.   ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_2_features/interaction_report.json       (Stage 3)    ║
║    • artifacts/phase_1_data/merged_data.parquet                (Phase 1)   ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/final_feature_matrix.parquet              ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)              ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — reads only disk artifacts from prior stages              ║
║    ✓ Config-driven — feature list sourced from interaction_report.json    ║
║    ✓ Atomic — writes to .tmp then renames                                 ║
║    ✓ Efficient — parallel TA generation (loky, n_jobs=6, cores=1)         ║
║    ✓ Traceable — row/column counts logged; Auditor lifecycle completed     ║
║    ✓ Temporal — sorted by (ticker, date); target computed per-ticker      ║
║    ✓ No leakage — raw unscaled features; scaling deferred to Phase 3      ║
║                                                                            ║
║  Version: 1.0.0                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# SUPPRESS PANDAS_TA UPSTREAM FUTUREWARNING (BEFORE IMPORT)
# ═══════════════════════════════════════════════════════════════
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated",
    module="pandas_ta_classic",
)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from joblib import Parallel, delayed

# ─── Project path bootstrap ────────────────────────────────────────────────
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas_ta_classic as ta                         # noqa: E402
from core.audit import Auditor                         # noqa: E402
from phases.phase2.config import Phase2Config          # noqa: E402


# ─── Constants ──────────────────────────────────────────────────────────────
VERSION = "1.0.0"
STAGE_NAME = "5_prepare_features"

PHASE1_ARTIFACT   = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase_2_features"
INTERACTION_REPORT_JSON = PHASE2_ARTIFACT_DIR / "interaction_report.json"
OUTPUT_PARQUET    = PHASE2_ARTIFACT_DIR / "final_feature_matrix.parquet"

REQUIRED_OHLCV_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

# Must match Stage 1/2/3/4
FORWARD_DAYS = 5

# Parallelism — same settings as Stages 2/3/4
N_PARALLEL_JOBS = 6
MIN_TICKER_ROWS  = 200

# Interaction column convention — must match Stage 3
_IX_PREFIX    = "IX_"
_IX_SEPARATOR = "_x_"


# ─── Indicator-to-column prefix map ────────────────────────────────────────
# Identical to all prior stages — maps indicator "kind" to TA output prefixes.
_KIND_TO_PREFIX: Dict[str, List[str]] = {
    "adx":            ["ADX_", "DMP_", "DMN_"],
    "chop":           ["CHOP_"],
    "aroon":          ["AROOND_", "AROONU_", "AROONOSC_"],
    "supertrend":     ["SUPERT_", "SUPERTd_", "SUPERTl_", "SUPERTs_"],
    "rsi":            ["RSI_"],
    "roc":            ["ROC_"],
    "macd":           ["MACD_", "MACDh_", "MACDs_"],
    "mom":            ["MOM_"],
    "willr":          ["WILLR_"],
    "natr":           ["NATR_"],
    "bbands":         ["BBL_", "BBM_", "BBU_", "BBB_", "BBP_"],
    "atr":            ["ATRr_"],
    "rvi":            ["RVI_"],
    "cmf":            ["CMF_"],
    "mfi":            ["MFI_"],
    "obv":            ["OBV"],
    "sma":            ["SMA_"],
    "ema":            ["EMA_"],
    "zscore":         ["ZS_"],
    "skew":           ["SKEW_"],
    "stdev":          ["STDEV_"],
    "log_return":     ["LOGRET_"],
    "percent_return": ["PCTRET_"],
}


# ─── Parallel Worker ────────────────────────────────────────────────────────
def _process_ticker_worker(
    ticker: str,
    ticker_df: pd.DataFrame,
    filtered_specs: List[Dict[str, Any]],
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Compute TA indicators for one ticker in a parallel worker process.

    Must be a module-level function (not a method) so joblib/loky can
    pickle it across process boundaries.

    loky backend is used throughout this project because pandas_ta_classic
    may spawn internal child processes. Standard multiprocessing uses
    daemonic workers which cannot have children — loky's non-daemonic
    workers avoid this crash.

    ``ticker_df.ta.cores = 1`` is set before calling `.strategy()` to
    prevent each worker from spawning all available CPU threads, which
    would cause OS thread thrashing (6 workers × 11 threads = 66 threads
    on 11 physical cores).

    Args:
        ticker: Ticker symbol string (for error reporting).
        ticker_df: OHLCV slice for this ticker — already .copy()'d by caller.
        filtered_specs: pandas_ta indicator spec dicts to compute.

    Returns:
        (ticker, result_df, None) on success.
        (ticker, None, error_string) on failure.
    """
    import warnings as _w
    _w.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="Setting an item of incompatible dtype",
    )

    try:
        strategy = ta.Strategy(
            name="Phase2_FinalFeatures",
            ta=filtered_specs,
        )
        ticker_df.ta.cores = 1                          # CRITICAL — no thrashing
        ticker_df.ta.strategy(strategy, verbose=False)
        return (ticker, ticker_df, None)
    except Exception as exc:
        return (ticker, None, str(exc))


# ─── Main Class ─────────────────────────────────────────────────────────────
class FeatureMatrixBuilder:
    """Build and cache the final feature matrix from validated Phase 2 outputs.

    Reads the feature list from interaction_report.json so the script
    requires zero code changes if the upstream feature set is ever updated.
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        # Populated during load_interaction_report()
        self._final_features: Optional[List[str]]       = None
        self._base_features: Optional[List[str]]        = None
        self._interaction_features: Optional[List[str]] = None
        self._filtered_specs: Optional[List[Dict[str, Any]]] = None

    # ── 1. Read feature list from interaction_report.json ───────────────────
    def load_interaction_report(self) -> None:
        """Parse interaction_report.json and split features into base / IX.

        Raises:
            FileNotFoundError: Stage 3 artifact missing.
            AssertionError: Report schema violated.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 5] Loading Validated Feature Set from Stage 3...")
        print(f"{'='*72}")

        if not INTERACTION_REPORT_JSON.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 3 artifact not found: {INTERACTION_REPORT_JSON}\n"
                f"  Run 4_interaction_discovery.py (v1.1.0+) before this script."
            )

        with open(INTERACTION_REPORT_JSON, "r", encoding="utf-8") as fh:
            report = json.load(fh)

        assert "final_feature_set" in report, (
            f"[SCHEMA VIOLATION] interaction_report.json missing "
            f"'final_feature_set'.\n  Keys present: {list(report.keys())}"
        )

        final_features = report["final_feature_set"]
        assert isinstance(final_features, list) and len(final_features) > 0, (
            f"[SCHEMA VIOLATION] final_feature_set must be a non-empty list, "
            f"got: {final_features}"
        )

        # Version guard — warn if report predates StandardScaler fix
        report_version = report.get("version", "unknown")
        if report_version < "1.1.0":
            print(f"  ⚠  WARNING: interaction_report.json is version "
                  f"{report_version} (pre-StandardScaler fix).")
            print(f"     Re-run 4_interaction_discovery.py (v1.1.0+) to "
                  f"ensure you are caching the correct feature set.")

        # Split into base TA features vs interaction product columns
        base_features        = [f for f in final_features
                                 if not f.startswith(_IX_PREFIX)]
        interaction_features = [f for f in final_features
                                 if f.startswith(_IX_PREFIX)]

        self._final_features        = final_features
        self._base_features         = base_features
        self._interaction_features  = interaction_features

        stage3_ic = report.get("final_score_mean_ic", "N/A")

        print(f"  ✓ Report version      : {report_version}")
        print(f"  ✓ Stage 3 Mean IC     : {stage3_ic}")
        print(f"  ✓ Total features      : {len(final_features)}")
        print(f"  ✓ Base features       : {len(base_features)}")
        for i, f in enumerate(base_features, 1):
            print(f"      {i}. {f}")
        if interaction_features:
            print(f"  ✓ Interaction features: {len(interaction_features)}")
            for i, ix in enumerate(interaction_features, 1):
                inner = ix[len(_IX_PREFIX):]
                parts = inner.split(_IX_SEPARATOR)
                desc  = f"{parts[0]} × {parts[1]}" if len(parts) == 2 else ix
                print(f"      +{i}. {ix}  ({desc})")

    # ── 2. Load Phase 1 OHLCV ───────────────────────────────────────────────
    def load_phase1_data(self) -> pd.DataFrame:
        """Load and validate the raw OHLCV parquet from Phase 1.

        Returns:
            pd.DataFrame sorted by (ticker, date).
        """
        print(f"\n{'='*72}")
        print(f"[Stage 5] Loading Phase 1 OHLCV Data...")
        print(f"{'='*72}")

        if not PHASE1_ARTIFACT.exists():
            raise FileNotFoundError(
                f"[FATAL] Phase 1 artifact not found: {PHASE1_ARTIFACT}\n"
                f"  Run Phase 1 pipeline before Phase 2."
            )

        t0 = perf_counter()
        df = pd.read_parquet(PHASE1_ARTIFACT)
        t_load = perf_counter() - t0

        df.columns = df.columns.str.lower().str.strip()
        print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols  "
              f"({t_load:.2f}s)")

        missing = REQUIRED_OHLCV_COLUMNS - set(df.columns)
        assert len(missing) == 0, (
            f"[SCHEMA VIOLATION] Missing OHLCV columns: {missing}"
        )
        print(f"  ✓ Schema valid: all required OHLCV columns present")

        # Auditor lifecycle — start here (first data contact)
        self.auditor.start()
        self.auditor.record_input(str(PHASE1_ARTIFACT), df)

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ── 3. Compute target ───────────────────────────────────────────────────
    @staticmethod
    def compute_target(df: pd.DataFrame) -> pd.Series:
        """Compute the 5-day forward log return per ticker.

        Formula:  target_t = ln( close_{t+FORWARD_DAYS} / close_t )

        Computed within each ticker group to prevent boundary contamination
        between consecutive tickers in the sorted DataFrame.

        Args:
            df: DataFrame sorted by (ticker, date) with 'close' column.

        Returns:
            pd.Series named 'target_log_return', index-aligned with df.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 5] Computing Target ({FORWARD_DAYS}-day forward log return)...")
        print(f"{'='*72}")

        t0 = perf_counter()

        future_close = df.groupby("ticker")["close"].shift(-FORWARD_DAYS)
        target = np.log(future_close / df["close"])
        target.name = "target_log_return"

        n_valid = int(target.notna().sum())
        n_nan   = int(target.isna().sum())
        print(f"  ✓ Computed: {n_valid:,} valid / {n_nan:,} NaN  "
              f"({perf_counter() - t0:.3f}s)")
        print(f"    NaN rows are expected at the tail of each ticker "
              f"(no future close available).")

        return target

    # ── 4. Regenerate base TA features in parallel ──────────────────────────
    def regenerate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the validated base TA features for all tickers in parallel.

        Filters curated_indicators to only the specs needed for the base
        features in self._base_features — skipping all others for speed.

        Args:
            df: OHLCV DataFrame (with target column already attached).

        Returns:
            pd.DataFrame: df augmented with base TA feature columns.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 5] Regenerating {len(self._base_features)} Base "
              f"Features (parallel, n_jobs={N_PARALLEL_JOBS})...")
        print(f"{'='*72}")

        t0 = perf_counter()

        # ── Filter curated_indicators to only needed specs ──────────────────
        all_specs   = self.config.curated_indicators
        base_set    = set(self._base_features)

        filtered_specs: List[Dict[str, Any]] = []
        for spec in all_specs:
            kind     = spec.get("kind", "")
            prefixes = _KIND_TO_PREFIX.get(kind, [])
            needed   = any(
                any(feat.startswith(pfx) for feat in base_set)
                for pfx in prefixes
            )
            if needed:
                filtered_specs.append(spec)

        self._filtered_specs = filtered_specs
        print(f"  ℹ Filtered {len(all_specs)} curated specs → "
              f"{len(filtered_specs)} needed for "
              f"{len(self._base_features)} base features")

        # ── Prepare ticker slices ───────────────────────────────────────────
        tickers        = sorted(df["ticker"].unique())
        ticker_groups: List[Tuple[str, pd.DataFrame]] = []
        skipped: List[str] = []

        for tkr in tickers:
            tdf = df[df["ticker"] == tkr].copy()
            if len(tdf) < MIN_TICKER_ROWS:
                skipped.append(tkr)
                continue
            ticker_groups.append((tkr, tdf))

        if skipped:
            print(f"  ⚠ Skipped {len(skipped)} tickers with < "
                  f"{MIN_TICKER_ROWS} rows: {skipped[:10]}"
                  f"{'...' if len(skipped) > 10 else ''}")

        print(f"  ℹ Processing {len(ticker_groups)} tickers "
              f"(parallel, backend=loky, n_jobs={N_PARALLEL_JOBS})")

        # ── Parallel TA generation ──────────────────────────────────────────
        parallel_results = Parallel(
            n_jobs=N_PARALLEL_JOBS,
            backend="loky",
            verbose=0,
        )(
            delayed(_process_ticker_worker)(tkr, tdf, filtered_specs)
            for tkr, tdf in ticker_groups
        )

        # ── Collect results ─────────────────────────────────────────────────
        frames: List[pd.DataFrame] = []
        errors: List[str]          = []

        for tkr, result_df, error in parallel_results:
            if error is not None:
                errors.append(tkr)
                if len(errors) <= 5:
                    warnings.warn(
                        f"[Stage 5] pandas_ta failed for '{tkr}': {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                frames.append(result_df)

        if not frames:
            raise RuntimeError(
                "[FATAL] All tickers failed TA generation. "
                "Check pandas_ta_classic installation and input data quality."
            )

        if errors:
            print(f"  ⚠ {len(errors)} tickers failed TA generation: "
                  f"{errors[:10]}{'...' if len(errors) > 10 else ''}")

        initial_cols = set(df.columns)
        result_df    = pd.concat(frames, axis=0, ignore_index=True)
        new_cols     = set(result_df.columns) - initial_cols
        print(f"  ✓ TA generation complete: {len(new_cols)} new columns "
              f"across {len(frames)} tickers  "
              f"({perf_counter() - t0:.1f}s)")

        # ── Verify all base features were produced ──────────────────────────
        generated  = set(result_df.columns)
        matched    = [c for c in self._base_features if c in generated]
        unmatched  = [c for c in self._base_features if c not in generated]

        # Case-insensitive fallback (pandas_ta occasionally changes casing)
        if unmatched:
            col_upper = {c.upper(): c for c in generated}
            for m in unmatched[:]:
                if m.upper() in col_upper:
                    actual = col_upper[m.upper()]
                    result_df.rename(columns={actual: m}, inplace=True)
                    matched.append(m)
                    unmatched.remove(m)

        if unmatched:
            raise RuntimeError(
                f"[FATAL] Base features not produced by TA generation: "
                f"{unmatched}\n"
                f"  Check that curated_indicators in Phase2Config includes "
                f"the correct specs for these features."
            )

        print(f"  ✓ Verified {len(matched)}/{len(self._base_features)} "
              f"base features present")

        return result_df

    # ── 5. Compute interaction columns ──────────────────────────────────────
    def compute_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction columns as element-wise products of base features.

        Parses the IX_ naming convention established in Stage 3:
            IX_<featureA>_x_<featureB>  →  df[featureA] * df[featureB]

        No normalisation is applied — raw products are saved. StandardScaler
        is applied inside each Phase 3 CV/walk-forward fold to prevent
        data leakage.

        Args:
            df: DataFrame with all base feature columns present.

        Returns:
            pd.DataFrame: df with interaction columns appended.
        """
        if not self._interaction_features:
            print(f"\n  ℹ No interaction features to compute.")
            return df

        print(f"\n{'='*72}")
        print(f"[Stage 5] Computing {len(self._interaction_features)} "
              f"Interaction Column(s)...")
        print(f"{'='*72}")

        t0 = perf_counter()

        for ix_name in self._interaction_features:
            # Parse  "IX_LOGRET_5_x_PCTRET_10"  →  ("LOGRET_5", "PCTRET_10")
            inner = ix_name[len(_IX_PREFIX):]       # "LOGRET_5_x_PCTRET_10"
            parts = inner.split(_IX_SEPARATOR)       # ["LOGRET_5", "PCTRET_10"]

            if len(parts) != 2:
                raise ValueError(
                    f"[FATAL] Cannot parse interaction name '{ix_name}'.\n"
                    f"  Expected format: IX_<featureA>_x_<featureB>\n"
                    f"  Got parts after split: {parts}"
                )

            feat_a, feat_b = parts

            for feat in (feat_a, feat_b):
                if feat not in df.columns:
                    raise RuntimeError(
                        f"[FATAL] Interaction '{ix_name}' requires base "
                        f"feature '{feat}' which is not in the DataFrame.\n"
                        f"  Check that regenerate_base_features() ran first."
                    )

            df[ix_name] = df[feat_a] * df[feat_b]
            print(f"  ✓ {ix_name}  =  {feat_a} × {feat_b}")

        print(f"  ✓ Interaction columns computed  "
              f"({perf_counter() - t0:.3f}s)")

        return df

    # ── 6. Assemble and save final matrix ───────────────────────────────────
    def save_feature_matrix(self, df: pd.DataFrame) -> None:
        """Select the final columns and save atomically to parquet.

        Output schema:
            ticker | date | <feature_1> | ... | <feature_N> | target_log_return

        Rows with NaN in ANY feature or the target are retained — downstream
        Phase 3 code handles its own .dropna() to avoid silently discarding
        data that might be valid for some features but not others.

        Atomic write pattern: write to .tmp, then rename, so a crash mid-write
        never leaves a corrupt parquet file that looks complete.

        Args:
            df: Fully assembled DataFrame with all features + target.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 5] Assembling and Saving Final Feature Matrix...")
        print(f"{'='*72}")

        target_col   = "target_log_return"
        keep_cols    = ["ticker", "date"] + self._final_features + [target_col]

        # Verify all expected columns are present before slicing
        missing_cols = [c for c in keep_cols if c not in df.columns]
        if missing_cols:
            raise RuntimeError(
                f"[FATAL] Cannot assemble final matrix. "
                f"Missing columns: {missing_cols}"
            )

        final_df = (
            df[keep_cols]
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        n_rows        = len(final_df)
        n_complete    = int(final_df.dropna().shape[0])
        n_any_nan     = n_rows - n_complete
        pct_complete  = n_complete / n_rows * 100 if n_rows > 0 else 0.0

        print(f"  ✓ Final matrix shape : {n_rows:,} rows × "
              f"{len(keep_cols)} cols")
        print(f"  ✓ Complete rows      : {n_complete:,} "
              f"({pct_complete:.1f}%)")
        print(f"  ℹ Rows with any NaN  : {n_any_nan:,} "
              f"(expected — target NaN at ticker tails)")
        print(f"  ✓ Columns saved      : {keep_cols}")

        # ── Atomic write ────────────────────────────────────────────────────
        PHASE2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = OUTPUT_PARQUET.with_suffix(".parquet.tmp")

        try:
            final_df.to_parquet(tmp_path, index=False, engine="pyarrow")
            tmp_path.replace(OUTPUT_PARQUET)
            size_mb = OUTPUT_PARQUET.stat().st_size / 1_048_576
            print(f"  ✓ Saved: {OUTPUT_PARQUET.name}  "
                  f"({size_mb:.1f} MB)")
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"[FATAL] Failed to write feature matrix: {exc}"
            ) from exc

        # ── Auditor lifecycle ────────────────────────────────────────────────
        try:
            self.auditor.record_output(final_df, self.config.to_snapshot())
            self.auditor.success()
            print(f"  ✓ manifest.json updated (via Auditor)")
        except Exception as exc:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {exc}")

    # ── Orchestrator ────────────────────────────────────────────────────────
    def run(self) -> None:
        """Execute the full feature matrix generation pipeline.

        Step order:
            1. Read validated feature list from interaction_report.json.
            2. Load raw OHLCV from Phase 1.
            3. Compute 5-day forward log return target.
            4. Regenerate base TA features in parallel.
            5. Compute interaction product columns.
            6. Assemble and save final_feature_matrix.parquet.
        """
        t_pipeline = perf_counter()

        # Step 1: Discover features from Stage 3 artifact
        self.load_interaction_report()

        # Step 2: Load raw OHLCV
        df = self.load_phase1_data()

        # Step 3: Compute target (attach before TA so ticker groups are intact)
        target = self.compute_target(df)
        df["target_log_return"] = target

        # Step 4: Regenerate base TA features in parallel
        df = self.regenerate_base_features(df)

        # Step 5: Compute interaction product columns
        df = self.compute_interactions(df)

        # Step 6: Assemble and save
        self.save_feature_matrix(df)

        total = perf_counter() - t_pipeline
        print(f"\n{'='*72}")
        print(f"[Stage 5] COMPLETE — Total runtime: {total:.1f}s")
        print(f"  Output → {OUTPUT_PARQUET}")
        print(f"{'='*72}\n")


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 5 — Final Feature Matrix Generation       ║")
    print(f"║  Version: {VERSION:<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        config  = Phase2Config()
        builder = FeatureMatrixBuilder(config=config)
        builder.run()
        sys.exit(0)

    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as exc:
        print(f"\n[ASSERTION FAILED] {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Feature matrix generation aborted.",
              file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[UNHANDLED ERROR] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)
