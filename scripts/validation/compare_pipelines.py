#!/usr/bin/env python
"""
Compare old vs. new pipeline results.
Computes correlations, RMSE, and identifies divergences for validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_results(path):
    """Load aggregated pipeline output (final_data.rds or CSV equivalent)."""
    if path.endswith(".rds"):
        try:
            import pyreadr
            result = pyreadr.read_r(path)
            return result[None]  # Return first (only) dataset
        except ImportError:
            logger.error("pyreadr not installed. Install with: pip install pyreadr")
            return None
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        logger.error(f"Unsupported file type: {path}")
        return None

def compare_datasets(old_df, new_df, participant_col="PN", tolerance=0.05):
    """
    Compare old and new results.
    Returns a report with correlations, RMSE, and pass/fail status.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_participants": len(new_df),
        "feature_count": len([c for c in new_df.columns if c != participant_col]),
        "features": {}
    }
    
    # Find common columns (features)
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    common = old_cols & new_cols
    
    missing_old = new_cols - old_cols
    missing_new = old_cols - new_cols
    
    if missing_old:
        logger.warning(f"New pipeline has {len(missing_old)} new features: {missing_old}")
    if missing_new:
        logger.warning(f"Old pipeline had {len(missing_new)} features now missing: {missing_new}")
    
    # Compare common features
    divergences = []
    matches = 0
    
    for col in sorted(common):
        if col == participant_col or not np.issubdtype(new_df[col].dtype, np.number):
            continue
        
        # Align by participant
        merged = old_df[[participant_col, col]].merge(
            new_df[[participant_col, col]],
            on=participant_col,
            suffixes=("_old", "_new")
        )
        
        if len(merged) == 0:
            logger.warning(f"No matching participants for {col}")
            continue
        
        old_vals = merged[f"{col}_old"]
        new_vals = merged[f"{col}_new"]
        
        # Compute metrics
        corr = old_vals.corr(new_vals)
        rmse = np.sqrt(np.mean((old_vals - new_vals) ** 2))
        mae = np.mean(np.abs(old_vals - new_vals))
        pct_change = np.mean(np.abs((new_vals - old_vals) / (np.abs(old_vals) + 1e-9))) * 100
        
        status = "✓ OK" if abs(1 - corr) < (1 - tolerance) else "⚠ DIVERGE"
        
        report["features"][col] = {
            "correlation": float(corr),
            "rmse": float(rmse),
            "mae": float(mae),
            "pct_change": float(pct_change),
            "status": status,
            "n_samples": len(merged)
        }
        
        if status == "⚠ DIVERGE":
            divergences.append((col, corr, pct_change))
        else:
            matches += 1
    
    report["summary"] = {
        "matching_features": matches,
        "divergent_features": len(divergences),
        "match_rate": matches / (matches + len(divergences)) if (matches + len(divergences)) > 0 else 0
    }
    
    return report, divergences

def print_report(report, divergences):
    """Print a nicely formatted comparison report."""
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPARISON REPORT")
    logger.info("="*80)
    logger.info(f"Participants: {report['n_participants']}")
    logger.info(f"Features compared: {report['feature_count']}")
    logger.info(f"Match rate: {report['summary']['match_rate']*100:.1f}%")
    logger.info(f"Matching features: {report['summary']['matching_features']}")
    logger.info(f"Divergent features: {report['summary']['divergent_features']}")
    logger.info("="*80 + "\n")
    
    if divergences:
        logger.warning("DIVERGENT FEATURES (correlation < threshold):")
        logger.warning("-" * 80)
        for feat, corr, pct in sorted(divergences, key=lambda x: x[1])[:20]:
            logger.warning(f"  {feat:40s} | r={corr:.3f} | Δ={pct:.1f}%")
        logger.warning("-" * 80 + "\n")
    
    logger.info("TOP MATCHING FEATURES:")
    logger.info("-" * 80)
    sorted_feats = sorted(
        report["features"].items(),
        key=lambda x: x[1]["correlation"],
        reverse=True
    )[:10]
    for feat, metrics in sorted_feats:
        logger.info(f"  {feat:40s} | r={metrics['correlation']:.3f} | Δ={metrics['pct_change']:.1f}%")
    logger.info("-" * 80 + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare old vs. new pipeline results"
    )
    parser.add_argument(
        "--old",
        default="results/baseline_old/final_data.csv",
        help="Path to old results (CSV or RDS)"
    )
    parser.add_argument(
        "--new",
        default="output/final_data.csv",
        help="Path to new results (CSV or RDS)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Correlation tolerance for 'matching' (default 0.05 = r>0.95)"
    )
    parser.add_argument(
        "--output",
        default="output/comparison_report.json",
        help="Save report to JSON"
    )
    
    args = parser.parse_args()
    
    old_path = Path(args.old)
    new_path = Path(args.new)
    
    if not old_path.exists():
        logger.error(f"Old results not found: {old_path}")
        logger.info(f"Expected: {old_path}")
        return
    
    if not new_path.exists():
        logger.error(f"New results not found: {new_path}")
        logger.info(f"Expected: {new_path}")
        return
    
    logger.info(f"Loading old results: {old_path}")
    old_df = load_results(str(old_path))
    
    logger.info(f"Loading new results: {new_path}")
    new_df = load_results(str(new_path))
    
    if old_df is None or new_df is None:
        return
    
    logger.info(f"Old shape: {old_df.shape}, New shape: {new_df.shape}")
    
    report, divergences = compare_datasets(old_df, new_df, tolerance=args.tolerance)
    print_report(report, divergences)
    
    # Save report
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()
