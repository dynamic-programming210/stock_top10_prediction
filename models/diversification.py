"""
D3: Sector Diversification Constraints for Top-10 Selection
Ensures portfolio is diversified across sectors to reduce concentration risk
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SECTOR_MAP, SP500_SYMBOLS
from data.fetch_universe import get_sector_mapping
from utils import get_logger

logger = get_logger(__name__)

# Default diversification constraints
DEFAULT_MAX_PER_SECTOR = 3  # Max stocks from any single sector
DEFAULT_MIN_SECTORS = 4     # Minimum number of different sectors
DEFAULT_TOP_N = 10          # Number of stocks to select


class SectorDiversifier:
    """
    Applies sector diversification constraints to stock selection
    """
    
    def __init__(
        self,
        max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
        min_sectors: int = DEFAULT_MIN_SECTORS,
        sector_mapping: Dict[str, str] = None
    ):
        """
        Initialize diversifier
        
        Args:
            max_per_sector: Maximum stocks allowed from any single sector
            min_sectors: Minimum number of different sectors required
            sector_mapping: Dict mapping symbol -> sector (uses default if None)
        """
        self.max_per_sector = max_per_sector
        self.min_sectors = min_sectors
        self.sector_mapping = sector_mapping or get_sector_mapping()
        
        logger.info(f"SectorDiversifier initialized: max_per_sector={max_per_sector}, "
                   f"min_sectors={min_sectors}")
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        return self.sector_mapping.get(symbol, 'Unknown')
    
    def diversify_selection(
        self,
        ranked_df: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N,
        rank_col: str = 'predicted_rank',
        symbol_col: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Select top-N stocks with sector diversification constraints
        
        Uses a greedy algorithm:
        1. Sort by predicted rank (best first)
        2. Iterate through sorted list
        3. Add stock if sector constraint allows
        4. Continue until we have top_n stocks
        
        Args:
            ranked_df: DataFrame with predictions and rankings
            top_n: Number of stocks to select
            rank_col: Column with prediction/rank scores (higher = better)
            symbol_col: Column with stock symbols
            
        Returns:
            DataFrame with diversified top-N selection
        """
        # Add sector information
        df = ranked_df.copy()
        df['sector'] = df[symbol_col].map(self.sector_mapping).fillna('Unknown')
        
        # Sort by rank (descending - higher is better)
        df = df.sort_values(rank_col, ascending=False)
        
        # Track sector counts
        sector_counts = {}
        selected_indices = []
        selected_symbols = []
        
        for idx, row in df.iterrows():
            symbol = row[symbol_col]
            sector = row['sector']
            
            # Check sector constraint
            current_count = sector_counts.get(sector, 0)
            if current_count < self.max_per_sector:
                selected_indices.append(idx)
                selected_symbols.append(symbol)
                sector_counts[sector] = current_count + 1
                
                if len(selected_indices) >= top_n:
                    break
        
        # Check if we met minimum sectors requirement
        n_sectors = len(sector_counts)
        if n_sectors < self.min_sectors and len(df) >= top_n:
            logger.warning(f"Only {n_sectors} sectors in selection (min: {self.min_sectors})")
        
        # Get selected rows
        result = df.loc[selected_indices].copy()
        result['diversified_rank'] = range(1, len(result) + 1)
        
        # Log summary
        logger.info(f"Selected {len(result)} stocks across {n_sectors} sectors")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            logger.debug(f"  {sector}: {count} stocks")
        
        return result
    
    def analyze_sector_concentration(
        self,
        ranked_df: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N,
        symbol_col: str = 'symbol'
    ) -> Dict:
        """
        Analyze sector concentration in top-N without diversification
        
        Args:
            ranked_df: DataFrame sorted by rank
            top_n: Number of top stocks to analyze
            symbol_col: Column with stock symbols
            
        Returns:
            Dict with concentration metrics
        """
        top_df = ranked_df.head(top_n).copy()
        top_df['sector'] = top_df[symbol_col].map(self.sector_mapping).fillna('Unknown')
        
        sector_counts = top_df['sector'].value_counts()
        
        analysis = {
            'top_n': top_n,
            'n_sectors': len(sector_counts),
            'sector_counts': sector_counts.to_dict(),
            'max_concentration': int(sector_counts.max()),
            'concentration_pct': float(sector_counts.max() / top_n * 100),
            'most_concentrated_sector': sector_counts.idxmax(),
            'herfindahl_index': float((sector_counts / top_n).pow(2).sum())
        }
        
        return analysis
    
    def compare_diversified_vs_undiversified(
        self,
        ranked_df: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N,
        rank_col: str = 'predicted_rank',
        symbol_col: str = 'symbol'
    ) -> Dict:
        """
        Compare diversified vs undiversified selection
        
        Returns:
            Dict with comparison metrics
        """
        # Undiversified (pure rank-based)
        df = ranked_df.copy()
        df['sector'] = df[symbol_col].map(self.sector_mapping).fillna('Unknown')
        undiv_top = df.nlargest(top_n, rank_col)
        
        # Diversified
        div_top = self.diversify_selection(ranked_df, top_n, rank_col, symbol_col)
        
        comparison = {
            'undiversified': {
                'symbols': undiv_top[symbol_col].tolist(),
                'sectors': undiv_top['sector'].value_counts().to_dict(),
                'n_sectors': undiv_top['sector'].nunique(),
                'mean_rank': float(undiv_top[rank_col].mean()),
                'herfindahl': float((undiv_top['sector'].value_counts() / top_n).pow(2).sum())
            },
            'diversified': {
                'symbols': div_top[symbol_col].tolist(),
                'sectors': div_top['sector'].value_counts().to_dict(),
                'n_sectors': div_top['sector'].nunique(),
                'mean_rank': float(div_top[rank_col].mean()),
                'herfindahl': float((div_top['sector'].value_counts() / top_n).pow(2).sum())
            }
        }
        
        # Calculate overlap
        undiv_set = set(undiv_top[symbol_col])
        div_set = set(div_top[symbol_col])
        comparison['overlap'] = len(undiv_set & div_set)
        comparison['rank_sacrifice'] = comparison['undiversified']['mean_rank'] - comparison['diversified']['mean_rank']
        
        return comparison


def generate_diversified_top10(
    predictions_df: pd.DataFrame,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    min_sectors: int = DEFAULT_MIN_SECTORS,
    as_of_date: str = None
) -> pd.DataFrame:
    """
    Convenience function to generate diversified top-10
    
    Args:
        predictions_df: DataFrame with predictions
        max_per_sector: Max stocks per sector
        min_sectors: Min different sectors
        as_of_date: Date label for the selection
        
    Returns:
        DataFrame with diversified top-10
    """
    diversifier = SectorDiversifier(
        max_per_sector=max_per_sector,
        min_sectors=min_sectors
    )
    
    result = diversifier.diversify_selection(
        predictions_df,
        top_n=10,
        rank_col='predicted_rank',
        symbol_col='symbol'
    )
    
    if as_of_date:
        result['selection_date'] = as_of_date
    
    return result


def print_diversification_report(comparison: Dict):
    """Print a formatted diversification comparison report"""
    print("\n" + "=" * 60)
    print("SECTOR DIVERSIFICATION ANALYSIS")
    print("=" * 60)
    
    print("\n📊 UNDIVERSIFIED (Pure Rank-Based):")
    print(f"   Symbols: {', '.join(comparison['undiversified']['symbols'])}")
    print(f"   Sectors: {comparison['undiversified']['n_sectors']}")
    print(f"   Mean Rank Score: {comparison['undiversified']['mean_rank']:.4f}")
    print(f"   Herfindahl Index: {comparison['undiversified']['herfindahl']:.3f}")
    
    print("\n📊 DIVERSIFIED (With Sector Constraints):")
    print(f"   Symbols: {', '.join(comparison['diversified']['symbols'])}")
    print(f"   Sectors: {comparison['diversified']['n_sectors']}")
    print(f"   Mean Rank Score: {comparison['diversified']['mean_rank']:.4f}")
    print(f"   Herfindahl Index: {comparison['diversified']['herfindahl']:.3f}")
    
    print(f"\n📈 COMPARISON:")
    print(f"   Overlap: {comparison['overlap']}/10 stocks in common")
    print(f"   Rank Sacrifice: {comparison['rank_sacrifice']:+.4f} (lower is better)")
    print(f"   Diversification Gain: {comparison['undiversified']['herfindahl'] - comparison['diversified']['herfindahl']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sector Diversification Analysis")
    parser.add_argument('--max-per-sector', type=int, default=3,
                        help='Maximum stocks per sector (default: 3)')
    parser.add_argument('--min-sectors', type=int, default=4,
                        help='Minimum sectors required (default: 4)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze, do not apply diversification')
    
    args = parser.parse_args()
    
    # Try to load latest predictions
    from pathlib import Path
    from config import OUTPUTS_DIR
    
    pred_files = list(OUTPUTS_DIR.glob("top10_*.csv"))
    
    if not pred_files:
        print("No prediction files found. Run daily update first.")
        print("\nDemo with sample data:")
        
        # Create sample data for demo
        sample_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 
                      'AMZN', 'AMD', 'CRM', 'INTC', 'ORCL',
                      'JPM', 'BAC', 'GS', 'V', 'MA'],
            'predicted_rank': [1.0, 0.95, 0.90, 0.85, 0.80, 
                              0.75, 0.70, 0.65, 0.60, 0.55,
                              0.50, 0.45, 0.40, 0.35, 0.30]
        })
        
        diversifier = SectorDiversifier(
            max_per_sector=args.max_per_sector,
            min_sectors=args.min_sectors
        )
        
        comparison = diversifier.compare_diversified_vs_undiversified(sample_data)
        print_diversification_report(comparison)
    else:
        # Use latest prediction file
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading predictions from: {latest_file}")
        
        predictions = pd.read_csv(latest_file)
        
        # Need to have predicted_rank column
        if 'predicted_rank' not in predictions.columns:
            # Assume sorted by rank already
            predictions['predicted_rank'] = range(len(predictions), 0, -1)
        
        diversifier = SectorDiversifier(
            max_per_sector=args.max_per_sector,
            min_sectors=args.min_sectors
        )
        
        comparison = diversifier.compare_diversified_vs_undiversified(predictions)
        print_diversification_report(comparison)
