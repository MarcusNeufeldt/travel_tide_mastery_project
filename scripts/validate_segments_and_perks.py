import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both customer segments and perks assignment data."""
    segments_df = pd.read_csv('scripts/output/segments/customer_segments.csv')
    perks_df = pd.read_csv('scripts/output/segments/perks_assignment.csv')
    return segments_df, perks_df

def validate_data_consistency(segments_df: pd.DataFrame, perks_df: pd.DataFrame) -> Dict:
    """Check consistency between the two datasets."""
    consistency_report = {
        'total_customers': {
            'segments': len(segments_df),
            'perks': len(perks_df)
        },
        'user_id_match': len(set(segments_df['user_id']) - set(perks_df['user_id'])) == 0,
        'segment_distribution_match': (
            segments_df['segment'].value_counts().to_dict() == 
            perks_df['segment'].value_counts().to_dict()
        )
    }
    return consistency_report

def analyze_perk_distribution(perks_df: pd.DataFrame) -> Dict:
    """Analyze the distribution of perks across segments."""
    perk_analysis = {
        'perks_per_segment': perks_df.groupby('segment')['assigned_perks'].value_counts().to_dict(),
        'avg_perks_per_tier': perks_df.groupby('perk_tier')['assigned_perks'].apply(
            lambda x: x.str.count(',').mean() + 1
        ).to_dict(),
        'score_ranges': {
            col: {'min': perks_df[col].min(), 'max': perks_df[col].max()}
            for col in perks_df.columns if col.endswith('_score')
        }
    }
    return perk_analysis

def validate_business_rules(perks_df: pd.DataFrame) -> Dict:
    """Validate that business rules are being followed."""
    rules_validation = {
        'all_customers_have_perks': perks_df['assigned_perks'].notna().all(),
        'basic_benefits_only_basic': (
            perks_df[perks_df['assigned_perks'] == 'Basic Benefits']['perk_tier'] == 'Basic'
        ).all(),
        'platinum_multiple_perks': (
            perks_df[perks_df['perk_tier'] == 'Platinum']['assigned_perks'].str.count(',').mean() > 1
        ),
        'score_correlations': perks_df[[
            'total_perk_score', 'premium_support_score', 
            'priority_booking_score', 'discount_access_score', 
            'loyalty_bonus_score'
        ]].corr().to_dict()
    }
    return rules_validation

def plot_validation_results(segments_df: pd.DataFrame, perks_df: pd.DataFrame) -> None:
    """Create validation visualizations."""
    os.makedirs('scripts/output/validation', exist_ok=True)
    
    # 1. Score Distribution by Segment
    plt.figure(figsize=(12, 6))
    score_cols = [col for col in perks_df.columns if col.endswith('_score')]
    for col in score_cols:
        sns.boxplot(data=perks_df, x='segment', y=col)
        plt.title(f'{col} Distribution by Segment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'scripts/output/validation/{col}_by_segment.png')
        plt.close()
    
    # 2. Perk Distribution
    plt.figure(figsize=(12, 6))
    perks_df['perk_count'] = perks_df['assigned_perks'].str.count(',') + 1
    sns.boxplot(data=perks_df, x='perk_tier', y='perk_count')
    plt.title('Number of Perks by Tier')
    plt.tight_layout()
    plt.savefig('scripts/output/validation/perks_by_tier.png')
    plt.close()

def main():
    """Main validation function."""
    print("Loading data...")
    segments_df, perks_df = load_data()
    
    print("\nChecking data consistency...")
    consistency_report = validate_data_consistency(segments_df, perks_df)
    print("\nData Consistency Report:")
    for key, value in consistency_report.items():
        print(f"{key}: {value}")
    
    print("\nAnalyzing perk distribution...")
    perk_analysis = analyze_perk_distribution(perks_df)
    print("\nPerk Analysis:")
    print(f"Average perks per tier: {perk_analysis['avg_perks_per_tier']}")
    print("\nScore ranges:")
    for score, ranges in perk_analysis['score_ranges'].items():
        print(f"{score}: {ranges}")
    
    print("\nValidating business rules...")
    rules_validation = validate_business_rules(perks_df)
    print("\nBusiness Rules Validation:")
    for rule, result in rules_validation.items():
        if rule != 'score_correlations':
            print(f"{rule}: {result}")
    
    print("\nGenerating validation plots...")
    plot_validation_results(segments_df, perks_df)
    
    # Additional sanity checks
    print("\nSanity Checks:")
    print(f"Total unique customers: {len(perks_df['user_id'].unique())}")
    print("\nSegment distribution:")
    print(perks_df['segment'].value_counts(normalize=True).round(3) * 100)
    print("\nMost common perk combinations:")
    print(perks_df['assigned_perks'].value_counts().head())
    print("\nAverage scores by segment:")
    print(perks_df.groupby('segment')['total_perk_score'].mean().round(3))

if __name__ == "__main__":
    main() 