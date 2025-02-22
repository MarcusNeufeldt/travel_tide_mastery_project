# Mastery_Project/scripts/03_customer_segmentation.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from utils import setup_logging
from scipy.stats import percentileofscore
from typing import Dict, List, Tuple

# Setup logging
logger = setup_logging(__name__)

# Create output directories
os.makedirs('scripts/output/segments', exist_ok=True)
os.makedirs('scripts/output/metrics', exist_ok=True)

try:
    os.makedirs('scripts/output/metrics', exist_ok=True)
    md_file = open('scripts/output/metrics/segmentation_results.md', 'w', encoding='utf-8')
    logger.info('Created markdown file for results')
except Exception as e:
    logger.error(f'Error creating markdown file: {e}')
    raise

def write_to_md(text):
    """Write text to markdown file with error handling"""
    try:
        md_file.write(text + '\n')
    except Exception as e:
        logger.error(f'Error writing to markdown file: {e}')
        raise

# Define customer perks and their requirements
PERKS_CONFIG = {
    'premium_support': {
        'min_spend': 5000,
        'min_bookings': 5,
        'weight': 0.3
    },
    'priority_booking': {
        'min_frequency_score': 0.7,
        'weight': 0.2
    },
    'discount_access': {
        'min_discount_score': 0.6,
        'weight': 0.25
    },
    'loyalty_bonus': {
        'min_monetary_score': 0.8,
        'weight': 0.25
    }
}

def load_metrics():
    """Load the user metrics data from CSV."""
    try:
        # Try the Windows-style path first
        return pd.read_csv('scripts/output/metrics/user_metrics.csv')
    except FileNotFoundError:
        # Fall back to Unix-style path
        return pd.read_csv('output/metrics/user_metrics.csv')

def plot_metric_distributions(df, metrics):
    """Plot distributions of key metrics"""
    os.makedirs('scripts/output/segments', exist_ok=True)
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, len(metrics), i)
        sns.histplot(df[metric], bins=30)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.tight_layout()
    plt.savefig('scripts/output/segments/metric_distributions.png')
    plt.close()

def plot_correlations(df, metrics):
    """Plot correlation matrix for metrics"""
    os.makedirs('scripts/output/segments', exist_ok=True)
    correlations=df[metrics].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Metric Correlations')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/metric_correlations.png')
    plt.close()
    return correlations

def create_threshold_segments(df):
    """Create segments using thresholding approach."""
    # Calculate percentiles for bargain hunter index
    df['bargain_percentile'] = df['bargain_hunter_index'].rank(pct=True)
    # Create segments based on percentiles
    df['segment'] = pd.cut(
        df['bargain_percentile'],
        bins=[0, 0.5, 0.8, 0.9, 1.0],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    return df

def analyze_segments(df):
    """Analyze the characteristics of each segment"""
    segment_stats = df.groupby('segment').agg({
        # Booking behavior
        'total_flights': 'mean',
        'total_hotels': 'mean',
        'total_bookings': 'mean',
        # Monetary value
        'total_flight_spend': 'mean',
        'total_hotel_spend': 'mean',
        'total_spend': 'mean',
        # Discount behavior
        'discount_flight_proportion': 'mean',
        'average_flight_discount': 'mean',
        'ads_per_km': 'mean',
        # Component scores
        'monetary_score': 'mean',
        'frequency_score': 'mean',
        'discount_score': 'mean',
        'final_score': 'mean',
        # Customer count
        'user_id': 'count'
    }).round(3)
    
    # Calculate percentage of customers in each segment
    segment_stats['customer_percentage'] = (
        segment_stats['user_id'] / segment_stats['user_id'].sum() * 100
    ).round(2)
    
    # Write segment statistics to markdown
    write_to_md("\n## Segment Statistics\n")
    write_to_md(segment_stats.to_markdown())
    
    return segment_stats

def validate_segments(df):
    """Validate segment quality with comprehensive metrics."""
    
    # Data quality checks
    logger.info("\nData Quality Checks:")
    
    # Check for negative values
    negative_spend = (df['total_spend'] < 0).sum()
    if negative_spend > 0:
        logger.warning(f"Found {negative_spend} customers with negative total spend!")
    
    # Check for unreasonable values
    high_spend = df['total_spend'].quantile(0.99)
    outliers = (df['total_spend'] > high_spend).sum()
    logger.info(f"Number of customers with spend > 99th percentile (${high_spend:.2f}): {outliers}")
    
    # Check booking counts
    zero_bookings = (df['total_bookings'] == 0).sum()
    if zero_bookings > 0:
        logger.warning(f"Found {zero_bookings} customers with zero bookings!")
    
    # Segment stability
    logger.info("\nSegment Sizes:")
    sizes = df['segment'].value_counts(normalize=True)
    logger.info(sizes)
    
    # Segment characteristics
    logger.info("\nSegment Profiles:")
    profile = df.groupby('segment').agg({
        # Monetary metrics
        'total_spend': ['min', 'mean', 'max'],
        'total_bookings': ['min', 'mean', 'max'],
        # Discount behavior
        'discount_flight_proportion': ['min', 'mean', 'max'],
        'average_flight_discount': ['min', 'mean', 'max'],
        'ads_per_km': ['min', 'mean', 'max'],
        # Component scores
        'monetary_score': 'mean',
        'frequency_score': 'mean',
        'discount_score': 'mean',
        'final_score': ['min', 'mean', 'max']
    }).round(3)
    logger.info(profile)
    
    # Segment separation check
    logger.info("\nSegment Separation:")
    for metric in ['monetary_score', 'frequency_score', 'discount_score']:
        overlap = df.groupby('segment')[metric].agg(['min', 'max'])
        logger.info(f"\n{metric} ranges by segment:")
        logger.info(overlap)

def calculate_perk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate perk eligibility scores for each customer using fuzzy logic.
    
    Args:
        df: DataFrame containing customer metrics
    
    Returns:
        DataFrame with added perk score columns
    """
    # Calculate premium support score
    df['premium_support_score'] = (
        (df['total_spend'] >= PERKS_CONFIG['premium_support']['min_spend']).astype(float) * 0.6 +
        (df['total_bookings'] >= PERKS_CONFIG['premium_support']['min_bookings']).astype(float) * 0.4
    )
    
    # Calculate priority booking score
    df['priority_booking_score'] = np.where(
        df['frequency_score'] >= PERKS_CONFIG['priority_booking']['min_frequency_score'],
        df['frequency_score'],
        df['frequency_score'] * 0.5  # Partial credit for lower frequency
    )
    
    # Calculate discount access score
    df['discount_access_score'] = np.where(
        df['discount_score'] >= PERKS_CONFIG['discount_access']['min_discount_score'],
        df['discount_score'],
        df['discount_score'] * 0.7  # Partial credit for lower discount engagement
    )
    
    # Calculate loyalty bonus score
    df['loyalty_bonus_score'] = np.where(
        df['monetary_score'] >= PERKS_CONFIG['loyalty_bonus']['min_monetary_score'],
        df['monetary_score'],
        df['monetary_score'] * 0.6  # Partial credit for lower monetary value
    )
    
    return df

def assign_perks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign perks to customers based on their scores and handle edge cases.
    
    Args:
        df: DataFrame with perk scores
        
    Returns:
        DataFrame with assigned perks
    """
    # Calculate weighted total perk score
    df['total_perk_score'] = (
        df['premium_support_score'] * PERKS_CONFIG['premium_support']['weight'] +
        df['priority_booking_score'] * PERKS_CONFIG['priority_booking']['weight'] +
        df['discount_access_score'] * PERKS_CONFIG['discount_access']['weight'] +
        df['loyalty_bonus_score'] * PERKS_CONFIG['loyalty_bonus']['weight']
    )
    
    # Handle tie cases by using secondary metrics
    df['tiebreaker_score'] = (
        df['total_spend'] * 0.4 +
        df['total_bookings'] * 0.3 +
        df['customer_value'] * 0.3
    )
    
    # Assign minimum perk level based on total score
    df['perk_tier'] = pd.qcut(
        df['total_perk_score'],
        q=[0, 0.5, 0.8, 0.9, 1.0],
        labels=['Basic', 'Silver', 'Gold', 'Platinum'],
        duplicates='drop'  # Handle ties
    )
    
    # Assign specific perks
    df['assigned_perks'] = df.apply(lambda x: get_customer_perks(x), axis=1)
    
    return df

def get_customer_perks(row) -> List[str]:
    """
    Determine specific perks for a customer based on their scores and tier.
    
    Args:
        row: Series containing customer metrics and scores
        
    Returns:
        List of perks the customer is eligible for
    """
    perks = []
    
    # Assign perks based on tier and scores
    if row['perk_tier'] == 'Platinum':
        # Platinum customers get all perks they qualify for, with lower thresholds
        if row['premium_support_score'] >= 0.6:
            perks.append('Premium Support')
        if row['priority_booking_score'] >= 0.7:
            perks.append('Priority Booking')
        if row['discount_access_score'] >= 0.65:
            perks.append('Enhanced Discounts')
        if row['loyalty_bonus_score'] >= 0.75:
            perks.append('Loyalty Rewards')
        # Ensure Platinum customers get at least 2 perks
        if len(perks) < 2:
            scores = {
                'Premium Support': row['premium_support_score'],
                'Priority Booking': row['priority_booking_score'],
                'Enhanced Discounts': row['discount_access_score'],
                'Loyalty Rewards': row['loyalty_bonus_score']
            }
            # Add top scoring perks until we have at least 2
            while len(perks) < 2:
                top_perk = max(scores.items(), key=lambda x: x[1] if x[0] not in perks else -1)[0]
                if top_perk not in perks:
                    perks.append(top_perk)
    
    elif row['perk_tier'] == 'Gold':
        # Gold customers get 1-2 perks with medium thresholds
        if row['premium_support_score'] >= 0.65:
            perks.append('Premium Support')
        if row['priority_booking_score'] >= 0.75:
            perks.append('Priority Booking')
        if row['discount_access_score'] >= 0.7:
            perks.append('Enhanced Discounts')
        if row['loyalty_bonus_score'] >= 0.8:
            perks.append('Loyalty Rewards')
        # Ensure Gold customers get at least 1 perk
        if not perks:
            scores = {
                'Premium Support': row['premium_support_score'],
                'Priority Booking': row['priority_booking_score'],
                'Enhanced Discounts': row['discount_access_score'],
                'Loyalty Rewards': row['loyalty_bonus_score']
            }
            top_perk = max(scores.items(), key=lambda x: x[1])[0]
            perks.append(top_perk)
    
    elif row['perk_tier'] == 'Silver':
        # Silver customers get 1 perk with higher thresholds
        if row['premium_support_score'] >= 0.7:
            perks.append('Premium Support')
        if row['priority_booking_score'] >= 0.8:
            perks.append('Priority Booking')
        if row['discount_access_score'] >= 0.75:
            perks.append('Enhanced Discounts')
        if row['loyalty_bonus_score'] >= 0.85:
            perks.append('Loyalty Rewards')
        # Ensure Silver customers get exactly 1 perk
        if not perks:
            scores = {
                'Premium Support': row['premium_support_score'],
                'Priority Booking': row['priority_booking_score'],
                'Enhanced Discounts': row['discount_access_score'],
                'Loyalty Rewards': row['loyalty_bonus_score']
            }
            top_perk = max(scores.items(), key=lambda x: x[1])[0]
            perks.append(top_perk)
        elif len(perks) > 1:
            # Keep only the highest scoring perk
            scores = {perk: row[{
                'Premium Support': 'premium_support_score',
                'Priority Booking': 'priority_booking_score',
                'Enhanced Discounts': 'discount_access_score',
                'Loyalty Rewards': 'loyalty_bonus_score'
            }[perk]] for perk in perks}
            top_perk = max(scores.items(), key=lambda x: x[1])[0]
            perks = [top_perk]
    
    # Basic tier gets only basic benefits
    if not perks or row['perk_tier'] == 'Basic':
        perks = ['Basic Benefits']
    
    return perks

def validate_segmentation_model(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive validation of the segmentation model.
    
    Args:
        df: DataFrame with segmentation results
        
    Returns:
        Dictionary containing validation metrics
    """
    validation_results = {
        'data_quality': {},
        'segment_stability': {},
        'perk_distribution': {},
        'score_correlations': {}
    }
    
    # Data quality checks
    validation_results['data_quality'].update({
        'missing_values': df.isnull().sum().to_dict(),
        'negative_values': {
            col: (df[col] < 0).sum() 
            for col in ['total_spend', 'total_bookings']
        },
        'outliers': {
            col: len(df[df[col] > df[col].quantile(0.99)])
            for col in ['total_spend', 'total_bookings', 'total_perk_score']
        }
    })
    
    # Segment stability
    validation_results['segment_stability'].update({
        'segment_sizes': df['segment'].value_counts().to_dict(),
        'segment_proportions': df['segment'].value_counts(normalize=True).to_dict()
    })
    
    # Perk distribution
    validation_results['perk_distribution'].update({
        'perk_counts': df['assigned_perks'].apply(len).value_counts().to_dict(),
        'perk_types': df['assigned_perks'].explode().value_counts().to_dict()
    })
    
    # Score correlations
    score_cols = ['monetary_score', 'frequency_score', 'discount_score', 'total_perk_score']
    validation_results['score_correlations'] = df[score_cols].corr().to_dict()
    
    return validation_results

def create_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a final customer score using an enhanced RFM approach with fuzzy logic.
    
    The score combines:
    1. Monetary Value (M): Total spend across flights and hotels
    2. Frequency (F): Total number of bookings
    3. Discount Behavior (D): Combines discount proportion and average discount
    4. Customer Engagement (E): Session metrics and interaction patterns
    
    Each component is scaled 0-1 and weighted based on business importance:
    - Monetary: 35% (indicates customer value)
    - Frequency: 25% (indicates engagement)
    - Discount: 25% (indicates price sensitivity)
    - Engagement: 15% (indicates platform interaction)
    
    Returns:
        DataFrame with added columns for component scores and final score
    """
    # Previous implementation...
    df = create_base_scores(df)  # Previous scoring logic
    
    # Add engagement scoring
    df['engagement_score'] = calculate_engagement_score(df)
    
    # Enhanced final score calculation
    df['final_score'] = (
        df['monetary_score'] * 0.35 +
        df['frequency_score'] * 0.25 +
        df['discount_score'] * 0.25 +
        df['engagement_score'] * 0.15
    )
    
    # Calculate fuzzy segment membership
    df = calculate_fuzzy_membership(df)
    
    return df

def calculate_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate customer engagement score based on session metrics.
    
    Args:
        df: DataFrame with customer metrics
        
    Returns:
        Series containing engagement scores
    """
    # Normalize session metrics
    scaler = MinMaxScaler()
    
    session_score = scaler.fit_transform(
        df[['total_sessions', 'mean_session_time']].fillna(0)
    ).mean(axis=1)
    
    return pd.Series(session_score, index=df.index)

def calculate_fuzzy_membership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fuzzy membership scores for each segment.
    
    Args:
        df: DataFrame with customer metrics
        
    Returns:
        DataFrame with added fuzzy membership columns
    """
    segments = ['Bronze', 'Silver', 'Gold', 'Platinum']
    
    for segment in segments:
        # Calculate membership score based on distance from segment center
        segment_center = df[df['segment'] == segment]['final_score'].mean()
        df[f'{segment.lower()}_membership'] = 1 / (1 + np.abs(df['final_score'] - segment_center))
        
    return df

def plot_perk_distributions(df: pd.DataFrame) -> None:
    """
    Create visualizations for perk distribution and scores.
    
    Args:
        df: DataFrame with perk information
    """
    os.makedirs('scripts/output/segments', exist_ok=True)
    
    # 1. Perk Score Distributions
    plt.figure(figsize=(15, 5))
    perk_scores = ['premium_support_score', 'priority_booking_score', 
                  'discount_access_score', 'loyalty_bonus_score']
    
    for i, score in enumerate(perk_scores, 1):
        plt.subplot(1, 4, i)
        sns.histplot(df[score], bins=30)
        plt.title(f'{score.replace("_", " ").title()}')
        plt.xlabel('Score')
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/perk_score_distributions.png')
    plt.close()
    
    # 2. Perk Distribution by Segment
    plt.figure(figsize=(12, 6))
    perk_counts = df.groupby('segment')['assigned_perks'].apply(
        lambda x: pd.Series([item for sublist in x for item in sublist]).value_counts()
    ).unstack().fillna(0)
    
    perk_counts.plot(kind='bar', stacked=True)
    plt.title('Perk Distribution by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.legend(title='Perks', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/perk_segment_distribution.png')
    plt.close()
    
    # 3. Fuzzy Membership Scores
    plt.figure(figsize=(12, 6))
    membership_cols = ['bronze_membership', 'silver_membership', 
                      'gold_membership', 'platinum_membership']
    
    for col in membership_cols:
        sns.kdeplot(data=df, x=col, label=col.replace('_membership', '').title())
    
    plt.title('Fuzzy Membership Score Distributions')
    plt.xlabel('Membership Score')
    plt.ylabel('Density')
    plt.legend(title='Segment')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/fuzzy_membership_distributions.png')
    plt.close()
    
    # 4. Perk Eligibility Heatmap
    plt.figure(figsize=(10, 6))
    perk_metrics = ['total_spend', 'total_bookings', 'frequency_score', 
                   'discount_score', 'monetary_score']
    
    correlation_matrix = df[perk_metrics + perk_scores].corr().loc[perk_metrics, perk_scores]
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', cbar_kws={'label': 'Correlation'})
    
    plt.title('Perk Eligibility Correlations')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/perk_eligibility_correlations.png')
    plt.close()

def plot_segment_profiles(df):
    """Create visualizations of segment profiles."""
    os.makedirs('scripts/output/segments', exist_ok=True)
    
    # 1. Component Scores by Segment
    plt.figure(figsize=(12, 6))
    scores = ['monetary_score', 'frequency_score', 'discount_score']
    
    for i, score in enumerate(scores, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='segment', y=score, data=df, order=['Bronze', 'Silver', 'Gold', 'Platinum'])
        plt.title(f'{score.replace("_", " ").title()}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/segment_scores.png')
    plt.close()
    
    # 2. Spend and Booking Distribution
    plt.figure(figsize=(12, 6))
    
    # Spend distribution
    plt.subplot(1, 2, 1)
    sns.boxplot(x='segment', y='total_spend', data=df, order=['Bronze', 'Silver', 'Gold', 'Platinum'])
    plt.title('Total Spend by Segment')
    plt.xticks(rotation=45)
    plt.yscale('log')  # Log scale for better visualization
    
    # Booking distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(x='segment', y='total_bookings', data=df, order=['Bronze', 'Silver', 'Gold', 'Platinum'])
    plt.title('Total Bookings by Segment')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/segment_distributions.png')
    plt.close()
    
    # 3. Discount Behavior
    plt.figure(figsize=(12, 6))
    metrics = ['discount_flight_proportion', 'average_flight_discount', 'ads_per_km']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='segment', y=metric, data=df, order=['Bronze', 'Silver', 'Gold', 'Platinum'])
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/discount_behavior.png')
    plt.close()
    
    # 4. Radar Chart of Segment Profiles
    plt.figure(figsize=(10, 10))
    
    # Calculate mean scores for each segment
    radar_metrics = [
        'monetary_score', 
        'frequency_score', 
        'discount_score',
        'discount_flight_proportion',
        'average_flight_discount'
    ]
    
    segment_means = df.groupby('segment')[radar_metrics].mean()
    
    # Number of variables
    num_vars = len(radar_metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Plot data
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors for each segment
    for idx, segment in enumerate(['Bronze', 'Silver', 'Gold', 'Platinum']):
        values = segment_means.loc[segment].values.flatten().tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=segment, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Segment Profiles Radar Chart', pad=20)
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/segment_radar.png')
    plt.close()
    
    # Add new perk-related visualizations
    plot_perk_distributions(df)

def plot_segment_profiles(df):
    """Create visualizations of segment profiles"""
    # Plot segment distributions
    plt.figure(figsize=(12, 6))
    sns.countplot(x='segment', data=df, order=['Bronze', 'Silver', 'Gold', 'Platinum'])
    plt.title('Customer Segment Distribution')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/segment_distribution.png')
    plt.close()
    write_to_md("\n![Segment Distribution](./segments/segment_distribution.png)\n")

    # Plot segment characteristics
    metrics = ['monetary_score', 'frequency_score', 'discount_score', 'final_score']
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='segment', y=metric, data=df, 
                    order=['Bronze', 'Silver', 'Gold', 'Platinum'])
        plt.title(f'{metric} by Segment')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/segment_characteristics.png')
    plt.close()
    write_to_md("\n![Segment Characteristics](./segments/segment_characteristics.png)\n")

def create_base_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate base RFM scores for customer segmentation.
    
    Args:
        df: DataFrame containing customer metrics
        
    Returns:
        DataFrame with added base score columns
    """
    # Filter out customers with zero bookings
    logger.info(f"Customers before zero booking filter: {len(df)}")
    df = df[
        (df['total_flights'] > 0) | 
        (df['total_hotels'] > 0)
    ].copy()
    logger.info(f"Customers after zero booking filter: {len(df)}")
    
    # Create scaler
    scaler = MinMaxScaler()
    
    # 1. Monetary Score
    # Scale total spend (combine flights and hotels)
    df['total_spend'] = df['total_flight_spend'] + df['total_hotel_spend']
    df['total_spend'] = df['total_spend'].clip(lower=0)
    
    # Use percentile ranking for spend to reduce impact of outliers
    df['spend_rank'] = df['total_spend'].rank(pct=True)
    df['monetary_score'] = df['spend_rank']
    
    # 2. Frequency Score
    # Scale total bookings
    df['total_bookings'] = df['total_flights'] + df['total_hotels']
    
    # Use percentile ranking for bookings
    df['booking_rank'] = df['total_bookings'].rank(pct=True)
    df['frequency_score'] = df['booking_rank']
    
    # 3. Discount Behavior Score
    # First normalize each metric individually using percentile ranking
    df['discount_prop_rank'] = df['discount_flight_proportion'].rank(pct=True)
    df['discount_avg_rank'] = df['average_flight_discount'].rank(pct=True)
    df['ads_per_km_rank'] = df['ads_per_km'].rank(pct=True)
    
    # Combine normalized discount metrics
    df['discount_score'] = (
        df['discount_prop_rank'] * 0.4 +    # How often they seek discounts
        df['discount_avg_rank'] * 0.3 +     # How big discounts they get
        df['ads_per_km_rank'] * 0.3         # How efficiently they find deals
    )
    
    # Create initial segments using percentile thresholds
    df['segment'] = pd.qcut(
        df['monetary_score'],
        q=[0, 0.5, 0.8, 0.9, 1.0],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    
    return df

def get_key_metrics() -> List[str]:
    """Return list of key metrics for analysis and visualization."""
    return [
        'total_flights',
        'total_flight_spend',
        'total_hotels',
        'total_hotel_spend',
        'discount_flight_proportion',
        'average_flight_discount',
        'ads_per_km',
        'bargain_hunter_index'
    ]

def save_perks_assignment(df: pd.DataFrame) -> None:
    """
    Create and save a dedicated CSV file for the perks/rewards program assignments.
    
    Args:
        df: DataFrame with customer segmentation and perk information
    """
    # Create a focused DataFrame with just the perks-related information
    perks_df = df[[
        'user_id',
        'segment',
        'perk_tier',
        'assigned_perks',
        'total_perk_score',
        'premium_support_score',
        'priority_booking_score',
        'discount_access_score',
        'loyalty_bonus_score'
    ]].copy()
    
    # Convert assigned_perks list to a comma-separated string
    perks_df['assigned_perks'] = perks_df['assigned_perks'].apply(lambda x: ', '.join(x))
    
    # Add segment descriptions
    segment_descriptions = {
        'Bronze': 'Basic tier with essential benefits',
        'Silver': 'Mid-tier with enhanced service access',
        'Gold': 'Premium tier with priority services',
        'Platinum': 'Elite tier with exclusive benefits'
    }
    perks_df['segment_description'] = perks_df['segment'].map(segment_descriptions)
    
    # Round scores to 3 decimal places for clarity
    score_columns = [col for col in perks_df.columns if col.endswith('_score')]
    perks_df[score_columns] = perks_df[score_columns].round(3)
    
    # Save to CSV
    output_path = 'scripts/output/segments/perks_assignment.csv'
    perks_df.to_csv(output_path, index=False)
    logger.info(f"Perks assignment data saved to {output_path}")
    
    # Log summary statistics
    logger.info("\nPerks Assignment Summary:")
    logger.info(f"Total customers: {len(perks_df)}")
    logger.info("\nDistribution by perk tier:")
    logger.info(perks_df['perk_tier'].value_counts(normalize=True).round(3) * 100)
    logger.info("\nMost common perk combinations:")
    logger.info(perks_df['assigned_perks'].value_counts().head())

def main():
    """Main execution function with enhanced segmentation and validation."""
    try:
        # Write header to markdown file
        write_to_md("# Customer Segmentation Results\n")
        write_to_md(f"Analysis generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        write_to_md("This report contains the results of customer segmentation and perk assignment analysis.\n")
        logger.info('Wrote header to markdown file')
    except Exception as e:
        logger.error(f'Error writing markdown header: {e}')
        raise

    # Create output directory
    os.makedirs('scripts/output/segments', exist_ok=True)
    
    # Load and preprocess data
    df = load_metrics()
    logger.info(f"Loaded {len(df)} customer records")
    
    # Calculate base scores and segments
    df = create_final_score(df)
    
    # Calculate and assign perks
    df = calculate_perk_scores(df)
    df = assign_perks(df)
    
    # Validate results
    validation_results = validate_segmentation_model(df)
    logger.info("Validation Results:")
    logger.info(validation_results)
    
    # Generate visualizations
    plot_metric_distributions(df, get_key_metrics())
    plot_correlations(df, get_key_metrics())
    plot_segment_profiles(df)
    
    # Save results
    df.to_csv('scripts/output/segments/customer_segments.csv', index=False)
    logger.info("Segmentation complete - results saved to customer_segments.csv")
    
    # Save perks assignment
    save_perks_assignment(df)

    try:
        # Write validation results
        write_to_md("\n## Validation Results\n")
        write_to_md(f"```\n{validation_results}\n```")
        
        # Write perk assignment summary
        write_to_md("\n## Perk Assignment Summary\n")
        write_to_md(f"Total customers: {len(df)}\n")
        write_to_md("\n### Distribution by perk tier:\n")
        write_to_md(df['perk_tier'].value_counts(normalize=True).round(3).mul(100).to_markdown())
        write_to_md("\n### Most common perk combinations:\n")
        write_to_md(df['assigned_perks'].value_counts().head().to_markdown())
        
        md_file.close()
        logger.info('Successfully wrote all results to markdown file')
    except Exception as e:
        logger.error(f'Error writing final results to markdown file: {e}')
        raise

if __name__ == "__main__":
    main()