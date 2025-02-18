# Mastery_Project/scripts/02_feature_engineering.py
import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from utils import safe_db_decorator, db_params, setup_logging, haversine_distance
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logger = setup_logging(__name__)

# Create output directory
os.makedirs('scripts/output/metrics', exist_ok=True)
os.makedirs('scripts/output/segments', exist_ok=True)

class DatabaseConnection:
    def __init__(self):
        self.conn = None

    def __enter__(self):
        self.conn = psycopg2.connect(**db_params)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

@safe_db_decorator
def run_query(query, description="", connection=None):
    """Execute query and return results as a pandas DataFrame"""
    if description:
        logger.info(f"=== {description} ===")
    try:
        if connection:
            result = pd.read_sql_query(query, connection)
        else:
            with psycopg2.connect(**db_params) as conn:
                result = pd.read_sql_query(query, conn)
        if description:
            logger.info(f"Result Head:\n{result.head()}")
        return result
    except Exception as e:
        logger.error(f"Error running query: {e}")
        raise

@safe_db_decorator
def create_temp_cohort_table(connection):
    """Creates a temporary table for the cohort."""
    logger.info("Creating temporary cohort table...")
    query = """
        DROP TABLE IF EXISTS temp_cohort;
        CREATE TEMP TABLE temp_cohort AS
        SELECT DISTINCT u.user_id
        FROM users u
        JOIN sessions s ON u.user_id = s.user_id
        WHERE s.session_start >= '2023-01-04'
        GROUP BY u.user_id
        HAVING COUNT(DISTINCT s.session_id) > 7;
    """
    try:
        with connection.cursor() as cur:
            cur.execute(query)
            connection.commit()
        logger.info("Temporary cohort table created successfully.")
    except Exception as e:
        logger.error(f"Error creating temporary cohort table: {e}")
        raise

@safe_db_decorator
def calculate_basic_metrics(connection):
    """Calculate basic customer metrics, filtered by the cohort."""
    logger.info("Calculating basic metrics...")
    return run_query("""
        SELECT
            u.user_id,
            COUNT(DISTINCT f.trip_id) as total_flights,
            ABS(SUM(f.base_fare_usd)) as total_flight_spend,
            ABS(AVG(f.base_fare_usd)) as avg_flight_spend,
            COUNT(DISTINCT h.trip_id) as total_hotels,
            ABS(SUM(h.hotel_per_room_usd * h.nights)) as total_hotel_spend,
            ABS(AVG(h.hotel_per_room_usd * h.nights)) as avg_hotel_spend,
            COUNT(s.*) as total_sessions,
            AVG(EXTRACT(EPOCH FROM (s.session_end - s.session_start))) as mean_session_time,
            ABS(
                (COALESCE(AVG(f.base_fare_usd), 0) + COALESCE(AVG(h.hotel_per_room_usd * h.nights), 0)) *
                (NULLIF(COUNT(DISTINCT f.trip_id) + COUNT(DISTINCT h.trip_id), 0))
            ) as customer_value
        FROM users u
        JOIN temp_cohort c ON u.user_id = c.user_id  -- Join with the cohort
        LEFT JOIN sessions s ON u.user_id = s.user_id
        LEFT JOIN flights f ON s.trip_id = f.trip_id
        LEFT JOIN hotels h ON s.trip_id = h.trip_id
        GROUP BY u.user_id
    """, "Basic Metrics", connection)

@safe_db_decorator
def calculate_discount_metrics(connection):
    """Calculate discount-related metrics, filtered by the cohort."""
    logger.info("Calculating discount metrics...")
    
    return run_query("""
    WITH flight_metrics AS (
        SELECT
            u.user_id,
            COUNT(*) as total_flights,
            COUNT(*) FILTER (WHERE s.flight_discount) as discount_flights,
            AVG(CASE WHEN s.flight_discount THEN s.flight_discount_amount ELSE 0 END) as avg_discount,
            SUM(
                CASE 
                    WHEN s.flight_discount THEN 
                        s.flight_discount_amount * f.base_fare_usd 
                    ELSE 0 
                END
            ) as total_discount_value,
            SUM(
                haversine_distance(
                    u.home_airport_lat,
                    u.home_airport_lon,
                    f.destination_airport_lat,
                    f.destination_airport_lon
                )
            ) as total_distance
        FROM users u
        JOIN temp_cohort c ON u.user_id = c.user_id
        JOIN sessions s ON u.user_id = s.user_id
        JOIN flights f ON s.trip_id = f.trip_id
        WHERE s.flight_booked = TRUE
        GROUP BY u.user_id
    )
    SELECT
        user_id,
        COALESCE(discount_flights::float / NULLIF(total_flights, 0), 0) as discount_flight_proportion,
        COALESCE(avg_discount, 0) as average_flight_discount,
        COALESCE(total_discount_value / NULLIF(total_distance, 0), 0) as ads_per_km
    FROM flight_metrics
    """, "Discount Metrics", connection)

@safe_db_decorator
def remove_outliers(df, columns, n_std=3):
    """Remove outliers using different methods based on column type"""
    logger.info(f"Removing outliers from columns: {columns}")
    
    # Separate spending columns from others
    spending_columns = [col for col in columns if 'spend' in col.lower()]
    other_columns = [col for col in columns if 'spend' not in col.lower()]
    
    # For spending columns, use IQR method (more robust for skewed data)
    for col in spending_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Log outlier information
            outliers = df[df[col] > upper_bound]
            if len(outliers) > 0:
                logger.info(f"Removing {len(outliers)} outliers from {col}")
                logger.info(f"Upper bound for {col}: {upper_bound:.2f}")
                logger.info(f"Max value was: {df[col].max():.2f}")
            
            # Apply filter
            df = df[df[col] <= upper_bound]
    
    # For other columns, use standard deviation method
    for col in other_columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division by zero
                df = df[abs(df[col] - mean) <= (n_std * std)]
    
    return df

def plot_metric_distributions(metrics_df):
    """Create separate plots for different metric categories."""
    plt.style.use('seaborn')
    
    # 1. Booking Metrics Plot
    plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(data=metrics_df, x='total_flights', ax=ax1)
    ax1.set_title('Distribution of Total Flights')
    ax1.set_xlabel('Number of Flights')
    
    sns.histplot(data=metrics_df, x='total_hotels', ax=ax2)
    ax2.set_title('Distribution of Total Hotels')
    ax2.set_xlabel('Number of Hotel Bookings')
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/booking_metrics.png')
    plt.close()
    
    # 2. Spending Metrics Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.histplot(data=metrics_df, x='total_flight_spend', ax=ax1)
    ax1.set_title('Total Flight Spend')
    ax1.set_xlabel('USD')
    
    sns.histplot(data=metrics_df, x='total_hotel_spend', ax=ax2)
    ax2.set_title('Total Hotel Spend')
    ax2.set_xlabel('USD')
    
    sns.histplot(data=metrics_df, x='avg_flight_spend', ax=ax3)
    ax3.set_title('Average Flight Spend')
    ax3.set_xlabel('USD per Flight')
    
    sns.histplot(data=metrics_df, x='avg_hotel_spend', ax=ax4)
    ax4.set_title('Average Hotel Spend')
    ax4.set_xlabel('USD per Stay')
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/spending_metrics.png')
    plt.close()
    
    # 3. Discount Behavior Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.histplot(data=metrics_df, x='discount_flight_proportion', ax=ax1)
    ax1.set_title('Proportion of Discounted Flights')
    ax1.set_xlabel('Discount Proportion')
    
    sns.histplot(data=metrics_df, x='average_flight_discount', ax=ax2)
    ax2.set_title('Average Discount Amount')
    ax2.set_xlabel('Discount Percentage')
    
    sns.histplot(data=metrics_df, x='ads_per_km', ax=ax3)
    ax3.set_title('Average Discount Savings per KM')
    ax3.set_xlabel('USD/KM')
    
    sns.histplot(data=metrics_df, x='bargain_hunter_index', ax=ax4)
    ax4.set_title('Bargain Hunter Index')
    ax4.set_xlabel('Index Value')
    
    plt.tight_layout()
    plt.savefig('scripts/output/segments/discount_behavior.png')
    plt.close()
    
    # 4. Correlation Matrix
    plt.figure(figsize=(12, 10))
    correlation_metrics = [
        'total_flights', 'total_hotels', 
        'total_flight_spend', 'total_hotel_spend',
        'discount_flight_proportion', 'average_flight_discount',
        'bargain_hunter_index'
    ]
    sns.heatmap(
        metrics_df[correlation_metrics].corr(),
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f'
    )
    plt.title('Metric Correlations')
    plt.tight_layout()
    plt.savefig('scripts/output/segments/metric_correlations.png')
    plt.close()

def main():
    with DatabaseConnection() as conn:
        # Create the temporary cohort table
        create_temp_cohort_table(conn)

        # Calculate metrics
        basic_metrics = calculate_basic_metrics(conn)
        discount_metrics = calculate_discount_metrics(conn)

        # Merge metrics
        metrics = basic_metrics.merge(discount_metrics, on='user_id', how='left')

        # Fill NaN values with 0
        metrics = metrics.fillna(0)

        # Double-check to ensure all spending values are positive
        spend_columns = ['total_flight_spend', 'avg_flight_spend', 'total_hotel_spend', 'avg_hotel_spend', 'customer_value']
        for col in spend_columns:
            metrics[col] = metrics[col].abs()

        # Remove outliers
        metrics = remove_outliers(metrics, [
            'total_flights', 'total_flight_spend', 'avg_flight_spend',
            'total_hotels', 'total_hotel_spend', 'avg_hotel_spend',
            'total_sessions', 'mean_session_time', 'customer_value',
            'discount_flight_proportion', 'average_flight_discount',
            'ads_per_km'
        ])

        # Calculate bargain hunter index
        metrics['bargain_hunter_index'] = (
            metrics['discount_flight_proportion']
            * metrics['average_flight_discount']
            * metrics['ads_per_km']
        )

        # Calculate total spend
        metrics['total_spend'] = metrics['total_flight_spend'] + metrics['total_hotel_spend']
        metrics['total_spend'] = metrics['total_spend'].clip(lower=0)

        # Create metric distribution plots
        plot_metric_distributions(metrics)

        # Save results
        metrics.to_csv('scripts/output/metrics/user_metrics.csv', index=False)
        logger.info("Metrics saved to user_metrics.csv")
        logger.info(f"Sample of final metrics:\n{metrics.head()}")

        # Print correlations
        logger.info("Correlations with bargain_hunter_index:")
        correlations = metrics.corr()['bargain_hunter_index'].sort_values(ascending=False)
        logger.info(correlations)

if __name__ == "__main__":
    main()