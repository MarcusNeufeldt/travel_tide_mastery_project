# Mastery_Project/scripts/02_feature_engineering.py
import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from utils import safe_db_decorator, db_params, setup_logging, haversine_distance

# Setup logging
logger = setup_logging(__name__)

# Create output directory
os.makedirs('scripts/output/metrics', exist_ok=True)

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
    """Remove outliers based on standard deviation"""
    logger.info(f"Removing outliers from columns: {columns}, n_std={n_std}")
    for col in columns:
        if col in df.columns:  # Check if column exists
            mean = df[col].mean()
            std = df[col].std()
            if std > 0: #Avoid division by zero
                df = df[abs(df[col] - mean) <= (n_std * std)]
        else: #Log a message
            logger.warning(f"Column {col} not found in DataFrame. Skipping outlier removal for this column")
    return df

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