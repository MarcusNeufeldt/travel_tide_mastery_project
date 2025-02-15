import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create output directory if it doesn't exist
os.makedirs('output/eda', exist_ok=True)

# Database connection parameters
db_params = {
    'dbname': 'TravelTide',
    'user': 'Test',
    'password': 'bQNxVzJL4g6u',
    'host': 'ep-noisy-flower-846766.us-east-2.aws.neon.tech',
    'port': '5432'
}

def run_query(query, description=""):
    """Execute query and return results as a pandas DataFrame"""
    if description:
        print(f"\n=== {description} ===")
    try:
        conn = psycopg2.connect(**db_params)
        result = pd.read_sql_query(query, conn)
        print(result)
        return result
    finally:
        conn.close()

def main():
    # Set up plotting style
    plt.style.use('seaborn')
    
    # 1. Table Sizes
    run_query("""
        SELECT 'users' as table_name, COUNT(*) as row_count FROM users
        UNION ALL
        SELECT 'sessions', COUNT(*) FROM sessions
        UNION ALL
        SELECT 'flights', COUNT(*) FROM flights
        UNION ALL
        SELECT 'hotels', COUNT(*) FROM hotels;
    """, "Table Sizes")

    # 2. Data Quality Check - Null Values
    run_query("""
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN birthdate IS NULL THEN 1 ELSE 0 END) as null_birthdate,
            SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) as null_gender,
            SUM(CASE WHEN married IS NULL THEN 1 ELSE 0 END) as null_married,
            SUM(CASE WHEN has_children IS NULL THEN 1 ELSE 0 END) as null_has_children
        FROM users;
    """, "Null Values Check in Users Table")

    # 3. Hotel Names Convention
    hotel_names = run_query("""
        SELECT DISTINCT hotel_name 
        FROM hotels 
        LIMIT 5;
    """, "Sample Hotel Names")

    # 4. User Demographics
    demographics = run_query("""
        SELECT 
            gender,
            married,
            has_children,
            COUNT(*) as user_count,
            ROUND(COUNT(*)::decimal / SUM(COUNT(*)) OVER () * 100, 2) as percentage
        FROM users
        GROUP BY gender, married, has_children
        ORDER BY user_count DESC;
    """, "User Demographics")

    # 5. Birth Year Distribution and 2006 Analysis
    birth_years = run_query("""
        SELECT 
            EXTRACT(YEAR FROM birthdate) as birth_year,
            COUNT(*) as count
        FROM users
        GROUP BY birth_year
        ORDER BY birth_year;
    """, "Birth Year Distribution")

    # Plot birth year distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=birth_years, x='birth_year', y='count')
    plt.title('Distribution of Birth Years')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/eda/birth_year_distribution.png')
    plt.close()

    # 6. Customer Age Analysis
    run_query("""
        SELECT 
            AVG(DATE_PART('day', age(CURRENT_DATE, sign_up_date))/30.0) as avg_months_since_signup,
            MIN(sign_up_date) as earliest_signup,
            MAX(sign_up_date) as latest_signup
        FROM users;
    """, "Customer Age Analysis")

    # 7. Top 10 Hotels Analysis
    popular_hotels = run_query("""
        SELECT 
            hotel_name,
            COUNT(*) as bookings,
            ROUND(AVG(nights), 2) as avg_nights,
            ROUND(AVG(hotel_per_room_usd), 2) as avg_price_per_night
        FROM hotels
        GROUP BY hotel_name
        ORDER BY bookings DESC
        LIMIT 10;
    """, "Top 10 Popular Hotels")

    # 8. Most Expensive Hotels
    run_query("""
        SELECT 
            hotel_name,
            ROUND(AVG(hotel_per_room_usd), 2) as avg_price_per_night,
            COUNT(*) as bookings,
            ROUND(AVG(nights), 2) as avg_nights
        FROM hotels
        GROUP BY hotel_name
        HAVING COUNT(*) > 10  -- Filter out hotels with few bookings
        ORDER BY avg_price_per_night DESC
        LIMIT 10;
    """, "Top 10 Most Expensive Hotels")

    # 9. Airlines Analysis
    run_query("""
        WITH last_6_months AS (
            SELECT MAX(departure_time) - INTERVAL '6 months' as cutoff_date
            FROM flights
        )
        SELECT 
            trip_airline,
            COUNT(*) as flight_count,
            ROUND(AVG(base_fare_usd), 2) as avg_fare,
            ROUND(AVG(seats), 2) as avg_seats
        FROM flights
        WHERE departure_time >= (SELECT cutoff_date FROM last_6_months)
        GROUP BY trip_airline
        ORDER BY flight_count DESC
        LIMIT 10;
    """, "Top Airlines (Last 6 months)")

    # 10. Seasonal Price Variation
    seasonal_prices = run_query("""
        SELECT 
            EXTRACT(MONTH FROM departure_time) as month,
            ROUND(AVG(base_fare_usd), 2) as avg_fare
        FROM flights
        GROUP BY month
        ORDER BY month;
    """, "Seasonal Price Variation")

    # Plot seasonal price variation
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=seasonal_prices, x='month', y='avg_fare')
    plt.title('Average Flight Prices by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Fare (USD)')
    plt.tight_layout()
    plt.savefig('output/eda/seasonal_prices.png')
    plt.close()

if __name__ == "__main__":
    main() 