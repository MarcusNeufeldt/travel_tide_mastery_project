import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np

# Create output directory if it doesn't exist
os.makedirs('scripts/output/eda', exist_ok=True)

# Create or open markdown file for writing
md_file = open('scripts/output/eda/analysis_results.md', 'w', encoding='utf-8')

# Database connection parameters
db_params = {
    'dbname': 'TravelTide',
    'user': 'Test',
    'password': 'bQNxVzJL4g6u',
    'host': 'ep-noisy-flower-846766.us-east-2.aws.neon.tech',
    'port': '5432'
}

def write_to_md(text):
    """Write text to markdown file"""
    md_file.write(text + '\n')

def format_number(x):
    """Format numbers to avoid scientific notation"""
    if isinstance(x, (int, np.int64)):
        return f"{x:,}"
    elif isinstance(x, (float, np.float64)):
        return f"{x:,.2f}"
    return x

def run_query(query, description="", summary=""):
    """Execute query and return results as a pandas DataFrame"""
    if description:
        write_to_md(f"\n## {description}\n")
    if summary:
        write_to_md(f"{summary}\n")
    try:
        conn = psycopg2.connect(**db_params)
        result = pd.read_sql_query(query, conn)
        
        # Create a copy for markdown formatting
        markdown_df = result.copy()
        for col in markdown_df.select_dtypes(include=['int64', 'float64']).columns:
            markdown_df[col] = markdown_df[col].apply(format_number)
        
        # Write formatted table to markdown
        write_to_md(markdown_df.to_markdown(index=False))
        return result
    finally:
        conn.close()

def main():
    # Write header
    write_to_md("# TravelTide Data Analysis Results\n")
    write_to_md(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    write_to_md("This report provides a comprehensive analysis of the TravelTide database, including user demographics, booking patterns, and pricing trends.\n")
    
    # Set up plotting style
    plt.style.use('seaborn')
    
    # 1. Table Sizes
    # This query counts the number of records in each main table (users, sessions, flights, hotels)
    # to give an overview of the database size and distribution of data across tables
    run_query("""
        SELECT 'users' as table_name, COUNT(*) as row_count FROM users
        UNION ALL
        SELECT 'sessions', COUNT(*) FROM sessions
        UNION ALL
        SELECT 'flights', COUNT(*) FROM flights
        UNION ALL
        SELECT 'hotels', COUNT(*) FROM hotels;
    """, "Table Sizes", "Overview of the number of records in each table of the database.")

    # 2. Data Quality Check - Null Values
    # This query checks for missing values in key user demographic fields
    # to ensure data quality and completeness for analysis
    run_query("""
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN birthdate IS NULL THEN 1 ELSE 0 END) as null_birthdate,
            SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) as null_gender,
            SUM(CASE WHEN married IS NULL THEN 1 ELSE 0 END) as null_married,
            SUM(CASE WHEN has_children IS NULL THEN 1 ELSE 0 END) as null_has_children
        FROM users;
    """, "Data Quality Check", "Analysis of missing values in the users table. All fields are complete with no null values.")

    # 3. Hotel Names Convention
    # This query retrieves sample hotel names to understand the naming convention
    # which follows a 'Brand - location' format
    run_query("""
        SELECT DISTINCT hotel_name 
        FROM hotels 
        LIMIT 5;
    """, "Hotel Names Convention", "Sample of hotel names showing the naming convention: 'Brand - location' format.")

    # 4. User Demographics
    # This query analyzes user distribution by gender, marital status, and children
    # to understand the customer base composition
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
    """, "User Demographics", "Distribution of users by gender, marital status, and whether they have children. The majority of users are single without children.")

    # 5. Birth Year Distribution
    # This query extracts birth years from user data to analyze age distribution
    # and identify any unusual patterns that might need investigation
    birth_years = run_query("""
        SELECT 
            EXTRACT(YEAR FROM birthdate)::integer as birth_year,
            COUNT(*) as count
        FROM users
        GROUP BY birth_year
        ORDER BY birth_year;
    """, "Birth Year Distribution", "Distribution of user birth years. Note the unusual spike in 2006 which may require further investigation.")

    # Plot birth year distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=birth_years, x='birth_year', y='count')
    plt.title('Distribution of Birth Years')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('scripts/output/eda/birth_year_distribution.png')
    plt.close()
    write_to_md("\n![Birth Year Distribution](./eda/birth_year_distribution.png)\n")

    # 6. Customer Age Analysis
    # This query calculates how long users have been customers by analyzing sign-up dates
    # and provides insights into customer retention and account age
    run_query("""
        SELECT 
            ROUND(CAST(AVG(DATE_PART('day', age(CURRENT_DATE, sign_up_date))/30.0) as numeric), 2) as avg_months_since_signup,
            MIN(sign_up_date) as earliest_signup,
            MAX(sign_up_date) as latest_signup
        FROM users;
    """, "Customer Age Analysis", "Analysis of user sign-up dates and average account age.")

    # 7. Top 10 Hotels Analysis
    # This query identifies the most frequently booked hotels and analyzes their
    # booking patterns, average stay duration, and pricing
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
    """, "Top 10 Popular Hotels", "Most frequently booked hotels, all located in New York, showing similar booking patterns and pricing.")

    # 8. Most Expensive Hotels
    # This query identifies hotels with the highest average room rates,
    # filtering out hotels with fewer than 10 bookings to ensure reliability
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
    """, "Top 10 Most Expensive Hotels", "Hotels with the highest average room rates (minimum 10 bookings). European hotels dominate this category.")

    # 9. Airlines Analysis
    # This query analyzes airline performance over the last 6 months,
    # including number of flights, average fares, and seat capacity
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
    """, "Top Airlines (Last 6 months)", "Most active airlines in the past 6 months by number of flights. Traditional carriers dominate the top positions.")

    # 10. Seasonal Price Variation
    # This query analyzes flight price trends by month
    # to identify seasonal patterns and pricing strategies
    seasonal_prices = run_query("""
        SELECT 
            EXTRACT(MONTH FROM departure_time)::integer as month,
            ROUND(AVG(base_fare_usd), 2) as avg_fare
        FROM flights
        GROUP BY month
        ORDER BY month;
    """, "Seasonal Price Variation", "Average flight prices by month showing clear seasonal patterns with peaks in late summer and fall.")

    # Plot seasonal price variation
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=seasonal_prices, x='month', y='avg_fare')
    plt.title('Average Flight Prices by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Fare (USD)')
    plt.tight_layout()
    plt.savefig('scripts/output/eda/seasonal_prices.png')
    plt.close()
    write_to_md("\n![Seasonal Price Variation](./eda/seasonal_prices.png)\n")

    # Close the markdown file
    md_file.close()

if __name__ == "__main__":
    main() 