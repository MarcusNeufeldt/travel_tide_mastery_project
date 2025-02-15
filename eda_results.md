# TravelTide Data Analysis Results

## 1. Database Overview
The database contains four main tables with the following sizes:
- Users: 1,020,926 records
- Hotels: 1,918,617 records
- Flights: 1,901,038 records
- Sessions: 5,408,063 records

## 2. Data Quality
Analysis of the users table shows excellent data quality:
- No null values found in any key columns (birthdate, gender, married, has_children)
- All demographic data is complete and consistent

## 3. User Demographics

### Gender and Family Status Distribution
1. Top segments:
   - Single males without children: 25.18%
   - Single females without children: 19.60%
   - Married males without children: 12.51%
   - Married females without children: 10.82%

2. Other insights:
   - Total users identifying as 'Other' gender: 0.81%
   - About 31% of users have children
   - Approximately 39% of users are married

### Age Distribution
- Birth years range from 1931 to 2006
- Notable spike in 2006 (43,360 users) - requires investigation
- Most users were born between 1970-2000

### Customer Age on Platform
- Earliest signup: April 1, 2021
- Latest signup: July 20, 2023
- Average account age: ~0.5 months

## 4. Hotel Bookings Analysis

### Most Popular Hotels (Top 5)
All in New York:
1. Extended Stay: 14,075 bookings (avg. $178.54/night)
2. Radisson: 14,073 bookings (avg. $178.26/night)
3. Starwood: 14,029 bookings (avg. $176.56/night)
4. Conrad: 14,022 bookings (avg. $176.31/night)
5. Rosewood: 14,017 bookings (avg. $178.30/night)

### Most Expensive Hotels (Top 5)
1. Banyan Tree Copenhagen: $293.07/night
2. Rosewood Budapest: $289.86/night
3. Choice Hotels Rio de Janeiro: $278.55/night
4. Wyndham Batam: $269.08/night
5. Marriott Naples: $265.07/night

## 5. Flight Analysis

### Top Airlines (Last 6 months)
1. American Airlines: 2,591 flights (avg. fare $1,283.01)
2. Delta Air Lines: 2,437 flights (avg. fare $1,211.72)
3. United Airlines: 2,288 flights (avg. fare $1,215.95)
4. Ryanair: 1,732 flights (avg. fare $1,340.29)
5. Southwest Airlines: 989 flights (avg. fare $656.19)

### Seasonal Price Variation
Highest average fares:
- October: $948.67
- September: $936.82
- August: $796.57

Lowest average fares:
- June: $578.86
- July: $584.26
- February: $590.32

### Additional Flight Insights
- Average seats per booking across all airlines: ~1.90 seats
  * Very consistent across airlines (range: 1.89-1.95)
  * Air India has highest average at 1.95 seats per booking
  * AirTran Airways lowest at 1.89 seats per booking

### Hotel Name Convention
Hotels are named using format "[Brand] - [City]", examples:
- Accor - abu dhabi
- Accor - accra
- Accor - agra
This naming convention allows for easy analysis of hotel locations and brand distribution.

## Key Insights
1. **Market Concentration**: New York dominates the hotel bookings
2. **Pricing Patterns**: Clear seasonal variation in flight prices, with peak in September-October
3. **Demographics**: Platform attracts more single users than married users
4. **Airline Market**: Traditional carriers dominate, with American Airlines leading
5. **Data Quality**: Excellent data completeness with no null values in key fields

## Recommendations for Further Analysis
1. Investigate the unusual spike in 2006 birth year
2. Analyze the correlation between discounts and booking rates
3. Examine the relationship between hotel prices and booking duration
4. Study the geographical distribution of users vs. destinations
5. Analyze the impact of family status on travel patterns 