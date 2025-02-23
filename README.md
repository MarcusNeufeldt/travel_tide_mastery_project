# TravelTide Customer Segmentation Analysis

## Project Description
Analysis of TravelTide customer behavior to develop a targeted rewards program aimed at increasing conversion rates from 10% to 20%. The project identifies customer segments based on spending patterns, discount usage, and travel behavior to inform strategic perk assignments.

## Project Summary

### Key Points and Insights
- Identified four distinct customer segments with clear upgrade paths:
  - Platinum Elite (10%): High-value frequent travelers ($2.17M annual spend)
  - Gold Premium (10%): Balanced value-seekers ($1.84M annual spend)
  - Silver Growth (30%): Emerging frequent travelers ($4.18M annual spend)
  - Bronze Core (50%): Occasional travelers ($3.11M annual spend)
- Strong correlation between engagement and spend (0.85)
- Clear seasonal pricing patterns identified (Peak: Sept-Oct, $936-948 avg. fare)
- Projected 25% increase in customer engagement through personalized perks

### Links to Files
- [Executive Summary](executive_summary.md)
- [Project Report](project_report.pdf)

## Installation

1. Database Configuration
   ```python
   db_params = {
       'dbname': 'TravelTide',
       'user': '<your_username>',
       'password': '<your_password>',
       'host': '<host>',
       'port': '5432'
   }
   ```

2. Dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
```
TravelTide/
├── scripts/
│   ├── 01_eda_analysis.py          # Initial data exploration
│   ├── 02_feature_engineering.py    # Feature creation and metrics
│   ├── 03_customer_segmentation.py  # Customer segmentation
│   ├── 04_final_presentation.py     # Visualization generation
│   └── utils.py                     # Utility functions
├── output/
│   ├── eda/                         # EDA outputs
│   ├── metrics/                     # Feature engineering outputs
│   ├── segments/                    # Segmentation outputs
│   └── presentation/                # Final visualization outputs
├── Docs/
│   ├── project_report.pdf          # Detailed technical report
│   └── db_schema.md                # Database schema documentation
└── executive_summary.md            # Business-focused summary
```

## Usage
Execute the analysis pipeline in sequence:
```bash
python scripts/01_eda_analysis.py
python scripts/02_feature_engineering.py
python scripts/03_customer_segmentation.py
python scripts/04_final_presentation.py
```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- psycopg2

## Analysis Process
1. **Data Exploration**
   - Analyzed user demographics and behavior
   - Investigated travel patterns and preferences
   - Validated data quality and completeness

2. **Feature Engineering**
   - Created monetary value metrics
   - Developed engagement scores
   - Built discount sensitivity indicators

3. **Customer Segmentation**
   - Implemented rule-based segmentation
   - Validated segment stability
   - Assigned targeted perks

4. **Results & Recommendations**
   - Generated comprehensive visualizations
   - Created implementation timeline
   - Developed upgrade paths for each segment 