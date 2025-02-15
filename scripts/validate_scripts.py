import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_eda_outputs() -> Dict:
    """
    Validate the outputs and data quality from EDA analysis.
    
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'file_checks': {},
        'data_quality': {},
        'visualization_checks': {}
    }
    
    logger.info("Validating EDA outputs...")
    
    # 1. Check required output directories and files
    required_files = [
        'scripts/output/eda/birth_year_distribution.png',
        'scripts/output/eda/seasonal_prices.png'
    ]
    
    for file_path in required_files:
        validation_results['file_checks'][file_path] = os.path.exists(file_path)
        if not validation_results['file_checks'][file_path]:
            logger.warning(f"Missing required file: {file_path}")
    
    # 2. Validate visualization properties
    for img_path in required_files:
        if validation_results['file_checks'][img_path]:
            try:
                img = plt.imread(img_path)
                validation_results['visualization_checks'][img_path] = {
                    'shape': img.shape,
                    'valid_image': True
                }
            except Exception as e:
                logger.error(f"Error validating image {img_path}: {e}")
                validation_results['visualization_checks'][img_path] = {
                    'valid_image': False,
                    'error': str(e)
                }
    
    return validation_results

def investigate_negative_spending(metrics_df: pd.DataFrame) -> Dict:
    """
    Perform detailed investigation of customers with negative spending values.
    
    Args:
        metrics_df: DataFrame containing customer metrics
        
    Returns:
        Dictionary containing investigation results
    """
    investigation_results = {
        'negative_spending_analysis': {},
        'correlations': {},
        'patterns': {},
        'recommendations': []
    }
    
    # 1. Identify all customers with any negative spending
    negative_spend_mask = (
        (metrics_df['total_hotel_spend'] < 0) |
        (metrics_df['total_flight_spend'] < 0) |
        (metrics_df['customer_value'] < 0)
    )
    problem_customers = metrics_df[negative_spend_mask].copy()
    
    # 2. Detailed analysis of each spending metric
    spend_metrics = ['total_hotel_spend', 'total_flight_spend', 'customer_value']
    for metric in spend_metrics:
        neg_customers = metrics_df[metrics_df[metric] < 0]
        if len(neg_customers) > 0:
            investigation_results['negative_spending_analysis'][metric] = {
                'count': len(neg_customers),
                'min_value': neg_customers[metric].min(),
                'max_value': neg_customers[metric].max(),
                'mean_value': neg_customers[metric].mean(),
                'customer_details': []
            }
            
            # Detailed customer analysis
            for _, customer in neg_customers.iterrows():
                customer_detail = {
                    'user_id': customer.get('user_id', 'N/A'),
                    'total_flights': customer.get('total_flights', 0),
                    'total_hotels': customer.get('total_hotels', 0),
                    'total_flight_spend': customer.get('total_flight_spend', 0),
                    'total_hotel_spend': customer.get('total_hotel_spend', 0),
                    'customer_value': customer.get('customer_value', 0),
                    'bargain_hunter_index': customer.get('bargain_hunter_index', 0)
                }
                investigation_results['negative_spending_analysis'][metric]['customer_details'].append(customer_detail)
    
    # 3. Look for patterns
    if len(problem_customers) > 0:
        # Check booking patterns
        investigation_results['patterns']['booking_patterns'] = {
            'avg_flights': problem_customers['total_flights'].mean(),
            'avg_hotels': problem_customers['total_hotels'].mean(),
            'typical_spend_range': {
                'flights': {
                    'min': problem_customers['total_flight_spend'].min(),
                    'max': problem_customers['total_flight_spend'].max()
                },
                'hotels': {
                    'min': problem_customers['total_hotel_spend'].min(),
                    'max': problem_customers['total_hotel_spend'].max()
                }
            }
        }
        
        # Check for correlations with other metrics
        if 'bargain_hunter_index' in problem_customers.columns:
            investigation_results['correlations']['bargain_hunter'] = {
                'with_hotel_spend': problem_customers['total_hotel_spend'].corr(problem_customers['bargain_hunter_index']),
                'with_flight_spend': problem_customers['total_flight_spend'].corr(problem_customers['bargain_hunter_index'])
            }
    
    # 4. Generate recommendations
    if investigation_results['negative_spending_analysis']:
        # General recommendations
        investigation_results['recommendations'].append({
            'type': 'data_validation',
            'description': 'Implement validation checks in feature engineering pipeline',
            'details': 'Add checks to prevent negative spending values during calculation'
        })
        
        # Specific recommendations based on patterns
        for metric, analysis in investigation_results['negative_spending_analysis'].items():
            if analysis['count'] > 0:
                if 'hotel' in metric.lower():
                    investigation_results['recommendations'].append({
                        'type': 'hotel_spend_calculation',
                        'description': 'Review hotel spend calculation logic',
                        'details': f'Focus on {analysis["count"]} customers with negative {metric}, particularly customer ID {analysis["customer_details"][0]["user_id"]}'
                    })
                elif 'flight' in metric.lower():
                    investigation_results['recommendations'].append({
                        'type': 'flight_spend_calculation',
                        'description': 'Review flight spend calculation logic',
                        'details': f'Focus on {analysis["count"]} customers with negative {metric}'
                    })
                elif 'value' in metric.lower():
                    investigation_results['recommendations'].append({
                        'type': 'customer_value_calculation',
                        'description': 'Review customer value calculation logic',
                        'details': f'Focus on {analysis["count"]} customers with negative {metric}, particularly customer ID {analysis["customer_details"][0]["user_id"]}'
                    })
    
    return investigation_results

def validate_feature_engineering_outputs() -> Dict:
    """
    Validate the feature engineering outputs and transformations.
    
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'file_checks': {},
        'data_quality': {},
        'metric_checks': {},
        'critical_issues': [],
        'warnings': [],
        'negative_spending_investigation': {}
    }
    
    logger.info("Validating feature engineering outputs...")
    
    # 1. Check if metrics file exists
    metrics_path = 'scripts/output/metrics/user_metrics.csv'
    validation_results['file_checks']['metrics_file'] = os.path.exists(metrics_path)
    
    if validation_results['file_checks']['metrics_file']:
        try:
            # Load metrics data
            metrics_df = pd.read_csv(metrics_path)
            
            # 2. Data quality checks
            validation_results['data_quality'].update({
                'row_count': len(metrics_df),
                'null_counts': metrics_df.isnull().sum().to_dict(),
            })
            
            # Check for negative values with detailed reporting
            negative_values = {}
            negative_spending = {}
            for col in metrics_df.select_dtypes(include=[np.number]).columns:
                neg_count = (metrics_df[col] < 0).sum()
                if neg_count > 0:
                    negative_values[col] = neg_count
                    if 'spend' in col.lower() or 'value' in col.lower():
                        negative_spending[col] = {
                            'count': neg_count,
                            'min_value': metrics_df[col].min(),
                            'affected_rows': metrics_df[metrics_df[col] < 0].index.tolist()
                        }
            
            validation_results['data_quality']['negative_values'] = negative_values
            
            # Flag critical spending issues
            if negative_spending:
                validation_results['critical_issues'].append({
                    'type': 'negative_spending',
                    'details': negative_spending,
                    'recommendation': 'Investigate and correct negative spending values in feature engineering pipeline'
                })
            
            # 3. Enhanced metric range checks
            required_metrics = [
                'total_flights', 'total_flight_spend', 'total_hotels',
                'total_hotel_spend', 'bargain_hunter_index'
            ]
            
            # Detailed metric analysis
            for metric in required_metrics:
                if metric in metrics_df.columns:
                    metric_stats = {
                        'min': metrics_df[metric].min(),
                        'max': metrics_df[metric].max(),
                        'mean': metrics_df[metric].mean(),
                        'median': metrics_df[metric].median(),
                        'std': metrics_df[metric].std(),
                        'skew': metrics_df[metric].skew(),
                        'outliers': len(metrics_df[
                            (metrics_df[metric] > metrics_df[metric].mean() + 3 * metrics_df[metric].std()) |
                            (metrics_df[metric] < metrics_df[metric].mean() - 3 * metrics_df[metric].std())
                        ])
                    }
                    
                    # Add distribution checks
                    if 'spend' in metric or 'value' in metric:
                        metric_stats['zero_values'] = (metrics_df[metric] == 0).sum()
                        metric_stats['negative_values'] = (metrics_df[metric] < 0).sum()
                        
                        # Flag potential issues
                        if metric_stats['negative_values'] > 0:
                            validation_results['warnings'].append({
                                'metric': metric,
                                'issue': 'negative_values',
                                'count': metric_stats['negative_values'],
                                'recommendation': f'Review {metric} calculation logic'
                            })
                        
                        if metric_stats['outliers'] > len(metrics_df) * 0.01:  # More than 1% outliers
                            validation_results['warnings'].append({
                                'metric': metric,
                                'issue': 'high_outliers',
                                'count': metric_stats['outliers'],
                                'recommendation': f'Consider outlier treatment for {metric}'
                            })
                    
                    validation_results['metric_checks'][metric] = metric_stats
            
            # 4. Check for expected columns
            validation_results['data_quality']['missing_columns'] = [
                metric for metric in required_metrics
                if metric not in metrics_df.columns
            ]
            
            # 5. Check for data consistency
            validation_results['data_quality']['consistency_checks'] = {
                'total_spend_matches': (
                    abs(metrics_df['total_flight_spend'] + metrics_df['total_hotel_spend'] 
                        - metrics_df['total_spend']).max() < 0.01
                    if all(col in metrics_df.columns for col in 
                          ['total_flight_spend', 'total_hotel_spend', 'total_spend'])
                    else False
                ),
                'booking_counts_match': (
                    (metrics_df['total_flights'] >= 0).all() and
                    (metrics_df['total_hotels'] >= 0).all()
                    if all(col in metrics_df.columns for col in ['total_flights', 'total_hotels'])
                    else False
                )
            }
            
            # 6. Add correlation analysis for key metrics
            if len(required_metrics) > 1:
                correlation_matrix = metrics_df[required_metrics].corr()
                validation_results['metric_checks']['correlations'] = correlation_matrix.to_dict()
            
            # Add detailed investigation for negative spending
            if negative_spending:
                investigation_results = investigate_negative_spending(metrics_df)
                validation_results['negative_spending_investigation'] = investigation_results
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating metrics file: {e}")
            validation_results['error'] = str(e)
            validation_results['critical_issues'].append({
                'type': 'validation_error',
                'details': str(e),
                'recommendation': 'Review feature engineering pipeline for errors'
            })
    
    return validation_results

def validate_segmentation_outputs() -> Dict:
    """
    Validate the customer segmentation outputs and assignments.
    
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'file_checks': {},
        'segment_quality': {},
        'perk_distribution': {},
        'visualization_checks': {}
    }
    
    logger.info("Validating segmentation outputs...")
    
    # 1. Check required files
    required_files = [
        'scripts/output/segments/customer_segments.csv',
        'scripts/output/segments/perks_assignment.csv',
        'scripts/output/segments/segment_radar.png',
        'scripts/output/segments/perk_segment_distribution.png'
    ]
    
    for file_path in required_files:
        validation_results['file_checks'][file_path] = os.path.exists(file_path)
    
    # 2. Validate segmentation results
    if validation_results['file_checks']['scripts/output/segments/customer_segments.csv']:
        try:
            segments_df = pd.read_csv('scripts/output/segments/customer_segments.csv')
            
            # Check segment distribution
            segment_dist = segments_df['segment'].value_counts(normalize=True)
            validation_results['segment_quality'].update({
                'segment_distribution': segment_dist.to_dict(),
                'total_customers': len(segments_df),
                'segment_counts': segments_df['segment'].value_counts().to_dict()
            })
            
            # Validate segment proportions
            expected_proportions = {
                'Bronze': 0.5,
                'Silver': 0.3,
                'Gold': 0.1,
                'Platinum': 0.1
            }
            
            validation_results['segment_quality']['proportion_errors'] = {
                segment: abs(segment_dist.get(segment, 0) - expected)
                for segment, expected in expected_proportions.items()
            }
            
        except Exception as e:
            logger.error(f"Error validating segmentation file: {e}")
            validation_results['segment_quality']['error'] = str(e)
    
    # 3. Validate perk assignments
    if validation_results['file_checks']['scripts/output/segments/perks_assignment.csv']:
        try:
            perks_df = pd.read_csv('scripts/output/segments/perks_assignment.csv')
            
            # Check perk distribution
            validation_results['perk_distribution'].update({
                'perk_tier_distribution': perks_df['perk_tier'].value_counts().to_dict(),
                'total_assignments': len(perks_df),
                'unique_perks': perks_df['assigned_perks'].nunique()
            })
            
            # Validate business rules
            validation_results['perk_distribution']['business_rules'] = {
                'all_customers_have_perks': perks_df['assigned_perks'].notna().all(),
                'platinum_multiple_perks': (
                    perks_df[perks_df['perk_tier'] == 'Platinum']['assigned_perks']
                    .str.count(',').mean() > 1
                ),
                'basic_benefits_only_basic': (
                    perks_df[perks_df['assigned_perks'] == 'Basic Benefits']['perk_tier'] == 'Basic'
                ).all()
            }
            
        except Exception as e:
            logger.error(f"Error validating perks file: {e}")
            validation_results['perk_distribution']['error'] = str(e)
    
    return validation_results

def validate_all_scripts() -> Dict:
    """
    Run all validation checks and compile results.
    
    Returns:
        Dictionary containing all validation results
    """
    all_results = {
        'eda_validation': validate_eda_outputs(),
        'feature_engineering_validation': validate_feature_engineering_outputs(),
        'segmentation_validation': validate_segmentation_outputs()
    }
    
    # Check for critical errors
    critical_errors = []
    
    # 1. Check EDA outputs
    if not all(all_results['eda_validation']['file_checks'].values()):
        critical_errors.append("Missing required EDA visualization files")
    
    # 2. Check feature engineering outputs
    if not all_results['feature_engineering_validation']['file_checks']['metrics_file']:
        critical_errors.append("Missing user metrics file")
    
    # 3. Check segmentation outputs
    if not all(all_results['segmentation_validation']['file_checks'].values()):
        critical_errors.append("Missing required segmentation output files")
    
    # Add validation summary
    all_results['validation_summary'] = {
        'critical_errors': critical_errors,
        'validation_status': 'FAILED' if critical_errors else 'PASSED'
    }
    
    return all_results

def main():
    """Main execution function."""
    try:
        # Run all validations
        validation_results = validate_all_scripts()
        
        # Log results
        logger.info("\n=== Validation Results ===")
        
        # Log validation status
        status = validation_results['validation_summary']['validation_status']
        logger.info(f"\nOverall Validation Status: {status}")
        
        if validation_results['validation_summary']['critical_errors']:
            logger.error("\nCritical Errors Found:")
            for error in validation_results['validation_summary']['critical_errors']:
                logger.error(f"- {error}")
        
        # Log detailed results
        logger.info("\nEDA Validation:")
        logger.info(f"Files present: {validation_results['eda_validation']['file_checks']}")
        
        logger.info("\nFeature Engineering Validation:")
        if 'data_quality' in validation_results['feature_engineering_validation']:
            logger.info(f"Number of customers: {validation_results['feature_engineering_validation']['data_quality'].get('row_count', 'N/A')}")
            
            # Log negative spending investigation results
            if 'negative_spending_investigation' in validation_results['feature_engineering_validation']:
                investigation = validation_results['feature_engineering_validation']['negative_spending_investigation']
                logger.info("\nNegative Spending Investigation Results:")
                
                # Check if there are any negative spending issues
                if investigation.get('negative_spending_analysis', {}):
                    # Log analysis for each metric
                    for metric, analysis in investigation['negative_spending_analysis'].items():
                        logger.info(f"\n{metric} Analysis:")
                        logger.info(f"- Number of affected customers: {analysis['count']}")
                        logger.info(f"- Value range: {analysis['min_value']} to {analysis['max_value']}")
                        logger.info("\nAffected Customers:")
                        for customer in analysis['customer_details'][:5]:  # Show first 5 customers
                            logger.info(f"- Customer ID: {customer['user_id']}")
                            logger.info(f"  Flight spend: {customer['total_flight_spend']}")
                            logger.info(f"  Hotel spend: {customer['total_hotel_spend']}")
                            logger.info(f"  Customer value: {customer['customer_value']}")
                else:
                    logger.info("No negative spending values found - all spending metrics are positive.")
                
                # Log recommendations if any exist
                if investigation.get('recommendations'):
                    logger.info("\nRecommendations:")
                    for rec in investigation['recommendations']:
                        logger.info(f"- {rec['description']}")
                        logger.info(f"  Details: {rec['details']}")
        
        logger.info("\nSegmentation Validation:")
        if 'segment_quality' in validation_results['segmentation_validation']:
            logger.info("Segment Distribution:")
            logger.info(validation_results['segmentation_validation']['segment_quality'].get('segment_distribution', {}))
        
        # Save validation results
        os.makedirs('scripts/output/validation', exist_ok=True)
        with open('scripts/output/validation/validation_results.txt', 'w') as f:
            f.write("=== Validation Summary ===\n")
            f.write(f"Status: {status}\n\n")
            
            for category, results in validation_results.items():
                f.write(f"\n=== {category} ===\n")
                if category == 'feature_engineering_validation':
                    # Write critical issues first
                    if results.get('critical_issues'):
                        f.write("\nCritical Issues:\n")
                        for issue in results['critical_issues']:
                            f.write(f"- {issue['type']}: {issue['recommendation']}\n")
                            f.write(f"  Details: {issue['details']}\n")
                    
                    # Write warnings
                    if results.get('warnings'):
                        f.write("\nWarnings:\n")
                        for warning in results['warnings']:
                            f.write(f"- {warning['metric']}: {warning['issue']} ({warning['count']} instances)\n")
                            f.write(f"  Recommendation: {warning['recommendation']}\n")
                
                f.write(str(results))
        
        logger.info("\nValidation results saved to scripts/output/validation/validation_results.txt")
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 