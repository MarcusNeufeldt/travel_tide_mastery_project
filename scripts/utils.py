import logging
import os
from functools import wraps
import time
import psycopg2
from psycopg2 import Error
import math

# Database connection parameters
db_params = {
    'dbname': 'TravelTide',
    'user': 'Test',
    'password': 'bQNxVzJL4g6u',
    'host': 'ep-noisy-flower-846766.us-east-2.aws.neon.tech',
    'port': '5432'
}

# Set up logging
def setup_logging(name):
    """Set up logging for the given module."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid adding handlers multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def safe_db_execute(conn, query, params=None):
    """Safely execute a database query with error handling."""
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        return cur
    except Error as e:
        logging.error(f"Database error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        raise

def safe_db_decorator(func):
    """Decorator for safe database operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except psycopg2.Error as e:
            logging.error(f"Database error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper 

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (in degrees)
        lat2, lon2: Latitude and longitude of point 2 (in degrees)
        
    Returns:
        Distance between the points in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c 