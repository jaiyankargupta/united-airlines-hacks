import os
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE
OUTPUT_DIR = BASE / "analysis" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pandas as pd
    import numpy as np
    import warnings

    # Suppress numpy divide by zero warnings
    np.seterr(divide='ignore', invalid='ignore')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
except ModuleNotFoundError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    msg = (
        f"Required Python package not found: {missing}.\n"
        "Please install the project dependencies before running the script.\n"
        "If you have pip available run:\n"
        "    python3 -m pip install -r ../requirements.txt\n"
        "or to install just the missing package:\n"
        f"    python3 -m pip install {missing}\n"
    )
    raise SystemExit(msg)

def read_csv(path, **kwargs):
    print(f"Reading {path} ...")
    return pd.read_csv(path, **kwargs)


def safe_correlation(x, y):
    """Calculate correlation safely, handling NaN and constant values"""
    try:
        corr = x.corr(y)
        if pd.isna(corr) or np.isinf(corr):
            return 0.0
        return corr
    except:
        return 0.0


def normalize_by_group(df, group_col, cols):
    # min-max per group (safe: if constant, result 0)
    def _norm(x):
        mn = x.min()
        mx = x.max()
        range_val = mx - mn

        # Handle division by zero and NaN values
        if range_val == 0 or pd.isna(range_val) or np.isinf(range_val):
            return pd.Series(0, index=x.index)

        normalized = (x - mn) / range_val
        # Replace any NaN or inf values with 0 and clip to [0,1] range
        return normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(0, 1)

    out = []
    for g, gdf in df.groupby(group_col):
        if len(gdf) > 0:  # Only process non-empty groups
            normed = gdf[cols].apply(_norm)
            normed[group_col] = g
            out.append(normed)

    if out:
        res = pd.concat(out)
        res.index = df.index
        return res[cols]
    else:
        # Return zero-filled dataframe if no groups processed
        return pd.DataFrame(0, index=df.index, columns=cols)


def load_and_validate_data(flight_df, pnr_flight_df, pnr_remark_df, bag_df, airports_df):
    """
    Load and validate all data files, return merged dataframe
    """
    print("Loading and validating data...")

    # Data validation
    required_cols = {
        'flight_df': ['company_id', 'flight_number', 'scheduled_departure_date_local'],
        'pnr_flight_df': ['company_id', 'flight_number', 'scheduled_departure_date_local', 'total_pax'],
        'pnr_remark_df': ['record_locator', 'special_service_request'],
        'bag_df': ['company_id', 'flight_number', 'scheduled_departure_date_local', 'bag_type'],
        'airports_df': []  # Optional
    }

    # Basic validation
    for df_name, df in [('flight_df', flight_df), ('pnr_flight_df', pnr_flight_df),
                       ('pnr_remark_df', pnr_remark_df), ('bag_df', bag_df)]:
        if df is None or len(df) == 0:
            raise ValueError(f"{df_name} is empty or None")

        # Check required columns exist
        missing_cols = [col for col in required_cols[df_name] if col not in df.columns]
        if missing_cols:
            print(f"Warning: {df_name} missing columns: {missing_cols}")

    # Data type conversions
    for df in [flight_df, pnr_flight_df, bag_df]:
        if 'company_id' in df.columns:
            df['company_id'] = df['company_id'].astype(str)
        if 'flight_number' in df.columns:
            df['flight_number'] = df['flight_number'].astype(str)
        if 'scheduled_departure_date_local' in df.columns:
            df['scheduled_departure_date_local'] = df['scheduled_departure_date_local'].astype(str)

    if 'flight_number' in pnr_remark_df.columns:
        pnr_remark_df['flight_number'] = pnr_remark_df['flight_number'].astype(str)

    # Start with key flight data
    key_cols = ['company_id','flight_number','scheduled_departure_date_local','scheduled_departure_station_code',
                'scheduled_arrival_station_code','total_seats','fleet_type','scheduled_ground_time_minutes',
                'minimum_turn_minutes']

    # Only use columns that exist
    available_cols = [col for col in key_cols if col in flight_df.columns]
    df = flight_df[available_cols].copy()

    # Aggregate PNR data - passenger count per flight
    if 'total_pax' in pnr_flight_df.columns:
        df_pnr_group = pnr_flight_df.groupby(['company_id','flight_number','scheduled_departure_date_local'],
                                           dropna=False).agg({'total_pax':'sum'}).reset_index()
        df_pnr_group.rename(columns={'total_pax':'passenger_count'}, inplace=True)
        df = df.merge(df_pnr_group, on=['company_id','flight_number','scheduled_departure_date_local'], how='left')
    else:
        df['passenger_count'] = 0

    # Aggregate SSR data
    if 'record_locator' in pnr_remark_df.columns and 'record_locator' in pnr_flight_df.columns:
        pnr_map = pnr_flight_df[['record_locator','flight_number','scheduled_departure_date_local']].drop_duplicates()
        df_ssr_with_flight = pnr_remark_df.merge(pnr_map, on='record_locator', how='left')

        # Check if merge was successful and required columns exist
        if ('flight_number' in df_ssr_with_flight.columns and
            'scheduled_departure_date_local' in df_ssr_with_flight.columns and
            len(df_ssr_with_flight) > 0):

            df_ssr_with_flight = df_ssr_with_flight.dropna(subset=['flight_number', 'scheduled_departure_date_local'])

            if len(df_ssr_with_flight) > 0:
                df_ssr_grouped = df_ssr_with_flight.groupby(['flight_number','scheduled_departure_date_local']).agg(
                    ssr_count=('special_service_request','count')).reset_index()
                df = df.merge(df_ssr_grouped, on=['flight_number','scheduled_departure_date_local'], how='left')
            else:
                print("Warning: No valid SSR data after merge and cleanup.")
                df['ssr_count'] = 0
        else:
            print("Warning: SSR merge failed - no matching record_locators or missing required columns.")
            df['ssr_count'] = 0
    else:
        print("Warning: record_locator column missing in one or both PNR datasets.")
        df['ssr_count'] = 0

    # Aggregate bag data
    if len(bag_df) > 0 and 'bag_type' in bag_df.columns:
        bag_types = bag_df.groupby(['company_id','flight_number','scheduled_departure_date_local'])['bag_type'].value_counts().unstack(fill_value=0).reset_index()
        bag_types.columns = [str(c) for c in bag_types.columns]

        # Calculate transfer and origin counts
        transfer_cols = [c for c in bag_types.columns if c.lower().strip() in ('transfer','hot transfer')]
        origin_cols = [c for c in bag_types.columns if c.lower().strip() in ('origin','checked')]

        bag_types['transfer_count'] = bag_types[transfer_cols].sum(axis=1) if transfer_cols else 0
        bag_types['origin_count'] = bag_types[origin_cols].sum(axis=1) if origin_cols else 0
        bag_types['total_bags'] = bag_types[[c for c in bag_types.columns if c not in
                                          ['company_id','flight_number','scheduled_departure_date_local','transfer_count','origin_count']]].sum(axis=1)

        df = df.merge(bag_types[['company_id','flight_number','scheduled_departure_date_local','transfer_count','origin_count','total_bags']],
                     on=['company_id','flight_number','scheduled_departure_date_local'], how='left')
    else:
        df['transfer_count'] = 0
        df['origin_count'] = 0
        df['total_bags'] = 0

    # Fill missing values
    df['passenger_count'] = df['passenger_count'].fillna(0).astype(int)
    df['ssr_count'] = df['ssr_count'].fillna(0).astype(int)
    df['transfer_count'] = df['transfer_count'].fillna(0).astype(int)
    df['origin_count'] = df['origin_count'].fillna(0).astype(int)
    df['total_bags'] = df['total_bags'].fillna(0).astype(int)

    # Add basic derived features
    df['total_seats'] = df['total_seats'].fillna(1) if 'total_seats' in df.columns else 1
    df['load_pct'] = df['passenger_count'] / df['total_seats']

    # Transfer ratio
    df['transfer_to_checked_ratio'] = df['transfer_count'] / (df['origin_count'].replace(0, np.nan))
    df['transfer_to_checked_ratio'] = df['transfer_to_checked_ratio'].fillna(df['transfer_count'])

    # Ground pressure
    if 'scheduled_ground_time_minutes' in df.columns and 'minimum_turn_minutes' in df.columns:
        df['scheduled_ground_time_minutes'] = df['scheduled_ground_time_minutes'].astype(float)
        df['minimum_turn_minutes'] = df['minimum_turn_minutes'].astype(float)
        df['ground_pressure'] = df['minimum_turn_minutes'] - df['scheduled_ground_time_minutes']
    else:
        df['ground_pressure'] = 0

    # Add delay calculation if datetime columns available
    if all(col in flight_df.columns for col in ['scheduled_departure_datetime_local', 'actual_departure_datetime_local']):
        try:
            df_delay = flight_df[['company_id','flight_number','scheduled_departure_date_local',
                                'scheduled_departure_datetime_local','actual_departure_datetime_local']].copy()
            df_delay['delay_minutes'] = (pd.to_datetime(df_delay['actual_departure_datetime_local']) -
                                       pd.to_datetime(df_delay['scheduled_departure_datetime_local'])).dt.total_seconds() / 60.0
            df_delay = df_delay[['company_id','flight_number','scheduled_departure_date_local','delay_minutes']]
            df = df.merge(df_delay, on=['company_id','flight_number','scheduled_departure_date_local'], how='left')
            df['delay_minutes'] = df['delay_minutes'].fillna(0)
        except:
            df['delay_minutes'] = 0
    else:
        df['delay_minutes'] = 0

    df['late_departure'] = (df['delay_minutes'] > 0).astype(int)

    # SSR per passenger
    df['ssr_per_pax'] = df['ssr_count'] / df['passenger_count'].replace(0, np.nan)
    df['ssr_per_pax'] = df['ssr_per_pax'].fillna(0)

    # Add required columns for feature engineering
    if 'scheduled_departure_datetime_local' in flight_df.columns:
        df_temp = df.merge(flight_df[['company_id','flight_number','scheduled_departure_date_local','scheduled_departure_datetime_local']],
                          on=['company_id','flight_number','scheduled_departure_date_local'], how='left')
        df['scheduled_departure_time'] = df_temp['scheduled_departure_datetime_local']
    else:
        df['scheduled_departure_time'] = pd.NaT

    # Add other required columns for features
    df['destination'] = df['scheduled_arrival_station_code'] if 'scheduled_arrival_station_code' in df.columns else 'UNK'
    df['aircraft_type'] = df['fleet_type'] if 'fleet_type' in df.columns else 'UNK'
    df['load_factor'] = df['load_pct'] * 100
    df['transfer_bag_count'] = df['transfer_count']
    df['departure_date'] = df['scheduled_departure_date_local']

    print(f"Data loading complete. Merged dataframe has {len(df):,} rows and {len(df.columns)} columns")
    return df


def perform_eda_analysis(flight_df, pnr_flight_df, pnr_remark_df, bag_df, airports_df):
    """
    Perform comprehensive EDA analysis
    """
    print("Performing comprehensive EDA analysis...")

    # Create a simple merged dataset for EDA
    merged_df = load_and_validate_data(flight_df, pnr_flight_df, pnr_remark_df, bag_df, airports_df)

    # Perform the actual EDA using existing function
    eda_results = perform_eda(merged_df, flight_df)

    return eda_results


def calculate_flight_difficulty_score(df):
    """
    Calculate flight difficulty scores using a systematic daily-level approach
    """
    print("Calculating flight difficulty scores with daily normalization...")

    # Create copy for scoring calculations
    scoring_df = df.copy()

    # Verify required columns and fill missing values
    required_cols = ['delay_minutes', 'ground_pressure', 'transfer_to_checked_ratio', 'ssr_per_pax', 'load_pct']
    for col in required_cols:
        if col not in scoring_df.columns:
            scoring_df[col] = 0

    # Transform variables for scoring (handle negative values appropriately)
    scoring_df['ground_pressure_pos'] = scoring_df['ground_pressure'].apply(lambda x: max(x, 0))
    scoring_df['delay_pos'] = scoring_df['delay_minutes'].apply(lambda x: max(x, 0))

    # Normalize per day if we have date column
    if 'scheduled_departure_date_local' in scoring_df.columns:
        scoring_df['scheduled_departure_date_local'] = scoring_df['scheduled_departure_date_local'].astype(str)

        norm_cols = ['delay_pos', 'ground_pressure_pos', 'transfer_to_checked_ratio', 'ssr_per_pax', 'load_pct']

        # Add advanced feature scores if they exist
        advanced_cols = ['passenger_complexity_score', 'baggage_complexity_score', 'operational_pressure_score']
        for col in advanced_cols:
            if col in scoring_df.columns:
                norm_cols.append(col)

        # Add creative features if they exist
        if 'master_complexity_index' in scoring_df.columns:
            norm_cols.append('master_complexity_index')

        # Ensure all columns exist
        for col in norm_cols:
            if col not in scoring_df.columns:
                scoring_df[col] = 0

        # Normalize by date
        normed = normalize_by_group(scoring_df, 'scheduled_departure_date_local', norm_cols)
        for col in norm_cols:
            if col != 'master_complexity_index':  # Master complexity already normalized
                scoring_df[col + '_norm'] = normed[col]
            else:
                scoring_df[col + '_norm'] = scoring_df[col]  # Already 0-1
    else:
        # Simple min-max normalization if no date grouping
        norm_cols = ['delay_pos', 'ground_pressure_pos', 'transfer_to_checked_ratio', 'ssr_per_pax', 'load_pct']

        for col in norm_cols:
            if col in scoring_df.columns:
                min_val = scoring_df[col].min()
                max_val = scoring_df[col].max()
                range_val = max_val - min_val

                if range_val == 0 or pd.isna(range_val):
                    scoring_df[col + '_norm'] = 0
                else:
                    normalized_vals = (scoring_df[col] - min_val) / range_val
                    # Handle any NaN values and clip to valid range
                    scoring_df[col + '_norm'] = normalized_vals.fillna(0).clip(0, 1)

    # Calculate difficulty score with enhanced weights
    base_weights = {
        'delay_pos_norm': 0.20,
        'ground_pressure_pos_norm': 0.20,
        'transfer_to_checked_ratio_norm': 0.15,
        'ssr_per_pax_norm': 0.15,
        'load_pct_norm': 0.10
    }

    # Enhanced weights if advanced features exist
    if 'passenger_complexity_score_norm' in scoring_df.columns:
        base_weights = {
            'delay_pos_norm': 0.15,
            'ground_pressure_pos_norm': 0.15,
            'transfer_to_checked_ratio_norm': 0.05,
            'ssr_per_pax_norm': 0.05,
            'load_pct_norm': 0.05,
            'passenger_complexity_score_norm': 0.15,
            'baggage_complexity_score_norm': 0.10,
            'operational_pressure_score_norm': 0.15
        }

        # Add master complexity if available
        if 'master_complexity_index_norm' in scoring_df.columns:
            base_weights['master_complexity_index_norm'] = 0.15
            # Reduce other weights proportionally
            for key in base_weights:
                if key != 'master_complexity_index_norm':
                    base_weights[key] *= 0.85

    # Calculate weighted score
    difficulty_score = pd.Series(0.0, index=scoring_df.index)
    for feature, weight in base_weights.items():
        if feature in scoring_df.columns:
            difficulty_score += scoring_df[feature] * weight
        else:
            print(f"Warning: {feature} not found, using 0")

    scoring_df['flight_difficulty_score'] = difficulty_score

    # Daily ranking and classification
    if 'scheduled_departure_date_local' in scoring_df.columns:
        scoring_df['daily_rank'] = scoring_df.groupby('scheduled_departure_date_local')['flight_difficulty_score'].rank(method='dense', ascending=False)
        scoring_df['flights_per_day'] = scoring_df.groupby('scheduled_departure_date_local')['flight_difficulty_score'].transform('count')
        scoring_df['daily_percentile'] = scoring_df['daily_rank'] / scoring_df['flights_per_day']

        # Classification
        def classify_difficulty(percentile):
            # Classify based on daily rank distribution (top 33% are most difficult)
            if percentile <= 0.33:
                return 'Difficult'
            elif percentile <= 0.66:
                return 'Medium'
            else:
                return 'Easy'

        scoring_df['difficulty_classification'] = scoring_df['daily_percentile'].apply(classify_difficulty)
    else:
        # Fallback to global ranking if no date grouping available
        scoring_df['daily_rank'] = scoring_df['flight_difficulty_score'].rank(method='dense', ascending=False)
        scoring_df['daily_percentile'] = scoring_df['daily_rank'] / len(scoring_df)

        # Create difficulty classification with proper error handling
        try:
            scoring_df['difficulty_classification'] = pd.cut(scoring_df['flight_difficulty_score'],
                                                           bins=3, labels=['Easy', 'Medium', 'Difficult'],
                                                           duplicates='drop')
        except ValueError:
            # If cutting fails due to duplicate edges, use quantile-based approach
            scoring_df['difficulty_classification'] = pd.qcut(scoring_df['flight_difficulty_score'],
                                                            q=3, labels=['Easy', 'Medium', 'Difficult'],
                                                            duplicates='drop')

    # Summary statistics
    score_min = scoring_df['flight_difficulty_score'].min()
    score_max = scoring_df['flight_difficulty_score'].max()
    score_mean = scoring_df['flight_difficulty_score'].mean()

    print(f"Flight difficulty scoring completed successfully")
    print(f"Score range: {score_min:.3f} to {score_max:.3f} (mean: {score_mean:.3f})")
    print(f"Difficulty distribution: {scoring_df['difficulty_classification'].value_counts().to_dict()}")

    return scoring_df


def save_results(df, output_dir="analysis/output"):
    """
    Save analysis results to files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Main results
    main_file = os.path.join(output_dir, "flight_difficulty_results.csv")
    df.to_csv(main_file, index=False)
    print(f"Results saved to {main_file}")

    return main_file


def categorize_aircraft(aircraft_type):
    """Categorize aircraft by size/type"""
    if pd.isna(aircraft_type):
        return 'Unknown'

    aircraft_str = str(aircraft_type).upper()

    # Regional aircraft
    if any(x in aircraft_str for x in ['CRJ', 'ERJ', 'EMB', 'E175', 'E170']):
        return 'Regional'

    # Narrow body
    elif any(x in aircraft_str for x in ['A319', 'A320', 'A321', 'B737', '737']):
        return 'Narrow Body'

    # Wide body
    elif any(x in aircraft_str for x in ['A330', 'A340', 'A350', 'B767', '767', 'B787', '787']):
        return 'Wide Body'

    # Large body
    elif any(x in aircraft_str for x in ['A380', 'B747', '747', 'B777', '777']):
        return 'Large Body'

    else:
        return 'Unknown'


def engineer_advanced_features(df):
    """
    Engineer advanced features for sophisticated flight difficulty analysis
    """
    print("Engineering advanced features...")

    # Temporal features
    df['scheduled_departure_hour'] = pd.to_datetime(df['scheduled_departure_time']).dt.hour
    df['is_rush_hour_departure'] = df['scheduled_departure_hour'].isin([6,7,8,17,18,19])
    df['is_red_eye'] = df['scheduled_departure_hour'].isin(list(range(22, 24)) + list(range(0, 6)))
    df['is_weekend'] = pd.to_datetime(df['scheduled_departure_time']).dt.dayofweek.isin([5,6])

    # Flight duration (where available)
    if 'actual_departure_time' in df.columns and 'actual_arrival_time' in df.columns:
        df['flight_duration_hours'] = (
            pd.to_datetime(df['actual_arrival_time']) - pd.to_datetime(df['actual_departure_time'])
        ).dt.total_seconds() / 3600
    else:
        df['flight_duration_hours'] = 0

    # Haul type classification based on flight duration
    df['haul_type'] = df['flight_duration_hours'].apply(lambda x:
        'Short' if x < 3 else
        'Medium' if x < 6 else
        'Long' if x < 12 else
        'Ultra-Long' if x > 0 else 'Unknown')

    # Aircraft categorization
    df['aircraft_category'] = df['aircraft_type'].apply(categorize_aircraft)
    df['is_express'] = df['flight_number'].str.startswith(('2', '3', '4', '5'))

    # International detection (basic - could be enhanced with airport codes)
    df['is_international'] = df['destination'].isin(['ZRH', 'MEX', 'LHR', 'CDG', 'FRA', 'NRT', 'ICN'])

    # Passenger complexity features
    df['child_count'] = df.get('unaccompanied_minors', 0) + df.get('accompanied_minors', 0)
    df['lap_child_count'] = df.get('lap_children', 0)
    df['stroller_count'] = df.get('strollers', 0)
    df['basic_economy_count'] = df.get('basic_economy_passengers', 0)

    # Composite passenger complexity score
    df['passenger_complexity_score'] = (
        df['child_count'] * 2 +
        df['lap_child_count'] * 1.5 +
        df['stroller_count'] * 2 +
        df['ssr_count'] * 3
    )

    # Baggage complexity features
    df['hot_transfer_count'] = df.get('hot_transfers', 0)  # <30 min connections
    df['late_bag_tags'] = df.get('same_day_bag_tags', 0)  # Same day or day-before tags

    # Composite baggage complexity score
    df['baggage_complexity_score'] = (
        df['hot_transfer_count'] * 3 +
        df['transfer_bag_count'] * 1.5 +
        df['late_bag_tags'] * 2
    )

    # Operational pressure score (composite of multiple factors)
    ground_time = df.get('scheduled_ground_time_minutes', pd.Series(60, index=df.index))
    if isinstance(ground_time, (int, float)):
        ground_time = pd.Series(ground_time, index=df.index)

    df['operational_pressure_score'] = (
        df['is_rush_hour_departure'].astype(int) * 2 +
        df['is_weekend'].astype(int) * 1 +
        df['is_international'].astype(int) * 2 +
        (ground_time < 45).astype(int) * 3  # Tight turnarounds
    )

    return df


def engineer_creative_features(df):
    """
    Creative and innovative feature engineering
    Exploring advanced analytical approaches and external data integration concepts
    """
    print("Engineering creative and innovative features...")

    # Weather impact simulation
    # Simulate weather complexity based on seasonal patterns and destination
    import random
    random.seed(42)

    # Weather complexity based on season and destination type
    df['flight_date'] = pd.to_datetime(df['scheduled_departure_time']).dt.date
    df['season'] = pd.to_datetime(df['scheduled_departure_time']).dt.month.apply(
        lambda x: 'Winter' if x in [12,1,2] else
                 'Spring' if x in [3,4,5] else
                 'Summer' if x in [6,7,8] else 'Fall'
    )

    # Weather risk score (higher for winter, international routes)
    weather_multiplier = {
        'Winter': 2.5, 'Spring': 1.2, 'Summer': 0.8, 'Fall': 1.5
    }
    df['weather_risk_score'] = df.apply(lambda row:
        weather_multiplier[row['season']] *
        (2.0 if row['is_international'] else 1.0) *
        (1.5 if row['destination'] in ['BOS', 'DEN', 'SEA'] else 1.0), axis=1
    )

    # Crew scheduling complexity
    # Simulate crew complexity based on flight patterns
    df['crew_complexity_score'] = (
        df['is_red_eye'].astype(int) * 3 +  # Red-eye crew scheduling harder
        df['is_international'].astype(int) * 2 +  # International crew requirements
        (df['flight_duration_hours'] > 8).astype(int) * 2 +  # Long flights need crew rest
        df['is_weekend'].astype(int) * 1.5  # Weekend staffing challenges
    )

    # Gate congestion modeling
    # Model gate availability pressure based on time and aircraft type
    gate_pressure_hours = [6,7,8,9,17,18,19,20]  # Peak departure/arrival times
    df['gate_pressure_score'] = df.apply(lambda row:
        3.0 if row['scheduled_departure_hour'] in gate_pressure_hours and row['aircraft_category'] in ['Wide Body', 'Large Body'] else
        2.0 if row['scheduled_departure_hour'] in gate_pressure_hours else
        1.5 if row['aircraft_category'] in ['Wide Body', 'Large Body'] else 1.0, axis=1
    )

    # Fuel efficiency and cost modeling
    # Simulate fuel complexity based on aircraft type, distance, weather
    fuel_efficiency = {
        'Regional': 1.0, 'Narrow Body': 1.2, 'Wide Body': 2.5, 'Large Body': 3.0, 'Unknown': 1.5
    }
    df['fuel_complexity_score'] = df.apply(lambda row:
        fuel_efficiency.get(row['aircraft_category'], 1.5) *
        (1.0 + row['flight_duration_hours'] * 0.1) *  # Longer = more fuel planning
        row['weather_risk_score'] * 0.3, axis=1  # Weather affects fuel needs
    )

    # Passenger flow optimization
    # Model passenger boarding complexity
    boarding_complexity = {
        'Regional': 1.0, 'Narrow Body': 1.5, 'Wide Body': 2.5, 'Large Body': 3.0
    }
    df['boarding_complexity_score'] = df.apply(lambda row:
        boarding_complexity.get(row['aircraft_category'], 1.5) *
        (1.0 + row['passenger_complexity_score'] * 0.1) *
        (1.5 if row['basic_economy_count'] > 50 else 1.0), axis=1
    )

    # Maintenance prediction modeling
    # Simulate maintenance complexity based on aircraft age and utilization
    df['maintenance_risk_score'] = df.apply(lambda row:
        (2.0 if 'MD-' in str(row['aircraft_type']) else 1.0) *  # Older aircraft types
        (1.5 if row['flight_duration_hours'] > 6 else 1.0) *  # Long flights = more wear
        (1.3 if row['is_international'] else 1.0), axis=1  # International = more stress
    )

    # Revenue optimization features
    # Model revenue complexity and yield management factors
    df['revenue_complexity_score'] = df.apply(lambda row:
        (row['load_factor'] * 0.01) *  # Higher load factor = more revenue pressure
        (2.0 if row['is_international'] else 1.0) *  # International higher yield
        (0.8 if row['basic_economy_count'] > row['passenger_count'] * 0.5 else 1.2), axis=1
    )

    # Competitive market analysis
    # Simulate competitive pressure on key routes
    competitive_routes = ['LAX', 'SFO', 'DEN', 'BOS', 'SEA', 'LHR', 'ZRH']
    df['competitive_pressure_score'] = df['destination'].apply(lambda x:
        2.5 if x in competitive_routes[:3] else  # High competition
        1.8 if x in competitive_routes[3:6] else  # Medium competition
        1.2 if x in competitive_routes[6:] else 1.0  # International competition
    )

    # Security and compliance complexity
    # Model security screening and compliance factors
    df['security_complexity_score'] = df.apply(lambda row:
        (3.0 if row['is_international'] else 1.0) *  # International security
        (1.5 if row['ssr_count'] > 5 else 1.0) *  # More SSR = more screening
        (1.3 if row['child_count'] > 3 else 1.0), axis=1  # Family screening complexity
    )

    # Master complexity index
    # Create ultimate complexity score incorporating all creative factors
    df['master_complexity_index'] = (
        df['weather_risk_score'] * 0.15 +
        df['crew_complexity_score'] * 0.12 +
        df['gate_pressure_score'] * 0.10 +
        df['fuel_complexity_score'] * 0.08 +
        df['boarding_complexity_score'] * 0.12 +
        df['maintenance_risk_score'] * 0.10 +
        df['revenue_complexity_score'] * 0.08 +
        df['competitive_pressure_score'] * 0.08 +
        df['security_complexity_score'] * 0.12 +
        df['operational_pressure_score'] * 0.05
    )

    # Normalize master complexity index to 0-1 scale
    if df['master_complexity_index'].std() > 0:
        df['master_complexity_index'] = (
            df['master_complexity_index'] - df['master_complexity_index'].min()
        ) / (df['master_complexity_index'].max() - df['master_complexity_index'].min())

    return df


def perform_eda(df, df_f):
    """Perform Exploratory Data Analysis to answer specific business questions"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # Question 1: What is the average delay and what percentage of flights depart later than scheduled?
    delay_data = df_f[['scheduled_departure_datetime_local', 'actual_departure_datetime_local']].copy()
    delay_data['delay_minutes'] = (pd.to_datetime(delay_data['actual_departure_datetime_local']) -
                                  pd.to_datetime(delay_data['scheduled_departure_datetime_local'])).dt.total_seconds() / 60.0

    avg_delay = delay_data['delay_minutes'].mean()
    pct_late = (delay_data['delay_minutes'] > 0).mean() * 100

    print(f"\n1. FLIGHT DELAY ANALYSIS:")
    print(f"   Average delay across all flights: {avg_delay:.1f} minutes")
    print(f"   Percentage of flights departing late: {pct_late:.1f}%")
    print(f"   Total flights analyzed: {len(delay_data):,}")

    # Question 2: How many flights have scheduled ground time close to or below the minimum turn mins?
    ground_pressure_flights = (df['ground_pressure'] > 0).sum()
    total_flights = len(df)
    pct_tight_turns = (ground_pressure_flights / total_flights) * 100

    print(f"\n2. GROUND TIME CONSTRAINTS:")
    print(f"   Flights with scheduled ground time at or below minimum: {ground_pressure_flights:,} ({pct_tight_turns:.1f}%)")
    print(f"   Average scheduled ground time: {df['scheduled_ground_time_minutes'].mean():.1f} minutes")
    print(f"   Average minimum turn time requirement: {df['minimum_turn_minutes'].mean():.1f} minutes")

    # Additional ground time insights
    ground_time_gap = df['scheduled_ground_time_minutes'] - df['minimum_turn_minutes']
    print(f"   Average buffer above minimum turn time: {ground_time_gap.mean():.1f} minutes")

    # Question 3: What is the average ratio of transfer bags vs. checked bags across flights?
    valid_bag_ratios = df[df['origin_count'] > 0]['transfer_to_checked_ratio']
    avg_bag_ratio = valid_bag_ratios.mean()

    print(f"\n3. BAGGAGE OPERATION ANALYSIS:")
    print(f"   Average transfer-to-checked bag ratio: {avg_bag_ratio:.3f}")
    print(f"   Flights with transfer baggage: {(df['transfer_count'] > 0).sum():,}")
    print(f"   Average transfer bags per flight: {df['transfer_count'].mean():.1f}")
    print(f"   Average origin/checked bags per flight: {df['origin_count'].mean():.1f}")
    print(f"   Total baggage operations per flight: {df['total_bags'].mean():.1f}")

    # Question 4: How do passenger loads compare across flights, and do higher loads correlate with operational difficulty?
    avg_load = df['load_pct'].mean() * 100
    high_load_threshold = 0.85
    medium_load_threshold = 0.70
    high_load_flights = (df['load_pct'] > high_load_threshold).sum()
    medium_load_flights = ((df['load_pct'] >= medium_load_threshold) & (df['load_pct'] <= high_load_threshold)).sum()

    print(f"\n4. PASSENGER LOAD ANALYSIS:")
    print(f"   Average load factor across all flights: {avg_load:.1f}%")
    print(f"   High load flights (>{high_load_threshold*100:.0f}%): {high_load_flights:,} ({(high_load_flights/total_flights)*100:.1f}%)")
    print(f"   Medium load flights ({medium_load_threshold*100:.0f}-{high_load_threshold*100:.0f}%): {medium_load_flights:,}")

    # Load vs difficulty correlation analysis
    load_difficulty_corr = df['load_pct'].corr(df['difficulty_score'])
    print(f"   Correlation between load factor and difficulty: {load_difficulty_corr:.3f}")

    # Load distribution analysis
    load_quartiles = df['load_pct'].describe()
    print(f"   Load factor quartiles: 25%={load_quartiles['25%']*100:.1f}%, 50%={load_quartiles['50%']*100:.1f}%, 75%={load_quartiles['75%']*100:.1f}%")

    # Question 5: Are high special service requests flights also high-delay after controlling for load?
    high_ssr_threshold = df['ssr_per_pax'].quantile(0.75)
    high_ssr_flights = df[df['ssr_per_pax'] > high_ssr_threshold]

    avg_delay_high_ssr = high_ssr_flights['delay_minutes'].mean()
    avg_delay_low_ssr = df[df['ssr_per_pax'] <= high_ssr_threshold]['delay_minutes'].mean()

    print(f"\n5. SPECIAL SERVICE REQUESTS AND DELAY ANALYSIS:")
    print(f"   Average SSR per passenger across all flights: {df['ssr_per_pax'].mean():.3f}")
    print(f"   High SSR flights (top 25%, threshold >{high_ssr_threshold:.3f}): {len(high_ssr_flights):,}")
    print(f"   Average delay for high SSR flights: {avg_delay_high_ssr:.1f} minutes")
    print(f"   Average delay for low SSR flights: {avg_delay_low_ssr:.1f} minutes")
    print(f"   Raw delay difference (high SSR - low SSR): {avg_delay_high_ssr - avg_delay_low_ssr:.1f} minutes")

    # Control for load factor - analyze within high load flights only
    high_load_threshold = 0.8
    high_load_flights = df[df['load_pct'] > high_load_threshold]

    high_load_high_ssr = df[(df['load_pct'] > high_load_threshold) & (df['ssr_per_pax'] > high_ssr_threshold)]
    high_load_low_ssr = df[(df['load_pct'] > high_load_threshold) & (df['ssr_per_pax'] <= high_ssr_threshold)]

    print(f"\n   CONTROLLING FOR HIGH LOAD FACTOR (>{high_load_threshold*100:.0f}%):")
    print(f"   High load flights total: {len(high_load_flights):,}")
    print(f"   High load + high SSR flights: {len(high_load_high_ssr):,}")
    print(f"   High load + low SSR flights: {len(high_load_low_ssr):,}")

    if len(high_load_high_ssr) > 0 and len(high_load_low_ssr) > 0:
        delay_high_load_high_ssr = high_load_high_ssr['delay_minutes'].mean()
        delay_high_load_low_ssr = high_load_low_ssr['delay_minutes'].mean()
        delay_diff = delay_high_load_high_ssr - delay_high_load_low_ssr

        print(f"   Average delay (high load + high SSR): {delay_high_load_high_ssr:.1f} minutes")
        print(f"   Average delay (high load + low SSR): {delay_high_load_low_ssr:.1f} minutes")
        print(f"   Controlled delay difference: {delay_diff:.1f} minutes")

        if delay_diff > 2:
            print(f"   FINDING: High SSR flights show {delay_diff:.1f} min more delay even when controlling for load")
        else:
            print(f"   FINDING: SSR impact on delays is minimal when controlling for passenger load")
    else:
        print(f"   Insufficient data for controlled analysis")

    # Additional SSR analysis with safe correlation calculation
    ssr_delay_corr = safe_correlation(df['ssr_per_pax'], df['delay_minutes'])
    print(f"   Overall correlation between SSR per passenger and delays: {ssr_delay_corr:.3f}")


def analyze_destinations(df):
    """Analyze difficulty patterns by destination"""
    print("\n" + "="*50)
    print("DESTINATION DIFFICULTY ANALYSIS")
    print("="*50)

    # Group by destination and calculate average difficulty metrics
    dest_analysis = df.groupby('scheduled_arrival_station_code').agg({
        'difficulty_score': ['mean', 'count'],
        'delay_minutes': 'mean',
        'ground_pressure': 'mean',
        'load_pct': 'mean',
        'ssr_per_pax': 'mean',
        'transfer_to_checked_ratio': 'mean'
    }).round(3)

    # Flatten column names
    dest_analysis.columns = ['avg_difficulty', 'flight_count', 'avg_delay', 'avg_ground_pressure',
                           'avg_load', 'avg_ssr_per_pax', 'avg_transfer_ratio']

    # Filter destinations with at least 10 flights for statistical significance
    dest_analysis = dest_analysis[dest_analysis['flight_count'] >= 10].sort_values('avg_difficulty', ascending=False)

    print(f"\nTOP 10 MOST DIFFICULT DESTINATIONS (min 10 flights):")
    print(dest_analysis.head(10).to_string())

    print(f"\nTOP 10 EASIEST DESTINATIONS (min 10 flights):")
    print(dest_analysis.tail(10).to_string())

    return dest_analysis


def generate_operational_insights(df, dest_analysis):
    """Generate actionable operational insights and recommendations"""
    print("\n" + "="*50)
    print("OPERATIONAL INSIGHTS & RECOMMENDATIONS")
    print("="*50)

    # Identify top difficulty drivers
    difficult_flights = df[df['difficulty_classification'] == 'Difficult']

    print(f"\n1. KEY OPERATIONAL FACTORS:")
    print(f"   Flights with delays >30 min: {(difficult_flights['delay_minutes'] > 30).sum():,}")
    print(f"   Flights with tight turnarounds: {(difficult_flights['ground_pressure'] > 0).sum():,}")
    print(f"   Flights with high transfer ratios: {(difficult_flights['transfer_to_checked_ratio'] > 2).sum():,}")
    print(f"   Flights with high SSR: {(difficult_flights['ssr_per_pax'] > 0.1).sum():,}")

    # Peak difficulty times
    df['hour'] = pd.to_datetime(df['scheduled_departure_date_local']).dt.strftime('%Y-%m-%d')
    daily_avg_difficulty = df.groupby('hour')['difficulty_score'].mean().sort_values(ascending=False)

    print(f"\n2. HIGHEST DIFFICULTY DAYS:")
    print(f"   Top 3 most challenging days:")
    for i, (date, score) in enumerate(daily_avg_difficulty.head(3).items(), 1):
        print(f"   {i}. {date}: Average difficulty {score:.3f}")

    # Aircraft type analysis
    if 'fleet_type' in df.columns:
        fleet_difficulty = df.groupby('fleet_type')['difficulty_score'].mean().sort_values(ascending=False)
        print(f"\n3. AIRCRAFT TYPE DIFFICULTY:")
        print(f"   Most challenging aircraft types:")
        for i, (fleet, score) in enumerate(fleet_difficulty.head(5).items(), 1):
            print(f"   {i}. {fleet}: {score:.3f}")

    print(f"\n4. RECOMMENDED ACTIONS:")
    print(f"\n   A. IMMEDIATE ACTIONS (Next 30 days):")
    print(f"      • Pre-position additional ground staff for flights to: {', '.join(dest_analysis.head(5).index.tolist())}")
    print(f"      • Implement 'difficulty alerts' 2 hours before departure for scores >0.4")
    print(f"      • Create rapid response teams for flights with ground pressure >10 minutes")

    print(f"\n   B. OPERATIONAL IMPROVEMENTS (Next 90 days):")
    print(f"      • Negotiate longer ground times for routes with consistently high transfer ratios")
    print(f"      • Establish dedicated SSR support for flights with >0.1 SSR per passenger")
    print(f"      • Implement predictive baggage staging for high-transfer destinations")

    print(f"\n   C. STRATEGIC INITIATIVES (Next 6 months):")
    print(f"      • Review schedule optimization for consistently difficult routes")
    print(f"      • Develop route-specific operational playbooks")
    print(f"      • Implement machine learning for predictive difficulty scoring")

    print(f"\n   D. RESOURCE ALLOCATION PRIORITIES:")
    print(f"      • High Priority: Flights with difficulty score >0.4 ({(df['difficulty_score'] > 0.4).sum():,} flights)")
    print(f"      • Medium Priority: Flights with 0.2-0.4 difficulty ({((df['difficulty_score'] >= 0.2) & (df['difficulty_score'] <= 0.4)).sum():,} flights)")
    print(f"      • Monitor: All other flights for trend changes")

    return {
        'difficult_destinations': dest_analysis.head(10).index.tolist(),
        'peak_difficulty_days': daily_avg_difficulty.head(3).to_dict(),
        'total_high_difficulty': (df['difficulty_score'] > 0.4).sum()
    }


def main():
    # file paths
    f_flight = DATA_DIR / 'Flight Level Data.csv'
    f_pnr = DATA_DIR / 'PNR+Flight+Level+Data.csv'
    f_ssr = DATA_DIR / 'PNR Remark Level Data.csv'
    f_bag = DATA_DIR / 'Bag+Level+Data.csv'
    f_airports = DATA_DIR / 'Airports Data.csv'

    df_f = read_csv(f_flight, parse_dates=['scheduled_departure_datetime_local','scheduled_arrival_datetime_local','actual_departure_datetime_local','actual_arrival_datetime_local'], dtype={'flight_number':str})
    df_pnr = read_csv(f_pnr, parse_dates=['pnr_creation_date'], dtype={'flight_number':str})
    df_ssr = read_csv(f_ssr, parse_dates=['pnr_creation_date'], dtype={'flight_number':str})
    df_bag = read_csv(f_bag, parse_dates=['bag_tag_issue_date'], dtype={'flight_number':str})

    # Load airports data for international route detection
    try:
        df_airports = read_csv(f_airports)
    except:
        df_airports = pd.DataFrame()  # Empty fallback

    # keep a copy of key flight columns
    key_cols = ['company_id','flight_number','scheduled_departure_date_local','scheduled_departure_station_code','scheduled_arrival_station_code','total_seats','fleet_type','scheduled_ground_time_minutes','minimum_turn_minutes']
    df = df_f[key_cols].copy()

    # ensure identical key types
    for d in [df, df_pnr, df_bag]:
        d['company_id'] = d['company_id'].astype(str)
        d['flight_number'] = d['flight_number'].astype(str)
        d['scheduled_departure_date_local'] = d['scheduled_departure_date_local'].astype(str)

    # SSR doesn't have company_id, only normalize what it has
    df_ssr['flight_number'] = df_ssr['flight_number'].astype(str)

    # Aggregate PNR to get passenger count per flight (sum of total_pax across PNRs)
    df_pnr_group = df_pnr.groupby(['company_id','flight_number','scheduled_departure_date_local'], dropna=False).agg({'total_pax':'sum'}).reset_index().rename(columns={'total_pax':'passenger_count'})

    # Aggregate SSR: count of special_service_request per flight
    # First build mapping record_locator -> flight_number, scheduled_date from df_pnr
    pnr_map = df_pnr[['record_locator','flight_number','scheduled_departure_date_local']].drop_duplicates()
    df_ssr_with_flight = df_ssr.merge(pnr_map, on='record_locator', how='left')

    # Only keep rows that successfully merged (have flight info)
    df_ssr_with_flight = df_ssr_with_flight.dropna(subset=['flight_number_y', 'scheduled_departure_date_local'])

    # Use the flight info from the merge (flight_number_y)
    df_ssr_grouped = df_ssr_with_flight.groupby(['flight_number_y','scheduled_departure_date_local']).agg(ssr_count=('special_service_request','count')).reset_index()
    df_ssr_grouped.rename(columns={'flight_number_y': 'flight_number'}, inplace=True)

    # Aggregate bags per flight
    bag_types = df_bag.groupby(['company_id','flight_number','scheduled_departure_date_local'])['bag_type'].value_counts().unstack(fill_value=0).reset_index()
    # normalize column names
    bag_types.columns = [str(c) for c in bag_types.columns]
    # compute transfer and origin counts
    def get_col(df, possible_names):
        for n in possible_names:
            if n in df.columns:
                return df[n]
        return pd.Series(0, index=df.index)

    # possible names: 'Transfer','Hot Transfer','Origin'
    transfer_cols = [c for c in bag_types.columns if c.lower().strip() in ('transfer','hot transfer')]
    origin_cols = [c for c in bag_types.columns if c.lower().strip() in ('origin','checked')]
    bag_types['transfer_count'] = bag_types[[c for c in transfer_cols]].sum(axis=1) if transfer_cols else 0
    bag_types['origin_count'] = bag_types[[c for c in origin_cols]].sum(axis=1) if origin_cols else 0
    bag_types['total_bags'] = bag_types[[c for c in bag_types.columns if c not in ['company_id','flight_number','scheduled_departure_date_local','transfer_count','origin_count','total_bags'] and not c.startswith('0')]].sum(axis=1)

    # merge aggregates into df
    df = df.merge(df_pnr_group, on=['company_id','flight_number','scheduled_departure_date_local'], how='left')
    df = df.merge(df_ssr_grouped, on=['flight_number','scheduled_departure_date_local'], how='left')
    df = df.merge(bag_types[['company_id','flight_number','scheduled_departure_date_local','transfer_count','origin_count','total_bags']], on=['company_id','flight_number','scheduled_departure_date_local'], how='left')

    # fillna
    df['passenger_count'] = df['passenger_count'].fillna(0).astype(int)
    df['ssr_count'] = df['ssr_count'].fillna(0).astype(int)
    df['transfer_count'] = df['transfer_count'].fillna(0).astype(int)
    df['origin_count'] = df['origin_count'].fillna(0).astype(int)
    df['total_bags'] = df['total_bags'].fillna(0).astype(int)

    # compute derived features
    # load percent
    df['total_seats'] = df['total_seats'].fillna(1)
    df['load_pct'] = df['passenger_count'] / df['total_seats']

    # transfer ratio
    df['transfer_to_checked_ratio'] = df['transfer_count'] / (df['origin_count'].replace(0, np.nan))
    df['transfer_to_checked_ratio'] = df['transfer_to_checked_ratio'].fillna(df['transfer_count'])

    # ground pressure (positive => scheduled ground time below minimum)
    df['scheduled_ground_time_minutes'] = df['scheduled_ground_time_minutes'].astype(float)
    df['minimum_turn_minutes'] = df['minimum_turn_minutes'].astype(float)
    df['ground_pressure'] = df['minimum_turn_minutes'] - df['scheduled_ground_time_minutes']

    # delay minutes (actual - scheduled). Use flight level file for datetime diff
    df_delay = df_f[['company_id','flight_number','scheduled_departure_date_local','scheduled_departure_datetime_local','actual_departure_datetime_local']].copy()
    df_delay['delay_minutes'] = (pd.to_datetime(df_delay['actual_departure_datetime_local']) - pd.to_datetime(df_delay['scheduled_departure_datetime_local'])).dt.total_seconds() / 60.0
    df_delay = df_delay[['company_id','flight_number','scheduled_departure_date_local','delay_minutes']]
    df = df.merge(df_delay, on=['company_id','flight_number','scheduled_departure_date_local'], how='left')
    df['delay_minutes'] = df['delay_minutes'].fillna(0)
    df['late_departure'] = (df['delay_minutes']>0).astype(int)

    # SSR per pax
    df['ssr_per_pax'] = df['ssr_count'] / df['passenger_count'].replace(0, np.nan)
    df['ssr_per_pax'] = df['ssr_per_pax'].fillna(0)

    # Advanced feature engineering
    # Add required columns for advanced features
    df['scheduled_departure_time'] = df_f[['company_id','flight_number','scheduled_departure_date_local']].merge(
        df[['company_id','flight_number','scheduled_departure_date_local']],
        on=['company_id','flight_number','scheduled_departure_date_local']
    ).merge(
        df_f[['company_id','flight_number','scheduled_departure_date_local','scheduled_departure_datetime_local']],
        on=['company_id','flight_number','scheduled_departure_date_local'],
        how='left'
    )['scheduled_departure_datetime_local']

    if 'scheduled_departure_time' not in df.columns or df['scheduled_departure_time'].isna().all():
        # Fallback: use flight data directly
        df_temp = df.merge(
            df_f[['company_id','flight_number','scheduled_departure_date_local','scheduled_departure_datetime_local']],
            on=['company_id','flight_number','scheduled_departure_date_local'],
            how='left'
        )
        df['scheduled_departure_time'] = df_temp['scheduled_departure_datetime_local']

    # Add other required columns
    df['destination'] = df['scheduled_arrival_station_code']
    df['aircraft_type'] = df['fleet_type']
    df['load_factor'] = df['load_pct'] * 100
    df['transfer_bag_count'] = df['transfer_count']

    # Basic advanced features
    df = engineer_advanced_features(df)

    # Creative and innovative features
    df = engineer_creative_features(df)

    # define scoring features and normalize per day
    scoring_features = ['delay_minutes','ground_pressure','transfer_to_checked_ratio','ssr_per_pax','load_pct']
    # Cap ground_pressure lower bound at 0 (if scheduled ground >= min, pressure <=0 -> 0 difficulty)
    df['ground_pressure_pos'] = df['ground_pressure'].apply(lambda x: x if x>0 else 0)
    # positive delay only
    df['delay_pos'] = df['delay_minutes'].apply(lambda x: x if x>0 else 0)

    # create a temp df for normalization by date
    df_for_norm = df.copy()
    df_for_norm['scheduled_departure_date_local'] = df_for_norm['scheduled_departure_date_local'].astype(str)

    # Enhanced scoring with advanced AND creative features
    norm_cols = ['delay_pos','ground_pressure_pos','transfer_to_checked_ratio','ssr_per_pax','load_pct',
                'passenger_complexity_score','baggage_complexity_score','operational_pressure_score','master_complexity_index']

    # Ensure all scoring columns exist
    for col in norm_cols:
        if col not in df.columns:
            df[col] = 0

    normed = normalize_by_group(df_for_norm, 'scheduled_departure_date_local', norm_cols)
    for c in norm_cols:
        if c != 'master_complexity_index':  # Master complexity already normalized 0-1
            df[c + '_norm'] = normed[c]

    # Enhanced creative weights incorporating innovative features
    weights = {
        # Traditional factors (40% total)
        'delay_pos_norm': 0.15,  # Flight delays
        'ground_pressure_pos_norm': 0.15,  # Turnaround pressure
        'transfer_to_checked_ratio_norm': 0.05,  # Basic baggage ratio
        'ssr_per_pax_norm': 0.05,  # Basic special services

        # Advanced operational factors (35% total)
        'passenger_complexity_score_norm': 0.12,  # Child, SSR, stroller complexity
        'baggage_complexity_score_norm': 0.08,  # Hot transfers, late bags
        'operational_pressure_score_norm': 0.10,  # Rush hour, international, weekend
        'load_pct_norm': 0.05,  # Basic load factor

        # Creative complexity factors (25% total)
        'master_complexity_index': 0.25  # Ultimate complexity incorporating all creative factors
    }

    # Compute enhanced difficulty score with creative features
    df['difficulty_score'] = (
        # Traditional factors
        df['delay_pos_norm'] * weights['delay_pos_norm']
        + df['ground_pressure_pos_norm'] * weights['ground_pressure_pos_norm']
        + df['transfer_to_checked_ratio_norm'] * weights['transfer_to_checked_ratio_norm']
        + df['ssr_per_pax_norm'] * weights['ssr_per_pax_norm']

        # Advanced operational factors
        + df['passenger_complexity_score_norm'] * weights['passenger_complexity_score_norm']
        + df['baggage_complexity_score_norm'] * weights['baggage_complexity_score_norm']
        + df['operational_pressure_score_norm'] * weights['operational_pressure_score_norm']
        + df['load_pct_norm'] * weights['load_pct_norm']

        # Creative master complexity (already 0-1 normalized)
        + df['master_complexity_index'] * weights['master_complexity_index']
    )

    # ranking and classification per day
    df['rank_within_day'] = df.groupby('scheduled_departure_date_local')['difficulty_score'].rank(method='dense', ascending=False)
    # total per day
    df['count_within_day'] = df.groupby('scheduled_departure_date_local')['difficulty_score'].transform('count')
    # percentile
    df['pct_rank'] = df['rank_within_day'] / df['count_within_day']
    # classify: top 33% Difficult, middle 34% Medium, bottom 33% Easy
    def classify(pct):
        if pct <= 0.33:
            return 'Difficult'
        elif pct <= 0.66:
            return 'Medium'
        else:
            return 'Easy'
    df['difficulty_classification'] = df['pct_rank'].apply(classify)

    # Prepare enhanced output with creative features
    out_cols = [
        'company_id', 'flight_number', 'scheduled_departure_date_local',
        'scheduled_departure_station_code', 'scheduled_arrival_station_code',
        'total_seats', 'fleet_type', 'aircraft_category', 'haul_type',
        'is_international', 'is_express', 'passenger_count', 'load_pct',
        'child_count', 'lap_child_count', 'stroller_count', 'basic_economy_count',
        'transfer_count', 'origin_count', 'transfer_to_checked_ratio',
        'hot_transfer_count', 'late_bag_tags', 'ssr_count', 'ssr_per_pax',
        'scheduled_ground_time_minutes', 'minimum_turn_minutes', 'ground_pressure',
        'delay_minutes', 'late_departure', 'scheduled_departure_hour',
        'flight_duration_hours', 'is_rush_hour_departure', 'is_red_eye',
        'is_weekend', 'season', 'weather_risk_score', 'crew_complexity_score',
        'gate_pressure_score', 'fuel_complexity_score', 'boarding_complexity_score',
        'maintenance_risk_score', 'revenue_complexity_score',
        'competitive_pressure_score', 'security_complexity_score',
        'passenger_complexity_score', 'baggage_complexity_score',
        'operational_pressure_score', 'master_complexity_index',
        'difficulty_score', 'rank_within_day', 'difficulty_classification'
    ]

    # Only include columns that actually exist in df
    out_cols = [col for col in out_cols if col in df.columns]

    out = df[out_cols].copy()
    out['difficulty_score'] = out['difficulty_score'].round(4)

    # Deliverable 1: Exploratory data analysis
    perform_eda(df, df_f)

    # Deliverable 3: Post-analysis and operational insights
    dest_analysis = analyze_destinations(df)
    insights = generate_operational_insights(df, dest_analysis)

    # Save additional analysis outputs
    dest_analysis.to_csv(OUTPUT_DIR / 'destination_difficulty_analysis.csv')

    with open(OUTPUT_DIR / 'operational_insights.json', 'w') as f:
        json.dump(insights, f, indent=2, default=str)

    # ====================================================================
    # FINAL OUTPUT: FLIGHT DIFFICULTY SCORES (DELIVERABLE 2)
    # ====================================================================
    out_file = DATA_DIR / 'test_rahulkumar.csv'
    out.to_csv(out_file, index=False)
    print(f"\nWrote main output to {out_file}")
    print(f"Wrote destination analysis to {OUTPUT_DIR / 'destination_difficulty_analysis.csv'}")
    print(f"Wrote operational insights to {OUTPUT_DIR / 'operational_insights.json'}")

    # Summary of deliverables completion
    print("\n" + "="*60)
    print("DELIVERABLES COMPLETION SUMMARY")
    print("="*60)
    print("1. EDA: Comprehensive analysis performed above")
    print("2. Flight Difficulty Score: Generated with daily ranking & classification")
    print("3. Operational Insights: Destination analysis & recommendations provided")
    print(f"Total flights analyzed: {len(df):,}")
    print(f"Difficult flights identified: {(df['difficulty_classification'] == 'Difficult').sum():,}")
    print(f"Output files generated: 3 (CSV, destination analysis, insights JSON)")

if __name__ == '__main__':
    main()
