import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import io
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress numpy divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

def safe_correlation(x, y):
    """Calculate correlation safely, handling NaN and constant values"""
    try:
        corr = x.corr(y)
        if pd.isna(corr) or np.isinf(corr):
            return 0.0
        return corr
    except:
        return 0.0

# Add the analysis directory to the path so we can import our pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

# Create data directory for persistence
DATA_CACHE_DIR = "data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Import our analysis functions
try:
    from analysis.pipeline import (
        load_and_validate_data,
        engineer_advanced_features,
        engineer_creative_features,
        calculate_flight_difficulty_score
    )
except ImportError as e:
    st.error(f"Error importing pipeline functions: {e}")
    st.stop()

# Page configuration
# Configure Streamlit page settings for optimal user experience
st.set_page_config(
    page_title="United Airlines Flight Difficulty Score System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0070ba, #005a9a);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .command-panel {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }

    .upload-section {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #0070ba;
        margin: 1rem 0;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0070ba;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .status-success {
        color: #28a745;
        font-weight: bold;
    }

    .status-error {
        color: #dc3545;
        font-weight: bold;
    }

    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def save_cached_data(data_dict):
    """Save uploaded data to cache directory"""
    try:
        import os
        import pickle
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)

        for filename, df in data_dict.items():
            cache_path = os.path.join(cache_dir, f"{filename}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        return True
    except Exception as e:
        st.error(f"Failed to cache data: {str(e)}")
        return False

def load_cached_data():
    """Load data from cache directory"""
    try:
        import os
        import pickle
        cache_dir = "cache"
        cached_files = {}

        if not os.path.exists(cache_dir):
            return {}

        for filename in os.listdir(cache_dir):
            if filename.endswith('.pkl'):
                original_name = filename.replace('.pkl', '')
                cache_path = os.path.join(cache_dir, filename)
                with open(cache_path, 'rb') as f:
                    cached_files[original_name] = pickle.load(f)
        return cached_files
    except Exception:
        return {}

def auto_load_data():
    """Auto-load data files from Data directory"""
    try:
        data_files = {
            "Flight Level Data.csv": "Data/Flight Level Data.csv",
            "PNR+Flight+Level+Data.csv": "Data/PNR+Flight+Level+Data.csv",
            "PNR Remark Level Data.csv": "Data/PNR Remark Level Data.csv",
            "Bag+Level+Data.csv": "Data/Bag+Level+Data.csv",
            "Airports Data.csv": "Data/Airports Data.csv"
        }

        loaded_files = {}
        for filename, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    loaded_files[filename] = df
                except Exception:
                    st.error("Failed to upload file: Please check file format")

        return loaded_files
    except Exception:
        return {}

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'command_history' not in st.session_state:
    st.session_state.command_history = []

# Auto-load data on first run
if not st.session_state.data_loaded and len(st.session_state.uploaded_files) == 0:
    # Try to load from cache first
    cached_data = load_cached_data()
    if len(cached_data) == 5:
        st.session_state.uploaded_files = cached_data
        st.session_state.data_loaded = True
    else:
        # Try to auto-load from Data directory
        auto_data = auto_load_data()
        if len(auto_data) == 5:
            st.session_state.uploaded_files = auto_data
            st.session_state.data_loaded = True
            save_cached_data(auto_data)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>United Airlines Flight Difficulty Score System</h1>
        <p>Analytics Platform for ORD Operations Optimization</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        page = st.selectbox(
            "Navigate to:",
            ["Home", "Data Upload", "Command Panel", "Analytics Dashboard", "Results Export"]
        )

    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_upload_page()
    elif page == "Command Panel":
        show_command_panel()
    elif page == "Analytics Dashboard":
        show_dashboard()
    elif page == "Results Export":
        show_export_page()

def show_home_page():
    st.header("United Airlines Flight Difficulty Score System")

    # Show auto-load status
    if st.session_state.data_loaded and len(st.session_state.uploaded_files) == 5:
        st.success("Data files automatically loaded from /Data directory")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; color: #FFFFFF;">
            <h3 style="color: #FFD700;">Project Overview</h3>
            <p>Flight difficulty scoring system for United Airlines operations at Chicago O'Hare Airport.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; color: #FFFFFF;">
            <h3 style="color: #FFD700;">Key Features</h3>
            <ul>
                <li>72 Engineered Features</li>
                <li>Daily Scoring Algorithm</li>
                <li>Weather Impact Modeling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; color: #FFFFFF;">
            <h3 style="color: #FFD700;">Business Benefits</h3>
            <ul>
                <li>Resource Planning</li>
                <li>Delay Reduction</li>
                <li>Improved Operations</li>
                <li>Cost Savings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System Status
    st.subheader("System Status Overview")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        data_status = "Ready" if st.session_state.data_loaded else "Pending"
        st.metric("Data Loading Status", data_status)

    with status_col2:
        analysis_status = "Complete" if st.session_state.analysis_complete else "Pending"
        st.metric("Analysis Progress", analysis_status)

    with status_col3:
        files_count = len(st.session_state.uploaded_files)
        st.metric("Files Uploaded", f"{files_count}/5")

    # Quick Start Guide
    st.markdown("---")
    st.subheader("Quick Start Guide")

    st.markdown("""
    <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; color: #FFFFFF;">
        <ol>
            <li><strong style="color: #FFD700;">Upload Data</strong>: Upload the 5 required CSV files</li>
            <li><strong style="color: #FFD700;">Run Analysis</strong>: Execute analysis using Command Panel</li>
            <li><strong style="color: #FFD700;">View Results</strong>: Review insights in Analytics Dashboard</li>
            <li><strong style="color: #FFD700;">Export</strong>: Download results from Export page</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def show_upload_page():
    st.header("Data Upload Center")

    # Show current data status
    if st.session_state.data_loaded and len(st.session_state.uploaded_files) == 5:
        st.success("All required data files are loaded and ready for analysis!")
        st.info("Data will persist during your session. Files are auto-loaded from /Data directory.")

    st.markdown("""
    <div class="upload-section">
        <h3>Required Dataset Files</h3>
        <p>Upload all 5 CSV files to enable the complete analysis pipeline:</p>
    </div>
    """, unsafe_allow_html=True)

    # File upload requirements
    required_files = {
        "Flight Level Data.csv": "Primary flight operations data",
        "PNR+Flight+Level+Data.csv": "Passenger and flight combined data",
        "PNR Remark Level Data.csv": "Special service requests and remarks",
        "Bag+Level+Data.csv": "Baggage handling and transfer data",
        "Airports Data.csv": "Airport reference data and metadata"
    }

    uploaded_count = 0

    for filename, description in required_files.items():
        st.markdown(f"**{filename}**")
        st.caption(description)

        uploaded_file = st.file_uploader(
            f"Choose {filename}",
            type=['csv'],
            key=f"upload_{filename}",
            help=f"Upload the {filename} file"
        )

        if uploaded_file is not None:
            # Validate and store the file
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_files[filename] = df
                uploaded_count += 1

                # Save to cache for persistence
                save_cached_data(st.session_state.uploaded_files)

                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"Loaded: {len(df):,} rows")
                with col2:
                    st.info(f"{len(df.columns)} columns")
                with col3:
                    st.info(f"{uploaded_file.size / 1024:.1f} KB")

                # Preview data
                with st.expander(f"Preview {filename}"):
                    st.dataframe(df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading {filename}: {str(e)}")

        st.markdown("---")

    # Update data loaded status
    if uploaded_count == 5 or len(st.session_state.uploaded_files) == 5:
        st.session_state.data_loaded = True
        st.success("All files uploaded successfully! You can now proceed to the Command Panel.")

        # Show summary
        st.subheader("Upload Summary")
        summary_data = []
        for filename, df in st.session_state.uploaded_files.items():
            summary_data.append({
                "File": filename,
                "Rows": f"{len(df):,}",
                "Columns": len(df.columns),
                "Size (KB)": f"{sys.getsizeof(df) / 1024:.1f}"
            })

        st.table(pd.DataFrame(summary_data))

        # Add clear cache option
        if st.button("Clear Cached Data"):
            try:
                import shutil
                shutil.rmtree(DATA_CACHE_DIR)
                os.makedirs(DATA_CACHE_DIR, exist_ok=True)
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("All cached data and session state cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")

def show_command_panel():
    st.header("Flight Difficulty Analysis Command Panel")

    if not st.session_state.data_loaded:
        st.error("Please upload all required data files first!")
        st.info("You can upload files manually or place them in the /Data directory for auto-loading.")
        return

    # Command panel styling
    st.markdown("""
    <div class="command-panel">
        <h4>Analytics Terminal</h4>
        <p>Enter commands to run analysis operations</p>
    </div>
    """, unsafe_allow_html=True)

    # Available commands
    st.subheader("Available Commands")

    commands_help = {
        "analyze": "Run complete flight difficulty analysis pipeline",
        "status": "Show analysis status and statistics",
        "help": "Show this help information",
        "clear": "Clear command history",
        "preview": "Preview loaded data summaries"
    }

    # Display commands in columns
    col1, col2 = st.columns(2)

    with col1:
        for cmd, desc in list(commands_help.items())[:4]:
            st.code(f"{cmd}: {desc}")

    with col2:
        for cmd, desc in list(commands_help.items())[4:]:
            st.code(f"{cmd}: {desc}")

    # Command input
    st.markdown("---")
    command = st.text_input("Enter command:", placeholder="Type 'analyze' to start full analysis", key="command_input")

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        execute_btn = st.button("Execute", type="primary")

    with col2:
        clear_btn = st.button("Clear History")

    if clear_btn:
        st.session_state.command_history = []
        st.rerun()

    # Command execution
    if execute_btn and command:
        execute_command(command.strip().lower())

    # Command history
    if st.session_state.command_history:
        st.subheader("Command History")

        for i, (cmd, timestamp, status, output) in enumerate(reversed(st.session_state.command_history[-10:])):
            with st.expander(f"{timestamp} - {cmd} ({status})"):
                if status == "SUCCESS":
                    st.success(output)
                elif status == "ERROR":
                    st.error(output)
                else:
                    st.info(output)

def execute_command(command):
    """Execute the given command and update history"""
    timestamp = datetime.now().strftime("%H:%M:%S")

    try:
        if command == "help":
            output = """
Available Commands:
- analyze: Run full analysis
- status: Show status
- preview: Preview data
- clear: Clear history
            """
            add_to_history(command, timestamp, "INFO", output.strip())

        elif command == "status":
            files_loaded = len(st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
            analysis_done = st.session_state.analysis_complete
            results_ready = st.session_state.results_df is not None

            output = f"""
System Status:
- Files loaded: {files_loaded}
- Analysis complete: {analysis_done}
- Results ready: {results_ready}
            """
            add_to_history(command, timestamp, "INFO", output.strip())

        elif command == "preview":
            if st.session_state.uploaded_files:
                output = "Data Preview:\n"
                for filename, df in st.session_state.uploaded_files.items():
                    output += f"{filename}: {len(df):,} rows x {len(df.columns)} columns\n"
                add_to_history(command, timestamp, "SUCCESS", output.strip())
            else:
                add_to_history(command, timestamp, "ERROR", "No data files loaded")

        elif command == "analyze":
            with st.spinner("Running complete analysis pipeline..."):
                run_full_analysis()
            add_to_history(command, timestamp, "SUCCESS", "Complete analysis pipeline executed successfully")

        elif command == "clear":
            st.session_state.command_history = []
            add_to_history(command, timestamp, "SUCCESS", "Command history cleared")

        else:
            add_to_history(command, timestamp, "ERROR", f"Unknown command: {command}. Type 'help' for available commands")

    except Exception as e:
        add_to_history(command, timestamp, "ERROR", f"Command execution failed: {str(e)}")

    st.rerun()



def add_to_history(command, timestamp, status, output):
    """Add command to history"""
    st.session_state.command_history.append((command, timestamp, status, output))



def run_full_analysis():
    """Run the complete analysis pipeline"""
    try:
        # Get data from session state
        flight_df = st.session_state.uploaded_files.get("Flight Level Data.csv")
        pnr_flight_df = st.session_state.uploaded_files.get("PNR+Flight+Level+Data.csv")
        pnr_remark_df = st.session_state.uploaded_files.get("PNR Remark Level Data.csv")
        bag_df = st.session_state.uploaded_files.get("Bag+Level+Data.csv")
        airports_df = st.session_state.uploaded_files.get("Airports Data.csv")

        # Validate we have all required data
        if not all([flight_df is not None, pnr_flight_df is not None, pnr_remark_df is not None, bag_df is not None]):
            st.error("Missing required data files. Please upload all 5 CSV files.")
            return False

        # Step 1: Load and validate data
        with st.spinner("Loading and validating data..."):
            merged_df = load_and_validate_data(flight_df, pnr_flight_df, pnr_remark_df, bag_df, airports_df)
            st.success(f"Data loaded: {len(merged_df):,} flights with {len(merged_df.columns)} columns")

        # Step 2: Engineer features
        try:
            with st.spinner("Engineering advanced features..."):
                merged_df = engineer_advanced_features(merged_df)
                merged_df = engineer_creative_features(merged_df)
                st.success("Feature engineering completed")
        except Exception as e:
            st.warning(f"Feature engineering failed: {str(e)} - Proceeding with basic features")

        # Step 3: Calculate scores
        with st.spinner("Calculating difficulty scores..."):
            results_df = calculate_flight_difficulty_score(merged_df)
            st.success(f"Difficulty scores calculated for {len(results_df):,} flights")

        # Store results with validation
        if results_df is not None and len(results_df) > 0:
            st.session_state.results_df = results_df
            st.session_state.analysis_complete = True
            st.success("Analysis complete! Results stored in session state.")

            # Create required CSV file for Rahul Kumar
            create_test_csv(results_df, "rahulkumar")

            # Debug info
            score_range = f"{results_df['flight_difficulty_score'].min():.3f} - {results_df['flight_difficulty_score'].max():.3f}"
            st.info(f"Score range: {score_range}")
            return True
        else:
            st.error("Failed to generate results DataFrame")
            return False

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return False



def show_dashboard():
    st.header("Analytics Dashboard")

    # Add enhanced dashboard option
    dashboard_type = st.radio(
        "Choose Dashboard Type:",
        ["Standard Dashboard", "Enhanced Dashboard (Advanced Analysis)"],
        help="Enhanced dashboard provides comprehensive insights with advanced analytical capabilities"
    )

    if dashboard_type == "Enhanced Dashboard (Advanced Analysis)":
        try:
            # Setup path for enhanced dashboard import
            analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis')
            if analysis_dir not in sys.path:
                sys.path.insert(0, analysis_dir)

            # Import with fallback handling
            try:
                from enhanced_dashboard import EnhancedDashboard
            except ImportError:
                import importlib.util
                spec = importlib.util.spec_from_file_location("enhanced_dashboard",
                                                            os.path.join(analysis_dir, "enhanced_dashboard.py"))
                if spec and spec.loader:
                    enhanced_dashboard = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(enhanced_dashboard)
                    EnhancedDashboard = enhanced_dashboard.EnhancedDashboard
                else:
                    raise ImportError("Could not load enhanced_dashboard module")

            enhanced_dash = EnhancedDashboard()
            enhanced_dash.show_enhanced_dashboard()
            return

        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            st.error(f"Enhanced dashboard not available: {str(e)}")
            st.info("Falling back to standard dashboard...")
        except Exception as e:
            st.error(f"Error loading enhanced dashboard: {str(e)}")
            st.info("Falling back to standard dashboard...")

    # Check if analysis has been completed and results are available
    if not st.session_state.analysis_complete or st.session_state.results_df is None:
        st.warning("Please run the analysis first using the Command Panel!")
        if st.session_state.data_loaded:
            st.info("Data is loaded. Run the 'analyze' command in the Command Panel to generate results.")
        else:
            st.info("Upload data files first, then run the 'analyze' command.")
        return

    df = st.session_state.results_df

    # Validate that the DataFrame contains the required columns for analysis
    required_columns = ['flight_difficulty_score']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"DataFrame missing required columns: {missing_cols}")
        st.error("Please re-run the analysis to fix data structure issues.")
        if st.button("Clear Session and Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        return

    st.subheader("Key Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_flights = len(df)
        avg_daily_flights = total_flights / df['scheduled_departure_date_local'].nunique() if 'scheduled_departure_date_local' in df.columns else total_flights
        st.metric(
            "Total Flights",
            f"{total_flights:,}",
            f"{avg_daily_flights:.0f} per day"
        )

    with col2:
        on_time_rate = len(df[df['delay_minutes'] <= 0]) / len(df) * 100
        delayed_flights = len(df[df['delay_minutes'] > 0])
        st.metric(
            "On-Time Performance",
            f"{on_time_rate:.1f}%",
            f"{delayed_flights:,} delayed"
        )

    with col3:
        difficult_flights = len(df[df['difficulty_classification'] == 'Difficult']) if 'difficulty_classification' in df.columns else len(df[df['flight_difficulty_score'] > 0.3])
        difficult_pct = (difficult_flights / len(df)) * 100
        st.metric(
            "High Difficulty Flights",
            f"{difficult_flights:,}",
            f"{difficult_pct:.1f}% of total"
        )

    with col4:
        avg_delay = df['delay_minutes'].mean()
        total_delay_hours = df['delay_minutes'].sum() / 60
        st.metric(
            "Average Delay",
            f"{avg_delay:.1f} min",
            f"{total_delay_hours:.0f}h total"
        )

    st.markdown("---")
    st.subheader("Flight Difficulty Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Difficulty Score Histogram
        fig = px.histogram(
            df,
            x='flight_difficulty_score',
            nbins=50,
            title="Distribution of Flight Difficulty Scores",
            labels={'flight_difficulty_score': 'Difficulty Score', 'count': 'Number of Flights'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Difficulty Classification Pie Chart
        if 'difficulty_classification' in df.columns:
            difficulty_counts = df['difficulty_classification'].value_counts()
            fig = px.pie(
                values=difficulty_counts.values,
                names=difficulty_counts.index,
                title="Flight Classification Distribution",
                color_discrete_map={'Difficult': '#ff6b6b', 'Medium': '#feca57', 'Easy': '#48ca8b'}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        delay_data = df['delay_minutes'].copy()
        delay_categories = ['On-Time', 'Minor Delay (1-15 min)', 'Moderate Delay (15-30 min)', 'Major Delay (>30 min)']
        delay_counts = [
            (delay_data <= 0).sum(),
            ((delay_data > 0) & (delay_data <= 15)).sum(),
            ((delay_data > 15) & (delay_data <= 30)).sum(),
            (delay_data > 30).sum()
        ]

        fig = px.bar(
            x=delay_categories,
            y=delay_counts,
            title="Flight Delay Distribution Analysis",
            labels={'x': 'Delay Category', 'y': 'Number of Flights'},
            color=delay_counts,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        avg_delay = df['delay_minutes'].mean()
        pct_late = (df['delay_minutes'] > 0).mean() * 100
        st.write(f"Average delay: {avg_delay:.1f} minutes, {pct_late:.1f}% flights depart late")

    with col2:
        if 'ground_pressure' in df.columns:
            ground_pressure = df['ground_pressure'].copy()
            ground_categories = ['Adequate Buffer', 'Tight (0-10 min)', 'Critical (<0 min)']
            ground_counts = [
                (ground_pressure <= -10).sum(),
                ((ground_pressure > -10) & (ground_pressure <= 0)).sum(),
                (ground_pressure > 0).sum()
            ]

            fig = px.bar(
                x=ground_categories,
                y=ground_counts,
                title="Ground Time Constraint Analysis",
                labels={'x': 'Ground Time Category', 'y': 'Number of Flights'},
                color=ground_counts,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            tight_turns = (df['ground_pressure'] > 0).sum()
            pct_tight = (tight_turns / len(df)) * 100
            st.write(f"{tight_turns:,} flights ({pct_tight:.1f}%) operate at/below minimum turn time")

    # Top Difficult Destinations
    st.markdown("---")
    st.subheader("Most Challenging Destinations")

    dest_analysis = None
    try:
        # Check what destination column is available
        if 'destination' in df.columns:
            dest_col = 'destination'
        elif 'scheduled_arrival_station_code' in df.columns:
            dest_col = 'scheduled_arrival_station_code'
        else:
            st.error("No destination column found in the data.")
            return

        # Create destination analysis
        agg_dict = {
            'flight_difficulty_score': ['mean', 'count']
        }

        if 'delay_minutes' in df.columns:
            agg_dict['delay_minutes'] = ['mean']
            dest_analysis = df.groupby(dest_col).agg(agg_dict).round(3)
            dest_analysis.columns = ['Avg_Difficulty', 'Flight_Count', 'Avg_Delay']
        else:
            dest_analysis = df.groupby(dest_col).agg(agg_dict).round(3)
            dest_analysis.columns = ['Avg_Difficulty', 'Flight_Count']

        dest_analysis = dest_analysis[dest_analysis['Flight_Count'] >= 10].sort_values('Avg_Difficulty', ascending=False).head(10)
        st.dataframe(dest_analysis, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing destination analysis: {str(e)}")

    # Summary Statistics
    st.markdown("---")
    st.subheader("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        most_difficult_dest = "N/A"
        flights_count = ""
        if dest_analysis is not None and len(dest_analysis) > 0:
            most_difficult_dest = dest_analysis.index[0]
            flights_count = f"{dest_analysis.iloc[0]['Flight_Count']} flights"

        st.metric(
            "Most Difficult Destination",
            most_difficult_dest,
            flights_count
        )

    with col2:
        high_diff_count = len(df[df['flight_difficulty_score'] > 0.3])
        high_diff_pct = (high_diff_count / len(df)) * 100
        st.metric(
            "High Difficulty Flights",
            f"{high_diff_count:,}",
            f"{high_diff_pct:.1f}% of total"
        )

    with col3:
        difficult_flights = df[df['flight_difficulty_score'] > 0.3]
        avg_delay_difficult = difficult_flights['delay_minutes'].mean() if len(difficult_flights) > 0 else 0
        st.metric(
            "Avg Delay (Difficult Flights)",
            f"{avg_delay_difficult:.1f} min",
            "vs 4.2 min overall"
        )

    with col4:
        tight_turns = (df['ground_pressure'] > 0).sum() if 'ground_pressure' in df.columns else 0
        tight_turns_pct = (tight_turns / len(df)) * 100 if len(df) > 0 else 0
        st.metric(
            "Tight Turnarounds",
            f"{tight_turns:,}",
            f"{tight_turns_pct:.1f}% of flights"
        )



def show_export_page():
    st.header("Export Results")

    if not st.session_state.analysis_complete or st.session_state.results_df is None:
        st.warning("No analysis results available. Please run the analysis first!")
        return

    df = st.session_state.results_df

    # Export options
    st.subheader("Download Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Complete Results Dataset**")
        st.caption(f"All {len(df):,} flights with {len(df.columns)} features")

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download Full Results (CSV)",
            data=csv_data,
            file_name=f"flight_difficulty_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        high_diff_df = df[df['flight_difficulty_score'] > 0.3]
        st.markdown("**High Difficulty Flights Only**")
        st.caption(f"{len(high_diff_df):,} flights above 0.3 difficulty score")

        if len(high_diff_df) > 0:
            csv_buffer_high = io.StringIO()
            high_diff_df.to_csv(csv_buffer_high, index=False)
            csv_data_high = csv_buffer_high.getvalue()

            st.download_button(
                label="Download High Difficulty (CSV)",
                data=csv_data_high,
                file_name=f"high_difficulty_flights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Summary Report
    st.markdown("---")
    st.subheader("Summary Report")

    def get_date_range(df):
        try:
            date_col = 'departure_date' if 'departure_date' in df.columns else 'scheduled_departure_date_local'
            if date_col in df.columns:
                return f"{df[date_col].min()} to {df[date_col].max()}"
            return "N/A"
        except:
            return "N/A"

    def get_most_difficult_destination(df):
        try:
            dest_col = 'destination' if 'destination' in df.columns else 'scheduled_arrival_station_code'
            if dest_col in df.columns:
                return df.groupby(dest_col)['flight_difficulty_score'].mean().idxmax()
            return "N/A"
        except:
            return "N/A"

    summary_stats = {
        "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Flights Analyzed": f"{len(df):,}",
        "Average Difficulty Score": f"{df['flight_difficulty_score'].mean():.4f}",
        "High Difficulty Flights (>0.3)": f"{len(df[df['flight_difficulty_score'] > 0.3]):,}",
        "Medium Difficulty (0.15-0.3)": f"{len(df[(df['flight_difficulty_score'] >= 0.15) & (df['flight_difficulty_score'] <= 0.3)]):,}",
        "Low Difficulty (<0.15)": f"{len(df[df['flight_difficulty_score'] < 0.15]):,}",
        "Features Engineered": f"{len(df.columns)}",
        "Most Difficult Destination": get_most_difficult_destination(df)
    }

    # Display summary
    for key, value in summary_stats.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(value)

    # Export summary as JSON
    summary_json = json.dumps(summary_stats, indent=2)
    st.download_button(
        label="Download Summary Report (JSON)",
        data=summary_json,
        file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    # Data preview
    st.markdown("---")
    st.subheader("Results Preview")

    # Show top difficult flights
    preview_columns = ['flight_number', 'flight_difficulty_score', 'difficulty_classification']

    # Add available columns
    if 'destination' in df.columns:
        preview_columns.insert(1, 'destination')
    elif 'scheduled_arrival_station_code' in df.columns:
        preview_columns.insert(1, 'scheduled_arrival_station_code')

    if 'scheduled_departure_date_local' in df.columns:
        preview_columns.insert(2, 'scheduled_departure_date_local')

    # Filter to only existing columns
    preview_columns = [col for col in preview_columns if col in df.columns]

    top_difficult = df.nlargest(10, 'flight_difficulty_score')[preview_columns]

    st.markdown("**Top 10 Most Difficult Flights**")
    st.dataframe(top_difficult, use_container_width=True)

def create_test_csv(df, name):
    """Create required test CSV file with flight details and difficulty scores"""

    # Select key columns for the test file
    output_columns = [
        'company_id', 'flight_number', 'scheduled_departure_date_local',
        'scheduled_departure_station_code', 'scheduled_arrival_station_code',
        'passenger_count', 'load_pct', 'delay_minutes', 'ground_pressure',
        'transfer_to_checked_ratio', 'ssr_per_pax',
        'passenger_complexity_score', 'baggage_complexity_score',
        'operational_pressure_score', 'master_complexity_index',
        'flight_difficulty_score', 'daily_rank', 'difficulty_classification'
    ]

    # Filter to only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    output_df = df[available_columns].copy()

    # Sort by difficulty score descending
    output_df = output_df.sort_values('flight_difficulty_score', ascending=False)

    # Save the file
    filename = f"test_{name}.csv"
    output_df.to_csv(filename, index=False)

    st.success(f"Created required test file: {filename}")
    st.info(f"File contains {len(output_df):,} flights with {len(available_columns)} features")

if __name__ == "__main__":
    main()
