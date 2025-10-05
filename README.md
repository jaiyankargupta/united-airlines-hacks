# United Airlines Flight Difficulty Analytics System

A comprehensive data-driven system to calculate Flight Difficulty Scores for United Airlines flights departing from Chicago O'Hare International Airport (ORD). This system quantifies operational complexity to enable proactive resource planning and optimized airport operations.

## Overview

This system addresses the challenge of identifying high-difficulty flights by replacing subjective assessments with a systematic, data-driven scoring framework using historical operational data. The system processes multiple data sources to generate actionable insights for operational teams.

## Features

### Core Analytics
- **Comprehensive Flight Analysis**: 72+ engineered operational features
- **Daily Scoring Algorithm**: Systematic daily-level scoring with normalization
- **Multi-Factor Assessment**: Weather, delays, passenger load, baggage complexity
- **Route Performance Analysis**: Destination-specific difficulty patterns

### Professional Dashboard
- **Executive Summary**: Key performance indicators and metrics
- **Operational Analysis**: Temporal trends and route performance
- **Correlation Analysis**: Factor relationships and dependencies
- **Root Cause Analysis**: High vs normal difficulty comparisons
- **Data Quality Assessment**: Comprehensive data validation reports
- **Export Capabilities**: Professional reports and full datasets

### Streamlined Interface
- **Simple Commands**: Five essential commands for all operations
- **Professional UI**: Clean, business-focused interface
- **Real-time Processing**: Live analysis and visualization
- **Export Options**: Multiple format support for reporting

## System Requirements

### Minimum Specifications
- **OS**: Windows 10, macOS 10.14, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended for large datasets
- **Storage**: 2GB available space
- **Browser**: Modern browser (Chrome, Firefox, Safari, Edge)

### Performance Recommendations
- **RAM**: 16GB+ for optimal performance with large datasets
- **CPU**: Multi-core processor for faster analysis
- **SSD**: Recommended for improved file I/O performance

## Installation

### Quick Start

1. **Download/Clone the Project**
```bash
git clone https://github.com/jaiyankargupta/united-airlines-hacks
cd united-airlines-hacks
```

2. **Automated Setup (Recommended)**
```bash
python install_dependencies.py
```

3. **Start the Application**
```bash
./run_app.sh
```

### Manual Installation

```bash
# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start application
streamlit run app.py --server.port 8501
```

### Dependencies
- **pandas** (≥1.5.0) - Data processing and analysis
- **numpy** (≥1.21.0) - Numerical computations
- **streamlit** (≥1.28.0) - Web application framework
- **plotly** (≥5.15.0) - Interactive visualizations
- **scikit-learn** (≥1.3.0) - Statistical analysis tools

## Data Requirements

### Required Files (5 CSV files)

Place in the `Data/` directory:

1. **Flight Level Data.csv** - Core flight operations data
2. **PNR+Flight+Level+Data.csv** - Passenger and flight combined data
3. **PNR Remark Level Data.csv** - Special service requests and remarks
4. **Bag+Level+Data.csv** - Baggage handling and transfer operations
5. **Airports Data.csv** - Airport reference and metadata

### Analysis Module Details

**Core Processing:**
- `pipeline.py` - **Essential module** containing the main analysis pipeline with data loading, feature engineering, and scoring algorithms
- Functions: `load_and_validate_data()`, `engineer_advanced_features()`, `calculate_flight_difficulty_score()`

**Enhanced Analytics (Optional):**
- `enhanced_analytics.py` - Advanced statistical analysis including correlation matrices and root cause frameworks
- `enhanced_eda.py` - Comprehensive data quality assessment and exploratory analysis
- `enhanced_pipeline.py` - Extended pipeline with additional analytical capabilities

**Note:** The system can operate with just `pipeline.py`. Enhanced modules provide additional analytical depth but are not required for basic operation.

### Directory Structure
```
united-airlines-hacks/
├── Data/                          # Input data files (5 CSV files)
│   ├── Flight Level Data.csv     # Primary flight operations data
│   ├── PNR+Flight+Level+Data.csv # Passenger and flight combined data
│   ├── PNR Remark Level Data.csv # Special service requests and remarks
│   ├── Bag+Level+Data.csv        # Baggage handling and transfer data
│   └── Airports Data.csv         # Airport reference and metadata
├── analysis/                      # Core analysis modules
│   ├── pipeline.py               # Main processing pipeline (required)
│   ├── enhanced_analytics.py     # Advanced correlation & root cause analysis
│   ├── enhanced_eda.py          # Enhanced data quality & statistical analysis
│   ├── enhanced_pipeline.py     # Extended pipeline with comprehensive features
│   ├── __pycache__/             # Python compiled bytecode cache
│   └── output/                   # Generated reports directory
│       ├── destination_difficulty_analysis.csv  # Route-specific insights
│       ├── operational_insights.json           # Actionable recommendations
│       └── streamlit_results.csv              # Complete analysis results
├── cache/                         # Performance optimization cache
│   ├── Flight Level Data.csv.pkl # Processed flight data cache
│   ├── PNR+Flight+Level+Data.csv.pkl # Processed PNR data cache
│   ├── PNR Remark Level Data.csv.pkl # Processed remarks cache
│   ├── Bag+Level+Data.csv.pkl   # Processed baggage data cache
│   └── Airports Data.csv.pkl    # Processed airport data cache
├── data_cache/                   # Data processing cache directory
├── __pycache__/                  # Main application cache
├── .venv/                        # Python virtual environment (if created)
├── app.py                        # Main application interface
├── enhanced_dashboard.py         # Professional analytics dashboard
├── requirements.txt              # Python package dependencies
├── run_app.sh                   # Application startup script (Unix/Mac)
└── README.md                    # Complete system documentation
```

**Key Components:**
- **Required Files**: `app.py`, `enhanced_dashboard.py`, `analysis/pipeline.py`
- **Data Files**: All 5 CSV files in `Data/` directory are required
- **Cache System**: Automatically stores processed data as `.pkl` files for faster subsequent loads
- **Enhanced Modules**: Optional advanced analytics capabilities
- **Output Reports**: Generated automatically during analysis

### Cache System Benefits
The system includes an intelligent caching mechanism that:
- **Speeds up data loading** by 5-10x after first run
- **Stores processed data** in `cache/` directory as pickle files
- **Automatically detects** when source CSV files are updated
- **Reduces memory usage** during repeated analysis
- **Improves user experience** with faster response times

**Cache Management:**
- Cache files are created automatically on first data load
- Delete `cache/` folder to force fresh data processing
- Cache files are safe to delete - they will regenerate as needed

## Using the System

### Step-by-Step Workflow

1. **Launch Application**
   - Run `./run_app.sh` or `streamlit run app.py`
   - Navigate to http://localhost:8501

2. **Upload Data**
   - Go to "Data Upload" page
   - Upload all 5 required CSV files
   - Verify successful data loading

3. **Run Analysis**
   - Navigate to "Command Panel"
   - Execute: `analyze`
   - Monitor progress and completion

4. **Review Results**
   - Visit "Flight Difficulty Analytics Dashboard"
   - Explore all dashboard tabs
   - Generate reports as needed

### Available Commands

The system uses five streamlined commands:

| Command | Description | Usage |
|---------|-------------|--------|
| `analyze` | Run complete flight difficulty analysis pipeline | Primary analysis command |
| `status` | Show system status and data loading progress | Check system state |
| `preview` | Display summary of loaded data files | Verify data before analysis |
| `help` | Show available commands and descriptions | Reference guide |
| `clear` | Clear command history and reset interface | Clean up interface |

### Dashboard Features

#### Executive Summary
- **Key Metrics**: Total flights, average scores, high-risk flights
- **Distribution Analysis**: Score patterns and categorization
- **Performance Indicators**: System-wide operational metrics

#### Operational Analysis
- **Temporal Trends**: Daily and periodic difficulty patterns
- **Route Performance**: Most challenging destinations and routes
- **Factor Importance**: Key operational drivers and correlations

#### Correlation Analysis
- **Feature Relationships**: Statistical correlations with difficulty
- **Segmented Analysis**: Correlations by difficulty levels
- **Impact Assessment**: Quantified factor influences

#### Root Cause Analysis
- **Comparative Analysis**: High vs normal difficulty characteristics
- **Contributing Factors**: Statistical differences and impacts
- **Actionable Recommendations**: Specific improvement suggestions

#### Data Quality Assessment
- **Completeness Metrics**: Missing data and coverage analysis
- **Outlier Detection**: Anomaly identification and reporting
- **Quality Scoring**: Overall data reliability assessment

## Output and Results

### Generated Files

**Primary Deliverable:**
- `test_rahulkumar.csv` - Complete flight analysis with scores

**Detailed Reports:**
- `analysis/output/executive_summary.csv` - Key metrics summary
- `analysis/output/action_plan.csv` - Prioritized recommendations
- `analysis/output/full_analysis_results.csv` - Complete dataset

### Key Metrics

**Difficulty Scoring:**
- Score Range: 0.000 to 1.000
- Classification: Easy (0.0-0.33), Medium (0.34-0.66), Difficult (0.67-1.0)
- Daily normalization for fair temporal comparison

**Operational Insights:**
- Flight complexity assessment across multiple dimensions
- Resource allocation recommendations
- Process improvement opportunities
- Predictive indicators for operational planning

## Troubleshooting

### Common Issues and Solutions

**Installation Problems:**
```bash
# Update pip and try again
python -m pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

**Port Conflicts:**
```bash
# Use alternative port
streamlit run app.py --server.port 8502
```


**Data Loading Errors:**
- Verify all 5 CSV files are present in `Data/` directory
- Check file names match exactly (case-sensitive)
- Ensure files are not corrupted or locked

## Technical Architecture

### System Components
- **Analysis Pipeline**: Core processing and feature engineering
- **Web Interface**: Streamlit-based professional dashboard
- **Data Processing**: Pandas/NumPy numerical operations
- **Visualizations**: Plotly interactive charts and graphs
- **Export System**: Multi-format report generation

### Scalability Features
- **Modular Design**: Extensible for additional airports/hubs
- **Configurable Parameters**: Adaptable scoring weights
- **API-Ready**: Framework suitable for system integration
- **Performance Optimized**: Efficient processing for large datasets


**United Airlines Flight Difficulty Analytics System** provides a comprehensive, data-driven approach to operational complexity assessment. The system enables proactive resource management, improved operational efficiency, and data-informed decision-making for flight operations at ORD and other hub airports.

For technical questions or operational support, refer to the troubleshooting section above or consult the system documentation within the application interface.
