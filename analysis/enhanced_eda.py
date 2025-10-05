import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EnhancedEDA:
    """
    Enhanced Exploratory Data Analysis module inspired by customer satisfaction analysis
    Provides comprehensive EDA capabilities with advanced visualizations and statistical insights
    """

    def __init__(self):
        self.data_profile = {}
        self.outlier_info = {}
        self.missing_data_info = {}

    def comprehensive_data_profiling(self, df, target_col=None):
        """
        Comprehensive data profiling with statistical analysis approach
        Analyzes data structure, quality, and statistical properties
        """
        profile = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': df.dtypes.to_dict()
            },
            'missing_data': {},
            'numeric_summary': {},
            'categorical_summary': {},
            'outliers': {},
            'correlations': {}
        }

        # Missing data analysis
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        profile['missing_data'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_pct.to_dict(),
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            profile['numeric_summary'][col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': stats.skew(df[col].dropna()),
                'kurtosis': stats.kurtosis(df[col].dropna()),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            profile['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_counts': value_counts.head(10).to_dict()
            }

        # Outlier detection using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            profile['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers.tolist() if len(outliers) < 50 else outliers.head(50).tolist()
            }

        # Correlation analysis
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            profile['correlations']['high_correlations'] = high_corr_pairs

        # Target variable analysis if provided
        if target_col and target_col in df.columns:
            profile['target_analysis'] = self._analyze_target_variable(df, target_col)

        self.data_profile = profile
        return profile

    def _analyze_target_variable(self, df, target_col):
        """
        Analyze target variable characteristics and distribution properties
        Determines if target is numeric or categorical and computes relevant statistics
        """
        target_analysis = {}

        if df[target_col].dtype in ['int64', 'float64']:
            target_analysis = {
                'type': 'numeric',
                'distribution': {
                    'mean': df[target_col].mean(),
                    'median': df[target_col].median(),
                    'std': df[target_col].std(),
                    'skewness': stats.skew(df[target_col].dropna()),
                    'normality_test': stats.jarque_bera(df[target_col].dropna())
                }
            }
        else:
            value_counts = df[target_col].value_counts()
            target_analysis = {
                'type': 'categorical',
                'distribution': {
                    'unique_classes': df[target_col].nunique(),
                    'class_balance': (value_counts / len(df)).to_dict(),
                    'most_common': value_counts.index[0],
                    'least_common': value_counts.index[-1]
                }
            }

        return target_analysis

    def create_missing_data_visualization(self, df):
        """
        Create comprehensive missing data visualization
        Shows percentage of missing values for each column with missing data
        """
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        # Only show columns with missing data
        missing_data = missing_pct[missing_pct > 0].sort_values(ascending=True)

        if len(missing_data) == 0:
            return None

        fig = go.Figure()

        # Bar chart of missing percentages
        fig.add_trace(go.Bar(
            y=missing_data.index,
            x=missing_data.values,
            orientation='h',
            marker_color='lightcoral',
            text=[f'{x:.1f}%' for x in missing_data.values],
            textposition='auto'
        ))

        fig.update_layout(
            title='Missing Data Analysis',
            xaxis_title='Percentage Missing (%)',
            yaxis_title='Columns',
            height=max(400, len(missing_data) * 30)
        )

        return fig

    def create_distribution_analysis(self, df, target_col=None):
        """
        Create distribution analysis for numeric columns
        Generates histograms to visualize the shape and spread of numeric variables
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return None

        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.1
        )

        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1

            # Add histogram
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                row=row, col=col_pos
            )

        fig.update_layout(
            title='Distribution Analysis - Numeric Variables',
            height=n_rows * 300
        )

        return fig

    def create_categorical_analysis(self, df, max_categories=10):
        """
        Create analysis for categorical variables
        Shows frequency distributions of categorical values using horizontal bar charts
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(categorical_cols) == 0:
            return None

        figures = []

        for col in categorical_cols[:4]:  # Limit to first 4 categorical columns
            value_counts = df[col].value_counts().head(max_categories)

            fig = go.Figure(data=[
                go.Bar(x=value_counts.values, y=value_counts.index, orientation='h')
            ])

            fig.update_layout(
                title=f'Distribution of {col}',
                xaxis_title='Count',
                yaxis_title=col,
                height=max(400, len(value_counts) * 30)
            )

            figures.append(fig)

        return figures

    def create_correlation_heatmap(self, df, method='pearson'):
        """
        Enhanced correlation heatmap with statistical significance testing
        Displays correlations with asterisks indicating statistical significance
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)

        # Calculate p-values for correlations
        p_values = pd.DataFrame(np.zeros_like(corr_matrix),
                               columns=corr_matrix.columns,
                               index=corr_matrix.index)

        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                if i != j:
                    try:
                        _, p_val = stats.pearsonr(df[numeric_cols[i]].dropna(),
                                                df[numeric_cols[j]].dropna())
                        p_values.iloc[i, j] = p_val
                    except:
                        p_values.iloc[i, j] = 1.0

        # Create significance mask
        significance_mask = p_values < 0.05

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=[[f'{corr_matrix.iloc[i, j]:.2f}{"*" if significance_mask.iloc[i, j] else ""}'
                  for j in range(len(corr_matrix.columns))]
                 for i in range(len(corr_matrix.index))],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Correlation Matrix ({method.title()}) - * indicates p<0.05',
            width=800,
            height=800
        )

        return fig

    def outlier_analysis_comprehensive(self, df, methods=['iqr', 'zscore']):
        """
        Comprehensive outlier analysis using multiple statistical methods
        Combines IQR and Z-score methods to identify potentially problematic data points
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_summary = {}

        for col in numeric_cols:
            outlier_summary[col] = {}

            # IQR method
            if 'iqr' in methods:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_summary[col]['iqr'] = {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(df)) * 100,
                    'indices': iqr_outliers.index.tolist()
                }

            # Z-score method
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                zscore_outliers = df[col].dropna()[z_scores > 3]
                outlier_summary[col]['zscore'] = {
                    'count': len(zscore_outliers),
                    'percentage': (len(zscore_outliers) / len(df)) * 100,
                    'indices': zscore_outliers.index.tolist()
                }

        self.outlier_info = outlier_summary
        return outlier_summary

    def create_outlier_visualization(self, df, columns=None):
        """
        Create box plots for outlier visualization
        Shows distribution quartiles and highlights outlier points for easy identification
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        figures = []

        # Create box plots
        for col in columns[:6]:  # Limit to 6 columns for readability
            fig = go.Figure()

            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))

            fig.update_layout(
                title=f'Outlier Analysis - {col}',
                yaxis_title=col,
                height=400
            )

            figures.append(fig)

        return figures

    def advanced_feature_relationships(self, df, target_col=None):
        """
        Analyze advanced feature relationships using scatter plot matrices
        Examines pairwise relationships between the most important variables
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        # Create scatter plot matrix for top correlated features
        if target_col and target_col in numeric_cols:
            # Find top correlated features with target
            correlations = []
            for col in numeric_cols:
                if col != target_col:
                    corr = df[col].corr(df[target_col])
                    if not np.isnan(corr):
                        correlations.append((col, abs(corr)))

            # Sort by correlation and take top 4
            top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:4]
            feature_cols = [target_col] + [feat[0] for feat in top_features]
        else:
            # Use first 5 numeric columns
            feature_cols = numeric_cols[:5]

        # Create scatter plot matrix
        fig = ff.create_scatterplotmatrix(
            df[feature_cols].dropna(),
            diag='histogram',
            height=800, width=800
        )

        fig.update_layout(title='Feature Relationship Matrix')

        return fig

    def segment_based_analysis(self, df, segment_col, target_col=None):
        """
        Perform segment-based analysis similar to operational hub-spoke analysis
        Compares statistical properties across different operational segments
        """
        if segment_col not in df.columns:
            return None

        segments = df[segment_col].unique()
        analysis_results = {}

        for segment in segments:
            segment_data = df[df[segment_col] == segment]

            segment_analysis = {
                'count': len(segment_data),
                'percentage': (len(segment_data) / len(df)) * 100
            }

            # Numeric column analysis by segment
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in numeric_cols:
                segment_analysis['target_stats'] = {
                    'mean': segment_data[target_col].mean(),
                    'median': segment_data[target_col].median(),
                    'std': segment_data[target_col].std()
                }

            # Top correlations within segment
            if len(numeric_cols) > 1:
                segment_corr = segment_data[numeric_cols].corr()
                if target_col and target_col in segment_corr.columns:
                    target_correlations = segment_corr[target_col].drop(target_col).abs().sort_values(ascending=False)
                    segment_analysis['top_correlations'] = target_correlations.head(5).to_dict()

            analysis_results[segment] = segment_analysis

        return analysis_results

    def create_segment_comparison_viz(self, df, segment_col, target_col):
        """
        Create visualization comparing segments using box plots
        Shows distribution differences across operational segments
        """
        if segment_col not in df.columns or target_col not in df.columns:
            return None

        # Box plot comparison
        fig = go.Figure()

        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment][target_col].dropna()

            fig.add_trace(go.Box(
                y=segment_data,
                name=str(segment),
                boxpoints='outliers'
            ))

        fig.update_layout(
            title=f'{target_col} Distribution by {segment_col}',
            xaxis_title=segment_col,
            yaxis_title=target_col,
            height=500
        )

        return fig

    def data_quality_report(self, df):
        """
        Generate comprehensive data quality report
        Assesses missing data, duplicates, data types, and provides improvement recommendations
        """
        quality_issues = {
            'missing_data': {},
            'duplicates': {},
            'data_types': {},
            'outliers': {},
            'recommendations': []
        }

        # Missing data issues
        missing_counts = df.isnull().sum()
        quality_issues['missing_data'] = {
            'total_missing_values': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum()
        }

        # Duplicate analysis
        quality_issues['duplicates'] = {
            'total_duplicates': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }

        # Data type analysis
        quality_issues['data_types'] = {
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }

        # Generate recommendations
        recommendations = []

        if missing_counts.sum() > 0:
            high_missing_cols = missing_counts[missing_counts > len(df) * 0.5].index.tolist()
            if high_missing_cols:
                recommendations.append(f"Consider removing columns with >50% missing data: {high_missing_cols}")

        if df.duplicated().sum() > 0:
            recommendations.append(f"Remove {df.duplicated().sum()} duplicate rows")

        if len(df.select_dtypes(include=['object']).columns) > 0:
            recommendations.append("Consider encoding categorical variables for analysis")

        quality_issues['recommendations'] = recommendations

        return quality_issues

    def generate_eda_summary_report(self, df, target_col=None):
        """
        Generate comprehensive EDA summary report
        Combines all analytical findings into a unified summary with key insights
        """
        # Run all analyses
        data_profile = self.comprehensive_data_profiling(df, target_col)
        quality_report = self.data_quality_report(df)

        summary_report = {
            'dataset_overview': {
                'shape': df.shape,
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                'data_types': df.dtypes.value_counts().to_dict()
            },
            'data_quality': quality_report,
            'statistical_summary': data_profile,
            'key_insights': []
        }

        # Generate key insights
        insights = []

        # Missing data insights
        if data_profile['missing_data']['total_missing'] > 0:
            insights.append(f"Dataset has {data_profile['missing_data']['total_missing']} missing values across {len(data_profile['missing_data']['columns_with_missing'])} columns")

        # Correlation insights
        if 'high_correlations' in data_profile['correlations']:
            high_corr_count = len(data_profile['correlations']['high_correlations'])
            if high_corr_count > 0:
                insights.append(f"Found {high_corr_count} highly correlated feature pairs (|r| > 0.7)")

        # Outlier insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        total_outliers = sum([data_profile['outliers'][col]['count'] for col in numeric_cols])
        if total_outliers > 0:
            insights.append(f"Detected {total_outliers} potential outliers across numeric columns")

        # Target variable insights
        if target_col and 'target_analysis' in data_profile:
            target_info = data_profile['target_analysis']
            if target_info['type'] == 'numeric':
                skewness = target_info['distribution']['skewness']
                if abs(skewness) > 1:
                    insights.append(f"Target variable '{target_col}' is highly skewed (skewness: {skewness:.2f})")
            elif target_info['type'] == 'categorical':
                class_balance = target_info['distribution']['class_balance']
                min_class_pct = min(class_balance.values()) * 100
                if min_class_pct < 5:
                    insights.append(f"Target variable '{target_col}' has class imbalance (smallest class: {min_class_pct:.1f}%)")

        summary_report['key_insights'] = insights

        return summary_report
