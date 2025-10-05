import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnalytics:
    """
    Enhanced analytics module inspired by customer satisfaction analysis patterns
    Provides comprehensive correlation analysis, sentiment analysis, and root cause investigation
    """

    def __init__(self):
        self.correlation_results = {}
        self.sentiment_results = {}
        self.feature_importance = {}

    def safe_correlation(self, x, y, method='pearson'):
        """
        Safely compute correlation between two series with error handling
        Returns zero correlation if computation fails or insufficient data
        """
        try:
            # Remove NaN values
            mask = ~(pd.isna(x) | pd.isna(y))
            if mask.sum() < 2:
                return 0.0

            x_clean = x[mask]
            y_clean = y[mask]

            if method == 'pearson':
                corr, _ = stats.pearsonr(x_clean, y_clean)
            elif method == 'spearman':
                corr, _ = stats.spearmanr(x_clean, y_clean)
            elif method == 'kendall':
                corr, _ = stats.kendalltau(x_clean, y_clean)
            else:
                corr, _ = stats.pearsonr(x_clean, y_clean)

            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0

    def segmented_correlation_analysis(self, df, target_col, segment_col, features=None):
        """
        Perform correlation analysis by segments (similar to customer satisfaction analysis)
        This method analyzes how correlations vary across different operational segments

        Args:
            df: DataFrame containing the data to analyze
            target_col: Target variable for correlation analysis
            segment_col: Column to segment by (e.g., difficulty_level, operational_type)
            features: List of features to analyze, if None uses all numeric columns

        Returns:
            Dictionary with correlation results organized by segment
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in [target_col, segment_col]]

        results = {}
        segments = df[segment_col].unique()

        for segment in segments:
            segment_data = df[df[segment_col] == segment]
            correlations = []

            for feature in features:
                corr = self.safe_correlation(segment_data[feature], segment_data[target_col])
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'sample_size': len(segment_data.dropna(subset=[feature, target_col]))
                })

            # Sort by absolute correlation
            correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
            results[f'{segment_col}_{segment}'] = correlations[:10]  # Top 10 correlations

        self.correlation_results = results
        return results

    def create_correlation_heatmap(self, df, features=None, title="Feature Correlation Heatmap"):
        """
        Create an enhanced correlation heatmap with interactive features
        Displays correlation strengths with numerical values and color coding
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[features].corr()

        # Create plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title,
            width=800,
            height=800
        )

        return fig

    def feature_importance_analysis(self, df, target_col, features=None):
        """
        Analyze feature importance using multiple statistical methods
        Combines correlation-based and mutual information approaches for robust ranking
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != target_col]

        # Prepare data
        X = df[features].fillna(0)
        y = df[target_col].fillna(0)

        # Method 1: Correlation-based importance
        corr_importance = []
        for feature in features:
            corr = abs(self.safe_correlation(X[feature], y))
            corr_importance.append({'feature': feature, 'correlation_importance': corr})

        # Method 2: Mutual Information
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_importance = [{'feature': features[i], 'mutual_info_importance': score}
                           for i, score in enumerate(mi_scores)]
        except:
            mi_importance = [{'feature': f, 'mutual_info_importance': 0} for f in features]

        # Combine results
        importance_df = pd.DataFrame(corr_importance)
        mi_df = pd.DataFrame(mi_importance)
        importance_df = importance_df.merge(mi_df, on='feature')

        # Calculate combined score
        scaler = StandardScaler()
        importance_df['combined_importance'] = scaler.fit_transform(
            importance_df[['correlation_importance', 'mutual_info_importance']]
        ).mean(axis=1)

        importance_df = importance_df.sort_values('combined_importance', ascending=False)
        self.feature_importance = importance_df

        return importance_df

    def sentiment_analysis(self, text_series, text_col_name='comments'):
        """
        Perform sentiment analysis on text data (similar to customer feedback analysis)
        Uses TextBlob library to analyze polarity and subjectivity of text content
        """
        sentiments = []

        for text in text_series.dropna():
            if isinstance(text, str) and len(text.strip()) > 0:
                blob = TextBlob(text)
                sentiment = blob.sentiment

                sentiments.append({
                    'text': text,
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity,
                    'sentiment_label': self._classify_sentiment(sentiment.polarity)
                })
            else:
                sentiments.append({
                    'text': '',
                    'polarity': 0,
                    'subjectivity': 0,
                    'sentiment_label': 'neutral'
                })

        sentiment_df = pd.DataFrame(sentiments)
        self.sentiment_results = sentiment_df

        return sentiment_df

    def _classify_sentiment(self, polarity):
        """
        Classify sentiment based on polarity score
        Positive > 0.1, Negative < -0.1, otherwise Neutral
        """
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

    def generate_word_cloud(self, text_series, sentiment_filter=None, max_words=100):
        """
        Generate word cloud visualization for text analysis
        Helps identify common themes and frequently mentioned terms

        Args:
            text_series: Series containing text data to analyze
            sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral')
            max_words: Maximum number of words to display in the cloud
        """
        if hasattr(self, 'sentiment_results') and sentiment_filter:
            # Filter by sentiment
            filtered_texts = self.sentiment_results[
                self.sentiment_results['sentiment_label'] == sentiment_filter
            ]['text'].tolist()
            text_data = ' '.join(filtered_texts)
        else:
            text_data = ' '.join(text_series.dropna().astype(str).tolist())

        if len(text_data.strip()) == 0:
            return None

        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                max_words=max_words,
                background_color='white',
                colormap='viridis'
            ).generate(text_data)

            # Create plotly figure
            fig = go.Figure()
            fig.add_layout_image(
                dict(source=wordcloud.to_image(),
                     x=0, y=1, xref="paper", yref="paper",
                     sizex=1, sizey=1, layer="below")
            )
            fig.update_layout(
                title=f"Word Cloud - {sentiment_filter.title() if sentiment_filter else 'All'} Comments",
                xaxis={'visible': False},
                yaxis={'visible': False},
                width=800, height=400
            )

            return fig
        except:
            return None

    def root_cause_analysis_framework(self, df, target_col, max_depth=5):
        """
        Implement systematic root cause analysis framework using correlation chains
        Similar to the Five Whys methodology used in quality improvement

        Args:
            df: DataFrame containing the operational data
            target_col: Target variable to investigate (e.g., difficulty_score)
            max_depth: Maximum depth for cause chain analysis (default 5)

        Returns:
            Structured root cause analysis results with actionable insights
        """
        analysis_results = []

        # Find primary correlates
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlations = []

        for col in numeric_cols:
            if col != target_col:
                corr = abs(self.safe_correlation(df[col], df[target_col]))
                correlations.append((col, corr))

        correlations = sorted(correlations, key=lambda x: x[1], reverse=True)

        # Build cause-effect chain
        for i, (primary_factor, _) in enumerate(correlations[:3]):  # Top 3 factors
            chain = [f"Why is {target_col} problematic?"]
            current_factor = primary_factor

            for depth in range(max_depth):
                if depth == 0:
                    chain.append(f"Because {current_factor} shows strong correlation with {target_col}")
                else:
                    # Find what correlates with current factor
                    sub_correlations = []
                    for col in numeric_cols:
                        if col != current_factor and col != target_col:
                            corr = abs(self.safe_correlation(df[col], df[current_factor]))
                            sub_correlations.append((col, corr))

                    if sub_correlations:
                        next_factor = max(sub_correlations, key=lambda x: x[1])
                        if next_factor[1] > 0.1:  # Minimum correlation threshold
                            chain.append(f"Why does {current_factor} affect the outcome?")
                            chain.append(f"Because it's influenced by {next_factor[0]}")
                            current_factor = next_factor[0]
                        else:
                            break
                    else:
                        break

            analysis_results.append({
                'primary_factor': primary_factor,
                'cause_chain': chain
            })

        return analysis_results

    def create_pareto_analysis(self, df, category_col, value_col, title="Pareto Analysis"):
        """
        Create Pareto chart analysis following the 80/20 principle
        Helps identify the most significant factors contributing to issues
        """
        # Group and sort by value
        grouped = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)

        # Calculate cumulative percentage
        cumulative_pct = (grouped.cumsum() / grouped.sum() * 100)

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar chart
        fig.add_trace(
            go.Bar(x=grouped.index, y=grouped.values, name=value_col),
            secondary_y=False,
        )

        # Add line chart for cumulative percentage
        fig.add_trace(
            go.Scatter(x=grouped.index, y=cumulative_pct.values,
                      mode='lines+markers', name='Cumulative %'),
            secondary_y=True,
        )

        # Add 80% line
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                     annotation_text="80%", secondary_y=True)

        # Update layout
        fig.update_xaxes(title_text=category_col)
        fig.update_yaxes(title_text=value_col, secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True)
        fig.update_layout(title_text=title)

        return fig

    def advanced_segmentation_analysis(self, df, target_col, segment_cols):
        """
        Perform advanced segmentation analysis similar to operational hub-spoke analysis
        Examines how target variables behave across different operational segments
        """
        results = {}

        for segment_col in segment_cols:
            if segment_col in df.columns:
                segment_stats = df.groupby(segment_col)[target_col].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(4)

                segment_stats['cv'] = segment_stats['std'] / segment_stats['mean']  # Coefficient of variation
                results[segment_col] = segment_stats.to_dict('index')

        return results

    def generate_insights_report(self, df, target_col, text_col=None):
        """
        Generate comprehensive insights report combining all analytical methods
        Provides a unified view of statistical findings and actionable recommendations
        """
        report = {
            'summary_stats': {},
            'correlation_analysis': {},
            'feature_importance': {},
            'segmentation_insights': {},
            'recommendations': []
        }

        # Summary statistics
        report['summary_stats'] = {
            'total_records': len(df),
            'target_mean': df[target_col].mean(),
            'target_median': df[target_col].median(),
            'target_std': df[target_col].std(),
            'missing_values': df.isnull().sum().to_dict()
        }

        # Feature importance
        importance_df = self.feature_importance_analysis(df, target_col)
        report['feature_importance'] = importance_df.head(10).to_dict('records')

        # Correlation analysis if segmentation column exists
        potential_segment_cols = ['difficulty_level', 'classification', 'category', 'type']
        for col in potential_segment_cols:
            if col in df.columns:
                corr_results = self.segmented_correlation_analysis(df, target_col, col)
                report['correlation_analysis'][col] = corr_results
                break

        # Sentiment analysis if text column provided
        if text_col and text_col in df.columns:
            sentiment_results = self.sentiment_analysis(df[text_col])
            sentiment_summary = sentiment_results['sentiment_label'].value_counts().to_dict()
            report['sentiment_summary'] = sentiment_summary

        # Generate actionable recommendations based on statistical findings
        top_features = importance_df.head(5)['feature'].tolist()
        report['recommendations'] = [
            f"Focus operational improvements on {top_features[0]} as it shows highest correlation with {target_col}",
            f"Establish monitoring systems for {top_features[1]} and {top_features[2]} to gain operational insights",
            f"Consider segmented analysis by key categorical variables to identify specific improvement areas",
            f"Implement predictive modeling using top {len(top_features)} features for proactive management"
        ]

        return report

    def create_comprehensive_dashboard_data(self, df, target_col, text_col=None):
        """
        Prepare comprehensive data for dashboard visualization
        Organizes all analytical results into a format suitable for interactive dashboards
        """
        dashboard_data = {}

        # Basic statistics
        dashboard_data['basic_stats'] = self.generate_insights_report(df, target_col, text_col)

        # Correlation heatmap data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            dashboard_data['correlation_matrix'] = df[numeric_cols].corr().to_dict()

        # Distribution data
        dashboard_data['target_distribution'] = df[target_col].describe().to_dict()

        # Top correlates
        correlations = []
        for col in numeric_cols:
            if col != target_col:
                corr = self.safe_correlation(df[col], df[target_col])
                correlations.append({'feature': col, 'correlation': corr})

        dashboard_data['top_correlations'] = sorted(
            correlations, key=lambda x: abs(x['correlation']), reverse=True
        )[:10]

        return dashboard_data
