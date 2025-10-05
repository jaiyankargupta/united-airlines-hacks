import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

class EnhancedDashboard:
    """
    Advanced Flight Analytics Dashboard
    Professional interface for comprehensive flight difficulty analysis
    """

    def __init__(self):
        pass

    def show_enhanced_dashboard(self):
        """Main dashboard interface"""
        st.title("Flight Difficulty Analytics Dashboard")

        # Check if data is available
        if not st.session_state.get('analysis_complete', False) or st.session_state.get('results_df') is None:
            st.warning("Analysis data not available. Please run the analysis first.")

            with st.expander("Getting Started"):
                st.markdown("""
                **Setup Process:**
                1. Navigate to the Data Upload section
                2. Upload all required CSV files (5 files total)
                3. Go to Command Panel and run: `analyze`
                4. Return to this dashboard to view results

                **Dashboard Features:**
                - Executive summary with key performance indicators
                - Detailed operational analysis and insights
                - Data quality assessment and recommendations
                - Correlation analysis between operational factors
                - Root cause identification for high-difficulty flights
                - Export capabilities for reporting
                """)

            if st.button("Generate Sample Data for Testing", type="primary"):
                self._run_sample_analysis()
            return

        df = st.session_state.results_df

        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Executive Summary",
            "Operational Analysis",
            "Correlation Analysis",
            "Root Cause Analysis",
            "Data Quality",
            "Reports & Actions"
        ])

        with tab1:
            self._show_executive_summary(df)

        with tab2:
            self._show_operational_analysis(df)

        with tab3:
            self._show_correlation_analysis(df)

        with tab4:
            self._show_root_cause_analysis(df)

        with tab5:
            self._show_data_quality_report(df)

        with tab6:
            self._show_reports_actions(df)

    def _show_executive_summary(self, df):
        """Executive summary with key metrics"""
        st.header("Executive Summary")

        target_col = 'difficulty_score' if 'difficulty_score' in df.columns else 'flight_difficulty_score'

        # Key Performance Indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_flights = len(df)
            st.metric("Total Flights Analyzed", f"{total_flights:,}")

        with col2:
            if target_col in df.columns:
                avg_difficulty = df[target_col].mean()
                st.metric("Average Difficulty Score", f"{avg_difficulty:.3f}")
            else:
                st.metric("Average Difficulty Score", "N/A")

        with col3:
            if 'difficulty_classification' in df.columns:
                high_difficulty = len(df[df['difficulty_classification'] == 'Difficult'])
                pct_high = (high_difficulty / total_flights) * 100 if total_flights > 0 else 0
                st.metric("High Difficulty Flights", f"{high_difficulty:,}", f"{pct_high:.1f}%")
            elif target_col in df.columns:
                high_difficulty = len(df[df[target_col] > 0.7])
                pct_high = (high_difficulty / total_flights) * 100 if total_flights > 0 else 0
                st.metric("High Difficulty Flights", f"{high_difficulty:,}", f"{pct_high:.1f}%")
            else:
                st.metric("High Difficulty Flights", "N/A")

        with col4:
            delay_columns = ['delay_minutes', 'arrival_delay_minutes', 'departure_delay_minutes']
            delay_col = None
            for col in delay_columns:
                if col in df.columns:
                    delay_col = col
                    break

            if delay_col:
                avg_delay = df[delay_col].mean()
                st.metric("Average Delay (minutes)", f"{avg_delay:.1f}")
            else:
                st.metric("Average Delay (minutes)", "N/A")

        st.divider()

        # Distribution Analysis
        st.subheader("Difficulty Score Distribution")

        if target_col in df.columns:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Create difficulty categories
                df_viz = df.copy()
                try:
                    df_viz['difficulty_level'] = pd.cut(
                        df_viz[target_col],
                        bins=3,
                        labels=['Low', 'Medium', 'High'],
                        duplicates='drop'
                    )
                except ValueError:
                    thresholds = [df[target_col].quantile(0.33), df[target_col].quantile(0.67)]
                    df_viz['difficulty_level'] = pd.cut(
                        df_viz[target_col],
                        bins=[-np.inf] + thresholds + [np.inf],
                        labels=['Low', 'Medium', 'High'],
                        include_lowest=True
                    )

                # Distribution histogram
                fig = px.histogram(
                    df_viz,
                    x=target_col,
                    color='difficulty_level',
                    title="Flight Difficulty Distribution",
                    color_discrete_map={'Low': '#2E8B57', 'Medium': '#DAA520', 'High': '#DC143C'},
                    nbins=30
                )
                fig.update_layout(
                    xaxis_title="Difficulty Score",
                    yaxis_title="Number of Flights",
                    legend_title="Difficulty Level",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Summary Statistics")
                if 'difficulty_level' in df_viz.columns:
                    dist_summary = df_viz['difficulty_level'].value_counts().sort_index()
                    for level, count in dist_summary.items():
                        pct = (count / len(df_viz)) * 100
                        st.write(f"**{level}:** {count:,} flights ({pct:.1f}%)")

                st.write("")
                st.write("**Score Statistics:**")
                st.write(f"Mean: {df[target_col].mean():.3f}")
                st.write(f"Median: {df[target_col].median():.3f}")
                st.write(f"Std Dev: {df[target_col].std():.3f}")
                st.write(f"Range: {df[target_col].min():.3f} - {df[target_col].max():.3f}")

        st.divider()

        # Key Insights
        st.subheader("Key Insights")

        insights = []
        if target_col in df.columns:
            avg_score = df[target_col].mean()
            high_count = len(df[df[target_col] > 0.7])

            insights.extend([
                f"Analysis covers {len(df):,} flights with comprehensive difficulty scoring",
                f"Overall difficulty level is {'elevated' if avg_score > 0.5 else 'moderate' if avg_score > 0.3 else 'low'} (average: {avg_score:.3f})",
                f"{high_count:,} flights ({(high_count/len(df))*100:.1f}%) require priority attention (score > 0.7)"
            ])

        if insights:
            for i, insight in enumerate(insights, 1):
                st.write(f"{i}. {insight}")

    def _show_operational_analysis(self, df):
        """Detailed operational analysis"""
        st.header("Operational Analysis")

        target_col = 'difficulty_score' if 'difficulty_score' in df.columns else 'flight_difficulty_score'

        if target_col not in df.columns:
            st.warning("Difficulty score data not available for operational analysis.")
            return

        # Temporal Analysis
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            st.subheader("Temporal Trends")
            date_col = date_columns[0]

            try:
                df_temp = df.copy()
                if df_temp[date_col].dtype == 'object':
                    df_temp['parsed_date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
                else:
                    df_temp['parsed_date'] = df_temp[date_col]
                df_temp = df_temp.dropna(subset=['parsed_date'])
                daily_stats = df_temp.groupby('parsed_date')[target_col].agg(['mean', 'count']).reset_index()

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.line(
                        daily_stats,
                        x='parsed_date',
                        y='mean',
                        title="Daily Average Difficulty Score",
                        markers=True
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Average Difficulty Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        daily_stats,
                        x='parsed_date',
                        y='count',
                        title="Daily Flight Volume"
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Number of Flights"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception:
                st.info("Unable to parse date information for temporal analysis.")

        st.divider()

        # Route Analysis
        route_columns = [col for col in df.columns if any(term in col.lower() for term in ['origin', 'destination', 'route'])]
        if len(route_columns) >= 2:
            st.subheader("Route Performance Analysis")

            origin_col = route_columns[0]
            dest_col = route_columns[1] if len(route_columns) > 1 else route_columns[0]

            try:
                route_stats = df.groupby([origin_col, dest_col])[target_col].agg(['mean', 'count']).reset_index()
                route_stats = route_stats[route_stats['count'] >= 5].nlargest(15, 'mean')
            except Exception:
                st.info("Unable to perform route analysis with current data structure.")
                return

            if not route_stats.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    route_stats['route'] = route_stats[origin_col].astype(str) + ' → ' + route_stats[dest_col].astype(str)

                    fig = px.bar(
                        route_stats.head(10),
                        x='mean',
                        y='route',
                        orientation='h',
                        title="Top 10 Most Difficult Routes (min 5 flights)",
                        text='mean'
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(
                        xaxis_title="Average Difficulty Score",
                        yaxis_title="Route"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Route Statistics")
                    for _, row in route_stats.head(5).iterrows():
                        st.write(f"**{row[origin_col]} → {row[dest_col]}**")
                        st.write(f"Avg Score: {row['mean']:.3f}")
                        st.write(f"Flight Count: {row['count']}")
                        st.write("---")

        st.divider()

        # Feature Importance
        st.subheader("Operational Factor Importance")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if len(numeric_cols) > 0:
            correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
            correlations = correlations.dropna().sort_values(key=abs, ascending=False).head(12)

            if not correlations.empty:
                fig = px.bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    title="Feature Correlation with Difficulty Score",
                    color=correlations.values,
                    color_continuous_scale='RdBu_r'
                )
                fig.update_layout(
                    xaxis_title="Correlation Coefficient",
                    yaxis_title="Operational Factors",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                correlation_df = pd.DataFrame({
                    'Factor': correlations.index,
                    'Correlation': correlations.values,
                    'Strength': ['Strong' if abs(x) > 0.5 else 'Moderate' if abs(x) > 0.3 else 'Weak' for x in correlations.values]
                })
                st.dataframe(correlation_df, use_container_width=True)

    def _show_correlation_analysis(self, df):
        """Correlation analysis between operational factors"""
        st.header("Correlation Analysis")

        target_col = 'difficulty_score' if 'difficulty_score' in df.columns else 'flight_difficulty_score'

        if target_col not in df.columns:
            st.warning("Difficulty score data required for correlation analysis.")
            return

        # Overall correlation analysis
        self._show_basic_correlation_analysis(df, target_col)

        st.divider()

        # Segmented Analysis
        st.subheader("Segmented Correlation Analysis")

        df_analysis = df.copy()
        try:
            # Remove any rows with NaN values in target column
            df_analysis = df_analysis.dropna(subset=[target_col])

            if len(df_analysis) < 10:
                st.warning("Insufficient data for segmented correlation analysis.")
                return

            df_analysis['difficulty_segment'] = pd.qcut(
                df_analysis[target_col],
                q=3,
                labels=['Low_Difficulty', 'Medium_Difficulty', 'High_Difficulty'],
                duplicates='drop'
            )
        except (ValueError, TypeError):
            try:
                thresholds = [df_analysis[target_col].quantile(0.33), df_analysis[target_col].quantile(0.67)]
                df_analysis['difficulty_segment'] = pd.cut(
                    df_analysis[target_col],
                    bins=[-np.inf] + thresholds + [np.inf],
                    labels=['Low_Difficulty', 'Medium_Difficulty', 'High_Difficulty'],
                    include_lowest=True
                )
            except Exception:
                st.warning("Unable to create difficulty segments for correlation analysis.")
                return

        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if len(numeric_cols) > 0 and 'difficulty_segment' in df_analysis.columns:
            segments = df_analysis['difficulty_segment'].dropna().unique()

            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]

            for i, segment in enumerate(segments):
                with columns[i % 3]:
                    st.subheader(f"{segment.replace('_Difficulty', '')} Difficulty Flights")

                    segment_data = df_analysis[df_analysis['difficulty_segment'] == segment]
                    if len(segment_data) > 10:
                        try:
                            seg_corr = segment_data[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
                            top_corr = seg_corr.dropna().sort_values(key=abs, ascending=False).head(5)

                            if not top_corr.empty:
                                for feature, corr in top_corr.items():
                                    st.metric(feature[:25], f"{corr:.3f}")
                            else:
                                st.info("No correlations found")
                        except Exception:
                            st.info("Unable to calculate correlations")
                    else:
                        st.info("Insufficient data for analysis")

    def _show_root_cause_analysis(self, df):
        """Root cause analysis for flight difficulty"""
        st.header("Root Cause Analysis")

        target_col = 'difficulty_score' if 'difficulty_score' in df.columns else 'flight_difficulty_score'

        if target_col not in df.columns:
            st.warning("Difficulty score data required for root cause analysis.")
            return

        # High vs Normal Difficulty Comparison
        threshold = df[target_col].quantile(0.8)
        high_difficulty = df[df[target_col] >= threshold]
        normal_flights = df[df[target_col] < threshold]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Flight Categories")
            st.write(f"**Total Flights:** {len(df):,}")
            st.write(f"**High Difficulty (≥{threshold:.2f}):** {len(high_difficulty):,} ({len(high_difficulty)/len(df)*100:.1f}%)")
            st.write(f"**Normal Difficulty (<{threshold:.2f}):** {len(normal_flights):,} ({len(normal_flights)/len(df)*100:.1f}%)")
            st.write(f"**Average Score:** {df[target_col].mean():.3f}")
            st.write(f"**Score Range:** {df[target_col].min():.3f} - {df[target_col].max():.3f}")

        with col2:
            # Distribution visualization
            fig = px.histogram(
                df,
                x=target_col,
                nbins=25,
                title="Difficulty Score Distribution with Threshold"
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                         annotation_text=f"High Difficulty Threshold ({threshold:.2f})")
            fig.update_layout(
                xaxis_title="Difficulty Score",
                yaxis_title="Number of Flights"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Contributing Factors Analysis
        if len(high_difficulty) > 0 and len(normal_flights) > 0:
            st.subheader("Contributing Factors Analysis")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)

            if len(numeric_cols) > 0:
                differences = []
                for col in numeric_cols[:15]:
                    try:
                        if col in high_difficulty.columns and col in normal_flights.columns:
                            high_mean = high_difficulty[col].mean()
                            normal_mean = normal_flights[col].mean()

                            # Skip if means are NaN or normal_mean is 0
                            if pd.isna(high_mean) or pd.isna(normal_mean) or normal_mean == 0:
                                continue

                            pct_diff = ((high_mean - normal_mean) / normal_mean) * 100
                            differences.append({
                                'Factor': col,
                                'High_Difficulty_Avg': high_mean,
                                'Normal_Avg': normal_mean,
                                'Percent_Difference': pct_diff,
                                'Absolute_Difference': abs(pct_diff)
                            })
                    except Exception:
                        continue  # Skip problematic columns

                if differences:
                    diff_df = pd.DataFrame(differences)
                    diff_df = diff_df.sort_values('Absolute_Difference', ascending=False)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        top_factors = diff_df.head(8)

                        fig = px.bar(
                            top_factors,
                            x='Percent_Difference',
                            y='Factor',
                            orientation='h',
                            title="Key Differences: High vs Normal Difficulty Flights",
                            color='Percent_Difference',
                            color_continuous_scale='RdBu_r'
                        )
                        fig.update_layout(
                            xaxis_title="Percent Difference (%)",
                            yaxis_title="Operational Factors"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Top Contributing Factors")
                        for _, row in diff_df.head(8).iterrows():
                            direction = "higher" if row['Percent_Difference'] > 0 else "lower"
                            st.write(f"**{row['Factor'][:30]}**")
                            st.write(f"{abs(row['Percent_Difference']):.1f}% {direction}")
                            st.write("---")

        st.divider()

        # Recommendations
        st.subheader("Recommended Actions")

        recommendations = [
            "Monitor high-impact factors showing largest differences between difficult and normal flights",
            "Implement predictive alerts based on identified risk factors",
            "Allocate additional resources for flights predicted to be high difficulty",
            "Review and optimize operational processes for top contributing factors",
            "Establish continuous monitoring of difficulty score trends"
        ]

        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

    def _show_data_quality_report(self, df):
        """Data quality assessment report"""
        st.header("Data Quality Assessment")

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            st.metric("Total Columns", len(df.columns))

        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))

        with col4:
            missing_total = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_total:,}")

        st.divider()

        # Missing data analysis
        st.subheader("Data Completeness Analysis")

        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100

        col1, col2 = st.columns([2, 1])

        with col1:
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data[missing_data > 0].index,
                    'Missing_Count': missing_data[missing_data > 0].values,
                    'Missing_Percentage': missing_pct[missing_data > 0].values
                }).sort_values('Missing_Percentage', ascending=False)

                fig = px.bar(
                    missing_df.head(10),
                    x='Missing_Percentage',
                    y='Column',
                    orientation='h',
                    title="Missing Data by Column (Top 10)"
                )
                fig.update_layout(
                    xaxis_title="Missing Data Percentage (%)",
                    yaxis_title="Columns"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values detected in the dataset")

        with col2:
            st.subheader("Quality Metrics")

            completeness_score = (1 - missing_data.sum() / (len(df) * len(df.columns))) * 100

            if completeness_score > 95:
                quality_status = "Excellent"
            elif completeness_score > 85:
                quality_status = "Good"
            elif completeness_score > 70:
                quality_status = "Fair"
            else:
                quality_status = "Needs Improvement"

            st.metric("Data Completeness", f"{completeness_score:.1f}%", quality_status)

            # Duplicate analysis
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Records", duplicates)

            # Numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                st.write("**Data Type Distribution:**")
                st.write(f"Numeric: {len(numeric_cols)}")
                st.write(f"Text: {len(df.select_dtypes(include=['object']).columns)}")
                st.write(f"Other: {len(df.columns) - len(numeric_cols) - len(df.select_dtypes(include=['object']).columns)}")

        st.divider()

        # Outlier Analysis
        if len(numeric_cols) > 0:
            st.subheader("Outlier Analysis")

            outlier_summary = []
            for col in numeric_cols[:8]:  # Limit to prevent UI overload
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(df)) * 100

                outlier_summary.append({
                    'Column': col,
                    'Outlier_Count': outlier_count,
                    'Outlier_Percentage': outlier_pct
                })

            if outlier_summary:
                outlier_data = []
                for item in outlier_summary:
                    if item['Outlier_Count'] > 0:
                        outlier_data.append(item)

                if outlier_data:
                    outlier_data.sort(key=lambda x: x['Outlier_Percentage'], reverse=True)
                    outlier_df = pd.DataFrame(outlier_data)
                    st.dataframe(outlier_df, use_container_width=True)
                else:
                    st.info("No significant outliers detected using IQR method")

    def _show_reports_actions(self, df):
        """Reports and actionable insights"""
        st.header("Reports & Actionable Insights")

        target_col = 'difficulty_score' if 'difficulty_score' in df.columns else 'flight_difficulty_score'

        if target_col not in df.columns:
            st.warning("Difficulty score data required for generating insights.")
            return

        # Summary Statistics
        st.subheader("Analysis Summary")

        total_flights = len(df)
        high_difficulty_flights = len(df[df[target_col] > 0.7])
        avg_score = df[target_col].mean()

        summary_data = {
            'Metric': [
                'Total Flights Analyzed',
                'Average Difficulty Score',
                'High Risk Flights (>0.7)',
                'High Risk Percentage',
                'Score Standard Deviation'
            ],
            'Value': [
                f"{total_flights:,}",
                f"{avg_score:.3f}",
                f"{high_difficulty_flights:,}",
                f"{(high_difficulty_flights/total_flights)*100:.1f}%",
                f"{df[target_col].std():.3f}"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.divider()

        # Strategic Recommendations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Immediate Actions")
            immediate_actions = [
                "Prioritize resources for flights scoring above 0.7",
                "Implement daily difficulty score monitoring dashboard",
                "Review top 10% most difficult flights weekly",
                "Share difficulty insights with operational teams",
                "Develop response protocols for high-difficulty flights"
            ]

            for i, action in enumerate(immediate_actions, 1):
                st.write(f"{i}. {action}")

        with col2:
            st.subheader("Long-term Improvements")
            strategic_actions = [
                "Build predictive models for proactive planning",
                "Investigate root causes of consistently difficult routes",
                "Optimize resource allocation algorithms",
                "Train operational staff on difficulty indicators",
                "Implement continuous improvement feedback systems"
            ]

            for i, action in enumerate(strategic_actions, 1):
                st.write(f"{i}. {action}")

        st.divider()

        # Export Options
        st.subheader("Export Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export Executive Summary", type="primary"):
                self._export_executive_summary(df)

        with col2:
            if st.button("Export Action Plan"):
                self._export_action_plan(df)

        with col3:
            if st.button("Export Full Dataset"):
                self._export_full_analysis(df)

    def _show_basic_correlation_analysis(self, df, target_col):
        """Basic correlation analysis visualization"""
        st.subheader("Feature Correlation Analysis")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if len(numeric_cols) > 0:
            correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
            correlations = correlations.dropna().sort_values(key=abs, ascending=False).head(10)

            if not correlations.empty:
                fig = go.Figure(go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker_color=['#DC143C' if x < 0 else '#2E8B57' for x in correlations.values]
                ))

                fig.update_layout(
                    title="Top 10 Feature Correlations with Difficulty Score",
                    xaxis_title="Correlation Coefficient",
                    yaxis_title="Features",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Correlation interpretation
                st.write("**Correlation Strength Guide:**")
                st.write("Strong: |r| > 0.5, Moderate: 0.3 < |r| ≤ 0.5, Weak: |r| ≤ 0.3")
            else:
                st.info("No significant correlations found.")
        else:
            st.info("Insufficient numeric features for correlation analysis.")

    def _export_executive_summary(self, df):
        """Export executive summary to CSV"""
        try:
            target_col = 'difficulty_score' if 'difficulty_score' in df.columns else 'flight_difficulty_score'

            summary_data = {
                'Metric': ['Total Flights', 'Average Difficulty Score', 'High Difficulty Flights', 'Data Completeness'],
                'Value': [
                    len(df),
                    df[target_col].mean() if target_col in df.columns else 0,
                    len(df[df[target_col] > 0.7]) if target_col in df.columns else 0,
                    ((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            os.makedirs("analysis/output", exist_ok=True)

            output_path = "analysis/output/executive_summary.csv"
            summary_df.to_csv(output_path, index=False)
            st.success(f"Executive summary exported to {output_path}")

        except Exception as e:
            st.error(f"Error exporting executive summary: {str(e)}")

    def _export_action_plan(self, df):
        """Export action plan to CSV"""
        try:
            action_items = [
                "Review and allocate additional resources for high-difficulty flights",
                "Implement daily monitoring of flight difficulty scores",
                "Develop predictive models for proactive planning",
                "Train operational staff on difficulty indicators"
            ]

            action_df = pd.DataFrame({
                'Priority': ['High', 'High', 'Medium', 'Medium'],
                'Action': action_items,
                'Timeline': ['Immediate', 'This Week', 'Next Month', 'Next Quarter']
            })

            os.makedirs("analysis/output", exist_ok=True)

            output_path = "analysis/output/action_plan.csv"
            action_df.to_csv(output_path, index=False)
            st.success(f"Action plan exported to {output_path}")

        except Exception as e:
            st.error(f"Error exporting action plan: {str(e)}")

    def _export_full_analysis(self, df):
        """Export full analysis results to CSV"""
        try:
            os.makedirs("analysis/output", exist_ok=True)

            output_path = "analysis/output/full_analysis_results.csv"
            df.to_csv(output_path, index=False)
            st.success(f"Full analysis results exported to {output_path}")

        except Exception as e:
            st.error(f"Error exporting full analysis: {str(e)}")

    def _run_sample_analysis(self):
        """Generate sample data for testing"""
        try:
            st.info("Generating sample analysis data...")

            np.random.seed(42)
            n_flights = 1200

            # Generate airport codes and route data
            airports = ['ORD', 'DEN', 'SFO', 'LAX', 'JFK', 'DFW', 'ATL', 'BOS', 'SEA', 'MIA']

            sample_data = {
                'flight_number': [f'UA{1000 + i}' for i in range(n_flights)],
                'difficulty_score': np.random.beta(2, 5, n_flights),
                'delay_minutes': np.random.gamma(2, 2, n_flights),
                'passenger_load': np.random.uniform(0.5, 1.0, n_flights),
                'weather_impact': np.random.uniform(0.0, 1.0, n_flights),
                'ground_pressure': np.random.uniform(0.0, 0.8, n_flights),
                'operational_complexity': np.random.uniform(0.1, 0.9, n_flights),
                'origin': np.random.choice(airports, n_flights),
                'destination': np.random.choice(airports, n_flights),
                'flight_date': pd.date_range('2024-01-01', periods=30, freq='D').repeat(n_flights // 30 + 1)[:n_flights]
            }

            # Create realistic correlations
            for i in range(n_flights):
                if sample_data['difficulty_score'][i] > 0.7:
                    sample_data['delay_minutes'][i] *= 1.8
                    sample_data['passenger_load'][i] = min(1.0, sample_data['passenger_load'][i] * 1.3)
                    sample_data['operational_complexity'][i] *= 1.4

            sample_df = pd.DataFrame(sample_data)

            st.session_state.results_df = sample_df
            st.session_state.analysis_complete = True

            st.success("Sample analysis data generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")

def main():
    """Main function for standalone testing"""
    st.set_page_config(page_title="Flight Analytics Dashboard", layout="wide")

    dashboard = EnhancedDashboard()
    dashboard.show_enhanced_dashboard()

if __name__ == "__main__":
    main()
