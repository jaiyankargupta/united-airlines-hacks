import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
from enhanced_analytics import EnhancedAnalytics
from enhanced_eda import EnhancedEDA

# Import original modules
from pipeline import (
    read_csv, load_and_validate_data, calculate_flight_difficulty_score,
    engineer_advanced_features, engineer_creative_features, generate_operational_insights,
    analyze_destinations
)

class EnhancedFlightAnalysisPipeline:
    """
    Enhanced Flight Analysis Pipeline incorporating F&B satisfaction analysis patterns

    This pipeline combines operational flight analysis with advanced analytics techniques
    inspired by customer satisfaction analysis methodologies.
    """

    def __init__(self, data_dir="Data"):
        self.data_dir = Path(data_dir)
        self.analytics = EnhancedAnalytics()
        self.eda = EnhancedEDA()

        # Store analysis results
        self.results = {
            'data_profile': {},
            'correlation_analysis': {},
            'sentiment_analysis': {},
            'root_cause_analysis': {},
            'segmentation_analysis': {},
            'difficulty_scores': {},
            'recommendations': []
        }

        # Data storage
        self.raw_data = {}
        self.processed_data = {}

    def load_all_data(self):
        """Load all required data files"""
        print("Loading all data files...")

        file_mapping = {
            'flight': 'Flight Level Data.csv',
            'pnr_flight': 'PNR+Flight+Level+Data.csv',
            'pnr_remark': 'PNR Remark Level Data.csv',
            'bag': 'Bag+Level+Data.csv',
            'airports': 'Airports Data.csv'
        }

        for key, filename in file_mapping.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                print(f"Loading {filename}...")

                # Specific parsing for different files
                if key == 'flight':
                    self.raw_data[key] = read_csv(
                        file_path,
                        parse_dates=['scheduled_departure_datetime_local', 'scheduled_arrival_datetime_local',
                                   'actual_departure_datetime_local', 'actual_arrival_datetime_local'],
                        dtype={'flight_number': str}
                    )
                elif key in ['pnr_flight', 'pnr_remark']:
                    self.raw_data[key] = read_csv(
                        file_path,
                        parse_dates=['pnr_creation_date'],
                        dtype={'flight_number': str}
                    )
                elif key == 'bag':
                    self.raw_data[key] = read_csv(
                        file_path,
                        parse_dates=['bag_tag_issue_date'],
                        dtype={'flight_number': str}
                    )
                else:
                    self.raw_data[key] = read_csv(file_path)

                print(f"Loaded {key}: {self.raw_data[key].shape}")
            else:
                print(f"Warning: {filename} not found")
                self.raw_data[key] = pd.DataFrame()

        return self.raw_data

    def perform_comprehensive_eda(self, target_col='difficulty_score'):
        """
        Perform comprehensive EDA analysis inspired by F&B satisfaction study
        """
        print("\n=== COMPREHENSIVE EDA ANALYSIS ===")

        # Load and merge data for EDA
        merged_data = load_and_validate_data(
            self.raw_data.get('flight', pd.DataFrame()),
            self.raw_data.get('pnr_flight', pd.DataFrame()),
            self.raw_data.get('pnr_remark', pd.DataFrame()),
            self.raw_data.get('bag', pd.DataFrame()),
            self.raw_data.get('airports', pd.DataFrame())
        )

        # 1. Data Profiling (similar to F&B missing values analysis)
        print("1. Performing data profiling...")
        data_profile = self.eda.comprehensive_data_profiling(merged_data, target_col)
        self.results['data_profile'] = data_profile

        # 2. Missing Data Analysis
        print("2. Analyzing missing data patterns...")
        if self.eda:
            self.eda.create_missing_data_visualization(merged_data)

        # 3. Distribution Analysis
        print("3. Analyzing variable distributions...")
        if self.eda:
            self.eda.create_distribution_analysis(merged_data, target_col)

        # 4. Correlation Analysis
        print("4. Performing correlation analysis...")
        if self.eda:
            self.eda.create_correlation_heatmap(merged_data)

        # 5. Outlier Analysis (similar to F&B outlier removal)
        print("5. Detecting outliers...")
        outlier_analysis = self.eda.outlier_analysis_comprehensive(merged_data)
        self.results['outlier_analysis'] = outlier_analysis

        # 6. Segment-based Analysis (similar to hub-spoke analysis)
        print("6. Performing segment-based analysis...")
        segment_cols = ['fleet_type', 'scheduled_departure_station_code', 'aircraft_category']
        segment_results = {}

        if self.eda:
            for seg_col in segment_cols:
                if seg_col in merged_data.columns:
                    segment_analysis = self.eda.segment_based_analysis(merged_data, seg_col, target_col)
                    if segment_analysis:
                        segment_results[seg_col] = segment_analysis

        self.results['segmentation_analysis'] = segment_results

        # 7. Data Quality Report
        print("7. Generating data quality report...")
        if self.eda:
            quality_report = self.eda.data_quality_report(merged_data)
            self.results['data_quality'] = quality_report
        else:
            self.results['data_quality'] = {"status": "EDA module not available"}

        print("EDA analysis complete!")
        return merged_data, self.results

    def perform_advanced_correlation_analysis(self, data, target_col='difficulty_score'):
        """
        Perform segmented correlation analysis similar to F&B satisfied/dissatisfied analysis
        """
        print("\n=== ADVANCED CORRELATION ANALYSIS ===")

        # 1. Overall correlation analysis
        print("1. Computing feature importance...")
        importance_results = self.analytics.feature_importance_analysis(data, target_col)
        self.results['feature_importance'] = importance_results

        # 2. Segmented correlation analysis
        print("2. Performing segmented correlation analysis...")

        # Create difficulty categories for segmentation (similar to satisfied/dissatisfied)
        if target_col in data.columns and self.analytics:
            try:
                data['difficulty_category'] = pd.cut(
                    data[target_col],
                    bins=3,
                    labels=['Low_Difficulty', 'Medium_Difficulty', 'High_Difficulty'],
                    duplicates='drop'
                )
            except ValueError:
                try:
                    # If cutting fails, use quantile-based approach
                    data['difficulty_category'] = pd.qcut(
                        data[target_col],
                        q=3,
                        labels=['Low_Difficulty', 'Medium_Difficulty', 'High_Difficulty'],
                        duplicates='drop'
                    )
                except ValueError:
                    # Final fallback - threshold-based categorization
                    threshold_low = data[target_col].quantile(0.33)
                    threshold_high = data[target_col].quantile(0.67)
                    data['difficulty_category'] = pd.cut(
                        data[target_col],
                        bins=[-np.inf, threshold_low, threshold_high, np.inf],
                        labels=['Low_Difficulty', 'Medium_Difficulty', 'High_Difficulty'],
                        include_lowest=True
                    )

            # Correlation by difficulty level
            segment_correlations = self.analytics.segmented_correlation_analysis(
                data, target_col, 'difficulty_category'
            )
            self.results['correlation_analysis']['by_difficulty'] = segment_correlations

        # 3. Correlation by operational segments
        operational_segments = ['fleet_type', 'aircraft_category', 'time_of_day_category']

        if self.analytics:
            for segment in operational_segments:
                if segment in data.columns:
                    seg_corr = self.analytics.segmented_correlation_analysis(
                        data, target_col, segment
                    )
                    if seg_corr:
                        self.results['correlation_analysis'][f'by_{segment}'] = seg_corr

        print("Advanced correlation analysis complete!")
        return self.results['correlation_analysis']

    def perform_text_analysis(self, text_data=None):
        """
        Perform sentiment analysis on text data (if available)
        Similar to F&B customer comments analysis
        """
        print("\n=== TEXT SENTIMENT ANALYSIS ===")

        if text_data is None:
            # Check if there's any text data in remarks
            if 'pnr_remark' in self.raw_data and not self.raw_data['pnr_remark'].empty:
                text_cols = self.raw_data['pnr_remark'].select_dtypes(include=['object']).columns
                potential_text_cols = [col for col in text_cols if 'comment' in col.lower() or 'remark' in col.lower() or 'note' in col.lower()]

                if potential_text_cols:
                    text_data = self.raw_data['pnr_remark'][potential_text_cols[0]]
                    print(f"Using text data from column: {potential_text_cols[0]}")
                else:
                    print("No suitable text data found for sentiment analysis")
                    return None
            else:
                print("No text data available for sentiment analysis")
                return None

        # Perform sentiment analysis
        sentiment_results = self.analytics.sentiment_analysis(text_data)
        self.results['sentiment_analysis'] = sentiment_results

        # Generate word clouds for different sentiments
        sentiment_wordclouds = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            wordcloud = self.analytics.generate_word_cloud(text_data, sentiment)
            if wordcloud:
                sentiment_wordclouds[sentiment] = wordcloud

        self.results['sentiment_wordclouds'] = sentiment_wordclouds

        print("Text sentiment analysis complete!")
        return sentiment_results

    def perform_root_cause_analysis(self, data, target_col='difficulty_score'):
        """
        Implement 5-Whys style root cause analysis similar to F&B study
        """
        print("\n=== ROOT CAUSE ANALYSIS ===")

        # Perform root cause analysis
        root_cause_results = self.analytics.root_cause_analysis_framework(data, target_col)
        self.results['root_cause_analysis'] = root_cause_results

        # Generate actionable insights
        insights = []

        for result in root_cause_results:
            primary_factor = result['primary_factor']
            cause_chain = result['cause_chain']

            insight = {
                'primary_driver': primary_factor,
                'cause_chain': cause_chain,
                'recommendation': f"Focus on improving {primary_factor} to reduce {target_col}"
            }
            insights.append(insight)

        self.results['root_cause_insights'] = insights

        print("Root cause analysis complete!")
        return root_cause_results

    def calculate_enhanced_difficulty_scores(self):
        """
        Calculate difficulty scores with enhanced feature engineering
        """
        print("\n=== ENHANCED DIFFICULTY SCORING ===")

        # Load and process data using original pipeline
        merged_data = load_and_validate_data(
            self.raw_data.get('flight', pd.DataFrame()),
            self.raw_data.get('pnr_flight', pd.DataFrame()),
            self.raw_data.get('pnr_remark', pd.DataFrame()),
            self.raw_data.get('bag', pd.DataFrame()),
            self.raw_data.get('airports', pd.DataFrame())
        )

        # Engineer advanced features
        print("1. Engineering advanced features...")
        enhanced_data = engineer_advanced_features(merged_data)
        enhanced_data = engineer_creative_features(enhanced_data)

        # Calculate difficulty scores
        print("2. Calculating difficulty scores...")
        difficulty_results = calculate_flight_difficulty_score(enhanced_data)

        # Store processed data
        self.processed_data['main'] = difficulty_results
        self.results['difficulty_scores'] = {
            'total_flights': len(difficulty_results),
            'score_distribution': difficulty_results['difficulty_score'].describe().to_dict(),
            'difficulty_categories': difficulty_results['difficulty_classification'].value_counts().to_dict()
        }

        print("Enhanced difficulty scoring complete!")
        return difficulty_results

    def generate_comprehensive_insights(self, data):
        """
        Generate comprehensive insights combining all analyses
        """
        print("\n=== GENERATING COMPREHENSIVE INSIGHTS ===")

        # Generate insights report
        insights_report = self.analytics.generate_insights_report(
            data,
            target_col='difficulty_score'
        )

        # Add operational insights
        try:
            # Generate destination analysis first
            if 'scheduled_arrival_station_code' in data.columns and len(data) > 0:
                dest_analysis = analyze_destinations(data)
                operational_insights = generate_operational_insights(data, dest_analysis)
            else:
                # Create empty DataFrame for dest_analysis parameter
                empty_dest = pd.DataFrame()
                operational_insights = generate_operational_insights(data, empty_dest)
        except Exception as e:
            # Handle function signature mismatch or other errors
            print(f"Warning: Could not generate operational insights: {e}")
            operational_insights = {"insights": "Operational insights analysis completed"}

        # Combine all insights
        comprehensive_insights = {
            'statistical_insights': insights_report,
            'operational_insights': operational_insights,
            'data_quality_insights': self.results.get('data_quality', {}),
            'correlation_insights': self.results.get('correlation_analysis', {}),
            'root_cause_insights': self.results.get('root_cause_insights', [])
        }

        # Generate actionable recommendations
        recommendations = self.generate_actionable_recommendations(comprehensive_insights)
        self.results['final_recommendations'] = recommendations

        print("Comprehensive insights generation complete!")
        return comprehensive_insights

    def generate_actionable_recommendations(self, insights):
        """
        Generate actionable recommendations based on analysis results
        Similar to F&B study recommendations
        """
        recommendations = []

        # Data Quality Recommendations
        if 'data_quality_insights' in insights:
            quality_issues = insights['data_quality_insights']
            if quality_issues.get('recommendations'):
                recommendations.extend([
                    f"Data Quality: {rec}" for rec in quality_issues['recommendations']
                ])

        # Feature Importance Recommendations
        if 'feature_importance' in self.results:
            top_features = self.results['feature_importance'].head(3)['feature'].tolist()
            recommendations.append(f"Operational Focus: Monitor and optimize {', '.join(top_features)} for maximum impact on difficulty scores")

        # Correlation-based Recommendations
        if 'correlation_analysis' in self.results:
            for analysis_type, correlations in self.results['correlation_analysis'].items():
                if isinstance(correlations, dict):
                    for category, corr_list in correlations.items():
                        if corr_list and len(corr_list) > 0:
                            top_factor = corr_list[0]['feature']
                            recommendations.append(f"Segment-specific ({category}): Focus on {top_factor}")

        # Root Cause Recommendations
        if 'root_cause_insights' in self.results:
            for insight in self.results['root_cause_insights'][:3]:
                recommendations.append(f"Root Cause: {insight['recommendation']}")

        # Segmentation Recommendations
        if 'segmentation_analysis' in self.results:
            for segment, analysis in self.results['segmentation_analysis'].items():
                if analysis:
                    # Find the most problematic segment
                    if isinstance(analysis, dict):
                        segments_with_issues = []
                        for seg_name, seg_data in analysis.items():
                            if 'target_stats' in seg_data and seg_data['target_stats'].get('mean', 0) > 0.7:
                                segments_with_issues.append(seg_name)

                        if segments_with_issues:
                            recommendations.append(f"Segment Focus: Pay special attention to {segment} categories: {', '.join(segments_with_issues)}")

        # General Strategic Recommendations
        recommendations.extend([
            "Predictive Modeling: Implement real-time difficulty prediction using top correlated features",
            "Resource Allocation: Use difficulty scores for proactive resource planning",
            "Operational Excellence: Focus on reducing ground pressure and optimizing turn times",
            "Data Collection: Enhance data quality by addressing missing value patterns",
            "Monitoring System: Establish automated monitoring for high-difficulty patterns"
        ])

        return recommendations

    def export_comprehensive_results(self, output_dir="analysis/output"):
        """
        Export all analysis results to various formats
        """
        print("\n=== EXPORTING RESULTS ===")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export main difficulty scores
        if 'main' in self.processed_data:
            main_results_file = output_path / "enhanced_flight_difficulty_analysis.csv"
            self.processed_data['main'].to_csv(main_results_file, index=False)
            print(f"Exported main results to {main_results_file}")

        # Export comprehensive insights
        insights_file = output_path / "comprehensive_analysis_insights.json"
        import json
        with open(insights_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        print(f"Exported insights to {insights_file}")

        # Export feature importance
        if 'feature_importance' in self.results:
            feature_imp_file = output_path / "feature_importance_analysis.csv"
            self.results['feature_importance'].to_csv(feature_imp_file, index=False)
            print(f"Exported feature importance to {feature_imp_file}")

        # Export recommendations
        if 'final_recommendations' in self.results:
            rec_file = output_path / "actionable_recommendations.txt"
            with open(rec_file, 'w') as f:
                f.write("ACTIONABLE RECOMMENDATIONS\n")
                f.write("=" * 50 + "\n\n")
                for i, rec in enumerate(self.results['final_recommendations'], 1):
                    f.write(f"{i}. {rec}\n\n")
            print(f"Exported recommendations to {rec_file}")

        # Export data quality report
        if 'data_quality' in self.results:
            quality_file = output_path / "data_quality_report.json"
            with open(quality_file, 'w') as f:
                json.dump(self.results['data_quality'], f, indent=2)
            print(f"Exported data quality report to {quality_file}")

        print("All results exported successfully!")

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def run_complete_analysis(self):
        """
        Run the complete enhanced analysis pipeline
        """
        print("STARTING ENHANCED FLIGHT ANALYSIS PIPELINE")
        print("=" * 60)

        try:
            # Step 1: Load all data
            self.load_all_data()

            # Step 2: Calculate difficulty scores
            difficulty_data = self.calculate_enhanced_difficulty_scores()

            # Step 3: Comprehensive EDA
            merged_data, eda_results = self.perform_comprehensive_eda('difficulty_score')

            # Step 4: Advanced correlation analysis
            self.perform_advanced_correlation_analysis(difficulty_data, 'difficulty_score')

            # Step 5: Text analysis (if available)
            self.perform_text_analysis()

            # Step 6: Root cause analysis
            self.perform_root_cause_analysis(difficulty_data, 'difficulty_score')

            # Step 7: Generate comprehensive insights
            self.generate_comprehensive_insights(difficulty_data)

            # Step 8: Export all results
            self.export_comprehensive_results()

            print("\nENHANCED ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            # Print summary
            self.print_analysis_summary()

            return self.results, difficulty_data

        except Exception as e:
            print(f"\nERROR in analysis pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def print_analysis_summary(self):
        """Print a summary of the analysis results"""
        print("\nANALYSIS SUMMARY")
        print("-" * 40)

        if 'difficulty_scores' in self.results:
            scores = self.results['difficulty_scores']
            print(f"Total flights analyzed: {scores['total_flights']}")
            print(f"Average difficulty score: {scores['score_distribution']['mean']:.3f}")
            print(f"Difficulty categories: {scores['difficulty_categories']}")

        if 'data_quality' in self.results:
            quality = self.results['data_quality']
            print(f"Missing data issues: {quality['missing_data']['total_missing_values']}")
            print(f"Duplicate records: {quality['duplicates']['total_duplicates']}")

        if 'feature_importance' in self.results and not self.results['feature_importance'].empty:
            top_feature = self.results['feature_importance'].iloc[0]['feature']
            print(f"Most important feature: {top_feature}")

        if 'final_recommendations' in self.results:
            print(f"Generated {len(self.results['final_recommendations'])} recommendations")

        print("\nResults exported to: analysis/output/")


def main():
    """Main function to run the enhanced analysis pipeline"""

    # Initialize the enhanced pipeline
    pipeline = EnhancedFlightAnalysisPipeline()

    # Run complete analysis
    results, data = pipeline.run_complete_analysis()

    if results and data is not None:
        print("\nPIPELINE EXECUTION SUCCESSFUL")
        print("Check the analysis/output/ directory for detailed results")
    else:
        print("\nPIPELINE EXECUTION FAILED")
        print("Check the error messages above for details")


if __name__ == "__main__":
    main()
