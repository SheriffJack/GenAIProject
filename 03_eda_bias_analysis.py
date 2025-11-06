import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDABiasAnalyzer:
    """
    Comprehensive EDA with bias detection and fairness analysis
    """
    
    def __init__(self, data_path='data/processed/cleaned_data.csv', output_dir='reports'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bias_report = {
            'platform_bias': {},
            'temporal_bias': {},
            'geographic_bias': {},
            'author_bias': {},
            'engagement_bias': {},
            'fairness_metrics': {}
        }
        
    def load_data(self):
        """Load featured dataset"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def analyze_platform_bias(self, df):
        """Detect platform-specific biases"""
        logger.info("="*60)
        logger.info("ANALYZING PLATFORM BIAS")
        logger.info("="*60)
        
        # Platform-wise misinformation rate
        platform_stats = df.groupby('platform').agg({
            'is_misinformation': ['count', 'sum', 'mean']
        }).round(3)
        platform_stats.columns = ['total_posts', 'misinfo_count', 'misinfo_rate']
        
        logger.info("\nPlatform-wise Misinformation Rates:")
        logger.info(platform_stats.to_string())
        
        # Statistical test: Chi-square for independence
        contingency_table = pd.crosstab(df['platform'], df['is_misinformation'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        logger.info(f"\nChi-square test: χ²={chi2:.4f}, p={p_value:.4e}")
        
        if p_value < 0.05:
            logger.warning("⚠️  SIGNIFICANT platform bias detected (p < 0.05)")
            bias_severity = "HIGH" if p_value < 0.001 else "MODERATE"
        else:
            logger.info("✅ No significant platform bias (p >= 0.05)")
            bias_severity = "LOW"
        
        # Identify biased platforms
        overall_rate = df['is_misinformation'].mean()
        biased_platforms = platform_stats[
            abs(platform_stats['misinfo_rate'] - overall_rate) > 0.15
        ]
        
        self.bias_report['platform_bias'] = {
            'statistics': platform_stats.to_dict(),
            'chi_square': float(chi2),
            'p_value': float(p_value),
            'severity': bias_severity,
            'biased_platforms': biased_platforms.index.tolist(),
            'overall_rate': float(overall_rate)
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot
        platform_stats['misinfo_rate'].plot(kind='bar', ax=axes[0], color='coral')
        axes[0].axhline(overall_rate, color='red', linestyle='--', label='Overall Rate')
        axes[0].set_title('Misinformation Rate by Platform')
        axes[0].set_ylabel('Misinformation Rate')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Count plot
        platform_stats['total_posts'].plot(kind='bar', ax=axes[1], color='skyblue')
        axes[1].set_title('Total Posts by Platform')
        axes[1].set_ylabel('Number of Posts')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'platform_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved visualization to {self.output_dir / 'platform_bias_analysis.png'}")
        
        return platform_stats
    
    def analyze_temporal_bias(self, df):
        """Detect temporal biases"""
        logger.info("="*60)
        logger.info("ANALYZING TEMPORAL BIAS")
        logger.info("="*60)
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year_month'] = df['timestamp'].dt.to_period('M')
        
        # Monthly trend
        monthly_stats = df.groupby('year_month').agg({
            'is_misinformation': ['count', 'mean']
        })
        monthly_stats.columns = ['count', 'misinfo_rate']
        
        logger.info("\nMonthly Misinformation Trends:")
        logger.info(monthly_stats.tail(10).to_string())
        
        # Test for temporal trend (correlation with time)
        df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        correlation, p_value = stats.pearsonr(df['time_numeric'], df['is_misinformation'])
        
        logger.info(f"\nTemporal correlation: r={correlation:.4f}, p={p_value:.4e}")
        
        if abs(correlation) > 0.1 and p_value < 0.05:
            logger.warning(f"⚠️  SIGNIFICANT temporal bias detected")
            bias_severity = "HIGH"
        else:
            logger.info("✅ No significant temporal bias")
            bias_severity = "LOW"
        
        # Day of week analysis
        dow_stats = df.groupby('day_of_week')['is_misinformation'].mean()
        
        self.bias_report['temporal_bias'] = {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'severity': bias_severity,
            'monthly_trend': monthly_stats.to_dict(),
            'day_of_week_rates': dow_stats.to_dict()
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Monthly trend
        monthly_stats['misinfo_rate'].plot(ax=axes[0], marker='o', color='orange')
        axes[0].set_title('Misinformation Rate Over Time')
        axes[0].set_ylabel('Misinformation Rate')
        axes[0].set_xlabel('Month')
        axes[0].grid(True, alpha=0.3)
        
        # Day of week
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_stats.plot(kind='bar', ax=axes[1], color='teal')
        axes[1].set_title('Misinformation Rate by Day of Week')
        axes[1].set_ylabel('Misinformation Rate')
        axes[1].set_xticklabels(dow_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved visualization")
        
        return monthly_stats
    
    def analyze_geographic_bias(self, df):
        """Detect geographic biases"""
        logger.info("="*60)
        logger.info("ANALYZING GEOGRAPHIC BIAS")
        logger.info("="*60)
        
        if 'country' not in df.columns:
            logger.warning("⚠️  No country column found, skipping geographic analysis")
            return None
        
        # Country-wise statistics
        country_stats = df.groupby('country').agg({
            'is_misinformation': ['count', 'sum', 'mean']
        }).round(3)
        country_stats.columns = ['total_posts', 'misinfo_count', 'misinfo_rate']
        
        logger.info("\nCountry-wise Misinformation Rates:")
        logger.info(country_stats.to_string())
        
        # Chi-square test
        contingency_table = pd.crosstab(df['country'], df['is_misinformation'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        logger.info(f"\nChi-square test: χ²={chi2:.4f}, p={p_value:.4e}")
        
        if p_value < 0.05:
            logger.warning("⚠️  SIGNIFICANT geographic bias detected")
            bias_severity = "HIGH"
        else:
            logger.info("✅ No significant geographic bias")
            bias_severity = "LOW"
        
        self.bias_report['geographic_bias'] = {
            'statistics': country_stats.to_dict(),
            'chi_square': float(chi2),
            'p_value': float(p_value),
            'severity': bias_severity
        }
        
        # Visualization
        plt.figure(figsize=(12, 6))
        country_stats['misinfo_rate'].plot(kind='bar', color='crimson')
        plt.title('Misinformation Rate by Country')
        plt.ylabel('Misinformation Rate')
        plt.xlabel('Country')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'geographic_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return country_stats
    
    def analyze_author_verification_bias(self, df):
        """Detect bias related to author verification status"""
        logger.info("="*60)
        logger.info("ANALYZING AUTHOR VERIFICATION BIAS")
        logger.info("="*60)
        
        if 'author_verified' not in df.columns:
            logger.warning("⚠️  No author_verified column found")
            return None
        
        # Verification status vs misinformation
        verification_stats = df.groupby('author_verified').agg({
            'is_misinformation': ['count', 'mean']
        }).round(3)
        verification_stats.columns = ['count', 'misinfo_rate']
        
        logger.info("\nVerification Status vs Misinformation:")
        logger.info(verification_stats.to_string())
        
        # T-test for difference
        verified = df[df['author_verified'] == 1]['is_misinformation']
        unverified = df[df['author_verified'] == 0]['is_misinformation']
        
        t_stat, p_value = stats.ttest_ind(verified, unverified)
        
        logger.info(f"\nT-test: t={t_stat:.4f}, p={p_value:.4e}")
        
        if p_value < 0.05:
            logger.warning("⚠️  SIGNIFICANT author verification bias detected")
            bias_severity = "HIGH"
        else:
            logger.info("✅ No significant author verification bias")
            bias_severity = "LOW"
        
        self.bias_report['author_bias'] = {
            'statistics': verification_stats.to_dict(),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'severity': bias_severity
        }
        
        return verification_stats
    
    def analyze_engagement_bias(self, df):
        """Analyze if high-engagement posts are disproportionately misinfo"""
        logger.info("="*60)
        logger.info("ANALYZING ENGAGEMENT BIAS")
        logger.info("="*60)
        
        # Correlation between engagement and misinformation
        correlation, p_value = stats.pearsonr(df['engagement'], df['is_misinformation'])
        
        logger.info(f"Engagement-Misinformation correlation: r={correlation:.4f}, p={p_value:.4e}")
        
        # Split into engagement quartiles
        df['engagement_quartile'] = pd.qcut(df['engagement'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_stats = df.groupby('engagement_quartile')['is_misinformation'].mean()
        
        logger.info("\nMisinformation Rate by Engagement Quartile:")
        logger.info(quartile_stats.to_string())
        
        if abs(correlation) > 0.15 and p_value < 0.05:
            logger.warning("⚠️  SIGNIFICANT engagement bias detected")
            bias_severity = "HIGH"
        else:
            logger.info("✅ No significant engagement bias")
            bias_severity = "LOW"
        
        self.bias_report['engagement_bias'] = {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'severity': bias_severity,
            'quartile_rates': quartile_stats.to_dict()
        }
        
        # Visualization
        plt.figure(figsize=(10, 6))
        quartile_stats.plot(kind='bar', color='purple')
        plt.title('Misinformation Rate by Engagement Level')
        plt.ylabel('Misinformation Rate')
        plt.xlabel('Engagement Quartile')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'engagement_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return quartile_stats
    
    def feature_correlation_analysis(self, df):
        """Analyze feature correlations"""
        logger.info("="*60)
        logger.info("ANALYZING FEATURE CORRELATIONS")
        logger.info("="*60)
        
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_misinformation']
        
        # Correlation with target
        target_corr = df[numeric_cols + ['is_misinformation']].corr()['is_misinformation'].drop('is_misinformation')
        target_corr = target_corr.abs().sort_values(ascending=False)
        
        logger.info("\nTop 15 Features by Correlation with Target:")
        logger.info(target_corr.head(15).to_string())
        
        # Feature correlation matrix (top features only)
        top_features = target_corr.head(20).index.tolist() + ['is_misinformation']
        corr_matrix = df[top_features].corr()
        
        # Visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix (Top 20 Features)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved correlation matrix")
        
        return target_corr
    
    def identify_top_predictive_features(self, df):
        """Identify top features with statistical significance"""
        logger.info("="*60)
        logger.info("IDENTIFYING TOP PREDICTIVE FEATURES")
        logger.info("="*60)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_misinformation']
        
        feature_stats = []
        
        for col in numeric_cols:
            try:
                # Split by target
                group_0 = df[df['is_misinformation'] == 0][col]
                group_1 = df[df['is_misinformation'] == 1][col]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)
                
                # Effect size (Cohen's d)
                mean_diff = group_1.mean() - group_0.mean()
                pooled_std = np.sqrt((group_0.std()**2 + group_1.std()**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                feature_stats.append({
                    'feature': col,
                    't_statistic': abs(t_stat),
                    'p_value': p_value,
                    'cohens_d': abs(cohens_d),
                    'mean_real': group_0.mean(),
                    'mean_misinfo': group_1.mean()
                })
            except:
                continue
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_stats)
        feature_df = feature_df.sort_values('cohens_d', ascending=False)
        
        # Filter significant features
        significant = feature_df[feature_df['p_value'] < 0.05]
        
        logger.info(f"\nTop 10 Most Predictive Features (by effect size):")
        logger.info(significant.head(10).to_string(index=False))
        
        # Save to file
        significant.to_csv(self.output_dir / 'top_predictive_features.csv', index=False)
        logger.info(f"✅ Saved feature statistics")
        
        return significant
    
    def calculate_fairness_metrics(self, df):
        """Calculate fairness metrics across groups"""
        logger.info("="*60)
        logger.info("CALCULATING FAIRNESS METRICS")
        logger.info("="*60)
        
        fairness_metrics = {}
        
        # Platform fairness (equal opportunity)
        if 'platform' in df.columns:
            platform_tpr = {}
            for platform in df['platform'].unique():
                platform_data = df[df['platform'] == platform]
                true_positives = ((platform_data['is_misinformation'] == 1)).sum()
                total_positives = (platform_data['is_misinformation'] == 1).sum()
                tpr = true_positives / total_positives if total_positives > 0 else 0
                platform_tpr[platform] = tpr
            
            fairness_metrics['platform_equal_opportunity'] = platform_tpr
            logger.info(f"\nPlatform True Positive Rates: {platform_tpr}")
        
        # Demographic parity (country)
        if 'country' in df.columns:
            country_rates = df.groupby('country')['is_misinformation'].mean().to_dict()
            fairness_metrics['country_demographic_parity'] = country_rates
            logger.info(f"\nCountry Misinformation Rates: {country_rates}")
        
        self.bias_report['fairness_metrics'] = fairness_metrics
        
        logger.info("✅ Fairness metrics calculated")
        
        return fairness_metrics
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive bias report"""
        logger.info("="*60)
        logger.info("GENERATING COMPREHENSIVE BIAS REPORT")
        logger.info("="*60)
    
    # Convert non-serializable objects to strings/primitives
        def make_json_serializable(obj):
            """Convert pandas/numpy objects to JSON-serializable types"""
            if isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):  # datetime/Period objects
                return str(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
    
    # Clean bias report for JSON serialization
        cleaned_report = make_json_serializable(self.bias_report)
    
    # Save bias report JSON
        report_path = self.output_dir / 'bias_detection_report.json'
        with open(report_path, 'w') as f:
            json.dump(cleaned_report, f, indent=2)
    
        logger.info(f"✅ Saved bias report to {report_path}")
    
    # Generate text summary
        summary_path = self.output_dir / 'bias_detection_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("BIAS DETECTION & FAIRNESS ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
        
            for bias_type, results in self.bias_report.items():
                f.write(f"\n{bias_type.upper().replace('_', ' ')}:\n")
                f.write("-" * 70 + "\n")
            
                if isinstance(results, dict) and 'severity' in results:
                    f.write(f"Severity: {results['severity']}\n")
                    f.write(f"P-value: {results.get('p_value', 'N/A')}\n")
            
                f.write("\n")
    
        logger.info(f"✅ Saved text summary to {summary_path}")
    
    # Overall bias assessment
        high_bias_count = sum(
            1 for results in self.bias_report.values()
            if isinstance(results, dict) and results.get('severity') == 'HIGH'
    )
    
        if high_bias_count > 2:
            logger.warning(f"⚠️  WARNING: {high_bias_count} types of HIGH bias detected!")
            logger.warning("    Consider bias mitigation strategies before deployment")
        elif high_bias_count > 0:
            logger.info(f"⚠️  {high_bias_count} type(s) of HIGH bias detected - monitor during training")
        else:
            logger.info("✅ No HIGH severity biases detected")
    
        return self.bias_report
    
    def run_full_analysis(self):
        """Execute complete EDA and bias analysis"""
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE EDA & BIAS ANALYSIS")
        logger.info("="*60)
        
        df = self.load_data()
        
        # Run all analyses
        self.analyze_platform_bias(df)
        self.analyze_temporal_bias(df)
        self.analyze_geographic_bias(df)
        self.analyze_author_verification_bias(df)
        self.analyze_engagement_bias(df)
        self.feature_correlation_analysis(df)
        top_features = self.identify_top_predictive_features(df)
        self.calculate_fairness_metrics(df)
        
        # Generate final report
        bias_report = self.generate_comprehensive_report()
        
        logger.info("="*60)
        logger.info("✅ EDA & BIAS ANALYSIS COMPLETED")
        logger.info("="*60)
        
        return bias_report, top_features


if __name__ == "__main__":
    # Create output directory
    Path('reports').mkdir(exist_ok=True)
    
    # Run analysis
    analyzer = EDABiasAnalyzer()
    bias_report, top_features = analyzer.run_full_analysis()
    
    print("\n" + "="*60)
    print("EDA & BIAS ANALYSIS COMPLETE")
    print("Check 'reports/' directory for visualizations and reports")
    print("="*60)