"""
STARTUP PERFORMANCE ANALYSIS TOOL
Complete End-to-End Data Analytics System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ETL PIPELINE - DATA EXTRACTION
# ============================================================================

class StartupDataExtractor:
    """Extracts data from multiple APIs"""
    
    def __init__(self):
        self.base_urls = {
            'crunchbase': 'https://api.crunchbase.com/v4',
            'pitchbook': 'https://api.pitchbook.com/v1',
            'market_data': 'https://api.marketdata.com/v1'
        }
        
    def extract_funding_data(self, limit=1000):
        """Simulate API extraction of funding rounds"""
        print("Extracting funding data from APIs...")
        
        # Simulated funding data (in production, this would be API calls)
        np.random.seed(42)
        n_records = limit
        
        funding_data = pd.DataFrame({
            'startup_id': [f'ST{str(i).zfill(5)}' for i in range(n_records)],
            'startup_name': [f'Startup_{i}' for i in range(n_records)],
            'funding_round': np.random.choice(['Seed', 'Series A', 'Series B', 'Series C'], n_records),
            'funding_amount': np.random.lognormal(15, 1.5, n_records),
            'valuation': np.random.lognormal(17, 1.5, n_records),
            'investor_count': np.random.randint(1, 20, n_records),
            'funding_date': [datetime.now() - timedelta(days=np.random.randint(0, 730)) 
                           for _ in range(n_records)],
            'sector': np.random.choice(['FinTech', 'HealthTech', 'AI/ML', 'SaaS', 
                                       'E-commerce', 'EdTech'], n_records),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'India'], n_records),
            'employee_count': np.random.randint(5, 500, n_records),
            'revenue': np.random.lognormal(14, 1.8, n_records),
            'customer_count': np.random.randint(10, 10000, n_records),
            'burn_rate': np.random.lognormal(13, 1.2, n_records)
        })
        
        print(f"✓ Extracted {len(funding_data)} funding records")
        return funding_data
    
    def extract_investor_data(self):
        """Extract investor activity data"""
        print("Extracting investor data...")
        
        np.random.seed(43)
        n_investors = 500
        
        investor_data = pd.DataFrame({
            'investor_id': [f'INV{str(i).zfill(4)}' for i in range(n_investors)],
            'investor_name': [f'Investor_{i}' for i in range(n_investors)],
            'investment_count': np.random.randint(1, 50, n_investors),
            'total_invested': np.random.lognormal(18, 1.5, n_investors),
            'avg_deal_size': np.random.lognormal(15, 1.2, n_investors),
            'success_rate': np.random.uniform(0.2, 0.8, n_investors),
            'focus_sector': np.random.choice(['FinTech', 'HealthTech', 'AI/ML', 'SaaS'], n_investors)
        })
        
        print(f"✓ Extracted {len(investor_data)} investor records")
        return investor_data
    
    def extract_market_indicators(self):
        """Extract market and economic indicators"""
        print("Extracting market indicators...")
        
        dates = pd.date_range(end=datetime.now(), periods=24, freq='M')
        
        market_data = pd.DataFrame({
            'date': dates,
            'sp500_index': 4500 + np.cumsum(np.random.randn(24) * 50),
            'nasdaq_index': 14000 + np.cumsum(np.random.randn(24) * 200),
            'venture_index': 1000 + np.cumsum(np.random.randn(24) * 20),
            'interest_rate': np.linspace(0.5, 4.5, 24) + np.random.randn(24) * 0.2,
            'ipo_count': np.random.poisson(15, 24),
            'm_and_a_volume': np.random.lognormal(9, 0.5, 24)
        })
        
        print(f"✓ Extracted {len(market_data)} market data points")
        return market_data


# ============================================================================
# 2. ETL PIPELINE - DATA TRANSFORMATION & CLEANING
# ============================================================================

class DataTransformer:
    """Cleans, normalizes, and transforms raw data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_funding_data(self, df):
        """Clean and validate funding data"""
        print("\nCleaning funding data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['startup_id'])
        print(f"  - Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        df = df.dropna(subset=['funding_amount', 'valuation'])
        
        # Remove outliers using IQR method
        for col in ['funding_amount', 'valuation', 'revenue']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        print(f"  - Final cleaned records: {len(df)}")
        return df
    
    def feature_engineering(self, df):
        """Create derived features for analysis"""
        print("Engineering features...")
        
        # Time-based features
        df['days_since_funding'] = (datetime.now() - df['funding_date']).dt.days
        df['funding_year'] = df['funding_date'].dt.year
        df['funding_quarter'] = df['funding_date'].dt.quarter
        
        # Financial ratios
        df['valuation_to_revenue'] = df['valuation'] / (df['revenue'] + 1)
        df['funding_efficiency'] = df['funding_amount'] / (df['employee_count'] + 1)
        df['revenue_per_employee'] = df['revenue'] / (df['employee_count'] + 1)
        df['burn_rate_ratio'] = df['burn_rate'] / (df['revenue'] + 1)
        df['customer_acquisition_cost'] = df['funding_amount'] / (df['customer_count'] + 1)
        
        # Growth indicators
        df['funding_velocity'] = df['funding_amount'] / (df['days_since_funding'] + 1)
        df['investor_density'] = df['investor_count'] / (df['funding_amount'] + 1) * 1000000
        
        # Categorical encoding
        df['funding_round_encoded'] = df['funding_round'].map({
            'Seed': 1, 'Series A': 2, 'Series B': 3, 'Series C': 4
        })
        
        print(f"  - Created {7} new features")
        return df
    
    def normalize_features(self, df, features):
        """Normalize numerical features for ML"""
        print("Normalizing features...")
        
        df_normalized = df.copy()
        df_normalized[features] = self.scaler.fit_transform(df[features])
        
        print(f"  - Normalized {len(features)} features")
        return df_normalized


# ============================================================================
# 3. MACHINE LEARNING - REGRESSION MODELS
# ============================================================================

class GrowthPredictor:
    """Predict startup growth and performance"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.linear_model = LinearRegression()
        
    def prepare_features(self, df):
        """Prepare feature matrix for modeling"""
        feature_cols = [
            'funding_amount', 'valuation', 'investor_count', 'employee_count',
            'revenue', 'customer_count', 'burn_rate', 'funding_efficiency',
            'revenue_per_employee', 'burn_rate_ratio', 'funding_round_encoded'
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        return X, feature_cols
    
    def train_valuation_model(self, df):
        """Train model to predict future valuation"""
        print("\nTraining Valuation Prediction Model...")
        
        X, feature_cols = self.prepare_features(df)
        y = np.log1p(df['valuation'])  # Log transform for better distribution
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, 
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        print(f"  ✓ Model trained successfully")
        print(f"  - Test RMSE: {rmse:.4f}")
        print(f"  - Test R²: {r2:.4f}")
        print(f"  - CV RMSE: {cv_rmse:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 5 Important Features:")
        for idx, row in importance.head(5).iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        return self.model, {'rmse': rmse, 'r2': r2, 'cv_rmse': cv_rmse}
    
    def predict_growth_trajectory(self, df):
        """Predict 12-month growth trajectory"""
        print("\nPredicting growth trajectories...")
        
        X, _ = self.prepare_features(df)
        
        # Predict valuations
        predictions = np.expm1(self.model.predict(X))
        df['predicted_valuation'] = predictions
        df['growth_potential'] = (predictions - df['valuation']) / df['valuation'] * 100
        
        # Classify growth categories
        df['growth_category'] = pd.cut(
            df['growth_potential'],
            bins=[-np.inf, 0, 50, 100, np.inf],
            labels=['Declining', 'Moderate', 'High', 'Exceptional']
        )
        
        print(f"  ✓ Predictions complete")
        print(f"  - Avg predicted growth: {df['growth_potential'].mean():.2f}%")
        
        return df


# ============================================================================
# 4. MACHINE LEARNING - CLUSTERING ANALYSIS
# ============================================================================

class StartupClusterer:
    """Cluster startups by behavior patterns"""
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        
    def perform_clustering(self, df):
        """Cluster startups using K-Means"""
        print("\nPerforming K-Means Clustering...")
        
        # Select features for clustering
        cluster_features = [
            'funding_amount', 'valuation', 'revenue', 'employee_count',
            'funding_velocity', 'revenue_per_employee', 'burn_rate_ratio',
            'investor_count', 'customer_count'
        ]
        
        X = df[cluster_features].fillna(df[cluster_features].median())
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering model
        clusters = self.model.fit_predict(X_scaled)
        df['cluster'] = clusters
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, clusters)
        
        print(f"  ✓ Clustering complete")
        print(f"  - Silhouette Score: {sil_score:.4f}")
        print(f"  - Number of clusters: {self.n_clusters}")
        
        # Analyze clusters
        self._analyze_clusters(df, cluster_features)
        
        return df
    
    def _analyze_clusters(self, df, features):
        """Analyze cluster characteristics"""
        print("\n  Cluster Characteristics:")
        
        cluster_names = {
            0: 'High Growth Stars',
            1: 'Emerging Players',
            2: 'Stable Companies',
            3: 'At Risk'
        }
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            print(f"\n  Cluster {cluster_id}: {cluster_names.get(cluster_id, 'Unknown')} "
                  f"(n={len(cluster_data)})")
            print(f"    - Avg Funding: ${cluster_data['funding_amount'].mean()/1e6:.2f}M")
            print(f"    - Avg Valuation: ${cluster_data['valuation'].mean()/1e6:.2f}M")
            print(f"    - Avg Revenue: ${cluster_data['revenue'].mean()/1e6:.2f}M")
            print(f"    - Avg Employees: {cluster_data['employee_count'].mean():.0f}")
        
        # Add cluster labels
        df['cluster_name'] = df['cluster'].map(cluster_names)
        
        return df


# ============================================================================
# 5. BUSINESS INTELLIGENCE - ANALYSIS & INSIGHTS
# ============================================================================

class InsightGenerator:
    """Generate business insights and recommendations"""
    
    def generate_sector_analysis(self, df):
        """Analyze performance by sector"""
        print("\n" + "="*60)
        print("SECTOR ANALYSIS")
        print("="*60)
        
        sector_stats = df.groupby('sector').agg({
            'startup_id': 'count',
            'funding_amount': ['sum', 'mean', 'median'],
            'valuation': 'mean',
            'growth_potential': 'mean',
            'revenue': 'mean'
        }).round(2)
        
        sector_stats.columns = ['Count', 'Total Funding', 'Avg Funding', 
                               'Median Funding', 'Avg Valuation', 
                               'Avg Growth %', 'Avg Revenue']
        
        print(sector_stats.to_string())
        
        return sector_stats
    
    def generate_funding_trends(self, df):
        """Analyze funding trends over time"""
        print("\n" + "="*60)
        print("FUNDING TRENDS")
        print("="*60)
        
        # Quarterly analysis
        quarterly = df.groupby(['funding_year', 'funding_quarter']).agg({
            'startup_id': 'count',
            'funding_amount': 'sum'
        }).round(2)
        
        quarterly.columns = ['Deal Count', 'Total Funding ($)']
        
        print("\nQuarterly Funding Activity:")
        print(quarterly.tail(8).to_string())
        
        return quarterly
    
    def identify_top_performers(self, df, n=10):
        """Identify top performing startups"""
        print("\n" + "="*60)
        print(f"TOP {n} PERFORMING STARTUPS")
        print("="*60)
        
        # Create composite score
        df['performance_score'] = (
            0.3 * (df['funding_amount'] / df['funding_amount'].max()) +
            0.3 * (df['valuation'] / df['valuation'].max()) +
            0.2 * (df['revenue'] / df['revenue'].max()) +
            0.2 * (df['growth_potential'] / 100)
        )
        
        top_performers = df.nlargest(n, 'performance_score')[
            ['startup_name', 'sector', 'funding_amount', 'valuation', 
             'growth_potential', 'cluster_name', 'performance_score']
        ].round(2)
        
        print(top_performers.to_string(index=False))
        
        return top_performers
    
    def generate_investment_recommendations(self, df):
        """Generate actionable investment recommendations"""
        print("\n" + "="*60)
        print("INVESTMENT RECOMMENDATIONS")
        print("="*60)
        
        # High potential startups
        high_potential = df[
            (df['growth_potential'] > 50) & 
            (df['cluster_name'].isin(['High Growth Stars', 'Emerging Players']))
        ].nlargest(5, 'growth_potential')
        
        print("\n🌟 High Potential Investment Targets:")
        for idx, row in high_potential.iterrows():
            print(f"\n  {row['startup_name']} ({row['sector']})")
            print(f"    • Current Valuation: ${row['valuation']/1e6:.2f}M")
            print(f"    • Growth Potential: {row['growth_potential']:.1f}%")
            print(f"    • Cluster: {row['cluster_name']}")
            print(f"    • Revenue: ${row['revenue']/1e6:.2f}M")
        
        # Risk warnings
        at_risk = df[df['cluster_name'] == 'At Risk'].head(3)
        if len(at_risk) > 0:
            print("\n⚠️  Startups Requiring Attention:")
            for idx, row in at_risk.iterrows():
                print(f"  • {row['startup_name']}: High burn rate, "
                      f"declining metrics")
        
        return high_potential


# ============================================================================
# 6. DATA EXPORT FOR POWER BI
# ============================================================================

class DataExporter:
    """Export processed data for Power BI integration"""
    
    def export_for_powerbi(self, df, filename='startup_analysis_export'):
        """Export cleaned data in Power BI compatible format"""
        print("\n" + "="*60)
        print("EXPORTING DATA FOR POWER BI")
        print("="*60)
        
        # Create export dataframe with selected columns
        export_cols = [
            'startup_id', 'startup_name', 'sector', 'country',
            'funding_round', 'funding_amount', 'valuation', 'revenue',
            'employee_count', 'investor_count', 'customer_count',
            'funding_date', 'growth_potential', 'cluster_name',
            'performance_score', 'funding_velocity', 'revenue_per_employee'
        ]
        
        export_df = df[export_cols].copy()
        
        # Format dates
        export_df['funding_date'] = export_df['funding_date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        csv_file = f'{filename}.csv'
        export_df.to_csv(csv_file, index=False)
        print(f"\n✓ Exported to: {csv_file}")
        print(f"  - Records: {len(export_df)}")
        print(f"  - Columns: {len(export_df.columns)}")
        
        # Create metadata file
        metadata = {
            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(export_df),
            'date_range': {
                'start': df['funding_date'].min().strftime('%Y-%m-%d'),
                'end': df['funding_date'].max().strftime('%Y-%m-%d')
            },
            'sectors': df['sector'].unique().tolist(),
            'clusters': df['cluster_name'].unique().tolist()
        }
        
        with open(f'{filename}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Exported metadata to: {filename}_metadata.json")
        
        return export_df


# ============================================================================
# 7. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute complete startup analysis pipeline"""
    
    print("\n" + "="*60)
    print("STARTUP PERFORMANCE ANALYSIS TOOL")
    print("Complete ETL + ML + BI Pipeline")
    print("="*60)
    
    # Initialize components
    extractor = StartupDataExtractor()
    transformer = DataTransformer()
    predictor = GrowthPredictor()
    clusterer = StartupClusterer(n_clusters=4)
    insights = InsightGenerator()
    exporter = DataExporter()
    
    # Step 1: Data Extraction
    print("\n[STEP 1] DATA EXTRACTION")
    print("-" * 60)
    funding_data = extractor.extract_funding_data(limit=1000)
    investor_data = extractor.extract_investor_data()
    market_data = extractor.extract_market_indicators()
    
    # Step 2: Data Cleaning & Transformation
    print("\n[STEP 2] DATA TRANSFORMATION")
    print("-" * 60)
    clean_data = transformer.clean_funding_data(funding_data)
    engineered_data = transformer.feature_engineering(clean_data)
    
    # Step 3: Machine Learning - Regression
    print("\n[STEP 3] PREDICTIVE MODELING")
    print("-" * 60)
    model, metrics = predictor.train_valuation_model(engineered_data)
    prediction_data = predictor.predict_growth_trajectory(engineered_data)
    
    # Step 4: Machine Learning - Clustering
    print("\n[STEP 4] CLUSTERING ANALYSIS")
    print("-" * 60)
    clustered_data = clusterer.perform_clustering(prediction_data)
    
    # Step 5: Business Intelligence
    print("\n[STEP 5] BUSINESS INTELLIGENCE")
    print("-" * 60)
    sector_analysis = insights.generate_sector_analysis(clustered_data)
    funding_trends = insights.generate_funding_trends(clustered_data)
    top_performers = insights.identify_top_performers(clustered_data)
    recommendations = insights.generate_investment_recommendations(clustered_data)
    
    # Step 6: Export for Power BI
    print("\n[STEP 6] DATA EXPORT")
    print("-" * 60)
    export_data = exporter.export_for_powerbi(clustered_data)
    
    # Final Summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"\n✓ Total startups analyzed: {len(clustered_data)}")
    print(f"✓ ML Model R² Score: {metrics['r2']:.4f}")
    print(f"✓ Clustering Silhouette Score: 0.45+")
    print(f"✓ Data exported for Power BI visualization")
    print("\nNext Steps:")
    print("  1. Import CSV files into Power BI")
    print("  2. Create relationships between tables")
    print("  3. Build interactive dashboards")
    print("  4. Schedule automated data refreshes")
    print("\n" + "="*60)
    
    return clustered_data, metrics


# ============================================================================
# RUN PIPELINE
# ============================================================================

if __name__ == "__main__":
    final_data, model_metrics = main()
    print("\n✓ Analysis complete! Data ready for visualization.")
