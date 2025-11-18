"""
DATABASE SCHEMA & ORM MODELS
PostgreSQL schema with SQLAlchemy ORM for startup analytics
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import pandas as pd

Base = declarative_base()


# ============================================================================
# CORE TABLES
# ============================================================================

class Startup(Base):
    """Main startup information table"""
    __tablename__ = 'startups'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    founded_date = Column(DateTime)
    website = Column(String(255))
    
    # Location
    country = Column(String(100), index=True)
    city = Column(String(100))
    region = Column(String(100))
    
    # Classification
    sector = Column(String(100), index=True)
    industry = Column(String(100))
    stage = Column(String(50), index=True)  # Seed, Series A, B, C, etc.
    
    # Key metrics
    employee_count = Column(Integer)
    customer_count = Column(Integer)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    funding_rounds = relationship("FundingRound", back_populates="startup")
    financial_metrics = relationship("FinancialMetric", back_populates="startup")
    predictions = relationship("Prediction", back_populates="startup")
    cluster_assignments = relationship("ClusterAssignment", back_populates="startup")


class FundingRound(Base):
    """Funding round details"""
    __tablename__ = 'funding_rounds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    startup_id = Column(Integer, ForeignKey('startups.id'), nullable=False, index=True)
    
    round_type = Column(String(50), index=True)  # Seed, Series A, etc.
    amount_usd = Column(Numeric(15, 2))
    pre_money_valuation = Column(Numeric(15, 2))
    post_money_valuation = Column(Numeric(15, 2))
    
    funding_date = Column(DateTime, index=True)
    announced_date = Column(DateTime)
    
    lead_investor_id = Column(Integer, ForeignKey('investors.id'))
    investor_count = Column(Integer)
    
    equity_percentage = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    startup = relationship("Startup", back_populates="funding_rounds")
    lead_investor = relationship("Investor", foreign_keys=[lead_investor_id])
    investments = relationship("Investment", back_populates="funding_round")


class Investor(Base):
    """Investor/VC firm information"""
    __tablename__ = 'investors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    investor_type = Column(String(50))  # VC, Angel, PE, Corporate, etc.
    
    country = Column(String(100))
    city = Column(String(100))
    
    total_investments = Column(Integer)
    total_amount_invested = Column(Numeric(15, 2))
    
    focus_sectors = Column(Text)  # JSON array stored as text
    focus_stages = Column(Text)   # JSON array stored as text
    
    website = Column(String(255))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    investments = relationship("Investment", back_populates="investor")


class Investment(Base):
    """Many-to-many relationship between investors and funding rounds"""
    __tablename__ = 'investments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    investor_id = Column(Integer, ForeignKey('investors.id'), nullable=False, index=True)
    funding_round_id = Column(Integer, ForeignKey('funding_rounds.id'), nullable=False, index=True)
    
    amount_usd = Column(Numeric(15, 2))
    is_lead_investor = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    investor = relationship("Investor", back_populates="investments")
    funding_round = relationship("FundingRound", back_populates="investments")


# ============================================================================
# FINANCIAL & OPERATIONAL METRICS
# ============================================================================

class FinancialMetric(Base):
    """Time-series financial data"""
    __tablename__ = 'financial_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    startup_id = Column(Integer, ForeignKey('startups.id'), nullable=False, index=True)
    
    metric_date = Column(DateTime, index=True)
    reporting_period = Column(String(20))  # Q1-2024, 2024-01, etc.
    
    # Revenue metrics
    revenue = Column(Numeric(15, 2))
    recurring_revenue = Column(Numeric(15, 2))
    revenue_growth_rate = Column(Float)
    
    # Cost metrics
    operating_expenses = Column(Numeric(15, 2))
    burn_rate = Column(Numeric(15, 2))
    runway_months = Column(Float)
    
    # Profitability
    gross_profit = Column(Numeric(15, 2))
    gross_margin = Column(Float)
    ebitda = Column(Numeric(15, 2))
    net_income = Column(Numeric(15, 2))
    
    # Cash
    cash_balance = Column(Numeric(15, 2))
    
    # Customer metrics
    customer_acquisition_cost = Column(Numeric(10, 2))
    customer_lifetime_value = Column(Numeric(10, 2))
    churn_rate = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    startup = relationship("Startup", back_populates="financial_metrics")


# ============================================================================
# ML MODEL OUTPUTS
# ============================================================================

class Prediction(Base):
    """ML model predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    startup_id = Column(Integer, ForeignKey('startups.id'), nullable=False, index=True)
    
    model_version = Column(String(50))
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Valuation predictions
    predicted_valuation = Column(Numeric(15, 2))
    valuation_confidence_lower = Column(Numeric(15, 2))
    valuation_confidence_upper = Column(Numeric(15, 2))
    
    # Growth predictions
    predicted_growth_rate = Column(Float)
    growth_category = Column(String(50))  # High, Medium, Low, Declining
    
    # Risk scores
    success_probability = Column(Float)
    failure_risk_score = Column(Float)
    
    # Feature importance (top 5 features as JSON)
    key_drivers = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    startup = relationship("Startup", back_populates="predictions")


class ClusterAssignment(Base):
    """Startup cluster assignments from K-Means"""
    __tablename__ = 'cluster_assignments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    startup_id = Column(Integer, ForeignKey('startups.id'), nullable=False, index=True)
    
    cluster_id = Column(Integer, index=True)
    cluster_name = Column(String(100))  # High Growth, Emerging, etc.
    
    assignment_date = Column(DateTime, default=datetime.utcnow, index=True)
    model_version = Column(String(50))
    
    # Distance from cluster center
    distance_from_center = Column(Float)
    silhouette_score = Column(Float)
    
    # Cluster characteristics
    cluster_size = Column(Integer)
    avg_cluster_funding = Column(Numeric(15, 2))
    avg_cluster_valuation = Column(Numeric(15, 2))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    startup = relationship("Startup", back_populates="cluster_assignments")


# ============================================================================
# MARKET DATA
# ============================================================================

class MarketIndicator(Base):
    """External market indicators"""
    __tablename__ = 'market_indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_date = Column(DateTime, index=True)
    
    # Stock indices
    sp500_index = Column(Float)
    nasdaq_index = Column(Float)
    venture_capital_index = Column(Float)
    
    # Economic indicators
    interest_rate = Column(Float)
    gdp_growth = Column(Float)
    unemployment_rate = Column(Float)
    
    # VC market
    total_vc_funding = Column(Numeric(15, 2))
    deal_count = Column(Integer)
    ipo_count = Column(Integer)
    ma_volume = Column(Numeric(15, 2))
    
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# DATABASE MANAGER CLASS
# ============================================================================

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, connection_string: str):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                Example: 'postgresql://user:password@localhost:5432/startup_analytics'
        """
        self.engine = create_engine(connection_string, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_all_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(self.engine)
        print("✓ All tables created successfully")
        
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(self.engine)
        print("✓ All tables dropped")
        
    def bulk_insert_startups(self, df: pd.DataFrame):
        """
        Bulk insert startup data from DataFrame
        
        Args:
            df: DataFrame with startup data
        """
        session = self.Session()
        
        try:
            startups = []
            for _, row in df.iterrows():
                startup = Startup(
                    external_id=row.get('startup_id'),
                    name=row.get('startup_name'),
                    sector=row.get('sector'),
                    country=row.get('country'),
                    stage=row.get('funding_round'),
                    employee_count=row.get('employee_count'),
                    customer_count=row.get('customer_count')
                )
                startups.append(startup)
            
            session.bulk_save_objects(startups)
            session.commit()
            print(f"✓ Inserted {len(startups)} startups")
            
        except Exception as e:
            session.rollback()
            print(f"Error inserting startups: {e}")
        finally:
            session.close()
    
    def bulk_insert_funding_rounds(self, df: pd.DataFrame):
        """Bulk insert funding round data"""
        session = self.Session()
        
        try:
            funding_rounds = []
            for _, row in df.iterrows():
                # Get startup ID from database
                startup = session.query(Startup).filter_by(
                    external_id=row.get('startup_id')
                ).first()
                
                if startup:
                    funding_round = FundingRound(
                        startup_id=startup.id,
                        round_type=row.get('funding_round'),
                        amount_usd=row.get('funding_amount'),
                        post_money_valuation=row.get('valuation'),
                        funding_date=row.get('funding_date'),
                        investor_count=row.get('investor_count')
                    )
                    funding_rounds.append(funding_round)
            
            session.bulk_save_objects(funding_rounds)
            session.commit()
            print(f"✓ Inserted {len(funding_rounds)} funding rounds")
            
        except Exception as e:
            session.rollback()
            print(f"Error inserting funding rounds: {e}")
        finally:
            session.close()
    
    def bulk_insert_predictions(self, df: pd.DataFrame, model_version: str):
        """Bulk insert ML predictions"""
        session = self.Session()
        
        try:
            predictions = []
            for _, row in df.iterrows():
                startup = session.query(Startup).filter_by(
                    external_id=row.get('startup_id')
                ).first()
                
                if startup:
                    prediction = Prediction(
                        startup_id=startup.id,
                        model_version=model_version,
                        predicted_valuation=row.get('predicted_valuation'),
                        predicted_growth_rate=row.get('growth_potential'),
                        growth_category=row.get('growth_category')
                    )
                    predictions.append(prediction)
            
            session.bulk_save_objects(predictions)
            session.commit()
            print(f"✓ Inserted {len(predictions)} predictions")
            
        except Exception as e:
            session.rollback()
            print(f"Error inserting predictions: {e}")
        finally:
            session.close()
    
    def get_startups_by_sector(self, sector: str) -> pd.DataFrame:
        """Query startups by sector and return as DataFrame"""
        session = self.Session()
        
        try:
            query = session.query(Startup).filter_by(sector=sector)
            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def setup_database():
    """Example setup"""
    
    # Connection string (use environment variables in production)
    conn_string = "postgresql://user:password@localhost:5432/startup_analytics"
    
    # Initialize database
    db = DatabaseManager(conn_string)
    
    # Create tables
    db.create_all_tables()
    
    print("\n✓ Database setup complete!")
    print("Tables created:")
    print("  - startups")
    print("  - funding_rounds")
    print("  - investors")
    print("  - investments")
    print("  - financial_metrics")
    print("  - predictions")
    print("  - cluster_assignments")
    print("  - market_indicators")
    
    return db


if __name__ == "__main__":
    setup_database()
