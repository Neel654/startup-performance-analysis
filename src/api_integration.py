"""
API INTEGRATION MODULE
Handles authentication and data retrieval from multiple startup data sources
"""

import requests
import time
from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta
import json

class APIConfig:
    """Configuration for various startup data APIs"""
    
    CRUNCHBASE_API = "https://api.crunchbase.com/api/v4"
    PITCHBOOK_API = "https://api.pitchbook.com/v1"
    DEALROOM_API = "https://api.dealroom.co/api/v1"
    GOOGLE_TRENDS_API = "https://trends.google.com/trends/api"
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds


class StartupAPIClient:
    """
    Unified client for multiple startup data APIs
    Handles authentication, rate limiting, and error handling
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize API client with credentials
        
        Args:
            api_keys: Dictionary containing API keys for different services
                     Example: {'crunchbase': 'xxx', 'pitchbook': 'yyy'}
        """
        self.api_keys = api_keys
        self.request_count = 0
        self.last_request_time = time.time()
        
    def _check_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < 60 and self.request_count >= APIConfig.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - elapsed
            print(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()
    
    def _make_request(self, url: str, headers: Dict, params: Dict = None) -> Optional[Dict]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: API endpoint URL
            headers: Request headers including auth
            params: Query parameters
            
        Returns:
            JSON response or None if failed
        """
        for attempt in range(APIConfig.RETRY_ATTEMPTS):
            try:
                self._check_rate_limit()
                
                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                
                self.request_count += 1
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{APIConfig.RETRY_ATTEMPTS}): {e}")
                
                if attempt < APIConfig.RETRY_ATTEMPTS - 1:
                    time.sleep(APIConfig.RETRY_DELAY * (attempt + 1))
                else:
                    print(f"Failed after {APIConfig.RETRY_ATTEMPTS} attempts")
                    return None
    
    def get_crunchbase_organizations(self, query_params: Dict) -> List[Dict]:
        """
        Fetch organization data from Crunchbase
        
        Args:
            query_params: Search parameters (e.g., location, funding_stage)
            
        Returns:
            List of organization records
        """
        url = f"{APIConfig.CRUNCHBASE_API}/entities/organizations"
        headers = {
            "X-cb-user-key": self.api_keys.get('crunchbase', ''),
            "Content-Type": "application/json"
        }
        
        # Example query structure
        payload = {
            "field_ids": [
                "identifier",
                "name",
                "short_description",
                "rank_org",
                "location_identifiers",
                "funding_total",
                "num_funding_rounds",
                "categories"
            ],
            "query": [
                {
                    "type": "predicate",
                    "field_id": "location_identifiers",
                    "operator_id": "includes",
                    "values": query_params.get('locations', [])
                }
            ],
            "limit": query_params.get('limit', 100)
        }
        
        print(f"Fetching Crunchbase organizations...")
        response = self._make_request(url, headers, params=None)
        
        if response and 'entities' in response:
            print(f"✓ Retrieved {len(response['entities'])} organizations")
            return response['entities']
        return []
    
    def get_funding_rounds(self, org_id: str) -> List[Dict]:
        """
        Get funding rounds for a specific organization
        
        Args:
            org_id: Organization identifier
            
        Returns:
            List of funding round data
        """
        url = f"{APIConfig.CRUNCHBASE_API}/entities/organizations/{org_id}/funding_rounds"
        headers = {
            "X-cb-user-key": self.api_keys.get('crunchbase', ''),
            "Content-Type": "application/json"
        }
        
        response = self._make_request(url, headers)
        
        if response and 'cards' in response:
            return response['cards'].get('funding_rounds', [])
        return []
    
    def get_pitchbook_data(self, company_name: str) -> Optional[Dict]:
        """
        Fetch company data from PitchBook
        
        Args:
            company_name: Name of the company
            
        Returns:
            Company data dictionary
        """
        url = f"{APIConfig.PITCHBOOK_API}/companies/search"
        headers = {
            "Authorization": f"Bearer {self.api_keys.get('pitchbook', '')}",
            "Content-Type": "application/json"
        }
        params = {"q": company_name}
        
        print(f"Searching PitchBook for: {company_name}")
        return self._make_request(url, headers, params)
    
    def get_investor_profile(self, investor_id: str) -> Optional[Dict]:
        """
        Get detailed investor profile
        
        Args:
            investor_id: Investor identifier
            
        Returns:
            Investor profile data
        """
        url = f"{APIConfig.CRUNCHBASE_API}/entities/investors/{investor_id}"
        headers = {
            "X-cb-user-key": self.api_keys.get('crunchbase', ''),
            "Content-Type": "application/json"
        }
        
        return self._make_request(url, headers)
    
    def get_market_trends(self, keywords: List[str], timeframe: str = '3m') -> Dict:
        """
        Get search trend data for startup-related keywords
        
        Args:
            keywords: List of keywords to track
            timeframe: Time period ('1m', '3m', '12m')
            
        Returns:
            Trend data
        """
        # Note: Google Trends requires additional setup
        # This is a simplified example
        print(f"Fetching trends for: {', '.join(keywords)}")
        
        # Simulated trend data
        return {
            'keywords': keywords,
            'timeframe': timeframe,
            'data': [{'date': datetime.now(), 'interest': 75}]
        }


class DataAggregator:
    """
    Aggregates data from multiple API sources into unified format
    """
    
    def __init__(self, api_client: StartupAPIClient):
        self.client = api_client
        
    def aggregate_startup_data(self, startup_identifiers: List[str]) -> List[Dict]:
        """
        Fetch and combine data from multiple sources for each startup
        
        Args:
            startup_identifiers: List of startup IDs or names
            
        Returns:
            List of enriched startup records
        """
        enriched_data = []
        
        for identifier in startup_identifiers:
            print(f"\nAggregating data for: {identifier}")
            
            # Fetch from multiple sources
            crunchbase_data = self.client.get_crunchbase_organizations(
                {'name': identifier}
            )
            funding_rounds = self.client.get_funding_rounds(identifier)
            pitchbook_data = self.client.get_pitchbook_data(identifier)
            
            # Combine data
            combined = {
                'identifier': identifier,
                'crunchbase': crunchbase_data,
                'funding_rounds': funding_rounds,
                'pitchbook': pitchbook_data,
                'aggregated_at': datetime.now().isoformat()
            }
            
            enriched_data.append(combined)
            
            # Brief pause between startups
            time.sleep(0.5)
        
        return enriched_data
    
    def save_raw_data(self, data: List[Dict], filename: str):
        """Save raw API responses for backup"""
        filepath = f"raw_data/{filename}_{datetime.now().strftime('%Y%m%d')}.json"
        
        os.makedirs('raw_data', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ Saved raw data to: {filepath}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the API integration module"""
    
    # Configure API keys (in production, use environment variables)
    api_keys = {
        'crunchbase': os.getenv('CRUNCHBASE_API_KEY', 'your_key_here'),
        'pitchbook': os.getenv('PITCHBOOK_API_KEY', 'your_key_here')
    }
    
    # Initialize client
    client = StartupAPIClient(api_keys)
    aggregator = DataAggregator(client)
    
    # Example: Fetch organizations
    query = {
        'locations': ['San Francisco', 'New York'],
        'limit': 50
    }
    
    orgs = client.get_crunchbase_organizations(query)
    
    # Example: Aggregate data for specific startups
    startup_ids = ['company-123', 'company-456', 'company-789']
    enriched_data = aggregator.aggregate_startup_data(startup_ids)
    
    # Save for processing
    aggregator.save_raw_data(enriched_data, 'startup_data')
    
    print("\n✓ API data collection complete!")
    return enriched_data


if __name__ == "__main__":
    example_usage()
