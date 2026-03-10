"""
Firebase initialization and database schema setup.
Critical for maintaining agent state, performance logs, and user management.
Using firebase-admin as required by ecosystem constraints.
"""
import firebase_admin
from firebase_admin import credentials, firestore
import logging
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Persistent agent belief system"""
    confidence_score: float = 0.5
    risk_tolerance: float = 0.3
    market_regime: str = "unknown"
    last_updated: datetime = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
        if self.performance_metrics is None:
            self.performance_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "total_signals": 0
            }

class FirebaseManager:
    """Singleton Firebase manager for all database operations"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_firebase()
            self._setup_collections()
            self._initialized = True
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase with service account credentials"""
        try:
            # Check for service account file
            service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json")
            
            if not os.path.exists(service_account_path):
                logger.error(f"Firebase service account file not found: {service_account_path}")
                raise FileNotFoundError(f"Service account file required at {service_account_path}")
            
            # Initialize Firebase
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise
    
    def _setup_collections(self) -> None:
        """Create required Firestore collections with validation rules"""
        # Collections will be created automatically on first write
        # This method ensures schema consistency
        self.collections = {
            "agent_state": self.db.collection("agent_state"),
            "performance_logs": self.db.collection("performance_logs"),
            "circuit_breaker_rules": self.db.collection("circuit_breaker_rules"),
            "user_sessions": self.db.collection("user_sessions"),
            "signal_history": self.db.collection("signal_history")
        }
        
        # Initialize default circuit breaker rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Set up default circuit breaker safety rules"""
        default_rules = {
            "confidence_threshold": 0.7,
            "max_daily_loss_pct": 5.0,
            "consecutive_failures": 3,
            "volatility_threshold": 2.0,
            "data_freshness_seconds": 300,
            "hallucination_detection_enabled": True
        }
        
        try:
            rules_ref = self.collections["circuit_breaker_rules"].document("default")
            if not rules_ref.get().exists:
                rules_ref.set(default_rules)
                logger.info("Default circuit breaker rules initialized")
        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker rules: {str(e)}")
    
    def save_agent_state(self, agent_id: str, state: AgentState) -> bool:
        """Save agent state to Firestore with error handling"""
        try:
            state_dict = asdict(state)
            state_dict["last_updated"] = firestore.SERVER_TIMESTAMP
            
            self.collections["agent_state"].document(agent_id).set(state_dict)
            logger.debug(f"Agent state saved for {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save agent state: {str(e)}")
            return False
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Retrieve agent state from Firestore"""
        try:
            doc = self.collections["agent_state"].document(agent_id).get()
            if doc.exists:
                data = doc.to_dict()
                return AgentState(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve agent state: {str(e)}")
            return None
    
    def log_performance(self, metrics: Dict[str, Any]) -> bool:
        """Log performance metrics with timestamp"""
        try:
            metrics["timestamp"] = firestore.SERVER_TIMESTAMP
            self.collections["performance_logs"].add(metrics)
            logger.info(f"Performance logged: {metrics.get('metric_name', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to log performance: {str(e)}")
            return False
    
    def get_circuit_breaker_rules(self) -> Dict[str, Any]:
        """Retrieve current circuit breaker rules"""
        try:
            doc = self.collections["circuit_breaker_rules"].document("default").get()
            return doc.to_dict() if doc.exists else {}
        except Exception as e:
            logger.error(f"Failed to get circuit breaker rules: {str(e)}")
            return {}

# Global instance
firebase_manager = FirebaseManager()

# Test function for initialization
def test_firebase_connection() -> bool:
    """Test Firebase connection and basic operations"""
    try:
        manager = FirebaseManager()
        test_state = AgentState(confidence_score=0.8, market_regime="bullish")
        success = manager.save_agent_state("test_agent", test_state)
        
        if success:
            retrieved = manager.get_agent_state("test_agent")
            if retrieved and retrieved.confidence_score == 0.8:
                logger.info("Firebase test passed")
                return True
        
        logger.warning("Firebase test failed")
        return False
        
    except Exception as e:
        logger.error(f"Firebase test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    # Run connection test
    if test_firebase_connection():
        print("✅ Firebase setup successful")
    else:
        print("❌ Firebase setup failed")