"""
Core sentiment analysis engine using pre-trained transformer models.
Implements multi-source signal fusion with confidence scoring.
Includes hallucination detection and confidence calibration.
"""
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
import re

from firebase_init import firebase_manager, AgentState

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Structured signal output"""
    asset: str
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float
    rationale: str
    sources: List[str]
    timestamp: datetime
    risk_level: str = "medium"
    
    def validate(self) -> bool:
        """Validate signal integrity"""
        valid_directions = {"bullish", "bearish", "neutral"}
        valid_risk = {"low", "medium", "high"}
        
        if self.direction not in valid_directions:
            logger.warning(f"Invalid direction: {self.direction}")
            return False
        
        if self.risk_level not in valid_risk:
            logger.warning(f"Invalid risk level: {self.risk_level}")
            return False
        
        if not 0 <= self.confidence <= 1:
            logger.warning(f"Invalid confidence: {self.confidence}")
            return False
        
        if not self.asset or not isinstance(self.asset, str):
            logger.warning("Invalid asset")
            return False
        
        return True

class MarketSentimentOracle:
    """Main sentiment analysis engine with multi-source fusion"""
    
    def __init__(self, model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
        """Initialize with pre-trained financial sentiment model"""
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize agent state
            self.agent_state = AgentState()
            self.load_state()
            
            # Confidence calibration parameters
            self.calibration_factor = 1.0
            self.min_confidence_threshold = 0.6
            
            logger.info(f"Sentiment Oracle initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment oracle: {str(e)}")
            raise
    
    def load_state(self) -> None:
        """Load agent state from Firebase"""
        try:
            state = firebase_manager.get_agent_state("sentiment_oracle")
            if state:
                self.agent_state = state
                logger.info("Agent state loaded from Firebase")
        except Exception as e:
            logger.error(f"Failed to load agent state: {str(e)}")
    
    def save_state(self) -> bool:
        """Save current agent state to Firebase"""
        try:
            return firebase_manager.save_agent_state("sentiment_oracle", self.agent_state)
        except Exception as e:
            logger.error(f"Failed to save agent state: {str(e)}")
            return False
    
    def analyze_text(self, text: str, source: str = "unknown") -> Tuple[str, float]:
        """
        Analyze single text for sentiment with confidence score
        
        Returns: (sentiment_label, confidence_score)
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for analysis")
            return "neutral", 0.0
        
        try:
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(predictions, dim=1)
                
                # Convert to labels (model-specific mapping)
                label_map = {0: "bearish", 1: "bullish", 2: "neutral"}
                sentiment = label_map.get(predicted_class.item(), "neutral")
                confidence_score = confidence.item()
                
                # Apply confidence calibration
                confidence_score = self._calibrate_confidence(
                    confidence_score, 
                    source,
                    len(text.split())
                )
                
                logger.debug(f"Analyzed text from {source}: {sentiment} ({confidence_score:.2f})")
                return sentiment, confidence_score
                
        except Exception as e:
            logger.error(f"Failed to analyze text: {str(e)}")
            return "neutral", 0.0
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Truncate if too long
        if len(text.split()) > 500:
            text = ' '.join(text.split()[:500])
        
        return text.strip()
    
    def _calibrate_confidence(self, raw_confidence: float, source: str, text_length: int) -> float:
        """Adjust confidence based on source reliability and text quality"""
        try:
            calibrated = raw_confidence
            
            # Source reliability weights
            source_weights = {
                "twitter": 0.9,
                "news": 0.95,
                "reddit": 0.7,
                "telegram": 0.6,
                "unknown": 0.8
            }
            
            weight = source_weights.get(source, 0.8)
            calibrated *= weight
            
            # Text length adjustment (very short texts are less reliable)
            if text_length < 10:
                calibrated *= 0.7
            elif text_length > 100:
                calibrated *= 1.1
            
            # Apply learned calibration factor from agent state
            calibrated *= self.calibration_factor
            
            # Ensure within bounds
            calibrated = max(0.0, min(1.0, calibrated))
            
            return calibrated
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {str(e)}")
            return raw_confidence
    
    def synthesize_signals(self, data_sources: Dict[str, List[str]]) -> List[Signal]:
        """
        Synthesize signals from multiple data sources
        
        Args:
            data_sources: Dict mapping source names to lists of text snippets
            
        Returns: List of validated signals
        """
        signals = []
        source_analysis = {}
        
        try:
            # Analyze each source independently
            for source, texts in data_sources.items():
                if not texts:
                    continue
                    
                sentiments = []
                confidences = []
                
                for text in texts[:10]:  # Limit to 10 texts per source
                    sentiment, confidence = self.analyze_text(text, source)
                    if confidence > self.min_confidence_threshold:
                        sentiments.append(sentiment)
                        confidences.append(confidence)
                
                if sentiments:
                    # Aggregate source sentiment
                    source_sentiment = max(set(sentiments), key=sentiments.count)
                    avg_confidence = mean(confidences) if confidences else 0.0
                    
                    source_analysis[source] = {
                        "sentiment": source_sentiment