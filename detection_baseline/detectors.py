from collections.abc import Iterable
from typing import Any, Optional


class FirstSeenDetector:
    """
    Track first-seen entities and flag novelty in context.
    """
    
    def __init__(self, baseline_period_days: int = 30):
        self.known_entities = set()
        self.first_seen_dates = {}
        self.baseline_period = baseline_period_days
        
    def train(self, historical_entities: Iterable[Any], timestamps: Optional[list[Any]] = None) -> "FirstSeenDetector":
        """
        Learn from historical data.
        
        Parameters:
        -----------
        historical_entities : iterable
            The collection of historical entities.
        timestamps : list, optional
            Corresponding timestamps for when each entity was seen.
            
        Returns:
        --------
        self
        """
        for i, entity in enumerate(historical_entities):
            if entity not in self.known_entities:
                self.known_entities.add(entity)
                if timestamps and i < len(timestamps):
                    self.first_seen_dates[entity] = timestamps[i]
        
        print(f'Trained on {len(self.known_entities)} unique entities')
        return self
    
    def evaluate(self, entity: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Evaluate if an entity is novel and suspicious.
        
        Parameters:
        -----------
        entity : any
            The entity to check (e.g., username, IP, process name).
        context : dict, optional
            Context factors: 'hour' (0-23), 'is_weekend' (bool),
            'action' (str), 'user_role' (str).
            
        Returns:
        --------
        dict
            Dictionary indicating if the entity is novel, the risk level, and risk factors.
        """
        is_novel = entity not in self.known_entities
        
        if not is_novel:
            return {'entity': entity, 'is_novel': False, 'risk': 'low'}
        
        # It's novel - assess risk based on context
        risk_factors = []
        
        if context:
            # Night time activity
            if context.get('hour', 12) < 6 or context.get('hour', 12) > 22:
                risk_factors.append('off_hours')
            
            # Weekend activity
            if context.get('is_weekend', False):
                risk_factors.append('weekend')
            
            # Sensitive action
            if context.get('action', '') in ['admin', 'delete', 'export', 'download_bulk']:
                risk_factors.append('sensitive_action')
            
            # High-privilege user
            if context.get('user_role', '') in ['admin', 'root', 'system']:
                risk_factors.append('privileged_user')
        
        # Determine risk level
        if len(risk_factors) >= 2:
            risk = 'high'
        elif len(risk_factors) == 1:
            risk = 'medium'
        else:
            risk = 'low'
        
        return {
            'entity': entity,
            'is_novel': True,
            'risk': risk,
            'risk_factors': risk_factors
        }
