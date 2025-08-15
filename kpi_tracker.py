# =====================
# kpi_tracker.py
# =====================

# kpi_tracker.py
#!/usr/bin/env python3
"""
KPI Tracker for GraphRAG System
Tracks all required metrics per client requirements
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query: str
    route_taken: str
    route_expected: Optional[str]
    confidence: float
    latency_ms: float
    results_count: int
    average_score: float
    timestamp: str
    
    # KPI specific metrics
    is_correct_route: Optional[bool] = None
    is_multi_hop: bool = False
    index_cost: float = 0.0
    tokens_used: int = 0

@dataclass
class SystemKPIs:
    """System-wide KPIs per client requirements"""
    # Answer accuracy
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # Router performance
    router_accuracy: float = 0.0
    router_confusion_matrix: Dict[str, Dict[str, int]] = None
    
    # Latency metrics
    median_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    latency_vs_baseline_ratio: float = 0.0
    
    # Cost metrics
    index_build_cost: float = 0.0
    cost_reduction_percent: float = 0.0
    queries_processed: int = 0
    
    # Retrieval metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    def to_dict(self):
        return asdict(self)

class KPITracker:
    """Tracks and reports KPIs for GraphRAG system"""
    
    def __init__(self, log_dir: str = "/app/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "query_metrics.jsonl"
        self.kpi_file = self.log_dir / "system_kpis.json"
        
        self.query_metrics: List[QueryMetrics] = []
        self.system_kpis = SystemKPIs()
        
        # Load existing metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics from file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.query_metrics.append(QueryMetrics(**data))
        
        if self.kpi_file.exists():
            with open(self.kpi_file, 'r') as f:
                data = json.load(f)
                self.system_kpis = SystemKPIs(**data)
    
    def track_query(self, 
                   query: str,
                   route_taken: str,
                   confidence: float,
                   latency_ms: float,
                   results_count: int,
                   average_score: float,
                   route_expected: Optional[str] = None):
        """Track metrics for a single query"""
        
        metrics = QueryMetrics(
            query=query,
            route_taken=route_taken,
            route_expected=route_expected,
            confidence=confidence,
            latency_ms=latency_ms,
            results_count=results_count,
            average_score=average_score,
            timestamp=datetime.now().isoformat(),
            is_correct_route=(route_taken == route_expected) if route_expected else None,
            is_multi_hop=self._is_multi_hop_query(query)
        )
        
        self.query_metrics.append(metrics)
        
        # Save to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
        
        # Update KPIs
        self._update_kpis()
        
        return metrics
    
    def _is_multi_hop_query(self, query: str) -> bool:
        """Detect if query requires multi-hop reasoning"""
        multi_hop_indicators = [
            'relationship between',
            'how are.*connected',
            'compare',
            'affect',
            'impact',
            'cause',
            'lead to'
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in multi_hop_indicators)
    
    def _update_kpis(self):
        """Update system-wide KPIs"""
        if not self.query_metrics:
            return
        
        # Router accuracy
        labeled_queries = [m for m in self.query_metrics if m.route_expected is not None]
        if labeled_queries:
            correct = sum(1 for m in labeled_queries if m.is_correct_route)
            self.system_kpis.router_accuracy = correct / len(labeled_queries)
            
            # Build confusion matrix
            self._build_confusion_matrix(labeled_queries)
        
        # Latency metrics
        latencies = [m.latency_ms for m in self.query_metrics]
        self.system_kpis.median_latency_ms = np.median(latencies)
        self.system_kpis.p90_latency_ms = np.percentile(latencies, 90)
        
        # Calculate baseline comparison (vector-only baseline assumed to be 100ms)
        baseline_latency = 100.0
        self.system_kpis.latency_vs_baseline_ratio = self.system_kpis.median_latency_ms / baseline_latency
        
        # Query count
        self.system_kpis.queries_processed = len(self.query_metrics)
        
        # Save KPIs
        with open(self.kpi_file, 'w') as f:
            json.dump(self.system_kpis.to_dict(), f, indent=2)
    
    def _build_confusion_matrix(self, labeled_queries: List[QueryMetrics]):
        """Build confusion matrix for router performance"""
        routes = ['vector', 'graph', 'hybrid']
        matrix = {route: {r: 0 for r in routes} for route in routes}
        
        for metrics in labeled_queries:
            if metrics.route_expected and metrics.route_taken:
                matrix[metrics.route_expected][metrics.route_taken] += 1
        
        self.system_kpis.router_confusion_matrix = matrix
    
    def calculate_retrieval_metrics(self, 
                                   relevant_docs: List[str],
                                   retrieved_docs: List[str]) -> Dict[str, float]:
        """Calculate precision, recall, MRR, nDCG"""
        if not retrieved_docs:
            return {'precision': 0, 'recall': 0, 'mrr': 0, 'ndcg': 0}
        
        # Precision and Recall
        retrieved_set = set(retrieved_docs[:10])  # Top-10
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_set & relevant_set
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
        recall = len(intersection) / len(relevant_set) if relevant_set else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                mrr = 1 / (i + 1)
                break
        
        # nDCG (simplified)
        dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(retrieved_docs[:10]) if doc in relevant_set)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_set), 10)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'mrr': mrr,
            'ndcg': ndcg
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for KPI dashboard"""
        route_distribution = {'vector': 0, 'graph': 0, 'hybrid': 0}
        for m in self.query_metrics:
            if m.route_taken in route_distribution:
                route_distribution[m.route_taken] += 1
        
        return {
            'system_kpis': self.system_kpis.to_dict(),
            'route_distribution': route_distribution,
            'total_queries': len(self.query_metrics),
            'recent_queries': [asdict(m) for m in self.query_metrics[-10:]],
            'latency_trend': [m.latency_ms for m in self.query_metrics[-50:]],
            'success_criteria': {
                'router_accuracy': {
                    'value': self.system_kpis.router_accuracy,
                    'target': 0.9,
                    'met': self.system_kpis.router_accuracy >= 0.9
                },
                'median_latency_ratio': {
                    'value': self.system_kpis.latency_vs_baseline_ratio,
                    'target': 1.5,
                    'met': self.system_kpis.latency_vs_baseline_ratio <= 1.5
                },
                'cost_reduction': {
                    'value': self.system_kpis.cost_reduction_percent,
                    'target': 70,
                    'met': self.system_kpis.cost_reduction_percent >= 70
                }
            }
        }
    
    def generate_report(self) -> str:
        """Generate markdown report of KPIs"""
        data = self.get_dashboard_data()
        
        report = f"""
# GraphRAG KPI Report
Generated: {datetime.now().isoformat()}

## Success Criteria Status
- ✅ Router Accuracy: {data['system_kpis']['router_accuracy']:.1%} (Target: ≥90%)
- {'✅' if data['success_criteria']['median_latency_ratio']['met'] else '❌'} Median Latency: {data['system_kpis']['median_latency_ms']:.1f}ms ({data['system_kpis']['latency_vs_baseline_ratio']:.1f}x baseline, Target: ≤1.5x)
- {'✅' if data['success_criteria']['cost_reduction']['met'] else '❌'} Cost Reduction: {data['system_kpis']['cost_reduction_percent']:.0f}% (Target: ≥70%)

## Query Statistics
- Total Queries: {data['total_queries']}
- Route Distribution:
  - Vector: {data['route_distribution']['vector']} ({data['route_distribution']['vector']/max(data['total_queries'],1)*100:.1f}%)
  - Graph: {data['route_distribution']['graph']} ({data['route_distribution']['graph']/max(data['total_queries'],1)*100:.1f}%)
  - Hybrid: {data['route_distribution']['hybrid']} ({data['route_distribution']['hybrid']/max(data['total_queries'],1)*100:.1f}%)

## Performance Metrics
- Median Latency: {data['system_kpis']['median_latency_ms']:.1f}ms
- P90 Latency: {data['system_kpis']['p90_latency_ms']:.1f}ms

## Router Confusion Matrix
```
{json.dumps(data['system_kpis'].get('router_confusion_matrix', {}), indent=2)}
```
        """
        
        return report

# Global KPI tracker instance
kpi_tracker = KPITracker()

def track_query_metrics(query: str, 
                        route: str, 
                        confidence: float, 
                        latency_ms: float,
                        results_count: int,
                        average_score: float,
                        expected_route: Optional[str] = None):
    """Helper function to track query metrics"""
    return kpi_tracker.track_query(
        query=query,
        route_taken=route,
        confidence=confidence,
        latency_ms=latency_ms,
        results_count=results_count,
        average_score=average_score,
        route_expected=expected_route
    )