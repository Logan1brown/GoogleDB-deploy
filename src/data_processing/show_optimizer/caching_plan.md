# Show Optimizer Caching Consolidation Plan

## 1. Centralize All Caching in OptimizerCache

### Current Issues:
- ConceptAnalyzer has its own separate caching system
- NetworkAnalyzer receives but doesn't effectively use the OptimizerCache
- No data-driven approach to determine what should be cached

### Phase 1: Implement Performance Profiling

1. **Add profiling to key methods**:
   - Matcher.find_matches_with_fallback
   - ConceptAnalyzer._find_matching_shows
   - NetworkAnalyzer.get_network_matches
   - ScoreCalculator.calculate methods
   - RecommendationEngine.generate_recommendations

2. **Collect real usage data**:
   - Run the optimizer with various criteria combinations
   - Track execution time of each operation
   - Track parameter repetition to identify potential cache hits
   - Identify the most expensive operations

3. **Analyze profiling results**:
   - Which operations take the most time?
   - Which operations are called repeatedly with the same parameters?
   - Calculate potential time savings from caching

### Phase 2: Enhance OptimizerCache with Targeted Caching

1. **Add criteria-based caching to OptimizerCache**:
```python
def cache_matching_shows(self, criteria_hash: str, matching_shows, confidence_info):
    """Cache matching shows for specific criteria."""
    if not hasattr(self, '_matching_shows_cache'):
        self._matching_shows_cache = {}
        self._matching_shows_last_update = datetime.now()
    
    self._matching_shows_cache[criteria_hash] = (matching_shows, confidence_info)

def get_cached_matching_shows(self, criteria_hash: str, force_refresh: bool = False):
    """Get cached matching shows for specific criteria if available."""
    if force_refresh or not hasattr(self, '_matching_shows_cache'):
        return None
        
    if criteria_hash in self._matching_shows_cache:
        return self._matching_shows_cache[criteria_hash]
    
    return None
```

2. **Add network-specific caching**:
```python
def cache_network_matches(self, criteria_hash: str, network_matches):
    """Cache network matches for specific criteria."""
    if not hasattr(self, '_network_matches_cache'):
        self._network_matches_cache = {}
        self._network_matches_last_update = datetime.now()
    
    self._network_matches_cache[criteria_hash] = network_matches

def get_cached_network_matches(self, criteria_hash: str, force_refresh: bool = False):
    """Get cached network matches for specific criteria if available."""
    if force_refresh or not hasattr(self, '_network_matches_cache'):
        return None
        
    if criteria_hash in self._network_matches_cache:
        return self._network_matches_cache[criteria_hash]
    
    return None
```

3. **Add component score caching**:
```python
def cache_component_scores(self, criteria_hash: str, component_scores):
    """Cache component scores for specific criteria."""
    if not hasattr(self, '_component_scores_cache'):
        self._component_scores_cache = {}
        self._component_scores_last_update = datetime.now()
    
    self._component_scores_cache[criteria_hash] = component_scores

def get_cached_component_scores(self, criteria_hash: str, force_refresh: bool = False):
    """Get cached component scores for specific criteria if available."""
    if force_refresh or not hasattr(self, '_component_scores_cache'):
        return None
        
    if criteria_hash in self._component_scores_cache:
        return self._component_scores_cache[criteria_hash]
    
    return None
```

### Phase 3: Migrate ConceptAnalyzer to Use OptimizerCache

1. **Update ConceptAnalyzer._find_matching_shows**:
```python
def _find_matching_shows(self, criteria, integrated_data, force_refresh=False):
    # Generate a hash for the criteria
    criteria_hash = self._get_criteria_hash(criteria)
    
    # Try to get from cache
    if self.optimizer_cache:
        cached_result = self.optimizer_cache.get_cached_matching_shows(criteria_hash, force_refresh)
        if cached_result:
            st.write(f"Using cached matching shows ({len(cached_result[0])} shows)")
            return cached_result
    
    # Perform the search
    # [existing search logic]
    
    # Cache the result
    if self.optimizer_cache:
        self.optimizer_cache.cache_matching_shows(criteria_hash, matching_shows, confidence_info)
    
    return matching_shows, confidence_info
```

2. **Remove ConceptAnalyzer's internal cache**:
   - Remove `_matching_shows_cache` and `_matching_shows_last_update`
   - Remove `_is_matching_shows_cache_valid` method

### Phase 4: Enhance NetworkAnalyzer to Use OptimizerCache

1. **Update NetworkAnalyzer.get_network_matches**:
```python
def get_network_matches(self, criteria, matching_shows=None, force_refresh=False):
    # Generate a hash for the criteria
    criteria_hash = self._get_criteria_hash(criteria)
    
    # Try to get from cache
    if self.optimizer_cache:
        cached_result = self.optimizer_cache.get_cached_network_matches(criteria_hash, force_refresh)
        if cached_result:
            st.write(f"Using cached network matches ({len(cached_result)} networks)")
            return cached_result
    
    # Perform the network matching
    # [existing matching logic]
    
    # Cache the result
    if self.optimizer_cache:
        self.optimizer_cache.cache_network_matches(criteria_hash, network_matches)
    
    return network_matches
```

## 2. Implement Instrumentation for Data-Driven Decisions

### Add Profiling to Key Methods:

1. **Update ShowOptimizer to enable profiling**:
```python
def __init__(self):
    # Existing initialization
    self.profiling_enabled = OptimizerConfig.PERFORMANCE.get('enable_profiling', False)
```

2. **Wrap key methods with profiling**:
```python
def analyze_concept(self, criteria):
    if self.profiling_enabled:
        from .optimizer_profiler import profile_operation
        criteria_hash = self._get_criteria_hash(criteria)
        return profile_operation(
            "analyze_concept", 
            lambda: self.concept_analyzer.analyze_concept(criteria),
            param_hash=criteria_hash
        )
    else:
        return self.concept_analyzer.analyze_concept(criteria)
```

3. **Add profiling results display**:
```python
def display_performance_stats(self):
    """Display performance statistics if profiling is enabled."""
    if self.profiling_enabled:
        from .optimizer_profiler import display_profiling_results
        display_profiling_results()
```

## 3. Potential Caching Targets (to be validated by profiling)

Based on code analysis, these operations are likely candidates for caching:

1. **Matching Shows by Criteria**:
   - High computation cost
   - Often repeated with the same criteria
   - Already being cached by ConceptAnalyzer

2. **Network Matching**:
   - Depends on criteria and matching shows
   - Likely repeated for the same criteria

3. **Component Score Calculation**:
   - Depends on matching shows
   - Potentially expensive calculations
   - Often repeated for the same matching shows

4. **Success Probability Calculation**:
   - Complex calculation based on component scores
   - Likely repeated for the same component scores

## 4. Implementation Timeline

1. **Week 1**: Implement profiling infrastructure and instrument key methods
2. **Week 2**: Collect real usage data and analyze performance bottlenecks
3. **Week 3**: Enhance OptimizerCache with targeted caching methods
4. **Week 4**: Migrate ConceptAnalyzer and NetworkAnalyzer to use OptimizerCache
5. **Week 5**: Test and optimize the caching strategy based on real performance data

## 5. Success Metrics

- **Performance Improvement**: Measure end-to-end response time before and after caching
- **Cache Hit Rate**: Track percentage of requests served from cache
- **Time Saved**: Calculate total time saved through cache hits
- **Memory Usage**: Monitor memory consumption to ensure caching doesn't cause issues
