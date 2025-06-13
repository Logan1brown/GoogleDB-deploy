# Show Optimizer Caching with Streamlit

## 1. Profiling Results and Insights

### Key Findings from Profiling:
- **Major bottlenecks identified**:
  - `identify_success_factors`: 56.20 seconds (18.3% of runtime)
  - `_get_matching_shows`: 52.29 seconds (17.0% of runtime)
  - `find_matches_with_fallback`: 51.88 seconds (16.9% of runtime)

- **Call relationships**:
  - `identify_success_factors` is calling `_get_matching_shows` 2,637 times
  - Each call to `_get_matching_shows` triggers a call to `find_matches_with_fallback`
  - This cascading effect is causing the performance issue

## 2. Streamlit Caching Implementation

Streamlit provides built-in caching decorators that are perfect for our use case:

- **`@st.cache_data`**: For caching function return values (DataFrames, lists, dictionaries)
- **`@st.cache_resource`**: For caching resources (database connections, ML models)

### Primary Targets for Caching:
1. `_get_matching_shows` in CriteriaScorer (called 2,637 times)
2. `find_matches_with_fallback` in Matcher (called 2,638 times)
3. `identify_success_factors` in RecommendationEngine (takes 56.2 seconds)
4. `fetch_and_integrate_data` in ShowOptimizer (database queries)

### Implementation Approach

1. **Data Fetching Layer**
   ```python
   # In ShowsAnalyzer (optimizer-specific component)
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['data_cache_ttl'])
   def fetch_optimizer_data(self):
       # Existing data fetching code
       return shows_df
   
   # In ShowOptimizer
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['data_cache_ttl'])
   def fetch_and_integrate_data(self):
       # Existing data fetching and integration code
       return integrated_data
   
   # Create optimizer-specific cached version for SuccessAnalyzer only
   # In ShowOptimizer
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['data_cache_ttl'])
   def _fetch_success_data_for_optimizer(self):
       return self.success_analyzer.fetch_success_data()
   ```

2. **Show Matching Layer**
   ```python
   # In Matcher
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['matcher_cache_ttl'])
   def find_matches_with_fallback(self, criteria, data):
       # Existing matching code
       return matches, confidence_info
   
   # In CriteriaScorer
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['matcher_cache_ttl'])
   def _get_matching_shows(self, criteria, data):
       # Existing matching code
       return matching_shows, confidence_info
   ```

3. **Analysis Layer**
   ```python
   # In ConceptAnalyzer
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['analysis_cache_ttl'])
   def analyze_concept(self, criteria, integrated_data):
       # Existing analysis code
       return optimization_summary
   
   # In NetworkAnalyzer
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['analysis_cache_ttl'])
   def analyze_networks(self, matching_shows, criteria):
       # Existing network analysis code
       return network_rankings
   ```

4. **Recommendation Layer**
   ```python
   # In RecommendationEngine
   @st.cache_data(ttl=OptimizerConfig.PERFORMANCE['recommendation_cache_ttl'])
   def identify_success_factors(self, criteria, matching_shows, integrated_data, limit=5):
       # Existing success factor identification code
       return success_factors
   ```

### Configuration Updates

Add the following cache TTL settings to OptimizerConfig.PERFORMANCE:

```python
# Performance settings
PERFORMANCE = {
    'success_threshold': 0.6,      # Default threshold for success rate calculation
    'min_criteria_coverage': 0.5,  # Minimum criteria coverage for valid analysis
    'min_confidence_score': 0.3,   # Minimum confidence score for valid results
    'incremental_threshold': 0.7,  # Threshold for incremental vs full recalculation
    
    # Cache TTL settings (in seconds)
    'data_cache_ttl': 3600,        # 1 hour for base data
    'matcher_cache_ttl': 1800,     # 30 minutes for matching results
    'analysis_cache_ttl': 1800,    # 30 minutes for analysis results
    'recommendation_cache_ttl': 900 # 15 minutes for recommendations
}
```

## 3. Implementation Steps

1. **Update OptimizerConfig**
   - Add the cache TTL settings to the PERFORMANCE dictionary

2. **Import Streamlit Cache Decorators**
   - Add `from streamlit import cache_data` to each file that needs caching

3. **Apply Caching to Data Fetching**
   - Add caching to ShowsAnalyzer.fetch_optimizer_data
   - Create a cached wrapper for SuccessAnalyzer.fetch_success_data in ShowOptimizer
   - Add caching to ShowOptimizer.fetch_and_integrate_data

4. **Apply Caching to Matching and Analysis**
   - Add caching to Matcher.find_matches_with_fallback
   - Add caching to CriteriaScorer._get_matching_shows
   - Add caching to ConceptAnalyzer.analyze_concept
   - Add caching to NetworkAnalyzer.analyze_networks

5. **Apply Caching to Recommendations**
   - Add caching to RecommendationEngine.identify_success_factors

## 4. Benefits of Streamlit Caching

1. **Performance Improvements**
   - Significant reduction in computation time for repeated operations
   - Faster UI response times for users exploring similar concepts

2. **Simplified Implementation**
   - No custom cache class to maintain
   - Automatic cache invalidation based on TTL
   - Built-in cache key generation based on function inputs

3. **Reduced Database Load**
   - Fewer repeated queries to the database
   - Lower server resource utilization

4. **Better User Experience**
   - More responsive UI with cached results
   - Consistent performance across sessions

5. **Maintainability**
   - Cleaner code without custom caching logic
   - Easier to debug and extend
   - Streamlit handles cache management internally
