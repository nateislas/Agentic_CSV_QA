# Agentic CSV QA - Test User Flows

## Overview
This document contains comprehensive test scenarios and user flows to validate the full agentic capabilities of the CSV analysis system. These flows are designed to test the system's ability to handle complex, multi-turn conversations and sophisticated data analysis operations.

## Test Data Context
We're testing with crime data containing 23 columns:
- `lsoa_code`, `borough`, `major_category`, `minor_category`, `value`
- `year`, `month`, `crime_date`, `latitude`, `longitude`
- `crime_id`, `reported_date`, `location_type`, `outcome`, `police_force`
- `neighborhood`, `crime_severity`, `victim_count`, `property_damage`
- `weapon_used`, `time_period`, `day_of_week`, `season`

## Test Categories

### 1. Basic Data Exploration
**Goal**: Test fundamental data understanding capabilities

#### Flow 1.1: Initial Data Assessment
```
User: "What are the column names in this dataset?"
Expected: List of all 23 columns with brief descriptions

User: "Show me the first 10 rows of data"
Expected: Formatted table with sample data

User: "What are the data types of each column?"
Expected: Detailed breakdown of numeric, text, date columns

User: "How many rows and columns does this dataset have?"
Expected: Total count with file size information
```

#### Flow 1.2: Data Quality Assessment
```
User: "Are there any missing values in the dataset?"
Expected: Summary of null values per column

User: "Show me the unique values in the borough column"
Expected: List of all boroughs with counts

User: "What's the date range of the crime data?"
Expected: Earliest and latest crime dates
```

### 2. Statistical Analysis
**Goal**: Test aggregation and statistical capabilities

#### Flow 2.1: Basic Statistics
```
User: "Calculate the total number of crimes by borough"
Expected: Table with borough names and crime counts

User: "What's the average number of victims per crime?"
Expected: Statistical summary with mean, median, min, max

User: "Show me the top 5 most common crime categories"
Expected: Ranked list of major/minor categories with counts
```

#### Flow 2.2: Advanced Statistics
```
User: "Calculate the crime rate per borough (crimes per 1000 people)"
Expected: Table with borough, crime count, and calculated rate

User: "What's the correlation between property damage and victim count?"
Expected: Correlation coefficient with interpretation

User: "Show me the distribution of crimes by day of the week"
Expected: Bar chart or table showing crime frequency by day
```

### 3. Time-Based Analysis
**Goal**: Test temporal analysis capabilities

#### Flow 3.1: Temporal Patterns
```
User: "Show me crime trends by month for the last year"
Expected: Time series data with monthly crime counts

User: "Which months have the highest crime rates?"
Expected: Ranked list of months with crime statistics

User: "What's the average time between crime occurrence and reporting?"
Expected: Statistical analysis of time differences
```

#### Flow 3.2: Seasonal Analysis
```
User: "Compare crime rates between summer and winter months"
Expected: Comparative analysis with statistical significance

User: "Show me the seasonal patterns in different crime categories"
Expected: Breakdown by season and crime type
```

### 4. Geographic Analysis
**Goal**: Test spatial analysis capabilities

#### Flow 4.1: Location-Based Analysis
```
User: "Which boroughs have the highest crime rates?"
Expected: Ranked list with crime statistics per borough

User: "Show me crimes by neighborhood within each borough"
Expected: Hierarchical breakdown of crime data

User: "What are the most dangerous areas based on crime severity?"
Expected: Analysis using crime_severity and location data
```

#### Flow 4.2: Spatial Patterns
```
User: "Are there geographic clusters of similar crime types?"
Expected: Analysis of crime type distribution by location

User: "Which areas have the highest property damage rates?"
Expected: Geographic analysis of property damage patterns
```

### 5. Multi-Turn Complex Analysis
**Goal**: Test sophisticated multi-step analysis capabilities

#### Flow 5.1: Progressive Analysis
```
User: "Create a table showing crime statistics by borough"
Expected: Initial table with basic stats

User: "Now add the percentage of violent crimes for each borough"
Expected: Enhanced table with additional calculated column

User: "Filter to show only boroughs with more than 1000 crimes"
Expected: Filtered table with threshold applied

User: "Sort the results by crime rate in descending order"
Expected: Final sorted table with all requested features
```

#### Flow 5.2: Comparative Analysis
```
User: "Compare crime patterns between weekdays and weekends"
Expected: Comparative analysis table

User: "Now break this down by crime category"
Expected: Enhanced comparison with category breakdown

User: "Show me the top 3 most dangerous areas for each day of the week"
Expected: Complex filtered and ranked results
```

### 6. Predictive and Insight Analysis
**Goal**: Test advanced analytical capabilities

#### Flow 6.1: Pattern Recognition
```
User: "What factors are most strongly associated with high crime rates?"
Expected: Correlation analysis of multiple variables

User: "Are there any unusual patterns in the crime data?"
Expected: Anomaly detection and pattern analysis

User: "Which crime categories show the strongest seasonal patterns?"
Expected: Time series analysis by category
```

#### Flow 6.2: Risk Assessment
```
User: "Create a risk score for each borough based on crime severity and frequency"
Expected: Calculated risk metric with explanation

User: "Which areas should receive increased police patrols?"
Expected: Prioritized list based on crime analysis
```

### 7. Data Quality and Validation
**Goal**: Test data integrity and quality assessment

#### Flow 7.1: Data Validation
```
User: "Are there any inconsistencies in the crime data?"
Expected: Data quality report with issues identified

User: "Check if all crime dates are within reasonable ranges"
Expected: Date validation with outliers identified

User: "Are there any duplicate crime records?"
Expected: Duplicate detection and analysis
```

#### Flow 7.2: Data Completeness
```
User: "How complete is the victim count data?"
Expected: Completeness analysis with missing data patterns

User: "Which columns have the most missing values?"
Expected: Missing data analysis by column
```

### 8. Advanced Aggregation and Pivot Analysis
**Goal**: Test complex data transformation capabilities

#### Flow 8.1: Multi-Dimensional Analysis
```
User: "Create a pivot table showing crimes by borough and crime category"
Expected: Cross-tabulation of crime data

User: "Now add the average victim count for each combination"
Expected: Enhanced pivot with calculated metrics

User: "Show me the percentage breakdown of crime types by borough"
Expected: Percentage-based pivot analysis
```

#### Flow 8.2: Hierarchical Analysis
```
User: "Group crimes by borough, then by month, then by crime category"
Expected: Multi-level hierarchical grouping

User: "Calculate the rolling 3-month average crime rate for each borough"
Expected: Time-based rolling calculations
```

### 9. Interactive Exploration
**Goal**: Test conversational data exploration

#### Flow 9.1: Guided Exploration
```
User: "I want to understand crime patterns in London. Start with an overview"
Expected: High-level summary of crime data

User: "Now focus on violent crimes specifically"
Expected: Filtered analysis of violent crime subset

User: "Which areas have the highest violent crime rates?"
Expected: Geographic analysis of violent crimes

User: "Show me the trend of violent crimes over time"
Expected: Time series analysis of violent crimes
```

#### Flow 9.2: Hypothesis Testing
```
User: "I suspect crime rates are higher on weekends. Can you test this?"
Expected: Statistical comparison of weekday vs weekend crime rates

User: "Is there evidence that certain neighborhoods are more dangerous?"
Expected: Statistical analysis of neighborhood crime patterns

User: "Do crime rates correlate with the time of day?"
Expected: Temporal analysis of crime patterns
```

### 10. Complex Multi-Step Analysis
**Goal**: Test the most sophisticated analysis capabilities

#### Flow 10.1: Comprehensive Crime Analysis
```
User: "Create a comprehensive crime analysis report"
Expected: Multi-section report with various analyses

User: "Now add a section on geographic hotspots"
Expected: Enhanced report with spatial analysis

User: "Include recommendations for police resource allocation"
Expected: Final report with actionable insights
```

#### Flow 10.2: Comparative Study
```
User: "Compare crime patterns between different time periods"
Expected: Comparative analysis across time periods

User: "Now break this down by crime severity"
Expected: Enhanced comparison with severity analysis

User: "Which areas show the most improvement or deterioration?"
Expected: Trend analysis with change detection
```

## Expected System Capabilities

### Core Requirements
- **Response Time**: <1 second for simple queries, <5 seconds for complex operations
- **Multi-turn Context**: Maintain conversation history and reference previous results
- **Smart Sampling**: Handle large datasets without overwhelming LLM context
- **Error Recovery**: Graceful handling of malformed data and edge cases

### Advanced Features
- **Progressive Analysis**: Build complex analyses step by step
- **Context Awareness**: Reference previous results in subsequent queries
- **Intelligent Filtering**: Apply filters based on data characteristics
- **Statistical Rigor**: Provide proper statistical analysis with confidence levels
- **Geographic Intelligence**: Handle spatial data and location-based analysis
- **Temporal Analysis**: Perform time-based analysis and trend detection

### Quality Indicators
- **Accuracy**: Results should be mathematically correct
- **Interpretability**: Provide clear explanations of findings
- **Completeness**: Handle edge cases and missing data appropriately
- **Performance**: Maintain fast response times even for complex operations
- **Robustness**: Handle various data quality issues gracefully

## Success Metrics

### Functional Requirements
- [ ] All basic exploration queries return accurate results
- [ ] Statistical calculations are mathematically correct
- [ ] Multi-turn conversations maintain context properly
- [ ] Complex aggregations work with large datasets
- [ ] Geographic and temporal analysis functions correctly

### Performance Requirements
- [ ] Simple queries complete in <1 second
- [ ] Complex operations complete in <5 seconds
- [ ] Large file processing completes in <30 seconds
- [ ] Memory usage remains reasonable for 100+ column datasets

### User Experience Requirements
- [ ] Clear, understandable responses
- [ ] Proper error messages for invalid queries
- [ ] Progressive disclosure of complex results
- [ ] Intuitive conversation flow
- [ ] Helpful suggestions for follow-up questions

## Testing Strategy

### Phase 1: Basic Functionality
Test all basic exploration and analysis capabilities with small datasets.

### Phase 2: Performance Testing
Test with larger datasets to validate performance requirements.

### Phase 3: Edge Cases
Test with malformed data, missing values, and unusual patterns.

### Phase 4: Multi-turn Complexity
Test complex, multi-step analysis scenarios.

### Phase 5: Production Readiness
Test with real-world crime datasets and validate all requirements.

## Notes for Implementation

1. **Data Privacy**: Ensure crime data is handled appropriately
2. **Performance**: Monitor response times and optimize as needed
3. **Accuracy**: Validate statistical calculations against known results
4. **Scalability**: Test with increasingly large datasets
5. **User Experience**: Ensure responses are clear and actionable

This test suite provides a comprehensive framework for validating the full agentic capabilities of the CSV analysis system, with particular focus on crime data analysis scenarios. 