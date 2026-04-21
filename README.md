# Solution Architecture - CCTS Complaint Analysis Pipeline

## Overview

This is a multi-step solution for comprehensive CCTS complaint analysis. The solution is organized into step-specific modules under the `src/` directory, allowing for modular development and easy integration of future analysis steps.

## Current Structure

```
src/
├── __init__.py                 # Package marker
├── main.py                     # Main orchestrator (entry point)
├── requirements.txt            # Dependencies
└── report_generation/          # Step 1: Report Generation
    ├── __init__.py             # Package marker
    ├── analyzer.py             # Core analysis engine
    ├── config.py               # Configuration management
    ├── utils.py                # Utility functions
    ├── prompts.py              # AI prompt templates
    └── Level_reports_generation.py  # Original script (reference)
```

## Step 1: Report Generation Module

### Purpose
Generates comprehensive complaint reports by analyzing customer interactions and creating detailed journeys.

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Analyzer** | `analyzer.py` | Core analysis engine with API integration |
| **Configuration** | `config.py` | Environment-based configuration management |
| **Utilities** | `utils.py` | JSON parsing, validation, helpers |
| **Prompts** | `prompts.py` | AI prompt templates for analysis |

### Output
- `interaction_analysis_{file_number}.json` - Detailed interaction-by-interaction analysis
- `journey_summary_{file_number}.json` - Aggregated journey and root cause analysis

## Future Steps

Space reserved for:
- Step 2: Pattern Analysis (identify systemic issues)
- Step 3: Predictive Modeling (forecast complaints)
- Step 4: Recommendations Engine (generate fixes)

## Getting Started

### Quick Start

```bash
# From src/ or project root
python src/main.py
```

### Configuration

Create `.env` file in project root:
```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
DATA_PICKLE_PATH=/path/to/interactions.pkl
```

## Module Design

### Separation of Concerns

Each step module is self-contained with:
- **config.py** - All configuration/settings
- **\*.py** - Step-specific business logic
- **utils.py** - Shared utilities
- **prompts.py** - AI prompt templates (if applicable)

### Import Pattern

```python
# From main.py
from report_generation import (
    ConfigManager,
    InteractionAnalyzer,
    save_analysis_results,
)

# To add new step
from analysis_step_2 import SomeAnalyzer
```

## Adding New Steps

To add Step 2 (or another step):

1. Create new folder: `src/analysis_step_2/`
2. Add core files:
   - `__init__.py` - Package exports
   - `analyzer.py` - Main logic
   - `config.py` - Configuration (if needed)
   - `utils.py` - Utilities

3. Update `src/main.py` to orchestrate the new step

Example:
```python
from analysis_step_2 import PatternAnalyzer

# In main():
pattern_analyzer = PatternAnalyzer()
patterns = pattern_analyzer.analyze(interaction_results)
```

## Dependencies

See `requirements.txt` for Python packages.

Core dependencies:
- `openai>=1.3.0` - Azure OpenAI integration
- `pandas>=1.5.0` - Data processing
- `python-dotenv>=1.0.0` - Environment variables

## Logging

All modules log to:
- Console (stdout)
- `level_reports.log` file

Log levels:
- DEBUG - Detailed diagnostic info
- INFO - Progress/operations
- WARNING - Warning conditions
- ERROR - Error conditions

## Error Handling

Each step module includes:
- **Retry logic** - Automatic retry with backoff
- **Validation** - Input/output validation
- **Error recovery** - Graceful degradation
- **Logging** - Detailed error context

## Configuration Hierarchy

1. Environment variables (`.env` file)
2. Default values in `config.py`
3. Command-line arguments (future enhancement)

## Testing

Modules are designed to be independently testable:

```python
from report_generation import ConfigManager, InteractionAnalyzer

# Test configuration
config = ConfigManager.load_processing_config()
assert config.temperature == 0.1

# Test analyzer
analyzer = InteractionAnalyzer(mock_client, config)
results = analyzer.analyze_all_interactions(test_df)
```

## Performance Considerations

- **API Rate Limiting** - Configurable delays between requests
- **Retry Strategy** - Exponential backoff for failed requests
- **Memory Usage** - Processes data in chunks
- **Logging Overhead** - Can be adjusted via log levels

## Troubleshooting

### Configuration Issues
```bash
# Check environment variables are loaded
python -c "from report_generation import ConfigManager; ConfigManager.load_azure_config()"
```

### Import Issues
```bash
# Ensure src/ is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### API Connection
```bash
# Test Azure OpenAI connection
python -c "from openai import AzureOpenAI; ..."
```

## Future Enhancements

- [ ] Parallelization of case processing
- [ ] Caching layer for API responses
- [ ] Database backend for results
- [ ] CLI arguments for step filtering
- [ ] Web dashboard for results
- [ ] Scheduled automation
- [ ] Performance metrics/monitoring

## Contributing New Steps

When adding a new analysis step:

1. **Follow naming convention**: `step_n_description/`
2. **Use consistent structure**: config, utils, main logic
3. **Add __init__.py**: Export public API
4. **Include logging**: All major operations
5. **Document outputs**: Data structures and formats
6. **Write docstrings**: All public functions
7. **Add type hints**: For IDE support

## Support

For issues:
1. Check `level_reports.log` for detailed errors
2. Review step-specific README
3. Verify configuration in `.env`
4. Check Azure OpenAI service status

---

**Version**: 2.0  
**Status**: Production-Ready  
**Last Updated**: January 2024
