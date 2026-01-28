# App module
from .update_daily import (
    run_daily_update,
    run_initial_setup,
    load_top10_history,
    save_top10_latest,
    save_top10_history,
    generate_quality_report,
    save_quality_report
)

# E5: REST API (lazy import to avoid FastAPI dependency if not needed)
def get_api_app():
    """Get FastAPI app instance"""
    from .api import app
    return app
