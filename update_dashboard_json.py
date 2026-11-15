"""Compatibility entry point for refreshing dashboard_data.json."""

from engine.run_evaluation_pipeline import run_full_pipeline


if __name__ == "__main__":
    run_full_pipeline()
