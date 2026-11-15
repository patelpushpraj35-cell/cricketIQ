"""Compatibility entry point; all player rankings now use the IPL score engine."""

from engine.run_evaluation_pipeline import run_full_pipeline


if __name__ == "__main__":
    run_full_pipeline()
