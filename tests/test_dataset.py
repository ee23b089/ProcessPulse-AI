from pathlib import Path
import pandas as pd

def test_dataset_schema_and_values():
    path = Path(__file__).parents[1] / "data" / "operations_data.csv"
    df = pd.read_csv(path)
    expected = {
        "record_id", "timestamp", "product", "process_step", "machine", "shift",
        "units_processed", "planned_minutes", "actual_minutes", "downtime_minutes",
        "defect_count", "energy_kwh"
    }
    assert expected.issubset(df.columns)
    assert len(df) >= 1000
    assert df["record_id"].is_unique
    assert (df["units_processed"] > 0).all()
    assert (df["planned_minutes"] > 0).all()
    assert (df["actual_minutes"] > 0).all()
    assert (df["defect_count"] >= 0).all()
    assert (df["defect_count"] <= df["units_processed"]).all()
