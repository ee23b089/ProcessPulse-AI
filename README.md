# ProcessPulse AI — Process Analytics & Delay-Risk Intelligence

**Portfolio project for a Data Analytics & AI / Process Informatics internship.**

ProcessPulse AI is an interactive operations analytics app that helps an operations team explore throughput, processing delays, downtime, defects, energy use, and possible bottlenecks. It also demonstrates a supervised machine-learning workflow that estimates whether an operation will exceed its planned processing time.

> **Data transparency:** The included dataset is synthetic and generated for demonstration. It does not represent any real company or plant. Model metrics on synthetic data are not evidence of real-world performance.

## Features

- Interactive filters for date range, process step, machine, and shift
- KPI summary: operation records, units processed, high-delay rate, defect rate
- Monthly delay trends and planned-vs-actual time comparisons
- Defect-rate analysis by process step
- Bottleneck-prioritization table and scatter plot
- Random Forest classifier for a high-delay flag (actual time > 115% of planned time)
- Held-out test metrics: accuracy, precision, recall, and F1
- Interactive scenario explorer with estimated delay probability
- CSV download for filtered or full data

## Tech stack

Python · Pandas · NumPy · Streamlit · Plotly · scikit-learn

## Run locally

Requires Python 3.10 or newer.

### Windows (PowerShell)

```powershell
cd process_pulse_ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Project structure

```text
process_pulse_ai/
├── app.py
├── generate_data.py
├── requirements.txt
├── README.md
├── data/
│   └── operations_data.csv
└── tests/
    └── test_dataset.py
```

## Data dictionary

| Field | Meaning |
|---|---|
| record_id | Unique operation record |
| timestamp | Synthetic event timestamp |
| product | Product family |
| process_step | Stage in the operation |
| machine | Machine identifier |
| shift | Morning, evening, or night |
| units_processed | Number of units processed |
| planned_minutes | Planned duration |
| actual_minutes | Actual duration |
| downtime_minutes | Downtime observed |
| defect_count | Number of defective units |
| energy_kwh | Estimated energy consumption |

The app derives `delay_minutes`, `delay_flag`, `defect_rate_pct`, and `month` from the raw columns.

## Analytical approach

1. **Prepare data:** parse timestamps and derive delay, delay flag, defect rate, and month.
2. **Monitor KPIs:** summarize volume, delays, defects, and downtime.
3. **Prioritize investigations:** rank process-step/machine combinations using a transparent composite score.
4. **Train a baseline model:** one-hot encode categorical features and train a Random Forest classifier.
5. **Evaluate:** hold out 25% of records and report multiple classification metrics.
6. **Explore scenarios:** allow users to change operation inputs and view model-estimated delay risk.

The composite bottleneck score is a prioritization heuristic, not a validated industrial standard. It should not be interpreted as proof of causation.

## Limitations and next steps

- Validate with real, authorized, representative operational data before drawing business conclusions.
- Add time-based validation to avoid leakage when using chronological data.
- Compare against simple baselines and calibrate probabilities.
- Add model monitoring, data-quality checks, and explainability.
- For a real deployment, review access control, privacy, data governance, and safety requirements.

## Description

An interactive process analytics dashboard with KPI monitoring, bottleneck prioritization, and a baseline machine-learning model for delay-risk estimation, built with Python and Streamlit using synthetic operations data.
