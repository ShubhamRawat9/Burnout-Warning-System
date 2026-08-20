# 🔥 Productivity Drift & Burnout Early Warning Platform

A machine-learning system that detects early signs of employee burnout from weekly
behavioral/productivity signals (task completion, focus sessions, context switches,
after-hours logins, error rates, etc.), explains *why* a risk score was raised using
SHAP, and surfaces alerts through an interactive dashboard and a Telegram bot.

## Features

- **Hybrid ML pipeline** — XGBoost (tabular features) + LSTM (16-week sequential
  behavior) combined into a hybrid burnout-risk probability.
- **Explainability** — SHAP values show which behavioral signals are driving each
  employee's risk score.
- **Interactive dashboard** (Streamlit) — org-wide risk distribution, per-employee
  deep dives, 16-week risk trajectories, drift heatmaps, and an alert log.
- **Telegram bot** — logs employee self-reported data to Google Sheets for
  ongoing data collection.

## Project structure

```
burnout_project/
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── data/
│   ├── raw/                   # Raw weekly productivity dataset
│   └── processed/             # Dataset with engineered drift features
├── models/                    # Trained XGBoost + LSTM models, encoders, SHAP explainer
├── notebooks/                 # Data generation, preprocessing, and model training notebooks
├── reports/                   # Generated charts (feature importance, SHAP, confusion matrix, etc.)
├── telegram_bot/
│   ├── bot.py                 # Telegram bot for data collection
│   └── .env.example           # Template for required environment variables
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd burnout_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

### Run the Telegram bot

The bot needs a Telegram bot token and a Google service-account credentials file
for writing to Google Sheets. **These are secrets and must never be committed to
git** — they're already excluded via `.gitignore`.

1. Copy `telegram_bot/.env.example` to `telegram_bot/.env` and fill in your values,
   or export the variables directly:
   ```bash
   export TELEGRAM_BOT_TOKEN="your-token-from-botfather"
   export GOOGLE_CREDENTIALS_PATH="./telegram_bot/credentials.json"
   ```
2. Place your Google service-account `credentials.json` in `telegram_bot/` (it's
   gitignored and will stay local).
3. Run the bot:
   ```bash
   python telegram_bot/bot.py
   ```

## ⚠️ Security note

This repository previously contained a hardcoded Telegram bot token and a Google
service-account private key. Both have been removed from the code (now loaded from
environment variables) and **should be treated as compromised — revoke and
regenerate them** before deploying:

- Telegram: talk to [@BotFather](https://t.me/BotFather) → `/revoke` (or generate
  a new token) for this bot.
- Google Cloud: delete the exposed key for the `burnoutbot@burnoutbot-491704.iam.gserviceaccount.com`
  service account in the [Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts)
  and create a new one.

## Model pipeline

See the notebooks for the full pipeline:

1. `00_environment_check.ipynb` — environment/dependency check
2. `01_data_generation.ipynb` — synthetic dataset generation
3. `02_preprocessing.ipynb` — feature engineering / drift features
4. `03_xgboost_model.ipynb` — XGBoost tabular model
5. `04_lstm_model.ipynb` — LSTM sequential model + hybrid ensemble

## License

Add a license of your choice (e.g. MIT) — see [choosealicense.com](https://choosealicense.com/).
