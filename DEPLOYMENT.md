# Deployment Instructions

## Option 1: Streamlit Cloud (Recommended)
Because this app uses **Streamlit**, it requires a permanent server to run (WebSockets). Vercel Serverless functions typically time out or fail to support this interactivity.

**Steps:**
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect your GitHub Account.
3. Select this repository: `Spandanamk264/Uber_analysis-Unified-Mentor`
4. Click **Deploy**.

## Option 2: Vercel (Static / Limited)
We have included a `vercel.json` config, but due to platform limits (300MB upload size for serverless bundles) and protocol mismatch, this is not recommended for Data Science apps.

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
