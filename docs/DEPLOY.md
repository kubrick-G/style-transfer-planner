# Deployment Guide: Style Transfer Planner

This document provides step-by-step instructions to deploy Style Transfer Planner on **Streamlit Community Cloud**, **Hugging Face Spaces**, or via **Docker**.

---

## 1. Streamlit Community Cloud (Recommended & Free)

Streamlit Community Cloud is the fastest way to host the public interactive demo.

### Step-by-step Setup:
1. **Push your repository to GitHub**:
   - Create a repository (e.g. `style-transfer-planner`) on GitHub.
   - Push your code to the `master` or `main` branch:
     ```bash
     git remote add origin https://github.com/kubrick-G/style-transfer-planner.git
     git push -u origin master
     ```
2. **Sign in to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. **Deploy the App**:
   - Click **"Create app"** or **"Deploy an app"**.
   - Select your repository (`kubrick-G/style-transfer-planner`), branch (`master`), and Main file path (`app.py`).
4. **Configure Secrets & Public-Mode Settings**:
   - Click **"Advanced settings"** -> **"Secrets"**.
   - Paste the following configuration (replace with your rotated Gemini API key):
     ```toml
     GEMINI_API_KEY = "AIzaSy..."
     GEMINI_MODEL = "gemini-3.5-flash-lite"
     GEMINI_MIN_INTERVAL_S = "5"
     STP_PUBLIC_MODE = "1"
     STP_DAILY_GEMINI_LIMIT = "40"
     STP_MAX_CLIP_S = "30"
     STP_MAX_UPLOAD_MB = "50"
     ```
   - Click **"Save"** and **"Deploy"**.
5. **Verify Live Deployment**:
   - Once deployed, visit your public URL in an Incognito window.
   - Open a terminal and run the smoke test against your live URL:
     ```bash
     python scripts/smoke_test.py https://<your-app-subdomain>.streamlit.app
     ```

---

## 2. Hugging Face Spaces (Docker Alternative)

Hugging Face Spaces provides containerized isolation and free CPU tiers.

### Step-by-step Setup:
1. **Create a new Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter Space name (e.g. `style-transfer-planner`).
   - Select **Docker** as the Space SDK (Blank template).
   - Set Space hardware to free CPU tier.
2. **Push Code to HF**:
   - Clone your Hugging Face Space repository locally or push this repo directly to HF:
     ```bash
     git remote add hf https://huggingface.co/spaces/<your-hf-username>/style-transfer-planner
     git push hf master
     ```
3. **Configure Environment Secrets**:
   - In your HF Space, go to **Settings** -> **Variables and secrets**.
   - Add the following secrets:
     - `GEMINI_API_KEY`: your API key.
     - `STP_PUBLIC_MODE`: `1`.
     - `STP_DAILY_GEMINI_LIMIT`: `40`.
4. **Test Live Endpoint**:
   - Check the build logs; once healthy, run:
     ```bash
     python scripts/smoke_test.py https://<your-hf-username>-style-transfer-planner.hf.space
     ```

---

## 3. Local Docker Run

To run the container locally:

```bash
# Build the Docker image
docker build -t style-transfer-planner .

# Run container in public mode with daily usage limit
docker run -p 7860:7860 \
  -e STP_PUBLIC_MODE=1 \
  -e STP_DAILY_GEMINI_LIMIT=40 \
  -e GEMINI_API_KEY="your-api-key-here" \
  style-transfer-planner
```

Visit `http://localhost:7860` in your browser.

Run the local smoke test:
```bash
python scripts/smoke_test.py http://localhost:7860
```
