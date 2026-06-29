# Deploying Stock Analyzer to Render (free, always-on)

The app is ready to deploy. Files added: `requirements.txt`, `Procfile`,
`render.yaml`, `.gitignore`. You need two free accounts: **GitHub** (to host
the code) and **Render** (to run it).

## Step 1 — Put the code on GitHub (easiest: web upload)
1. Go to https://github.com and sign up (free) if you don't have an account.
2. Click the **+** (top right) → **New repository**.
   - Name: `stock-analyzer`
   - Set to **Private** (your call) → **Create repository**.
3. On the new repo page, click **uploading an existing file**.
4. Open Finder at `~/Desktop/StockAnalyzer`, select ALL files
   (`app.py`, `templates/`, `requirements.txt`, `Procfile`, `render.yaml`,
   `.gitignore`, `DEPLOY.md`) and drag them into the browser.
5. Click **Commit changes**.

## Step 2 — Deploy on Render
1. Go to https://render.com and sign up (free) — choose **Sign in with GitHub**.
2. Click **New** → **Blueprint**.
3. Pick your `stock-analyzer` repo. Render reads `render.yaml` automatically.
4. Click **Apply** / **Create**. First build takes ~3–5 minutes.
5. (Recommended) In the service's **Environment** tab add:
   - Key: `SEC_CONTACT_EMAIL`  Value: your email (SEC asks for a contact).
6. When it says **Live**, you get a URL like
   `https://stock-analyzer-xxxx.onrender.com` — open it from any device.

## Notes
- **Free tier sleeps** after ~15 min idle; first visit then takes ~30–60s to
  wake. Fine for personal use.
- **Yahoo Finance data** can occasionally rate-limit cloud servers. SEC EDGAR
  (the 10-yr history) is reliable. If some tickers act flaky after deploy, that's
  why — refresh/retry usually works.
- Every time you change the code: re-upload the changed file on GitHub (or use
  git push), and Render auto-redeploys.
