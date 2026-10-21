# Flint API (backend)

FastAPI backend for Flint. It signs users in with Clerk, stores their data in Neon Postgres, and serves market data, plans and the AI desk to the Next.js frontend (`algo-trader-frontend`).

This guide goes from a fresh machine to a deployed backend on Vercel. Commands are for **PowerShell** on Windows, run from the `algo-trader-backend` folder.

---

## 0. What you need

| Thing | Where to get it |
|---|---|
| Python 3.14 | python.org |
| A Neon project (Postgres) | neon.tech, Dashboard, Connect |
| A Clerk application | clerk.com, API keys |
| A Gemini API key | aistudio.google.com |
| A Vercel account linked to GitHub | vercel.com |

---

## 1. Install (once per machine)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If this venv previously had `xgboost` installed, swap it for the CPU build first so the two don't overwrite each other:

```powershell
.\venv\Scripts\python.exe -m pip uninstall -y xgboost
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 2. Create `.env` (once)

The app reads only `.env` and `.env.clerk`. It never reads `.env.example`, which is just a template.

Create `.env` with **one value per line and no comments on the same line**. A trailing `# comment` becomes part of the value.

```
DATABASE_URL_UNPOOLED=postgresql://neondb_owner:...@ep-xxxx.neon.tech/neondb?sslmode=require
CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
AUTHORIZED_PARTIES=http://localhost:3000
GEMINI_API_KEY=...
ADMIN_USER_IDS=
```

- `DATABASE_URL_UNPOOLED` is the **owner** connection string from Neon. It is used only to set up the database in step 3, never by the running API.
- `DATABASE_URL_APP` is not written by hand. Step 3 creates it.
- Leave `DATABASE_URL` out entirely unless you paste a real Neon URL into it.

> **Never commit real values in `.env.example`.** It is tracked by git. Keep only the names and blank values there.

---

## 3. Set up the database (once per Neon database)

```powershell
.\venv\Scripts\python.exe -m db.migrate --app-role
```

Expected output:

```
flint_app role ready; DATABASE_URL_APP written to .env
applied 0001_init.sql
applied 0002_fills_fx_and_idempotency.sql
```

This does three things:
1. Creates `flint_app`, a limited login role the API runs as. Row-level security means it can only see the signed-in user's rows.
2. Writes `DATABASE_URL_APP` (the `flint_app` connection string) into `.env`.
3. Creates the tables.

**Later, when new migration files are added**, run it without the flag:

```powershell
.\venv\Scripts\python.exe -m db.migrate
```

Running `--app-role` again only gives `flint_app` a new password and rewrites `DATABASE_URL_APP`. If you do that, update the value on Vercel too (step 6).

---

## 4. Run locally

```powershell
.\venv\Scripts\python.exe -m uvicorn flint_api.main:app --host 127.0.0.1 --port 5000
```

(In Git Bash, `./run.sh` does the same.)

Check it in a browser: <http://127.0.0.1:5000/api/health> should show `{"status":"ok"}`.

- `http://127.0.0.1:5000/` shows 404. That is normal: there is no page at `/`.
- Routes like `/api/me` return **401 "Sign in to continue"** when opened directly. That is also normal: they need a Clerk login and are meant to be called through the frontend.
- Stop the server with **Ctrl+C**.

Then start the frontend in a second terminal (`algo-trader-frontend\frontend`):

```powershell
npm run dev
```

Open <http://localhost:3000> and sign in.

---

## 5. Deploy the backend to Vercel

The repo already contains what Vercel needs:

| File | Purpose |
|---|---|
| `index.py` | Exposes `flint_api.main:app` where Vercel looks for it. |
| `vercel.json` | Allows requests up to 300 s (the AI desk streams for a while) and keeps tests out of the bundle. |
| `.vercelignore` | Keeps `.env*`, venv and caches out of CLI uploads. |
| `.python-version` | Python 3.14. |
| `requirements.txt` | Uses `xgboost-cpu`, which keeps the bundle under Vercel's 500 MB limit. |

### 5.1 Push the code

```powershell
git push origin flint-os-v2
```

### 5.2 Create the Vercel project

1. In Vercel, click **Add New, then Project** and import `Sevith-NS/algo-trader-backend`.
2. **Framework Preset** should detect **FastAPI**. **Root Directory** stays as the repo root.
3. Open **Environment Variables** and add these. Copy each value from your local `.env` and `.env.clerk`:

   | Name | Value |
   |---|---|
   | `DATABASE_URL_APP` | the `flint_app` URL that step 3 wrote to `.env` |
   | `CLERK_PUBLISHABLE_KEY` | `pk_...` |
   | `CLERK_SECRET_KEY` | `sk_...` |
   | `GEMINI_API_KEY` | your key |
   | `AUTHORIZED_PARTIES` | `http://localhost:3000` for now (you add the frontend URL in step 6) |
   | `ADMIN_USER_IDS` | optional; your Clerk user id(s), comma-separated |

   **Do not** add `DATABASE_URL_UNPOOLED`. That is the owner login, used only from your machine for migrations.
4. Click **Deploy**.

### 5.3 Make it deploy from your branch

Vercel deploys the repo's default branch (`main`) to production. Either merge `flint-os-v2` into `main`, or set **Settings, Git, Production Branch** to `flint-os-v2`.

### 5.4 Check it

Open `https://<your-backend>.vercel.app/api/health` and look for `{"status":"ok"}`.

The first request after a while can take several seconds, because the instance is starting and loading pandas and xgboost. Later requests are fast.

---

## 6. Connect the frontend

In the **frontend's** Vercel project, under **Settings, Environment Variables**:

| Name | Value |
|---|---|
| `FLINT_API_URL` | `https://<your-backend>.vercel.app` (no trailing slash) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | same `pk_...` as the backend |
| `CLERK_SECRET_KEY` | same `sk_...` as the backend |

Redeploy the frontend.

Then go back to the **backend's** Vercel project and change `AUTHORIZED_PARTIES` to include the frontend's address, comma-separated:

```
https://<your-frontend>.vercel.app,http://localhost:3000
```

Redeploy the backend. **This step is required:** without it, every signed-in request returns 401 in production, because the backend rejects login tokens issued for sites it doesn't know.

Environment variable changes on Vercel only take effect after a redeploy (**Deployments, ⋯, Redeploy**).

---

## 7. Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| `relation "users" does not exist` | Tables were never created on this database. | Step 3. |
| `role "flint_app" does not exist` | Migrated without creating the role first. | `python -m db.migrate --app-role` (safe to re-run). |
| `missing "=" after ... @None` | `DATABASE_URL_APP` was built from a broken `DATABASE_URL` line, such as a comment pasted as the value. | Delete the `DATABASE_URL=` line from `.env`, then re-run step 3. |
| 500 errors on every signed-in page, health is fine | The database is unreachable or `DATABASE_URL_APP` is wrong. | Check the value on Vercel matches `.env`, then redeploy. |
| 401 on every signed-in request in production only | The frontend URL is missing from `AUTHORIZED_PARTIES`. | Step 6, then redeploy the backend. |
| The frontend gets a Vercel login page or 401 from the backend | `FLINT_API_URL` points to a **preview** URL, which Vercel protects. | Use the production URL, or turn off Deployment Protection for the backend. |
| Build fails with a function size error | The bundle is over 500 MB. | Make sure `requirements.txt` says `xgboost-cpu`, not `xgboost`, and that test tools stay in `requirements-dev.txt`. |
| An AI desk run cuts off around 5 minutes | Vercel's time limit. | `maxDuration` in `vercel.json` (300 s is the Hobby maximum; Pro allows more). |
| Market caches seem to reset | Vercel's disk is read-only, so caches live only as long as an instance. | Expected. Nothing breaks; the data is fetched again. |

---

## Reference

- Entry point: `flint_api/main.py` (`app`)
- Migrations: `db/migrations/*.sql`, applied by `db/migrate.py`
- Tests: `pip install -r requirements-dev.txt`, then `.\venv\Scripts\python.exe -m pytest`
- Environment variables: `flint_api/settings.py`
