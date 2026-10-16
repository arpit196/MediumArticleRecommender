# Next Read — session-based recommendation demo

A minimal reading UI for the existing MIND **news** recommender. It provides an
abstract reader, topic-filtered starter shelf, reading trail, anonymous sessions,
and up to four refreshed recommendations after an article is marked as read.
The FAISS/XGBoost serving application is unchanged.

## Run locally with the existing API

From `MINDlarge_train`, activate your Python environment, then:

```bash
python -m pip install -r demo/requirements.txt
# Start your existing recommender separately on port 8000, e.g. Docker.
python -m uvicorn demo.server:app --host 127.0.0.1 --port 8080
```

Open **http://127.0.0.1:8080**. The demo reads the repository's `news.tsv`.
Use the same metadata version as the model-serving API.

## Use your AWS API instead

WSL / Bash:

```bash
export RECOMMENDER_API_URL="http://mind-alb-1237783495.eu-north-1.elb.amazonaws.com"
python -m uvicorn demo.server:app --host 127.0.0.1 --port 8080
```

PowerShell:

```powershell
$env:RECOMMENDER_API_URL = "http://mind-alb-1237783495.eu-north-1.elb.amazonaws.com"
python -m uvicorn demo.server:app --host 127.0.0.1 --port 8080
```

The adapter calls AWS from your computer, so the ALB must permit your current
public IP. The browser only talks to the demo's own origin; no CORS changes to
`app.py` are needed. No upstream URL can be supplied by a browser visitor.
Each session load or read action makes one upstream request, with a 10-second
network inactivity timeout (3 seconds to connect), without automatic retries.
It sends the latest 100 article IDs, `top_k=4`, and `retrieve_k=100`.

## Demo flow

1. Open the page: a new reader receives cold-history recommendations.
2. Browse a topic or open a recommendation.
3. Read the dataset abstract; optionally open its original archive link.
4. Click **Mark as read**: the session history is saved before recommendations
   are refreshed, and already-read articles are excluded from the displayed set.
5. Reload: history persists. Use a different browser profile/incognito window
   to show a second, isolated reader. Tabs in the same browser share a session.
6. **New reader** starts a separate session. **Reset reading history** clears
   the current session only.

Marking an article as read is an explicit demo action, not verified reading or
automatic engagement tracking. Cold results use the API's existing fallback;
the starter shelf is a category-balanced selection from metadata, not model
output. No recommendations are invented if the upstream API is unavailable.
The sidebar displays API connectivity and the last server-reported inference
time; it is not a performance benchmark.

## Storage and deployment scope

Histories are server-side SQLite records associated with an opaque HttpOnly,
SameSite=Lax cookie. Sessions expire seven days after creation; expired records
are cleaned on subsequent session requests. No names or email are collected.
Set `DEMO_DB_PATH` to move the database, `DEMO_NEWS_PATH` for another metadata
file, and `DEMO_SECURE_COOKIE=1` when serving over HTTPS.

This is an anonymous, local demo rather than account authentication. Session
cookies act as bearer credentials. Keep it on loopback for presenting from your
computer. For a public multi-task ECS deployment, add HTTPS, rate limits, and
shared durable session storage instead of task-local SQLite. Browser-session
state has not been added to the production model container, and this task does
not redeploy or change AWS infrastructure.

## Tests

```bash
python -m pytest tests/test_demo.py
```

Tests use synthetic metadata and a mocked upstream HTTP transport. They verify
session isolation/persistence, history forwarding, four-result refresh, duplicate
reads, reset/new-reader behavior, validation, origin protection, and upstream
failure recovery. The existing GitHub Actions suite collects them automatically.
