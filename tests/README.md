# API tests and continuous integration

Run from the Git repository root (`MINDlarge_train`):

```bash
python -m venv .venv-test
source .venv-test/bin/activate  # PowerShell: .venv-test\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m pytest
```

To reproduce the CI coverage check:

```bash
python -m pytest --cov=recommender_api_under_test --cov-report=term-missing --cov-fail-under=85
```

The suite imports the unchanged `app.py` from a temporary working directory.
It isolates global engine state and redirects latency CSV writes to temporary
files. It never reads production artifacts, downloads MIND, calls AWS, or sends
load to an endpoint.

## What the tests establish

| Tests | Purpose |
| --- | --- |
| `test_api.py`: health and readiness | Preserve the distinction between HTTP 200 `not_ready` on `/health` and HTTP 503 on `/ready`; inference requires a loaded engine. |
| Request validation | Required history, empty/unknown histories, numeric boundaries, malformed JSON, and wrong HTTP method. Invalid requests must not call inference. |
| Recommendation contract | Defaults and explicit parameters reach the engine; response metadata, scores, and timing serialize correctly; empty results are permitted. |
| Failures and latency CSV | Inference exceptions return HTTP 500; successful calls write the expected request/count columns to an isolated CSV. |
| Warmup | Uses the first five metadata news IDs with `top_k=5`, `retrieve_k=25`; readiness and inference errors are handled as currently implemented. |
| `test_serving.py`: loaders | Required artifact failures, optional CTR fallback, metadata formats, missing columns, and invalid embedding-map type. |
| Retrieval and ranking | Real tiny FAISS search, index-to-news mapping, padding removal, recency weighting, seen-item exclusion, score ordering, empty/short candidate lists, and representative feature values. |
| Synthetic integration (`-m integration`) | Real FastAPI startup, disk loading, FAISS and XGBoost prediction, health endpoints, recommendations, and warmup using four synthetic articles and a two-tree test model. |

API unit tests inject a fake engine. They intentionally do not trigger startup.
Integration tests use `with TestClient(...)` to execute the actual startup event.
The synthetic regression model tests serving compatibility, not the quality of
your trained ranking model. Unknown IDs are accepted by the HTTP schema; they
are ignored when computing history embeddings. Scores need not be probabilities.

## CI behavior

`.github/workflows/ci.yml` runs on every push, pull request, and manual dispatch.
It uses Python 3.11 (matching the Dockerfile), installs the serving requirements
and test tools, checks dependency compatibility, and runs all tests. The job has
a 15-minute limit, read-only repository permissions, no AWS secrets, and an 85%
line coverage gate for `app.py`. JUnit, coverage XML, and installed dependency
versions are retained for seven days.

The production requirements remain unpinned; test tools are pinned. A full
runtime dependency lock is a separate release reproducibility improvement.
Deprecation warnings from the existing Pydantic/FastAPI APIs are visible; this
change does not migrate the application or alter model/application code.

Push these additions to GitHub to activate the workflow. In repository Settings
→ Rules / branch protection, make `API tests (Python 3.11)` a required check
after its first successful run if you want failing checks to block merges.
This configuration is CI only: it does not push images or update the ECS service.

The existing `performance/test_workload.py` is a separate Locust-harness suite;
it is not collected by this API suite. Production-model evaluation, Docker/ECS
smoke tests, long-running load tests, and A/B tests are outside this CI gate.
