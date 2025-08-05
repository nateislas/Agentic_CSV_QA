# Engineering and Scalability Test Plan

## 1. Purpose
This document outlines the test plan for validating the core engineering requirements of the Agentic CSV QA system, as specified by the CTO. The focus is on scalability, performance, concurrency, and robustness of the API and background job processing system. This plan complements the functional tests in `test_user_flows.md`.

## 2. Core Requirements from CTO
- **Scalability & Speed:** API calls should be fast (~1 second), with long-running tasks handled by background jobs.
- **Concurrency:** The system must handle multiple simultaneous users and requests.
- **Robustness:** The system must gracefully handle large and messy CSV files.
- **Architecture:** The design should reflect solid engineering principles (e.g., microservices, async processing).

---

## 3. Test Categories

### Test Suite 1: API Endpoint Validation

**Goal:** Ensure the two primary endpoints (`/upload`, `/query`) are robust and handle various inputs correctly.

| Test ID | Endpoint | Description | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **FT-1.1** | `POST /upload` | Upload a valid, clean CSV file (e.g., `london_crime.csv`). | `200 OK` with a `session_id`. |
| **FT-1.2** | `POST /upload` | Upload a very large CSV file (>100MB). | `202 Accepted` with a `job_id` for preprocessing. |
| **FT-1.3** | `POST /upload` | Upload a malformed CSV (inconsistent column numbers). | `400 Bad Request` with a clear error message. |
| **FT-1.4** | `POST /upload` | Upload a non-CSV file (e.g., `.txt`, `.png`). | `415 Unsupported Media Type` with a clear error. |
| **FT-1.5** | `POST /query` | Send a simple query (`"What are the columns?"`) to a valid session. | `200 OK` with the correct agent response. |
| **FT-1.6** | `POST /query` | Send a query to an invalid or expired `session_id`. | `404 Not Found`. |
| **FT-1.7** | `POST /query` | Send a query that is known to be computationally expensive. | `202 Accepted` with a `job_id` for async processing. |

### Test Suite 2: Performance & Scalability Testing

**Goal:** Verify the system meets the ~1 second response time requirement and scales under load.

| Test ID | Test Type | Description | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **PT-2.1** | Latency | Measure response time for a simple query on a large CSV. | p99 response time < 1 second. |
| **PT-2.2** | Load Test | Use a tool (e.g., k6, Locust) to simulate 50 concurrent users sending simple queries for 1 minute. | p95 response time remains < 1.5 seconds. No API errors. Server CPU/memory remain stable. |
| **PT-2.3** | Stress Test | Gradually increase concurrent users until the system fails. | Identify the breakpoint. System should fail gracefully (e.g., return `503 Service Unavailable`) without crashing. |

### Test Suite 3: Concurrency & Asynchronous Job Testing

**Goal:** Ensure the background job system works correctly and handles concurrent requests without conflicts.

| Test ID | Test Type | Description | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **CJ-3.1** | Concurrency | Send two different, long-running queries to the **same session** simultaneously. | Both jobs are created. Session state is not corrupted. Both jobs complete successfully with correct results. |
| **CJ-3.2** | Isolation | Send long-running queries to **10 different sessions** simultaneously. | All 10 jobs are created and processed independently. Results are not mixed between sessions. |
| **CJ-3.3** | Job Status | For a long-running job, poll a `GET /jobs/{job_id}/status` endpoint. | Status transitions correctly: `PENDING` -> `RUNNING` -> `SUCCESS`. |
| **CJ-3.4** | Job Failure | Send a query that will cause the agent to fail (e.g., ask it to perform an impossible calculation). | Job status transitions to `FAILED`. The status response contains a useful error message. |
| **CJ-3.5** | Result Retrieval | Once a job's status is `SUCCESS`, hit a `GET /jobs/{job_id}/result` endpoint. | `200 OK` with the final JSON response from the agent. |

### Test Suite 4: Advanced Agentic & Data Handling Tests

**Goal:** Verify the agent can handle the complex, messy, and multi-file scenarios mentioned by the CTO.

| Test ID | Flow Description | Steps | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **AAT-4.1** | Multi-CSV Merging | 1. Upload `london_crime.csv` (gets `session_A`).<br>2. Upload `borough_demographics.csv` (gets `session_B`).<br>3. Send query to `session_A`: `"Merge this data with the demographics data in session_B on the 'borough' column."` | The agent successfully performs the merge and can answer follow-up questions about the combined dataset. |
| **AAT-4.2** | Messy Data Handling | 1. Create a CSV with missing values, mixed data types, and extra empty columns.<br>2. Upload the messy CSV.<br>3. Send query: `"Summarize this data for me."` | The agent doesn't crash. It should ideally use its `DataQualityTool` to identify and report the issues before attempting analysis. |

---
This test plan provides a clear path to validating the engineering rigor of the application. I will create this file for you now.