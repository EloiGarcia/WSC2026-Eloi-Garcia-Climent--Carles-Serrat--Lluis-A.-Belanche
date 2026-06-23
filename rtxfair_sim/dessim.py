# revision_outputs/rtxfair_sim/dessim.py
"""
Discrete-event deployment simulation (optional P1 add-on).

Models an online credit-scoring-with-explanation service as an M/G/c FIFO queue:
Poisson applicant arrivals at rate `lam` (requests/s) are routed to either the
teacher+TreeSHAP path or the distilled student path, each with its own empirical
service-time distribution (sampled with replacement from measured per-request
latencies). We measure queue wait, sojourn (end-to-end) latency, throughput,
server utilisation, and SLA-violation rate under load, across replications.

Pure-Python event loop (no simpy dependency).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import heapq
import numpy as np


def simulate_mgc(lam, service_ms, n_requests=20000, n_servers=1,
                 sla_ms=50.0, seed=0, warmup=2000):
    """M/G/c FIFO queue.
    lam         : arrival rate (requests/second)
    service_ms  : 1-D array of empirical service times (milliseconds) to sample
    n_servers   : number of parallel servers (c)
    Returns dict of steady-state-ish metrics (after discarding `warmup` requests).
    """
    rng = np.random.RandomState(seed)
    service_ms = np.asarray(service_ms, dtype=float)
    # server availability times (seconds); min-heap of free timestamps
    servers = [0.0] * n_servers
    heapq.heapify(servers)

    t = 0.0
    waits, sojourns, svc = [], [], []
    busy_time = 0.0
    last_departure = 0.0
    for k in range(n_requests):
        t += rng.exponential(1.0 / lam)              # next arrival (s)
        s = float(service_ms[rng.randint(len(service_ms))]) / 1000.0  # service (s)
        free = heapq.heappop(servers)
        start = max(t, free)
        wait = start - t
        depart = start + s
        heapq.heappush(servers, depart)
        if k >= warmup:
            waits.append(wait); sojourns.append(s + wait); svc.append(s)
            busy_time += s
            last_departure = max(last_departure, depart)
    waits = np.array(waits); sojourns = np.array(sojourns) * 1000.0  # ms
    span = last_departure - 0.0
    return {
        "lam": lam, "n_servers": n_servers, "sla_ms": sla_ms, "seed": seed,
        "mean_wait_ms": float(np.mean(waits) * 1000),
        "p99_sojourn_ms": float(np.percentile(sojourns, 99)),
        "mean_sojourn_ms": float(np.mean(sojourns)),
        "throughput_rps": float(len(sojourns) / span) if span > 0 else np.nan,
        "utilization": float(busy_time / (span * n_servers)) if span > 0 else np.nan,
        "sla_violation_rate": float(np.mean(sojourns > sla_ms)),
        "offered_load": float(lam * np.mean(svc)),   # = rho * c (Erlangs)
    }


def sweep(service_ms_by_route, lambdas, seeds, n_servers=1, sla_ms=50.0,
          n_requests=20000):
    """Run the queue across routes x arrival-rates x seeds; return list of rows."""
    rows = []
    for route, svc in service_ms_by_route.items():
        for lam in lambdas:
            for sd in seeds:
                m = simulate_mgc(lam, svc, n_requests=n_requests,
                                 n_servers=n_servers, sla_ms=sla_ms, seed=sd)
                m["route"] = route
                rows.append(m)
    return rows
