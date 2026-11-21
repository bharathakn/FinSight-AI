"""Load test for the gateway (Plan §8.2 elasticity; Phase 8 acceptance).

    locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 60s --csv reports/load_remote

Traffic mix (weights): mostly scoring stored transactions, some what-if scoring, plus the read endpoints the
dashboard calls. Transaction IDs are fetched once from the API, so the test needs no local data.
"""
import random

from locust import HttpUser, between, events, task

IDS: list[str] = []


@events.test_start.add_listener
def load_ids(environment, **_):
    import httpx

    r = httpx.get(f"{environment.host}/transactions", params={"limit": 500, "sort": "transaction_id"}, timeout=120)
    IDS.extend(t["transaction_id"] for t in r.json()["items"])


class DashboardUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(10)
    def score(self):
        self.client.post("/score-transaction", json={"transaction_id": random.choice(IDS)}, name="/score-transaction")

    @task(2)
    def what_if(self):
        self.client.post("/score-transaction", json={"transaction_id": random.choice(IDS),
                                                     "overrides": {"amount_inr": random.choice([500, 20000, 95000]),
                                                                   "is_new_device": random.choice([0, 1])}},
                         name="/score-transaction (what-if)")

    @task(2)
    def browse(self):
        self.client.get("/transactions", params={"limit": 25, "label": random.choice(["FRAUDULENT", "BORDERLINE"])},
                        name="/transactions")

    @task(1)
    def patterns(self):
        self.client.get("/patterns", params={"consequent": "FRAUDULENT", "limit": 20}, name="/patterns")

    @task(1)
    def network(self):
        self.client.get(f"/network/ACC_IND_{random.randint(1, 1500):06d}", name="/network/{account_id}")
