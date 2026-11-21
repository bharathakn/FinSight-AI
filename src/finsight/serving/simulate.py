"""Attack simulator (Phase 9b): a normal day of payments with chosen fraud patterns injected, in upload format.

Background  legitimate payments from the bank history, moved to 31 Aug 2026 (the day after the history ends,
            so the history is genuine prior context) with new ids.
Attacks     fraud_ring      many senders pay one hub account within ~2 hours (fan-in)
            mule_chain      money arrives and is forwarded ≥ 90% within minutes, hop after hop (layering)
            burst           one sender, many payments to different receivers within 10 minutes
            structuring     one sender, several payments just under ₹50,000 in a day
            shared_device   one device used by many different accounts within a few hours
camouflage  True: behaviour looks normal (known device and payee, near home, usual amount) so only the network
            pattern gives it away — the same 50% camouflage idea as dataset v1.1 (D-002).
Injected rows are labelled FRAUDULENT with scenario "SIM <attack>", so the analysis can report detection rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finsight import config

ATTACKS = {
    "fraud_ring": "Fraud ring: many senders pay one hub account within two hours",
    "mule_chain": "Mule chain: money is received and forwarded within minutes, hop after hop",
    "burst": "Burst: one sender makes many payments to different people within 10 minutes",
    "structuring": "Structuring: several payments just under ₹50,000 to avoid the reporting threshold",
    "shared_device": "Shared device: one phone used by many different accounts",
}
DAY = pd.Timestamp("2026-08-31", tz=config.TIMEZONE)


def _background(ref, n: int, rng) -> pd.DataFrame:
    legit = ref.history.merge(ref.history_labels[["transaction_id", "risk_label"]], on="transaction_id")
    legit = legit[legit["risk_label"].astype(str) == "LEGITIMATE"]
    bg = legit.sample(min(n, len(legit)), random_state=int(rng.integers(1 << 30))).copy()
    tod = bg["timestamp"] - bg["timestamp"].dt.normalize()
    bg["timestamp"] = DAY + tod
    bg["transaction_id"] = [f"SIM_BG_{i:05d}" for i in range(len(bg))]
    bg["is_fraud"], bg["risk_label"], bg["scenario"] = 0, "LEGITIMATE", "S1 normal legitimate"
    return bg.drop(columns=[c for c in ("risk_label_x", "risk_label_y") if c in bg.columns])


class _Builder:
    def __init__(self, ref, rng, camouflage: bool):
        self.ref, self.rng, self.camo = ref, rng, camouflage
        self.template = ref.history.iloc[0]
        self.accounts = ref.accounts["account_id"].to_numpy()
        self.rows: list[dict] = []

    def accs(self, k: int) -> list[str]:
        return list(self.rng.choice(self.accounts, size=k, replace=False))

    def pay(self, attack: str, sender: str, receiver: str, amount: float, ts: pd.Timestamp, device: str | None = None) -> None:
        r, acc, cust = self.ref, self.ref.acc, self.ref.cust
        scid = acc.loc[sender, "customer_id"]
        home = cust.loc[scid]
        avg = float(acc.loc[sender, "historical_30d_avg_tx_amt_inr"])
        row = self.template.to_dict()
        rng = self.rng
        row.update({
            "transaction_id": f"SIM_{attack.upper()}_{len(self.rows):04d}", "timestamp": ts, "amount_inr": round(float(amount), 2),
            "channel": "UPI" if amount <= 100_000 else "RTGS", "sender_account_id": sender, "receiver_account_id": receiver,
            "sender_customer_id": scid, "receiver_customer_id": acc.loc[receiver, "customer_id"],
            "device_id": device or r.first_device.get(scid, f"DEV_UNKNOWN_{sender}"), "ip_address": f"ip-{sender}",
            "tx_city": home["home_city"], "tx_state": home["home_state"], "tx_lat": home["home_lat"], "tx_long": home["home_long"],
            "amount_ratio_vs_30d_avg": amount / avg if avg > 0 else 1.0,
            "account_balance_before": float(acc.loc[sender, "current_balance_inr"]) + amount,
            "tx_frequency_24h": r.medians["tx_frequency_24h"], "time_since_last_tx_sec": np.nan,
            "is_fraud": 1, "risk_label": "FRAUDULENT", "scenario": f"SIM {attack.replace('_', ' ')}",
        })
        if self.camo:      # behaviour looks ordinary: only the network structure is unusual
            row.update({"is_new_device": 0, "is_new_beneficiary": 0, "distance_from_home_km": float(rng.uniform(1, 15)),
                        "time_since_beneficiary_added_min": float(rng.uniform(20_000, 200_000))})
        else:              # classic red flags as well
            row.update({"is_new_device": int(rng.random() < 0.7), "is_new_beneficiary": 1,
                        "distance_from_home_km": float(rng.uniform(150, 900)),
                        "time_since_beneficiary_added_min": float(rng.uniform(2, 120))})
        row["account_balance_after"] = row["account_balance_before"] - row["amount_inr"]
        self.rows.append(row)

    # ---- attack patterns (size = intensity)
    def fraud_ring(self, size: int, t0: pd.Timestamp):
        hub, *senders = self.accs(size + 1)
        for i, s in enumerate(senders):
            self.pay("fraud_ring", s, hub, self.rng.uniform(15_000, 48_000), t0 + pd.Timedelta(minutes=int(self.rng.integers(0, 120))))

    def mule_chain(self, size: int, t0: pd.Timestamp):
        for c in range(max(1, size // 3)):
            chain = self.accs(4)
            amt, t = self.rng.uniform(30_000, 90_000), t0 + pd.Timedelta(minutes=40 * c)
            for a, b in zip(chain[:-1], chain[1:]):
                self.pay("mule_chain", a, b, amt, t)
                amt *= self.rng.uniform(0.92, 0.99)                 # forward almost everything …
                t += pd.Timedelta(minutes=int(self.rng.integers(3, 20)))   # … within minutes

    def burst(self, size: int, t0: pd.Timestamp):
        sender, *receivers = self.accs(size + 1)
        for i, rcv in enumerate(receivers):
            self.pay("burst", sender, rcv, self.rng.uniform(2_000, 12_000), t0 + pd.Timedelta(seconds=int(self.rng.integers(0, 600))))

    def structuring(self, size: int, t0: pd.Timestamp):
        sender, *receivers = self.accs(3)
        for i in range(size):
            self.pay("structuring", sender, receivers[i % len(receivers)], self.rng.uniform(45_000, 49_900),
                     t0 + pd.Timedelta(minutes=int(self.rng.integers(0, 600))))

    def shared_device(self, size: int, t0: pd.Timestamp):
        dev = f"DEV_SIM_SHARED_{int(self.rng.integers(1e6))}"
        senders = self.accs(size)
        receivers = self.accs(size)
        for i, (s, r) in enumerate(zip(senders, receivers)):
            self.pay("shared_device", s, r, self.rng.uniform(5_000, 40_000), t0 + pd.Timedelta(minutes=int(self.rng.integers(0, 240))), device=dev)


def simulate(ref, attacks: list[str], size: int = 10, camouflage: bool = False, background: int = 1500,
             seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    unknown = [a for a in attacks if a not in ATTACKS]
    if unknown or not attacks:
        raise ValueError(f"attacks must be a non-empty subset of {sorted(ATTACKS)}")
    size = int(np.clip(size, 3, 40))
    background = int(np.clip(background, 200, 5000))
    rng = np.random.default_rng(seed if seed is not None else config.SEED)
    bg = _background(ref, background, rng)
    b = _Builder(ref, rng, camouflage)
    for i, a in enumerate(attacks):
        getattr(b, a)(size, DAY + pd.Timedelta(hours=9 + 2 * i))
    att = pd.DataFrame(b.rows)
    cols = list(dict.fromkeys(list(bg.columns) + list(att.columns)))
    df = pd.concat([bg.reindex(columns=cols), att.reindex(columns=cols)], ignore_index=True).sort_values("timestamp")
    meta = {"attacks": attacks, "size": size, "camouflage": camouflage, "background_rows": int(len(bg)),
            "injected_rows": int(len(att)), "injected_by_attack": att["scenario"].value_counts().to_dict(),
            "descriptions": {a: ATTACKS[a] for a in attacks}}
    return df.reset_index(drop=True), meta
