"""Synthetic fraud detection data generator.

Generates realistic, interconnected datasets with embedded fraud patterns:
  - Shared addresses between unrelated customers (synthetic identity)
  - Device reuse across accounts (account takeover)
  - Burst transaction patterns (velocity abuse)
  - Unusual merchant concentration (money laundering)
  - Off-hours transaction clusters (automated fraud)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

logger = logging.getLogger("lakehouse_kg.helpers.fraud")


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic fraud data generation (fraud worked example).

    Lives with the fraud demo, not the core package: the core pipeline is
    domain-neutral, so scenario data generators belong under demos/.
    """

    catalog: str = "main"
    schema: str = "knowledge_graph"
    n_customers: int = 5000
    n_merchants: int = 500
    n_accounts: int = 6000
    n_transactions: int = 100_000
    n_devices: int = 4000
    n_alerts: int = 2000
    fraud_rate: float = 0.03
    shared_address_rate: float = 0.08
    shared_device_rate: float = 0.05
    seed: int = 42

    def table_ref(self, table_name: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{table_name}`"

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Carlos", "Maria", "Wei", "Yuki", "Ahmed", "Fatima", "Ivan", "Olga",
    "Raj", "Priya", "Miguel", "Ana", "Hans", "Ingrid", "Kofi", "Amara",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Chen", "Patel", "Kim", "Tanaka", "Ali", "Mueller", "Ivanov", "Okafor",
]

CITIES = [
    ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Philadelphia", "PA", "19101"),
    ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"),
    ("Dallas", "TX", "75201"), ("Miami", "FL", "33101"),
    ("Atlanta", "GA", "30301"), ("Boston", "MA", "02101"),
    ("Seattle", "WA", "98101"), ("Denver", "CO", "80201"),
    ("Portland", "OR", "97201"),
]

MERCHANT_CATEGORIES = [
    "Electronics", "Grocery", "Restaurant", "Gas Station", "Online Retail",
    "Travel", "Entertainment", "Healthcare", "Clothing", "Jewelry",
    "Cash Advance", "Wire Transfer", "Crypto Exchange", "Gift Cards",
    "Money Order",
]

# Higher-risk categories for fraud injection
HIGH_RISK_CATEGORIES = ["Cash Advance", "Wire Transfer", "Crypto Exchange", "Gift Cards", "Money Order"]


class FraudDataGenerator:
    """Generates synthetic fraud detection datasets with embedded fraud patterns."""

    def __init__(self, spark: SparkSession, config: SyntheticDataConfig):
        self.spark = spark
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        random.seed(config.seed)

    def generate_all(self) -> dict[str, str]:
        """Generate all tables and return a map of table_name -> full_table_ref."""
        self._ensure_schema()

        tables = {}
        customers_df = self._generate_customers()
        tables["customers"] = self._write_table("customers", customers_df)

        merchants_df = self._generate_merchants()
        tables["merchants"] = self._write_table("merchants", merchants_df)

        accounts_df = self._generate_accounts(customers_df)
        tables["accounts"] = self._write_table("accounts", accounts_df)

        devices_df = self._generate_devices(customers_df)
        tables["devices"] = self._write_table("devices", devices_df)

        transactions_df = self._generate_transactions(accounts_df, merchants_df)
        tables["transactions"] = self._write_table("transactions", transactions_df)

        alerts_df = self._generate_alerts(transactions_df)
        tables["alerts"] = self._write_table("alerts", alerts_df)

        logger.info(f"Generated {len(tables)} tables in {self.config.catalog}.{self.config.schema}")
        return tables

    def _ensure_schema(self):
        self.spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS {self.config.catalog}.{self.config.schema}"
        )

    def _write_table(self, name: str, pdf: pd.DataFrame) -> str:
        full_name = self.config.table_ref(name)
        sdf = self.spark.createDataFrame(pdf)
        sdf.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(full_name)
        logger.info(f"  Wrote {len(pdf)} rows to {full_name}")
        return full_name

    def _generate_customers(self) -> pd.DataFrame:
        n = self.config.n_customers
        customers = []
        addresses = self._generate_addresses(n)

        # Introduce shared addresses for fraud pattern
        n_shared = int(n * self.config.shared_address_rate)
        shared_indices = self.rng.choice(n, n_shared, replace=False)
        donor_indices = self.rng.choice(n, n_shared, replace=True)
        for i, donor in zip(shared_indices, donor_indices):
            if i != donor:
                addresses[i] = addresses[donor]

        for i in range(n):
            reg_date = datetime(2020, 1, 1) + timedelta(days=self.rng.randint(0, 1800))
            customers.append({
                "customer_id": f"CUST_{i:06d}",
                "first_name": random.choice(FIRST_NAMES),
                "last_name": random.choice(LAST_NAMES),
                "email": f"user{i}@{'gmail.com' if self.rng.random() > 0.3 else 'tempmail.xyz'}",
                "phone": f"+1{self.rng.randint(2000000000, 9999999999)}",
                "address": addresses[i]["address"],
                "city": addresses[i]["city"],
                "state": addresses[i]["state"],
                "zip_code": addresses[i]["zip"],
                "registration_date": reg_date.strftime("%Y-%m-%d"),
                "risk_score": round(float(self.rng.beta(2, 8)), 3),
            })

        return pd.DataFrame(customers)

    def _generate_addresses(self, n: int) -> list[dict]:
        addresses = []
        for _ in range(n):
            city, state, base_zip = random.choice(CITIES)
            addresses.append({
                "address": f"{self.rng.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Market', 'Broadway', 'Park'])} {'St' if self.rng.random() > 0.5 else 'Ave'}",
                "city": city,
                "state": state,
                "zip": str(int(base_zip) + self.rng.randint(0, 99)),
            })
        return addresses

    def _generate_merchants(self) -> pd.DataFrame:
        n = self.config.n_merchants
        merchants = []
        for i in range(n):
            category = random.choice(MERCHANT_CATEGORIES)
            merchants.append({
                "merchant_id": f"MERCH_{i:05d}",
                "merchant_name": f"{random.choice(['Quick', 'Express', 'Prime', 'Value', 'Metro', 'Global', 'Elite'])} {category} #{i}",
                "category": category,
                "city": (loc := random.choice(CITIES))[0],
                "state": loc[1],
                "risk_level": "HIGH" if category in HIGH_RISK_CATEGORIES else (
                    "MEDIUM" if self.rng.random() > 0.7 else "LOW"
                ),
                "registration_date": (
                    datetime(2019, 1, 1) + timedelta(days=self.rng.randint(0, 2000))
                ).strftime("%Y-%m-%d"),
            })
        return pd.DataFrame(merchants)

    def _generate_accounts(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_accounts
        customer_ids = customers_df["customer_id"].tolist()
        accounts = []
        for i in range(n):
            open_date = datetime(2020, 1, 1) + timedelta(days=self.rng.randint(0, 1800))
            accounts.append({
                "account_id": f"ACCT_{i:06d}",
                "customer_id": random.choice(customer_ids),
                "account_type": random.choice(["checking", "savings", "credit", "prepaid"]),
                "open_date": open_date.strftime("%Y-%m-%d"),
                "balance": round(float(self.rng.lognormal(8, 2)), 2),
                "status": random.choice(["active", "active", "active", "suspended", "closed"]),
                "daily_limit": round(float(self.rng.choice([1000, 2500, 5000, 10000, 25000])), 2),
            })
        return pd.DataFrame(accounts)

    def _generate_devices(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_devices
        customer_ids = customers_df["customer_id"].tolist()
        devices = []

        # Pre-generate some IPs that will be shared (botnet pattern)
        shared_ips = [f"192.168.{self.rng.randint(1,254)}.{self.rng.randint(1,254)}" for _ in range(20)]

        for i in range(n):
            is_shared = self.rng.random() < self.config.shared_device_rate
            if is_shared:
                ip = random.choice(shared_ips)
            else:
                ip = f"{self.rng.randint(1,223)}.{self.rng.randint(0,255)}.{self.rng.randint(0,255)}.{self.rng.randint(1,254)}"

            first_seen = datetime(2020, 6, 1) + timedelta(days=self.rng.randint(0, 1500))
            devices.append({
                "device_id": f"DEV_{i:06d}",
                "customer_id": random.choice(customer_ids),
                "device_type": random.choice(["mobile_ios", "mobile_android", "desktop_windows", "desktop_mac", "tablet"]),
                "ip_address": ip,
                "user_agent": random.choice([
                    "Mozilla/5.0 Chrome/120", "Mozilla/5.0 Safari/17",
                    "Mozilla/5.0 Firefox/121", "DalvikVM Android/14",
                ]),
                "first_seen": first_seen.strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": (first_seen + timedelta(days=self.rng.randint(1, 365))).strftime("%Y-%m-%d %H:%M:%S"),
            })

        return pd.DataFrame(devices)

    def _generate_transactions(
        self, accounts_df: pd.DataFrame, merchants_df: pd.DataFrame
    ) -> pd.DataFrame:
        n = self.config.n_transactions
        account_ids = accounts_df["account_id"].tolist()
        merchant_ids = merchants_df["merchant_id"].tolist()
        high_risk_merchants = merchants_df[
            merchants_df["risk_level"] == "HIGH"
        ]["merchant_id"].tolist()

        transactions = []
        n_fraud = int(n * self.config.fraud_rate)
        fraud_indices = set(self.rng.choice(n, n_fraud, replace=False))

        # Select some accounts as "fraud accounts" for realistic clustering
        n_fraud_accounts = max(10, n_fraud // 5)
        fraud_accounts = self.rng.choice(account_ids, n_fraud_accounts, replace=False).tolist()

        for i in range(n):
            is_fraud = i in fraud_indices
            base_date = datetime(2023, 1, 1) + timedelta(
                seconds=int(self.rng.uniform(0, 365 * 24 * 3600))
            )

            if is_fraud:
                acct = random.choice(fraud_accounts)
                merchant = random.choice(
                    high_risk_merchants if high_risk_merchants and self.rng.random() > 0.3
                    else merchant_ids
                )
                amount = round(float(self.rng.lognormal(7, 1.5)), 2)
                # Fraud tends toward off-hours
                if self.rng.random() > 0.4:
                    base_date = base_date.replace(hour=self.rng.randint(0, 5))
                channel = random.choice(["online", "online", "mobile", "atm"])
            else:
                acct = random.choice(account_ids)
                merchant = random.choice(merchant_ids)
                amount = round(float(self.rng.lognormal(4, 1.2)), 2)
                channel = random.choice(["pos", "online", "mobile", "atm", "pos", "pos"])

            transactions.append({
                "transaction_id": f"TXN_{i:08d}",
                "account_id": acct,
                "merchant_id": merchant,
                "amount": amount,
                "currency": "USD",
                "timestamp": base_date.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": channel,
                "status": random.choice(["completed", "completed", "completed", "pending", "declined"]),
                "is_fraud": is_fraud,
            })

        return pd.DataFrame(transactions)

    def _generate_alerts(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_alerts
        fraud_txns = transactions_df[transactions_df["is_fraud"] == True]["transaction_id"].tolist()
        legit_txns = transactions_df[transactions_df["is_fraud"] == False]["transaction_id"].tolist()

        alerts = []
        for i in range(n):
            # ~60% of alerts on actual fraud, 40% false positives
            if self.rng.random() < 0.6 and fraud_txns:
                txn_id = random.choice(fraud_txns)
            else:
                txn_id = random.choice(legit_txns)

            created = datetime(2023, 1, 1) + timedelta(days=self.rng.randint(0, 365))
            alerts.append({
                "alert_id": f"ALERT_{i:06d}",
                "transaction_id": txn_id,
                "alert_type": random.choice([
                    "velocity_breach", "amount_threshold", "geo_anomaly",
                    "device_anomaly", "pattern_match", "ml_model_flag",
                ]),
                "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
                "resolved": self.rng.random() > 0.3,
                "resolution": random.choice(["confirmed_fraud", "false_positive", "pending", "escalated"]),
            })

        return pd.DataFrame(alerts)
