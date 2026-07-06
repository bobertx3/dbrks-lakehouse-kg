# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Synthetic Data Generator (Fraud Worked Example)
# MAGIC
# MAGIC Generates realistic, interconnected datasets for testing the Lakehouse Knowledge Graph
# MAGIC starter kit. This is the fraud-detection worked example that pairs with the `fraud`
# MAGIC domain pack in notebook 02.
# MAGIC
# MAGIC **Tables created:**
# MAGIC | Table | Description | Patterns Embedded |
# MAGIC |-------|-------------|-------------------|
# MAGIC | `customers` | Customer profiles | Shared addresses (synthetic identity) |
# MAGIC | `accounts` | Bank accounts | Multiple accounts per customer |
# MAGIC | `merchants` | Merchant profiles | High-risk category merchants |
# MAGIC | `transactions` | Financial transactions | Velocity bursts, off-hours, high amounts |
# MAGIC | `devices` | Device/IP records | Shared IPs (botnet), device reuse |
# MAGIC | `alerts` | Fraud alerts | Mix of true/false positives |

# COMMAND ----------

# MAGIC %pip install scikit-learn networkx -q
# MAGIC %pip install -e /Workspace/Users/<your-user>/lakehouse-kg-starter --no-deps -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "knowledge_graph", "Schema")
dbutils.widgets.text("n_customers", "5000", "Number of Customers")
dbutils.widgets.text("n_transactions", "100000", "Number of Transactions")
dbutils.widgets.text("fraud_rate", "0.03", "Fraud Rate (0-1)")
dbutils.widgets.text("seed", "42", "Random Seed")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
n_customers = int(dbutils.widgets.get("n_customers"))
n_transactions = int(dbutils.widgets.get("n_transactions"))
fraud_rate = float(dbutils.widgets.get("fraud_rate"))
seed = int(dbutils.widgets.get("seed"))

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Customers: {n_customers:,}")
print(f"Transactions: {n_transactions:,}")
print(f"Fraud rate: {fraud_rate:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------

from lakehouse_kg.config import SyntheticDataConfig
from lakehouse_kg.generators.synthetic_fraud import FraudDataGenerator

config = SyntheticDataConfig(
    catalog=catalog,
    schema=schema,
    n_customers=n_customers,
    n_merchants=500,
    n_accounts=int(n_customers * 1.2),
    n_transactions=n_transactions,
    n_devices=int(n_customers * 0.8),
    n_alerts=int(n_transactions * 0.02),
    fraud_rate=fraud_rate,
    shared_address_rate=0.08,
    shared_device_rate=0.05,
    seed=seed,
)

generator = FraudDataGenerator(spark, config)
tables = generator.generate_all()

print("\nGenerated tables:")
for name, ref in tables.items():
    count = spark.table(ref).count()
    print(f"  {ref}: {count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Verification

# COMMAND ----------

from pyspark.sql import functions as F

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

# Verify fraud distribution
txn_ref = tables["transactions"]
txn_df = spark.table(txn_ref)
fraud_stats = txn_df.groupBy("is_fraud").agg(
    F.count("*").alias("count"),
    F.mean("amount").alias("avg_amount"),
    F.max("amount").alias("max_amount"),
)
print("\nTransaction fraud distribution:")
fraud_stats.show()

# Verify shared addresses
cust_ref = tables["customers"]
cust_df = spark.table(cust_ref)
addr_sharing = (
    cust_df.groupBy("address", "city", "state")
    .agg(F.count("*").alias("n_customers"))
    .filter(F.col("n_customers") > 1)
)
print(f"\nAddresses shared by multiple customers: {addr_sharing.count()}")

# Verify device/IP sharing
dev_ref = tables["devices"]
dev_df = spark.table(dev_ref)
ip_sharing = (
    dev_df.groupBy("ip_address")
    .agg(F.countDistinct("customer_id").alias("n_customers"))
    .filter(F.col("n_customers") > 1)
)
print(f"\nIPs shared by multiple customers: {ip_sharing.count()}")

# Channel distribution
print("\nTransaction channels:")
txn_df.groupBy("channel").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedded Fraud Patterns Summary
# MAGIC
# MAGIC The synthetic data contains these patterns for the agentic pipeline to discover:
# MAGIC
# MAGIC 1. **Synthetic Identity** — ~8% of customers share an address with an unrelated customer
# MAGIC 2. **Device Reuse / Botnets** — ~5% of devices share IP addresses across different customers
# MAGIC 3. **Velocity Abuse** — Fraud transactions cluster on specific "fraud accounts"
# MAGIC 4. **Off-Hours Activity** — 60% of fraud transactions occur between midnight and 5am
# MAGIC 5. **High-Risk Merchants** — Fraud transactions preferentially target Cash Advance, Wire Transfer, Crypto, Gift Card, and Money Order merchants
# MAGIC 6. **Amount Anomalies** — Fraud transactions have significantly higher amounts (lognormal μ=7 vs μ=4)

# COMMAND ----------

print("Synthetic fraud data generation complete!")
print(f"\nReady for agentic triplet generation pipeline.")
print(f"Run notebook 02_triplet_pipeline with:")
print(f"  catalog = {catalog}")
print(f"  schema = {schema}")
print(f"  domain = fraud")
