"""Synthetic Database Generator for Track H (DBMS Optimization).

Creates a realistic retail / order-processing SQLite database:
1. `customers` (10,000 rows): Customer profiles across segments and countries.
2. `products` (2,000 rows): Catalog across categories with prices and stock levels.
3. `orders` (50,000 rows): Orders across dates (2020-2024), amounts, and statuses.
4. `order_items` (100,000 rows): Line items with quantities, prices, and discounts.

Total rows: 162,000 records.
Uses transactional batching for sub-second generation.
"""

from __future__ import annotations
import os
import random
import sqlite3
import time
from typing import Optional, Tuple


DB_PATH = os.path.join(os.path.dirname(__file__), "results", "retail_benchmark.db")

SEGMENTS = ["CONSUMER", "CORPORATE", "ENTERPRISE", "SMALL_BUSINESS"]
COUNTRIES = ["US", "IN", "UK", "DE", "JP", "BR", "CA", "AU", "FR", "SG"]
CATEGORIES = ["ELECTRONICS", "APPAREL", "HOME", "BOOKS", "GARDEN", "SPORTS", "AUTOMOTIVE"]
ORDER_STATUSES = ["COMPLETED", "PENDING", "CANCELLED", "SHIPPED", "RETURNED"]


def create_schema(conn: sqlite3.Connection):
    """Creates the base relational tables without secondary indices (primary keys only)."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = MEMORY;")

    cursor.executescript("""
    DROP TABLE IF EXISTS order_items;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        cust_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        segment TEXT NOT NULL,
        country TEXT NOT NULL,
        created_year INTEGER NOT NULL
    );

    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL
    );

    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        cust_id INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        order_status TEXT NOT NULL,
        total_amount REAL NOT NULL
    );

    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        discount REAL NOT NULL
    );
    """)
    conn.commit()


def populate_database(
    db_path: str = DB_PATH,
    num_customers: int = 10_000,
    num_products: int = 2_000,
    num_orders: int = 50_000,
    num_order_items: int = 100_000,
    seed: int = 42
) -> Tuple[int, float]:
    """Populates the SQLite database with reproducible synthetic records."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    rng = random.Random(seed)

    t0 = time.perf_counter()
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    cursor = conn.cursor()

    # 1. Customers
    customers_data = [
        (
            cid,
            f"Customer_{cid}",
            rng.choice(SEGMENTS),
            rng.choice(COUNTRIES),
            rng.randint(2018, 2024)
        )
        for cid in range(1, num_customers + 1)
    ]
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        customers_data
    )

    # 2. Products
    products_data = [
        (
            pid,
            f"Product_{pid}",
            rng.choice(CATEGORIES),
            round(rng.uniform(10.0, 1500.0), 2),
            rng.randint(0, 500)
        )
        for pid in range(1, num_products + 1)
    ]
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
        products_data
    )

    # 3. Orders
    orders_data = []
    start_date_ts = int(time.mktime(time.strptime("2020-01-01", "%Y-%m-%d")))
    end_date_ts = int(time.mktime(time.strptime("2024-12-31", "%Y-%m-%d")))

    for oid in range(1, num_orders + 1):
        c_id = rng.randint(1, num_customers)
        rand_ts = rng.randint(start_date_ts, end_date_ts)
        o_date = time.strftime("%Y-%m-%d", time.localtime(rand_ts))
        o_status = rng.choice(ORDER_STATUSES)
        tot_amt = round(rng.uniform(20.0, 3500.0), 2)
        orders_data.append((oid, c_id, o_date, o_status, tot_amt))

    cursor.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        orders_data
    )

    # 4. Order Items
    order_items_data = [
        (
            iid,
            rng.randint(1, num_orders),
            rng.randint(1, num_products),
            rng.randint(1, 10),
            round(rng.uniform(10.0, 500.0), 2),
            round(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25]), 2)
        )
        for iid in range(1, num_order_items + 1)
    ]
    cursor.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)",
        order_items_data
    )

    conn.commit()
    conn.close()

    total_rows = num_customers + num_products + num_orders + num_order_items
    elapsed = time.perf_counter() - t0
    return total_rows, elapsed


if __name__ == "__main__":
    rows, sec = populate_database()
    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Generated {rows:,} rows in {sec:.2f}s | Database file: {size_mb:.2f} MB ({DB_PATH})")
