#!/usr/bin/env python3
"""
generate_mock_data.py
======================
Generates realistic, relationally-consistent mock data for a Text-to-SQL
portfolio project spanning two engines:

  1. SQLite  -> ecommerce_oltp.db        (OLTP: products, customers, orders, order_items)
  2. DuckDB  -> ecommerce_analytics.duckdb (OLAP: historical_sales_daily, category_margins)

Run directly:
    python generate_mock_data.py

Requires: faker, duckdb, pandas (standard lib: sqlite3, random, datetime)
    pip install faker duckdb pandas
"""

import os
import random
import sqlite3
from datetime import date, datetime, timedelta

import duckdb
import pandas as pd
from faker import Faker

# --------------------------------------------------------------------------
# Config / reproducibility
# --------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
Faker.seed(SEED)
fake = Faker()

SQLITE_PATH = "ecommerce_oltp.db"
DUCKDB_PATH = "ecommerce_analytics.duckdb"

N_PRODUCTS = 50
N_CUSTOMERS = 100
N_ORDERS = 300
ORDER_WINDOW_DAYS = 30           # orders placed within the last 30 days
HISTORY_YEARS = 2                # historical_sales_daily lookback

CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home & Kitchen",
    "Books",
    "Sports & Outdoors",
    "Beauty & Personal Care",
    "Toys & Games",
    "Grocery",
]

# Realistic product name pools per category (kept curated rather than
# randomly assembled so names actually look like products).
CATEGORY_PRODUCT_NAMES = {
    "Electronics": [
        "Wireless Earbuds", "Bluetooth Speaker", "Smartphone Stand", "USB-C Fast Charger",
        "27-inch 4K Monitor", "Mechanical Keyboard", "Wireless Mouse", "Portable SSD 1TB",
        "Fitness Smartwatch", "Noise Cancelling Headphones",
    ],
    "Clothing": [
        "Cotton Crew T-Shirt", "Denim Jacket", "Running Shorts", "Merino Wool Sweater",
        "Slim Fit Jeans", "Summer Floral Dress", "Packable Rain Jacket", "High-Waist Yoga Pants",
        "Graphic Pullover Hoodie", "Slim Formal Shirt",
    ],
    "Home & Kitchen": [
        "Stainless Steel Water Bottle", "Non-Stick Frying Pan", "Ceramic Coffee Mug Set",
        "Electric Kettle 1.7L", "Bamboo Cutting Board Set", "Glass Storage Container Set",
        "LED Adjustable Desk Lamp", "Memory Foam Pillow", "Egyptian Cotton Bed Sheets",
        "Digital Air Fryer",
    ],
    "Books": [
        "The Silent Detective (Mystery Novel)", "Atomic Focus (Self-Help Guide)",
        "The Home Cook's Bible (Cookbook)", "Beyond the Nebula (Sci-Fi Epic)",
        "A Life in Ink (Biography)", "The Curious Fox (Picture Book)",
        "Scaling Up (Business Strategy)", "Whispers of Autumn (Poetry Collection)",
        "Roads Less Traveled (Travel Memoir)", "Clean Code Handbook",
    ],
    "Sports & Outdoors": [
        "Non-Slip Yoga Mat", "Adjustable Dumbbell Set", "4-Person Camping Tent",
        "40L Hiking Backpack", "Resistance Bands Set", "Aero Cycling Helmet",
        "Insulated Steel Water Bottle", "Lightweight Running Shoes", "GPS Fitness Tracker",
        "Folding Camping Chair",
    ],
    "Beauty & Personal Care": [
        "Hydrating Face Cream", "Vitamin C Serum", "Sonic Electric Toothbrush",
        "Ionic Hair Dryer", "Shampoo & Conditioner Set", "Broad-Spectrum Sunscreen SPF50",
        "Tinted Lip Balm Set", "Gentle Foaming Cleanser", "Professional Makeup Brush Set",
        "Beard Grooming Kit",
    ],
    "Toys & Games": [
        "Classic Building Blocks Set", "Family Board Game", "Remote Control Race Car",
        "1000-Piece Jigsaw Puzzle", "Plush Teddy Bear", "Collectible Action Figure",
        "Kids Educational Tablet", "Strategy Card Game", "STEM Building Kit",
        "Deluxe Art & Craft Kit",
    ],
    "Grocery": [
        "Extra Virgin Olive Oil 1L", "Whole Wheat Pasta 500g", "Creamy Almond Butter",
        "Organic Green Tea Bags", "Basmati Rice 5kg", "Raw Wildflower Honey",
        "Mixed Nuts & Berries Pack", "Cold Pressed Orange Juice", "70% Dark Chocolate Bar",
        "Whole Grain Breakfast Cereal",
    ],
}

# Typical price band (min, max) per category, in USD.
CATEGORY_PRICE_RANGE = {
    "Electronics": (15.0, 350.0),
    "Clothing": (8.0, 90.0),
    "Home & Kitchen": (6.0, 150.0),
    "Books": (5.0, 35.0),
    "Sports & Outdoors": (10.0, 220.0),
    "Beauty & Personal Care": (4.0, 60.0),
    "Toys & Games": (5.0, 80.0),
    "Grocery": (2.0, 25.0),
}

# Rough profit-margin band per category, used for category_margins.
CATEGORY_MARGIN_RANGE = {
    "Electronics": (12.0, 22.0),
    "Clothing": (25.0, 45.0),
    "Home & Kitchen": (20.0, 38.0),
    "Books": (15.0, 30.0),
    "Sports & Outdoors": (18.0, 35.0),
    "Beauty & Personal Care": (30.0, 55.0),
    "Toys & Games": (20.0, 40.0),
    "Grocery": (8.0, 18.0),
}

CUSTOMER_TIERS = ["REGULAR", "GOLD", "PLATINUM"]
CUSTOMER_TIER_WEIGHTS = [0.70, 0.20, 0.10]

ORDER_STATUSES = ["PENDING", "PROCESSING", "SHIPPED", "CANCELLED"]
ORDER_STATUS_WEIGHTS = [0.15, 0.15, 0.60, 0.10]


# --------------------------------------------------------------------------
# SQLite (OLTP) data generation
# --------------------------------------------------------------------------
def generate_products(n=N_PRODUCTS):
    """Return a list of product dict rows, spread realistically across categories."""
    rows = []
    product_id = 1
    # Distribute n products roughly evenly across categories, sampling
    # without replacement from each category's curated name pool first,
    # then falling back to Faker-generated names if a category runs out.
    per_category = n // len(CATEGORIES)
    remainder = n % len(CATEGORIES)

    for i, category in enumerate(CATEGORIES):
        count = per_category + (1 if i < remainder else 0)
        name_pool = CATEGORY_PRODUCT_NAMES[category][:]
        random.shuffle(name_pool)
        chosen_names = name_pool[:count]
        # Pad with generated names if the category needs more than the pool has
        while len(chosen_names) < count:
            chosen_names.append(f"{fake.word().title()} {category.split(' ')[0]}")

        lo, hi = CATEGORY_PRICE_RANGE[category]
        for name in chosen_names:
            rows.append(
                {
                    "product_id": product_id,
                    "name": name,
                    "category": category,
                    "unit_price": round(random.uniform(lo, hi), 2),
                    "stock_quantity": random.randint(0, 500),
                }
            )
            product_id += 1

    return rows


def generate_customers(n=N_CUSTOMERS):
    rows = []
    for customer_id in range(1, n + 1):
        name = fake.name()
        rows.append(
            {
                "customer_id": customer_id,
                "name": name,
                "email": fake.unique.email(),
                "city": fake.city(),
                "tier": random.choices(CUSTOMER_TIERS, weights=CUSTOMER_TIER_WEIGHTS, k=1)[0],
            }
        )
    return rows


def generate_orders_and_items(products, customers, n_orders=N_ORDERS):
    """
    Builds orders + order_items together so totals and FKs stay consistent.
    Returns (order_rows, order_item_rows).
    """
    order_rows = []
    item_rows = []
    item_id = 1
    now = datetime.now()

    product_ids = [p["product_id"] for p in products]
    price_by_product = {p["product_id"]: p["unit_price"] for p in products}
    customer_ids = [c["customer_id"] for c in customers]

    for order_id in range(1, n_orders + 1):
        customer_id = random.choice(customer_ids)
        order_date = now - timedelta(
            days=random.randint(0, ORDER_WINDOW_DAYS - 1),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        status = random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS, k=1)[0]

        # 1-4 distinct products per order, averaging ~2 items/order
        n_items = random.choices([1, 2, 3, 4], weights=[0.20, 0.40, 0.30, 0.10], k=1)[0]
        chosen_products = random.sample(product_ids, k=min(n_items, len(product_ids)))

        order_total = 0.0
        for pid in chosen_products:
            quantity = random.randint(1, 5)
            unit_price = price_by_product[pid]
            line_total = round(quantity * unit_price, 2)
            order_total += line_total

            item_rows.append(
                {
                    "item_id": item_id,
                    "order_id": order_id,
                    "product_id": pid,
                    "quantity": quantity,
                    "line_total": line_total,
                }
            )
            item_id += 1

        order_rows.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                "order_status": status,
                "total_amount": round(order_total, 2),
            }
        )

    return order_rows, item_rows


def build_sqlite_db(path=SQLITE_PATH):
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.executescript(
        """
        CREATE TABLE products (
            product_id      INTEGER PRIMARY KEY,
            name            VARCHAR(100) NOT NULL,
            category        VARCHAR(50)  NOT NULL,
            unit_price      FLOAT        NOT NULL,
            stock_quantity  INTEGER      NOT NULL
        );

        CREATE TABLE customers (
            customer_id     INTEGER PRIMARY KEY,
            name            VARCHAR(100) NOT NULL,
            email           VARCHAR(150) NOT NULL UNIQUE,
            city            VARCHAR(100) NOT NULL,
            tier            VARCHAR(20)  NOT NULL CHECK (tier IN ('REGULAR','GOLD','PLATINUM'))
        );

        CREATE TABLE orders (
            order_id        INTEGER PRIMARY KEY,
            customer_id     INTEGER      NOT NULL,
            order_date      TIMESTAMP    NOT NULL,
            order_status    VARCHAR(20)  NOT NULL
                            CHECK (order_status IN ('PENDING','PROCESSING','SHIPPED','CANCELLED')),
            total_amount    FLOAT        NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        );

        CREATE TABLE order_items (
            item_id         INTEGER PRIMARY KEY,
            order_id        INTEGER      NOT NULL,
            product_id      INTEGER      NOT NULL,
            quantity        INTEGER      NOT NULL,
            line_total      FLOAT        NOT NULL,
            FOREIGN KEY (order_id)   REFERENCES orders (order_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        );
        """
    )

    products = generate_products()
    customers = generate_customers()
    orders, order_items = generate_orders_and_items(products, customers)

    cur.executemany(
        "INSERT INTO products (product_id, name, category, unit_price, stock_quantity) "
        "VALUES (:product_id, :name, :category, :unit_price, :stock_quantity)",
        products,
    )
    cur.executemany(
        "INSERT INTO customers (customer_id, name, email, city, tier) "
        "VALUES (:customer_id, :name, :email, :city, :tier)",
        customers,
    )
    cur.executemany(
        "INSERT INTO orders (order_id, customer_id, order_date, order_status, total_amount) "
        "VALUES (:order_id, :customer_id, :order_date, :order_status, :total_amount)",
        orders,
    )
    cur.executemany(
        "INSERT INTO order_items (item_id, order_id, product_id, quantity, line_total) "
        "VALUES (:item_id, :order_id, :product_id, :quantity, :line_total)",
        order_items,
    )

    conn.commit()

    counts = {
        "products": cur.execute("SELECT COUNT(*) FROM products").fetchone()[0],
        "customers": cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
        "orders": cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
        "order_items": cur.execute("SELECT COUNT(*) FROM order_items").fetchone()[0],
    }
    conn.close()
    return products, counts


# --------------------------------------------------------------------------
# DuckDB (OLAP) data generation
# --------------------------------------------------------------------------
def generate_historical_sales_daily(products, years=HISTORY_YEARS):
    """
    Aggregated daily sales by category, from `years` ago up to the end of
    last month. Not every category sells every day (mirrors real retail
    patterns) which keeps the row count close to ~2,500.
    """
    today = date.today()
    first_of_this_month = today.replace(day=1)
    end_date = first_of_this_month - timedelta(days=1)  # last day of previous month
    start_date = end_date.replace(year=end_date.year - years)

    # Baseline avg unit price per category, used to derive plausible revenue
    avg_price_by_category = {}
    for category in CATEGORIES:
        prices = [p["unit_price"] for p in products if p["category"] == category]
        avg_price_by_category[category] = sum(prices) / len(prices) if prices else 20.0

    rows = []
    current = start_date
    while current <= end_date:
        # Each day, a random subset of categories reports sales (2-5 of 8)
        n_categories_today = random.randint(2, 5)
        categories_today = random.sample(CATEGORIES, k=n_categories_today)

        # Light seasonality: weekends and Nov/Dec see a volume bump
        weekend_bump = 1.25 if current.weekday() >= 5 else 1.0
        holiday_bump = 1.4 if current.month in (11, 12) else 1.0
        seasonality = weekend_bump * holiday_bump

        for category in categories_today:
            base_units = random.randint(20, 250)
            units_sold = max(1, int(base_units * seasonality))
            avg_price = avg_price_by_category[category]
            # Some noise around the average selling price
            gross_revenue = round(units_sold * avg_price * random.uniform(0.9, 1.1), 2)
            return_count = int(units_sold * random.uniform(0.0, 0.06))  # up to ~6% returns

            rows.append(
                {
                    "date": current.isoformat(),
                    "category": category,
                    "units_sold": units_sold,
                    "gross_revenue": gross_revenue,
                    "return_count": return_count,
                }
            )

        current += timedelta(days=1)

    return rows


def generate_category_margins():
    rows = []
    for category in CATEGORIES:
        lo, hi = CATEGORY_MARGIN_RANGE[category]
        rows.append(
            {
                "category": category,
                "profit_margin_percentage": round(random.uniform(lo, hi), 2),
                "target_quarterly_growth": round(random.uniform(2.0, 15.0), 2),
            }
        )
    return rows


def build_duckdb_db(products, path=DUCKDB_PATH):
    if os.path.exists(path):
        os.remove(path)

    sales_rows = generate_historical_sales_daily(products)
    margin_rows = generate_category_margins()

    sales_df = pd.DataFrame(sales_rows)
    sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.date
    margins_df = pd.DataFrame(margin_rows)

    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE historical_sales_daily (
            date            DATE    NOT NULL,
            category        VARCHAR NOT NULL,
            units_sold      INTEGER NOT NULL,
            gross_revenue   DOUBLE  NOT NULL,
            return_count    INTEGER NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE category_margins (
            category                    VARCHAR NOT NULL,
            profit_margin_percentage    DOUBLE  NOT NULL,
            target_quarterly_growth     DOUBLE  NOT NULL
        );
        """
    )

    con.execute("INSERT INTO historical_sales_daily SELECT * FROM sales_df")
    con.execute("INSERT INTO category_margins SELECT * FROM margins_df")

    counts = {
        "historical_sales_daily": con.execute(
            "SELECT COUNT(*) FROM historical_sales_daily"
        ).fetchone()[0],
        "category_margins": con.execute("SELECT COUNT(*) FROM category_margins").fetchone()[0],
    }
    date_bounds = con.execute(
        "SELECT MIN(date), MAX(date) FROM historical_sales_daily"
    ).fetchone()
    con.close()
    return counts, date_bounds


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Generating SQLite OLTP database...")
    products, sqlite_counts = build_sqlite_db()
    print(f"  -> {SQLITE_PATH} created")
    for table, count in sqlite_counts.items():
        print(f"     {table:<15} {count:>6} rows")

    print("\nGenerating DuckDB OLAP database...")
    duckdb_counts, (min_date, max_date) = build_duckdb_db(products)
    print(f"  -> {DUCKDB_PATH} created")
    for table, count in duckdb_counts.items():
        print(f"     {table:<25} {count:>6} rows")
    print(f"     historical_sales_daily date range: {min_date} -> {max_date}")

    print("\nDone. Both databases are ready for the Text-to-SQL pipeline.")


if __name__ == "__main__":
    main()
