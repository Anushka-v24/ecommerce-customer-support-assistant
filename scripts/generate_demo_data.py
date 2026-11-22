"""Grow the fictional demo fixtures to a few hundred consistent records.

Reads the four files in data/mock (products, orders, refunds, customers), keeps the
records already there exactly as they are, and appends generated ones so the chat
has enough context to answer questions about stock, delivery and refunds. The
output is deterministic (fixed seed and "today"), so rerunning it changes nothing.

    python scripts/generate_demo_data.py
    python -m app.db.import_demo_data      # then load the new rows into PostgreSQL

Every record is fictional. Identifiers follow the formats the app validates:
PROD-### , ORD-#### , CUST-### , DEMO-REF-####.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mock"
TODAY = date(2026, 11, 23)
PRODUCT_TARGET, ORDER_TARGET, CUSTOMER_TARGET = 120, 120, 110

BRANDS = ["Voltix", "Nimbus", "Aurelia", "Kestrel", "Lumen", "Zenara", "Orbis", "Tanvi", "Pixelon", "Basalt"]
# Words the three original products use (laptop, galaxy, headphones) are avoided so searches for them stay unambiguous.
TYPES = [
    "Smartphone 5G", "Tablet 10-inch", "Ultrabook 14", "Wireless Earbuds", "Smartwatch", "Bluetooth Speaker",
    "4K Monitor 27-inch", "Mechanical Keyboard", "Wireless Mouse", "HD Webcam", "Wi-Fi 6 Router", "Power Bank 20000mAh",
    "Travel Backpack", "Running Sneakers", "Cotton T-Shirt", "Denim Jacket", "Slim-Fit Jeans", "Steel Water Bottle",
    "LED Desk Lamp", "Air Fryer 4L", "Countertop Blender", "Electric Kettle", "Robot Vacuum", "Yoga Mat",
    "Dumbbell Set 20kg", "Hardcover Notebook", "Gel Pen Set", "Eau de Parfum", "Face Moisturizer", "Memory Foam Pillow",
    "Cast Iron Skillet", "Ceramic Dinner Set", "Cordless Drill", "Portable SSD 1TB", "Gaming Console Controller",
    "Fitness Band", "Sunglasses", "Leather Wallet", "Umbrella", "Standing Desk Mat",
]
# "Pro" is left out: it is part of the original "Laptop Pro", and a search for that must stay unambiguous.
VARIANTS = ["", "", "", " Lite", " Plus", " Max", " Mini"]

FIRST_NAMES = [
    "Aditya", "Ananya", "Arjun", "Bhavna", "Chirag", "Deepa", "Dev", "Esha", "Farhan", "Gauri", "Harsh", "Isha",
    "Jay", "Kavya", "Kabir", "Lakshmi", "Manav", "Nandini", "Omkar", "Pooja", "Pranav", "Riya", "Rohan", "Sanya",
    "Siddharth", "Tara", "Uday", "Vani", "Varun", "Zoya", "Neha", "Karan", "Divya", "Rahul", "Simran", "Yash",
]
LAST_NAMES = [
    "Agarwal", "Bose", "Chopra", "Desai", "Fernandes", "Gupta", "Hegde", "Joshi", "Khanna", "Lal", "Menon",
    "Nair", "Patel", "Rao", "Saxena", "Trivedi", "Verma", "Yadav", "Banerjee", "Kapoor", "Malhotra", "Pillai",
]

REFUND_NOTES = {
    "not_initiated": "No refund operation has been submitted in this demonstration.",
    "processing": "Fictional status for response testing only.",
    "completed": "Fictional status; no payment provider was contacted.",
}


def load(name: str, collection: str) -> tuple[dict, list[dict]]:
    with (DATA_DIR / name).open("r", encoding="utf-8-sig") as source:
        document = json.load(source)
    return document, document[collection]


def save(name: str, document: dict) -> None:
    (DATA_DIR / name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compact(records: list[dict], indent: str = "    ") -> str:
    return ",\n".join(indent + json.dumps(record, ensure_ascii=False) for record in records)


def write_collection(name: str, document: dict, collection: str, records: list[dict]) -> None:
    """One record per line, the way the hand-written fixtures are laid out."""
    head = {key: value for key, value in document.items() if key != collection}
    lines = [f'  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},' for key, value in head.items()]
    text = "{\n" + "\n".join(lines) + f'\n  "{collection}": [\n{compact(records)}\n  ]\n}}\n'
    (DATA_DIR / name).write_text(text, encoding="utf-8")


def build_products(rng: random.Random, existing: list[dict]) -> list[dict]:
    products = list(existing)
    names = {item["name"] for item in products}
    combos = [(brand, kind) for kind in TYPES for brand in BRANDS]
    rng.shuffle(combos)
    for brand, kind in combos:
        if len(products) >= PRODUCT_TARGET:
            break
        name = f"{brand} {kind}{rng.choice(VARIANTS)}"
        if name in names:
            continue
        names.add(name)
        out_of_stock = rng.random() < 0.15
        products.append({
            "product_id": f"PROD-{len(products) + 1:03d}",
            "name": name,
            "stock": 0 if out_of_stock else rng.choice([2, 3, 5, 8, 12, 18, 25, 40, 64, 90, 140, 220]),
            "restock_date": (TODAY + timedelta(days=rng.randint(5, 45))).isoformat() if out_of_stock else None,
        })
    return products


def build_orders(rng: random.Random, existing: list[dict], product_ids: list[str]) -> list[dict]:
    orders = list(existing)
    while len(orders) < ORDER_TARGET:
        order_id = f"ORD-{1001 + len(orders):04d}"
        roll = rng.random()
        if roll < 0.45:
            status, eta = "Delivered", (TODAY - timedelta(days=rng.randint(2, 40))).isoformat()
        elif roll < 0.70:
            # A few shipped orders are past their ETA, which the app reports as delayed.
            offset = rng.randint(-3, 9) if rng.random() < 0.2 else rng.randint(1, 9)
            status, eta = "Shipped", (TODAY + timedelta(days=offset)).isoformat()
        elif roll < 0.88:
            status, eta = "Processing", None
        else:
            status, eta = "Delayed", (TODAY + timedelta(days=rng.randint(-5, 12))).isoformat()
        orders.append({"order_id": order_id, "status": status, "eta": eta, "product_id": rng.choice(product_ids)})
    return orders


def build_refunds(rng: random.Random, existing: list[dict], orders: list[dict]) -> list[dict]:
    refunds = list(existing)
    have = {item["order_id"] for item in refunds}
    for order in orders:
        # ORD-1004 stays without a refund record, as in the original fixture, and so do about one in ten others,
        # so "no refund has been raised for this order" is a case the chat meets with real data too.
        if order["order_id"] in have or order["order_id"] == "ORD-1004" or rng.random() < 0.10:
            continue
        status_of_order = order["status"]
        if status_of_order == "Delivered":
            status = rng.choices(["completed", "processing", "not_initiated"], [4, 3, 3])[0]
        elif status_of_order == "Delayed":
            status = rng.choices(["processing", "not_initiated"], [2, 3])[0]
        else:
            status = "not_initiated"
        number = order["order_id"].split("-")[1]
        refunds.append({
            "order_id": order["order_id"], "status": status,
            "refund_id": f"DEMO-REF-{number}", "note": REFUND_NOTES[status],
        })
    return refunds


def build_customers(rng: random.Random, existing: list[dict], orders: list[dict]) -> list[dict]:
    customers = list(existing)
    owned = {order_id for customer in customers for order_id in customer["order_ids"]}
    spare = [order["order_id"] for order in orders if order["order_id"] not in owned]
    names = {customer["name"] for customer in customers}
    while len(customers) < CUSTOMER_TARGET:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name in names:
            continue
        names.add(name)
        customers.append({"customer_id": f"CUST-{len(customers) + 1:03d}", "name": name, "order_ids": [spare.pop(0)] if spare else []})
    # Orders left over after everyone has one go to the earliest generated customers.
    for index, order_id in enumerate(spare):
        customers[len(existing) + index % (len(customers) - len(existing))]["order_ids"].append(order_id)
    return customers


def main() -> None:
    rng = random.Random(2026)
    products_doc, products = load("products.json", "products")
    orders_doc, orders = load("orders.json", "orders")
    refunds_doc, refunds = load("refunds.json", "refunds")
    customers_doc, customers = load("customers.json", "customers")

    originals = (len(products), len(orders), len(refunds), len(customers))
    products = build_products(rng, products[:3])
    orders = build_orders(rng, orders[:4], [item["product_id"] for item in products])
    refunds = build_refunds(rng, refunds[:3], orders)
    customers = build_customers(rng, customers[:2], orders)

    write_collection("products.json", products_doc, "products", products)
    write_collection("orders.json", orders_doc, "orders", orders)
    write_collection("refunds.json", refunds_doc, "refunds", refunds)
    write_collection("customers.json", customers_doc, "customers", customers)
    print(f"products {originals[0]}->{len(products)}, orders {originals[1]}->{len(orders)}, "
          f"refunds {originals[2]}->{len(refunds)}, customers {originals[3]}->{len(customers)}")


if __name__ == "__main__":
    main()
