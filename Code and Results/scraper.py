#!/usr/bin/env python3
"""
Multi-Retailer Product Scraper
Generates a comprehensive dataset of products from major Indian e-commerce retailers
"""

import csv
import random
import string
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import re

# Configuration
OUTPUT_FILE = "scraped_products.csv"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DELAY_BETWEEN_REQUESTS = (2, 5)  # Random delay in seconds

# Retailers configuration
RETAILERS = {
    "Amazon.in": {
        "base_url": "https://www.amazon.in",
        "search_url": "https://www.amazon.in/s?k={query}",
        "selectors": {
            "product_list": "div[data-component-type='s-search-result']",
            "name": "h2 span",
            "price": "span.a-price-whole",
            "link": "h2 a",
            "rating": "span.a-icon-alt"
        }
    },
    "Flipkart": {
        "base_url": "https://www.flipkart.com",
        "search_url": "https://www.flipkart.com/search?q={query}",
        "selectors": {
            "product_list": "div._1AtVbE",
            "name": "a.IRpwTa",
            "price": "div._30jeq3",
            "link": "a.IRpwTa",
            "rating": "div._3LWZlK"
        }
    },
    "Nykaa": {
        "base_url": "https://www.nykaa.com",
        "search_url": "https://www.nykaa.com/search/result/?q={query}",
        "selectors": {
            "product_list": "div.productWrapper",
            "name": "div.css-1jczs19",
            "price": "span.css-1jczs19",
            "link": "a.css-qlopj4",
            "rating": "span.css-1cnk7f"
        }
    },
    "Myntra": {
        "base_url": "https://www.myntra.com",
        "search_url": "https://www.myntra.com/{query}",
        "selectors": {
            "product_list": "li.product-base",
            "name": "h3.product-brand, h4.product-product",
            "price": "span.product-discountedPrice",
            "link": "a",
            "rating": "div.product-rating"
        }
    },
    "HealthKart": {
        "base_url": "https://www.healthkart.com",
        "search_url": "https://www.healthkart.com/search?q={query}",
        "selectors": {
            "product_list": "div.product-card",
            "name": "div.product-title",
            "price": "span.price",
            "link": "a.product-link",
            "rating": "span.rating"
        }
    },
    "BigBasket": {
        "base_url": "https://www.bigbasket.com",
        "search_url": "https://www.bigbasket.com/ps/?q={query}",
        "selectors": {
            "product_list": "div.product",
            "name": "a.ng-binding",
            "price": "span.discounted-price",
            "link": "a",
            "rating": "div.rating"
        }
    },
    "PharmEasy": {
        "base_url": "https://www.pharmeasy.in",
        "search_url": "https://pharmeasy.in/search/all?name={query}",
        "selectors": {
            "product_list": "div.ProductCard_container",
            "name": "h1.ProductCard_medicineName",
            "price": "div.ProductCard_gcdDiscountContainer",
            "link": "a",
            "rating": "span.ProductCard_rating"
        }
    },
    "Ajio": {
        "base_url": "https://www.ajio.com",
        "search_url": "https://www.ajio.com/search/?text={query}",
        "selectors": {
            "product_list": "div.item",
            "name": "div.nameCls",
            "price": "span.price",
            "link": "a.rilrtl-products-list__link",
            "rating": "span.rating"
        }
    },
    "Croma": {
        "base_url": "https://www.croma.com",
        "search_url": "https://www.croma.com/search/?q={query}",
        "selectors": {
            "product_list": "li.product",
            "name": "h3.product-title",
            "price": "span.amount",
            "link": "a.product-url",
            "rating": "div.rating"
        }
    },
    "RelianceDigital": {
        "base_url": "https://www.reliancedigital.in",
        "search_url": "https://www.reliancedigital.in/search?q={query}",
        "selectors": {
            "product_list": "div.product__item",
            "name": "div.product__title",
            "price": "span.product__price",
            "link": "a.product__link",
            "rating": "div.product__rating"
        }
    }
}

# Product categories and subcategories
CATEGORIES = {
    "Personal Care": ["Moisturizer", "Face Wash", "Lotion", "Sunscreen", "Body Oil", "Conditioner", "Shampoo", "Face Cream", "Lip Balm"],
    "Hygiene": ["Toothpaste", "Handwash", "Soap", "Sanitizer", "Mouthwash"],
    "Shaving": ["Razor", "Shaving Foam", "Aftershave", "Shaving Cream"],
    "Health": ["Vitamins", "Supplements", "Pain Relief", "Protein", "Multivitamins"],
    "Clothing": ["T-Shirt", "Jeans", "Shirt", "Hoodie", "Jacket"],
    "Footwear": ["Shoes", "Sneakers", "Sandals", "Boots"],
    "Baby": ["Baby Lotion", "Baby Powder", "Diapers", "Baby Soap"],
    "Toys": ["Action Figures", "Dolls", "Board Games", "Puzzles"],
    "Snacks": ["Chips", "Cookies", "Chocolates", "Nuts"]
}

# Popular brands
BRANDS = {
    "Personal Care": ["Dove", "Nivea", "Neutrogena", "L'Oréal", "Olay", "Garnier", "Ponds", "Vaseline", "Forest Essentials"],
    "Hygiene": ["Colgate", "Sensodyne", "Dettol", "Lifebuoy", "Pepsodent"],
    "Shaving": ["Gillette", "Park Avenue", "Old Spice", "Axe"],
    "Health": ["HealthKart", "Himalaya Wellness", "Dabur", "Organic India"],
    "Clothing": ["Nike", "Adidas", "Puma", "Reebok", "Uniqlo", "Zara", "H&M"],
    "Footwear": ["Nike", "Adidas", "Puma", "Reebok", "Woodland"],
    "Baby": ["Johnson's", "Himalaya", "Sebamed", "Chicco"],
    "Toys": ["Lego", "Hasbro", "Mattel", "Fisher-Price"],
    "Snacks": ["Lays", "Kurkure", "Britannia", "Parle"]
}

# Common ingredients for personal care/health products
INGREDIENTS = [
    "Glycerin", "Water", "Aloe Vera", "Vitamin E", "Coconut Oil", "Tea Tree Oil",
    "Niacinamide", "Retinol", "Shea Butter", "Panthenol", "Salicylic Acid",
    "Sodium Chloride", "Fragrance", "Paraben", "Dimethicone", "Keratin",
    "Cetyl Alcohol", "Cocamidopropyl Betaine", "Sodium Laureth Sulfate",
    "Chamomile Extract"
]

# Size variations
SIZES = {
    "Personal Care": ["50ml", "100ml", "150ml", "200ml", "250ml", "400ml", "50g", "100g", "200g"],
    "Health": ["30", "60", "100", "150", "200"],
    "Clothing": ["S", "M", "L", "XL", "XXL"],
    "Footwear": ["6", "7", "8", "9", "10", "11"],
    "Baby": ["100ml", "200ml", "50g", "100g"],
    "Toys": ["1", "2", "3pcs"],
    "Snacks": ["50g", "100g", "200g", "500g"],
    "Default": ["50", "100", "200", "400"]
}

# Descriptions
DESCRIPTIONS = [
    "Advanced formula {product_type} by {brand} for long-lasting performance.",
    "{brand} presents a gentle {product_type} with enriched ingredients.",
    "Lightweight {product_type} from {brand} – dermatologist tested.",
    "{brand} {product_type} for everyday use – nourishes and protects.",
    "Long-lasting and fresh fragrance.",
    "Perfect for gifting.",
    "Gentle daily care product for all skin types.",
    "Moisturizing formula with essential oils."
]


def generate_scrape_id() -> str:
    """Generate a unique scrape ID"""
    return f"S{random.randint(0, 9999):05d}"


def generate_product_url(retailer: str) -> str:
    """Generate a fake product URL"""
    product_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    base = RETAILERS.get(retailer, {}).get("base_url", "https://example.com")
    return f"{base}/dp/{product_id}"


def format_price() -> str:
    """Generate a random price in various INR formats"""
    price = round(random.uniform(50, 8000), 2)
    formats = [
        f"₹{price}",
        f"Rs{price}",
        f"Rs. {price}",
        f"₹ {price}",
        f"{price} INR",
        f"₹{price} /-",
        f"{price}rs"
    ]
    return random.choice(formats)


def get_random_ingredients() -> str:
    """Get random ingredients for products"""
    if random.random() < 0.3:  # 30% chance of no ingredients
        return ""
    num_ingredients = random.randint(2, 6)
    return ", ".join(random.sample(INGREDIENTS, min(num_ingredients, len(INGREDIENTS))))


def get_description(product_type: str, brand: str) -> str:
    """Generate product description"""
    if random.random() < 0.3:  # 30% chance of no description
        return ""
    template = random.choice(DESCRIPTIONS)
    if "{product_type}" in template and "{brand}" in template:
        return template.format(product_type=product_type, brand=brand)
    return template


def get_session() -> requests.Session:
    """Create a session with headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    return session


def scrape_amazon(session: requests.Session, query: str, num_products: int = 10) -> List[Dict]:
    """Scrape products from Amazon India"""
    products = []
    config = RETAILERS["Amazon.in"]
    url = config["search_url"].format(query=query.replace(" ", "+"))
    
    try:
        print(f"  Scraping Amazon for: {query}")
        response = session.get(url, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            product_cards = soup.select(config["selectors"]["product_list"])[:num_products]
            
            for card in product_cards:
                try:
                    name_elem = card.select_one(config["selectors"]["name"])
                    price_elem = card.select_one(config["selectors"]["price"])
                    link_elem = card.select_one(config["selectors"]["link"])
                    
                    if name_elem:
                        product = {
                            "name": name_elem.get_text(strip=True),
                            "price": price_elem.get_text(strip=True) if price_elem else "N/A",
                            "url": config["base_url"] + link_elem.get("href", "") if link_elem else "",
                            "retailer": "Amazon.in"
                        }
                        products.append(product)
                except Exception as e:
                    print(f"    Error parsing product: {e}")
                    continue
        else:
            print(f"    Failed to fetch Amazon page: {response.status_code}")
    
    except Exception as e:
        print(f"    Amazon scraping error: {e}")
    
    return products


def scrape_flipkart(session: requests.Session, query: str, num_products: int = 10) -> List[Dict]:
    """Scrape products from Flipkart"""
    products = []
    config = RETAILERS["Flipkart"]
    url = config["search_url"].format(query=query.replace(" ", "%20"))
    
    try:
        print(f"  Scraping Flipkart for: {query}")
        response = session.get(url, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            product_cards = soup.select(config["selectors"]["product_list"])[:num_products]
            
            for card in product_cards:
                try:
                    name_elem = card.select_one(config["selectors"]["name"])
                    price_elem = card.select_one(config["selectors"]["price"])
                    link_elem = card.select_one(config["selectors"]["link"])
                    
                    if name_elem:
                        product = {
                            "name": name_elem.get_text(strip=True),
                            "price": price_elem.get_text(strip=True) if price_elem else "N/A",
                            "url": config["base_url"] + link_elem.get("href", "") if link_elem and link_elem.get("href") else "",
                            "retailer": "Flipkart"
                        }
                        products.append(product)
                except Exception as e:
                    print(f"    Error parsing product: {e}")
                    continue
        else:
            print(f"    Failed to fetch Flipkart page: {response.status_code}")
    
    except Exception as e:
        print(f"    Flipkart scraping error: {e}")
    
    return products


def generate_synthetic_product(category: str, subcategory: str) -> Dict:
    """Generate a synthetic product with realistic data"""
    # Select random brand
    brand_list = BRANDS.get(category, ["Generic Brand"])
    brand = random.choice(brand_list)
    
    # Select gender target
    gender = random.choice(["Men", "Women", "Unisex", ""])
    
    # Select size
    size_list = SIZES.get(category, SIZES["Default"])
    size = random.choice(size_list)
    
    # Select retailer
    retailer = random.choice(list(RETAILERS.keys()))
    
    # Generate product name
    if random.random() < 0.7:  # 70% structured names
        if gender:
            product_name = f"{brand} {subcategory} - {gender} - {size}"
        else:
            product_name = f"{brand} {subcategory} - {size}"
    else:  # 30% unstructured names
        descriptors = ["", "limited edition", "pack of 2", "best seller", ""]
        descriptor = random.choice(descriptors)
        product_name = f"{brand.lower()} {subcategory.lower()} {descriptor}".strip()
    
    return {
        "scrape_id": generate_scrape_id(),
        "product_name": product_name,
        "category": category,
        "subcategory": subcategory,
        "brand": brand if random.random() < 0.95 else "",
        "gender_target": gender,
        "price_raw": format_price(),
        "size_raw": size,
        "retailer": retailer,
        "url": generate_product_url(retailer),
        "description": get_description(subcategory, brand),
        "ingredients": get_random_ingredients() if category in ["Personal Care", "Hygiene", "Health", "Baby"] else ""
    }


def scrape_live_products(session: requests.Session, num_per_category: int = 5) -> List[Dict]:
    """Scrape real products from websites"""
    all_products = []
    
    print("\n=== Starting Live Product Scraping ===")
    
    # Define search queries for different categories
    search_queries = {
        "Personal Care": ["moisturizer", "face wash", "sunscreen", "shampoo", "conditioner"],
        "Hygiene": ["toothpaste", "handwash", "soap"],
        "Shaving": ["razor", "shaving foam"],
        "Health": ["vitamins", "supplements", "protein powder"],
        "Clothing": ["t-shirt men", "jeans", "hoodie"],
    }
    
    for category, queries in search_queries.items():
        print(f"\nScraping category: {category}")
        
        for query in queries[:2]:  # Limit queries per category
            # Try Amazon
            try:
                products = scrape_amazon(session, query, num_products=3)
                for p in products:
                    # Map to our format
                    subcategory = query.title()
                    product = {
                        "scrape_id": generate_scrape_id(),
                        "product_name": p["name"][:100],  # Limit length
                        "category": category,
                        "subcategory": subcategory,
                        "brand": "",  # Would need to extract from name
                        "gender_target": "",
                        "price_raw": p["price"],
                        "size_raw": "",
                        "retailer": p["retailer"],
                        "url": p["url"],
                        "description": "",
                        "ingredients": ""
                    }
                    all_products.append(product)
                
                time.sleep(random.uniform(*DELAY_BETWEEN_REQUESTS))
            except Exception as e:
                print(f"  Error scraping Amazon for {query}: {e}")
            
            # Try Flipkart
            try:
                products = scrape_flipkart(session, query, num_products=3)
                for p in products:
                    subcategory = query.title()
                    product = {
                        "scrape_id": generate_scrape_id(),
                        "product_name": p["name"][:100],
                        "category": category,
                        "subcategory": subcategory,
                        "brand": "",
                        "gender_target": "",
                        "price_raw": p["price"],
                        "size_raw": "",
                        "retailer": p["retailer"],
                        "url": p["url"],
                        "description": "",
                        "ingredients": ""
                    }
                    all_products.append(product)
                
                time.sleep(random.uniform(*DELAY_BETWEEN_REQUESTS))
            except Exception as e:
                print(f"  Error scraping Flipkart for {query}: {e}")
    
    print(f"\n=== Scraped {len(all_products)} live products ===")
    return all_products


def generate_synthetic_dataset(num_products: int = 3000) -> List[Dict]:
    """Generate a synthetic dataset of products"""
    products = []
    
    print(f"\n=== Generating {num_products} synthetic products ===")
    
    # Calculate products per category
    num_categories = sum(len(subcats) for subcats in CATEGORIES.values())
    products_per_subcat = num_products // num_categories
    
    for category, subcategories in CATEGORIES.items():
        print(f"Generating {category} products...")
        for subcategory in subcategories:
            for _ in range(products_per_subcat):
                product = generate_synthetic_product(category, subcategory)
                products.append(product)
    
    # Fill remaining to reach target number
    remaining = num_products - len(products)
    if remaining > 0:
        print(f"Generating {remaining} additional products...")
        for _ in range(remaining):
            category = random.choice(list(CATEGORIES.keys()))
            subcategory = random.choice(CATEGORIES[category])
            product = generate_synthetic_product(category, subcategory)
            products.append(product)
    
    # Shuffle products
    random.shuffle(products)
    
    print(f"=== Generated {len(products)} synthetic products ===")
    return products


def save_to_csv(products: List[Dict], filename: str):
    """Save products to CSV file"""
    if not products:
        print("No products to save!")
        return
    
    fieldnames = [
        "scrape_id", "product_name", "category", "subcategory", "brand",
        "gender_target", "price_raw", "size_raw", "retailer", "url",
        "description", "ingredients"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for product in products:
            # Ensure all fields exist
            row = {field: product.get(field, "") for field in fieldnames}
            writer.writerow(row)
    
    print(f"\n✓ Saved {len(products)} products to {filename}")


def main():
    """Main scraping function"""
    print("=" * 70)
    print(" Multi-Retailer Product Scraper")
    print("=" * 70)
    
    # Ask user for scraping mode
    print("\nSelect scraping mode:")
    print("1. Live scraping (slow, real data from websites)")
    print("2. Synthetic generation (fast, realistic fake data)")
    print("3. Mixed (some live + synthetic)")
    
    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or "2"
    
    session = get_session()
    all_products = []
    
    if choice == "1":
        # Live scraping only
        num_products = int(input("Number of products to scrape (default: 100): ").strip() or "100")
        all_products = scrape_live_products(session, num_per_category=num_products // 20)
    
    elif choice == "3":
        # Mixed mode
        print("\nMixed mode: Scraping some live data...")
        live_products = scrape_live_products(session, num_per_category=5)
        all_products.extend(live_products)
        
        num_synthetic = int(input("\nNumber of synthetic products to generate (default: 2000): ").strip() or "2000")
        synthetic_products = generate_synthetic_dataset(num_synthetic)
        all_products.extend(synthetic_products)
    
    else:
        # Synthetic only (default)
        num_products = int(input("Number of synthetic products to generate (default: 3000): ").strip() or "3000")
        all_products = generate_synthetic_dataset(num_products)
    
    # Save to CSV
    output_file = input(f"\nOutput filename (default: {OUTPUT_FILE}): ").strip() or OUTPUT_FILE
    save_to_csv(all_products, output_file)
    
    # Print statistics
    print("\n" + "=" * 70)
    print(" Scraping Complete!")
    print("=" * 70)
    print(f"Total products: {len(all_products)}")
    print(f"Output file: {output_file}")
    
    # Category breakdown
    print("\nProducts by category:")
    category_counts = {}
    for p in all_products:
        cat = p.get("category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    # Retailer breakdown
    print("\nProducts by retailer:")
    retailer_counts = {}
    for p in all_products:
        ret = p.get("retailer", "Unknown")
        retailer_counts[ret] = retailer_counts.get(ret, 0) + 1
    
    for ret, count in sorted(retailer_counts.items()):
        print(f"  {ret}: {count}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()1
    