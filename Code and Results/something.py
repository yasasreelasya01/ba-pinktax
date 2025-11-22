#!/usr/bin/env python3
"""
Pink Tax Data Processing Pipeline
Stages:
1. EDA & Audit
2. Cleaning & Normalization
3. Deduplication
4. Gender Tagging & Filtering
5. Product Matching (Pairing)
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("="*80)
print("LOADING DATASET")
print("="*80)
df = pd.read_csv('/mnt/project/raw_scraped_data_3261_inr.csv')
print(f"Loaded {len(df)} records")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# STAGE 1: EDA & AUDIT
# ============================================================================
print("\n" + "="*80)
print("STAGE 1: EDA & AUDIT")
print("="*80)

print("\n1.1 Basic Info:")
print(f"   Total records: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n1.2 Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Missing Count', ascending=False)
print(missing_report[missing_report['Missing Count'] > 0])

print("\n1.3 Data Types:")
print(df.dtypes)

print("\n1.4 Sample Records:")
print(df.head(3).to_string())

print("\n1.5 Gender Distribution:")
print(df['gender_target'].value_counts(dropna=False))

print("\n1.6 Category Distribution (top 10):")
print(df['category'].value_counts().head(10))

print("\n1.7 Brand Distribution (top 10):")
print(df['brand'].value_counts().head(10))

print("\n1.8 Retailer Distribution:")
print(df['retailer'].value_counts())

print("\n1.9 Price Patterns:")
print(df['price_raw'].value_counts().head(10))

print("\n1.10 Size Patterns:")
print(df['size_raw'].value_counts().head(10))

# ============================================================================
# STAGE 2: CLEANING & NORMALIZATION
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: CLEANING & NORMALIZATION")
print("="*80)

df_clean = df.copy()

# 2.1 Clean Brand Names
print("\n2.1 Normalizing brand names...")
def clean_brand(brand):
    if pd.isna(brand):
        return 'Unknown'
    brand = str(brand).strip()
    # Remove special characters and extra spaces
    brand = re.sub(r'[^\w\s&\'-]', '', brand)
    brand = re.sub(r'\s+', ' ', brand)
    # Remove common suffixes
    brand = re.sub(r'\s+(Pvt|Ltd|Limited|India|Inc|Corp|Co|Private)(\s+|$)', '', brand, flags=re.IGNORECASE)
    # Capitalize properly
    brand = brand.title()
    # Handle specific brand normalizations
    brand_map = {
        "L'Oreal": "L'Oréal",
        "Loreal": "L'Oréal",
        "L'Orã©Al": "L'Oréal",
        "Park Avenue": "Park Avenue",
        "Forest Essential": "Forest Essentials",
        "Himalaya Wellness": "Himalaya",
    }
    for old, new in brand_map.items():
        if old.lower() in brand.lower():
            brand = new
            break
    return brand

df_clean['brand'] = df_clean['brand'].apply(clean_brand)
print(f"   Unique brands: {df_clean['brand'].nunique()}")

# 2.2 Normalize Category and Subcategory
print("\n2.2 Normalizing categories...")
def clean_category(cat):
    if pd.isna(cat):
        return 'Unknown'
    cat = str(cat).strip().title()
    # Remove extra spaces
    cat = re.sub(r'\s+', ' ', cat)
    return cat

df_clean['category'] = df_clean['category'].apply(clean_category)
df_clean['subcategory'] = df_clean['subcategory'].apply(clean_category)

# Filter out nonsensical categories
valid_categories = ['Personal Care', 'Clothing', 'Footwear', 'Shaving', 'Hygiene', 'Health']
df_clean = df_clean[df_clean['category'].isin(valid_categories)]
print(f"   Records after filtering valid categories: {len(df_clean)}")

# 2.3 Clean and Normalize Prices
print("\n2.3 Normalizing prices...")
def extract_price(price_str):
    if pd.isna(price_str):
        return np.nan
    # Remove currency symbols and extract number
    price_str = str(price_str)
    # Extract all numbers (including decimals)
    matches = re.findall(r'\d+\.?\d*', price_str)
    if matches:
        return float(matches[0])
    return np.nan

df_clean['price'] = df_clean['price_raw'].apply(extract_price)
print(f"   Valid prices: {df_clean['price'].notna().sum()}/{len(df_clean)}")

# 2.4 Clean and Normalize Sizes
print("\n2.4 Normalizing sizes...")
def extract_size_and_unit(size_str):
    if pd.isna(size_str):
        return np.nan, 'Unknown'
    size_str = str(size_str).strip()
    
    # Extract number and unit
    match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]+)', size_str)
    if match:
        size_value = float(match.group(1))
        unit = match.group(2).lower()
        
        # Normalize units
        unit_map = {
            'ml': 'ml', 'milliliter': 'ml', 'millilitre': 'ml',
            'l': 'ml', 'liter': 'ml', 'litre': 'ml',  # Will convert L to ml
            'g': 'g', 'gram': 'g', 'gm': 'g',
            'kg': 'g', 'kilogram': 'g',  # Will convert kg to g
            'oz': 'oz', 'ounce': 'oz',
            'count': 'count', 'pcs': 'count', 'pieces': 'count',
            's': 'size_s', 'm': 'size_m', 'l': 'size_l', 'xl': 'size_xl',
            'xxl': 'size_xxl'
        }
        
        normalized_unit = unit_map.get(unit, unit)
        
        # Convert units to base units
        if unit == 'l':
            size_value = size_value * 1000  # L to ml
            normalized_unit = 'ml'
        elif unit == 'kg':
            size_value = size_value * 1000  # kg to g
            normalized_unit = 'g'
        
        return size_value, normalized_unit
    
    # If no unit found, check if it's a clothing size
    size_str_upper = size_str.upper()
    if size_str_upper in ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL']:
        return 1, f'size_{size_str_upper.lower()}'
    
    # Check if it's just a number (for counts)
    if size_str.isdigit():
        return float(size_str), 'count'
    
    return np.nan, 'Unknown'

df_clean[['size', 'size_unit']] = df_clean['size_raw'].apply(
    lambda x: pd.Series(extract_size_and_unit(x))
)
print(f"   Valid sizes: {df_clean['size'].notna().sum()}/{len(df_clean)}")

# 2.5 Normalize Gender
print("\n2.5 Normalizing gender...")
def clean_gender(gender):
    if pd.isna(gender):
        return 'Unknown'
    gender = str(gender).strip().title()
    if gender in ['Men', 'Male', 'M']:
        return 'Men'
    elif gender in ['Women', 'Female', 'F', 'Woman']:
        return 'Women'
    elif gender in ['Unisex', 'Both', 'All']:
        return 'Unisex'
    return 'Unknown'

df_clean['gender_target'] = df_clean['gender_target'].apply(clean_gender)
print(df_clean['gender_target'].value_counts())

# 2.6 Clean Product Names
print("\n2.6 Cleaning product names...")
def clean_product_name(name):
    if pd.isna(name):
        return 'Unknown Product'
    name = str(name).strip()
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    return name

df_clean['product_name'] = df_clean['product_name'].apply(clean_product_name)

# 2.7 Calculate normalized price per unit
print("\n2.7 Calculating normalized price per unit...")
def calculate_price_per_unit(row):
    if pd.isna(row['price']) or pd.isna(row['size']) or row['size'] == 0:
        return np.nan
    
    # For volume/weight units, calculate price per unit
    if row['size_unit'] in ['ml', 'g', 'oz']:
        return round(row['price'] / row['size'], 2)
    # For clothing sizes and counts, price is already per item
    elif row['size_unit'].startswith('size_') or row['size_unit'] == 'count':
        return row['price']
    
    return np.nan

df_clean['normalized_price_per_unit'] = df_clean.apply(calculate_price_per_unit, axis=1)
print(f"   Calculated price per unit for {df_clean['normalized_price_per_unit'].notna().sum()} records")

print("\n2.8 Data after cleaning:")
print(f"   Total records: {len(df_clean)}")
print(f"   Valid prices: {df_clean['price'].notna().sum()}")
print(f"   Valid sizes: {df_clean['size'].notna().sum()}")
print(f"   Valid price per unit: {df_clean['normalized_price_per_unit'].notna().sum()}")

# ============================================================================
# STAGE 3: DEDUPLICATION
# ============================================================================
print("\n" + "="*80)
print("STAGE 3: DEDUPLICATION")
print("="*80)

print(f"\n3.1 Records before deduplication: {len(df_clean)}")

# Remove exact duplicates
df_clean = df_clean.drop_duplicates(
    subset=['product_name', 'brand', 'category', 'gender_target', 'price', 'size'],
    keep='first'
)
print(f"3.2 Records after removing exact duplicates: {len(df_clean)}")

# Remove near-duplicates (same product, brand, gender, but slightly different price/size)
# Keep the most recent one based on scraped_date
df_clean = df_clean.sort_values('scraped_date', ascending=False)
df_clean = df_clean.drop_duplicates(
    subset=['product_name', 'brand', 'category', 'subcategory', 'gender_target'],
    keep='first'
)
print(f"3.3 Records after removing near-duplicates: {len(df_clean)}")

# ============================================================================
# STAGE 4: GENDER TAGGING & FILTERING
# ============================================================================
print("\n" + "="*80)
print("STAGE 4: GENDER TAGGING & FILTERING")
print("="*80)

print("\n4.1 Gender distribution before filtering:")
print(df_clean['gender_target'].value_counts())

# Keep only records with clear gender tags (Men or Women)
df_gendered = df_clean[df_clean['gender_target'].isin(['Men', 'Women'])].copy()
print(f"\n4.2 Records after filtering for gendered products: {len(df_gendered)}")

print("\n4.3 Gender distribution after filtering:")
print(df_gendered['gender_target'].value_counts())

# ============================================================================
# STAGE 5: PRODUCT MATCHING (PAIRING)
# ============================================================================
print("\n" + "="*80)
print("STAGE 5: PRODUCT MATCHING (PAIRING)")
print("="*80)

print("\n5.1 Preparing for matching...")

# Remove gender-specific terms from product names for matching
def create_matching_key(row):
    name = row['product_name'].lower()
    # Remove gender indicators
    name = re.sub(r'\s*-?\s*(men|women|male|female|man|woman)\s*-?\s*', ' ', name, flags=re.IGNORECASE)
    # Remove size from name
    name = re.sub(r'\s*-?\s*\d+\s*(ml|g|l|oz|count|pcs|s|m|l|xl|xxl)\s*-?\s*', ' ', name, flags=re.IGNORECASE)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Create key: brand_category_subcategory_cleanedname_size_unit
    key = f"{row['brand']}_{row['category']}_{row['subcategory']}_{name}_{row['size']}_{row['size_unit']}"
    return key.lower()

df_gendered['matching_key'] = df_gendered.apply(create_matching_key, axis=1)

# Find potential pairs
print("\n5.2 Finding potential pairs...")
men_products = df_gendered[df_gendered['gender_target'] == 'Men'].copy()
women_products = df_gendered[df_gendered['gender_target'] == 'Women'].copy()

print(f"   Men's products: {len(men_products)}")
print(f"   Women's products: {len(women_products)}")

# Match products by matching key
pairs = []
pair_id = 1

for _, men_row in men_products.iterrows():
    matching_women = women_products[women_products['matching_key'] == men_row['matching_key']]
    
    if len(matching_women) > 0:
        # Take the first match (or could do more sophisticated matching)
        women_row = matching_women.iloc[0]
        
        pairs.append({
            'pair_id': f'PAIR{pair_id:05d}',
            'men_scrape_id': men_row['scrape_id'],
            'women_scrape_id': women_row['scrape_id'],
            'brand': men_row['brand'],
            'category': men_row['category'],
            'subcategory': men_row['subcategory'],
            'size': men_row['size'],
            'size_unit': men_row['size_unit'],
            'men_price': men_row['price'],
            'women_price': women_row['price'],
            'price_diff': women_row['price'] - men_row['price'],
            'price_diff_pct': ((women_row['price'] - men_row['price']) / men_row['price'] * 100) if men_row['price'] > 0 else 0
        })
        pair_id += 1

print(f"\n5.3 Total pairs found: {len(pairs)}")

# Create pairs dataframe
df_pairs = pd.DataFrame(pairs)

if len(df_pairs) > 0:
    print("\n5.4 Pair statistics:")
    print(f"   Average price difference: ₹{df_pairs['price_diff'].mean():.2f}")
    print(f"   Average price difference %: {df_pairs['price_diff_pct'].mean():.2f}%")
    print(f"   Products where women pay more: {(df_pairs['price_diff'] > 0).sum()}")
    print(f"   Products where men pay more: {(df_pairs['price_diff'] < 0).sum()}")
    print(f"   Products with same price: {(df_pairs['price_diff'] == 0).sum()}")

# ============================================================================
# CREATE FINAL DATASET
# ============================================================================
print("\n" + "="*80)
print("CREATING FINAL DATASET")
print("="*80)

# Prepare final dataset with all pairs
final_records = []
product_counter = 1

for _, pair in df_pairs.iterrows():
    # Get full records for men and women
    men_rec = men_products[men_products['scrape_id'] == pair['men_scrape_id']].iloc[0]
    women_rec = women_products[women_products['scrape_id'] == pair['women_scrape_id']].iloc[0]
    
    # Men's record
    final_records.append({
        'product_id': f'P{product_counter:05d}_M',
        'pair_id': pair['pair_id'],
        'product_name': men_rec['product_name'],
        'category': men_rec['category'],
        'subcategory': men_rec['subcategory'],
        'brand': men_rec['brand'],
        'gender_target': 'Men',
        'price': men_rec['price'],
        'size': men_rec['size'],
        'normalized_price_per_unit': men_rec['normalized_price_per_unit'],
        'retailer': men_rec['retailer'],
        'description': men_rec['description'] if pd.notna(men_rec['description']) else '',
        'ingredients': men_rec['ingredients'] if pd.notna(men_rec['ingredients']) else ''
    })
    
    # Women's record
    final_records.append({
        'product_id': f'P{product_counter:05d}_F',
        'pair_id': pair['pair_id'],
        'product_name': women_rec['product_name'],
        'category': women_rec['category'],
        'subcategory': women_rec['subcategory'],
        'brand': women_rec['brand'],
        'gender_target': 'Women',
        'price': women_rec['price'],
        'size': women_rec['size'],
        'normalized_price_per_unit': women_rec['normalized_price_per_unit'],
        'retailer': women_rec['retailer'],
        'description': women_rec['description'] if pd.notna(women_rec['description']) else '',
        'ingredients': women_rec['ingredients'] if pd.notna(women_rec['ingredients']) else ''
    })
    
    product_counter += 1

df_final = pd.DataFrame(final_records)

print(f"\n✓ Final dataset created with {len(df_final)} records ({len(df_final)//2} pairs)")
print(f"\nColumn structure:")
print(df_final.columns.tolist())

print("\nSample records:")
print(df_final.head(4).to_string())

# Save the final dataset
output_path = '/home/claude/pink_tax_cleaned_paired_dataset.csv'
df_final.to_csv(output_path, index=False)
print(f"\n✓ Final dataset saved to: {output_path}")

# Save the pairs summary
pairs_summary_path = '/home/claude/product_pairs_summary.csv'
df_pairs.to_csv(pairs_summary_path, index=False)
print(f"✓ Pairs summary saved to: {pairs_summary_path}")

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)

avg_price_diff = df_pairs['price_diff'].mean() if len(df_pairs) > 0 else 0
avg_pct_diff = df_pairs['price_diff_pct'].mean() if len(df_pairs) > 0 else 0
women_pay_more = (df_pairs['price_diff'] > 0).sum() if len(df_pairs) > 0 else 0
men_pay_more = (df_pairs['price_diff'] < 0).sum() if len(df_pairs) > 0 else 0
same_price = (df_pairs['price_diff'] == 0).sum() if len(df_pairs) > 0 else 0

summary = f"""
DATA PROCESSING PIPELINE - SUMMARY REPORT
{'='*80}

INPUT DATA:
  • Raw records loaded: {len(df)}
  • Columns: {len(df.columns)}

STAGE 1 - EDA & AUDIT:
  • Missing values identified: {missing[missing > 0].count()} columns with missing data
  • Data quality issues: Price format variations, size format inconsistencies
  
STAGE 2 - CLEANING & NORMALIZATION:
  • Records after category filtering: {len(df_clean)}
  • Brands normalized: {df_clean['brand'].nunique()} unique brands
  • Prices normalized: {df_clean['price'].notna().sum()} valid prices
  • Sizes normalized: {df_clean['size'].notna().sum()} valid sizes
  • Price per unit calculated: {df_clean['normalized_price_per_unit'].notna().sum()} records

STAGE 3 - DEDUPLICATION:
  • Records before: {len(df_clean)}
  • Records after: {len(df_clean)}
  • Duplicates removed: Exact and near-duplicates

STAGE 4 - GENDER FILTERING:
  • Records with clear gender tags: {len(df_gendered)}
  • Men's products: {len(men_products)}
  • Women's products: {len(women_products)}

STAGE 5 - PRODUCT PAIRING:
  • Total pairs found: {len(df_pairs)}
  • Final dataset size: {len(df_final)} records ({len(df_final)//2} pairs × 2)
  
PINK TAX INSIGHTS (from pairs):
  • Products where women pay MORE: {women_pay_more}
  • Products where men pay MORE: {men_pay_more}
  • Products with SAME price: {same_price}
  • Average price difference: ₹{avg_price_diff:.2f}
  • Average % difference: {avg_pct_diff:.2f}%

OUTPUT FILES:
  ✓ pink_tax_cleaned_paired_dataset.csv - Final dataset with all required columns
  ✓ product_pairs_summary.csv - Pair-level analysis

{'='*80}
"""

print(summary)

# Save summary report
with open('/home/claude/pipeline_summary_report.txt', 'w') as f:
    f.write(summary)

print("\n✓ Summary report saved to: /home/claude/pipeline_summary_report.txt")
print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)