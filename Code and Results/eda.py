#!/usr/bin/env python3
"""
Comprehensive EDA for Pink Tax Dataset
Performs extensive exploratory data analysis on the pink tax dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# File path
DATA_FILE = "pink_tax_dataset_2000_inr_fixed.csv"

def load_data():
    """Load the dataset"""
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("\n" + "=" * 80)
    print("1. BASIC DATASET INFORMATION")
    print("=" * 80)
    
    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    
    print("\n--- Column Names and Types ---")
    print(df.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Last 5 Rows ---")
    print(df.tail())
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n--- Duplicate Rows ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

def statistical_summary(df):
    """Statistical summary of numerical columns"""
    print("\n" + "=" * 80)
    print("2. STATISTICAL SUMMARY")
    print("=" * 80)
    
    print("\n--- Descriptive Statistics (All Columns) ---")
    print(df.describe(include='all'))
    
    print("\n--- Numerical Columns Statistics ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    print("\n--- Price Statistics by Gender ---")
    if 'gender_target' in df.columns and 'price' in df.columns:
        print(df.groupby('gender_target')['price'].describe())
    
    print("\n--- Normalized Price Statistics by Gender ---")
    if 'gender_target' in df.columns and 'normalized_price_per_unit' in df.columns:
        print(df.groupby('gender_target')['normalized_price_per_unit'].describe())

def pink_tax_analysis(df):
    """Analyze the pink tax - price difference between men and women products"""
    print("\n" + "=" * 80)
    print("3. PINK TAX ANALYSIS")
    print("=" * 80)
    
    # Filter for Men and Women products only
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    
    # Calculate price difference for each pair
    if 'pair_id' in df.columns:
        print("\n--- Price Difference by Pair ---")
        
        # Pivot to get men and women prices side by side
        pivot_df = df_gender.pivot_table(
            index='pair_id',
            columns='gender_target',
            values='price',
            aggfunc='first'
        ).reset_index()
        
        if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
            pivot_df['price_difference'] = pivot_df['Women'] - pivot_df['Men']
            pivot_df['price_difference_pct'] = ((pivot_df['Women'] - pivot_df['Men']) / pivot_df['Men']) * 100
            pivot_df['women_pay_more'] = pivot_df['price_difference'] > 0
            
            print(f"\nTotal product pairs: {len(pivot_df)}")
            print(f"Pairs where women pay more: {pivot_df['women_pay_more'].sum()} ({(pivot_df['women_pay_more'].sum()/len(pivot_df)*100):.2f}%)")
            print(f"Pairs where women pay less: {(~pivot_df['women_pay_more']).sum()} ({((~pivot_df['women_pay_more']).sum()/len(pivot_df)*100):.2f}%)")
            
            print(f"\nAverage price difference: ₹{pivot_df['price_difference'].mean():.2f}")
            print(f"Median price difference: ₹{pivot_df['price_difference'].median():.2f}")
            print(f"Average percentage difference: {pivot_df['price_difference_pct'].mean():.2f}%")
            print(f"Median percentage difference: {pivot_df['price_difference_pct'].median():.2f}%")
            
            print("\n--- Top 10 Products with Highest Pink Tax ---")
            top_pink_tax = pivot_df.nlargest(10, 'price_difference')[['pair_id', 'Men', 'Women', 'price_difference', 'price_difference_pct']]
            print(top_pink_tax)
            
            print("\n--- Products with Negative Pink Tax (Women pay less) ---")
            negative_pink_tax = pivot_df[pivot_df['price_difference'] < 0].nsmallest(10, 'price_difference')[['pair_id', 'Men', 'Women', 'price_difference', 'price_difference_pct']]
            print(negative_pink_tax)
    
    print("\n--- Overall Price Comparison ---")
    men_prices = df_gender[df_gender['gender_target'] == 'Men']['price']
    women_prices = df_gender[df_gender['gender_target'] == 'Women']['price']
    
    print(f"Men's products - Mean: ₹{men_prices.mean():.2f}, Median: ₹{men_prices.median():.2f}")
    print(f"Women's products - Mean: ₹{women_prices.mean():.2f}, Median: ₹{women_prices.median():.2f}")
    print(f"Average difference: ₹{women_prices.mean() - men_prices.mean():.2f}")
    
    # T-test
    t_stat, p_value = stats.ttest_ind(women_prices, men_prices)
    print(f"\nT-test results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")

def category_analysis(df):
    """Analyze products by category and subcategory"""
    print("\n" + "=" * 80)
    print("4. CATEGORY ANALYSIS")
    print("=" * 80)
    
    print("\n--- Product Count by Category ---")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    print("\n--- Product Count by Subcategory ---")
    subcategory_counts = df['subcategory'].value_counts().head(20)
    print(subcategory_counts)
    
    print("\n--- Average Price by Category ---")
    category_price = df.groupby('category')['price'].agg(['mean', 'median', 'min', 'max', 'count'])
    print(category_price.sort_values('mean', ascending=False))
    
    print("\n--- Average Price by Subcategory (Top 15) ---")
    subcategory_price = df.groupby('subcategory')['price'].agg(['mean', 'median', 'count'])
    print(subcategory_price.sort_values('mean', ascending=False).head(15))
    
    print("\n--- Pink Tax by Category ---")
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    category_gender_price = df_gender.groupby(['category', 'gender_target'])['price'].mean().unstack()
    if 'Men' in category_gender_price.columns and 'Women' in category_gender_price.columns:
        category_gender_price['pink_tax'] = category_gender_price['Women'] - category_gender_price['Men']
        category_gender_price['pink_tax_pct'] = (category_gender_price['pink_tax'] / category_gender_price['Men']) * 100
        print(category_gender_price.sort_values('pink_tax', ascending=False))
    
    print("\n--- Pink Tax by Subcategory (Top 15 by price difference) ---")
    subcategory_gender_price = df_gender.groupby(['subcategory', 'gender_target'])['price'].mean().unstack()
    if 'Men' in subcategory_gender_price.columns and 'Women' in subcategory_gender_price.columns:
        subcategory_gender_price['pink_tax'] = subcategory_gender_price['Women'] - subcategory_gender_price['Men']
        subcategory_gender_price['pink_tax_pct'] = (subcategory_gender_price['pink_tax'] / subcategory_gender_price['Men']) * 100
        print(subcategory_gender_price.sort_values('pink_tax', ascending=False).head(15))

def brand_analysis(df):
    """Analyze products by brand"""
    print("\n" + "=" * 80)
    print("5. BRAND ANALYSIS")
    print("=" * 80)
    
    print("\n--- Top 15 Brands by Product Count ---")
    brand_counts = df['brand'].value_counts().head(15)
    print(brand_counts)
    
    print("\n--- Average Price by Brand (Top 15 most expensive) ---")
    brand_price = df.groupby('brand')['price'].agg(['mean', 'median', 'min', 'max', 'count'])
    print(brand_price.sort_values('mean', ascending=False).head(15))
    
    print("\n--- Pink Tax by Brand (Top 15 brands with highest pink tax) ---")
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    brand_gender_price = df_gender.groupby(['brand', 'gender_target'])['price'].mean().unstack()
    if 'Men' in brand_gender_price.columns and 'Women' in brand_gender_price.columns:
        brand_gender_price['pink_tax'] = brand_gender_price['Women'] - brand_gender_price['Men']
        brand_gender_price['pink_tax_pct'] = (brand_gender_price['pink_tax'] / brand_gender_price['Men']) * 100
        brand_pink_tax = brand_gender_price.sort_values('pink_tax', ascending=False).head(15)
        print(brand_pink_tax)
    
    print("\n--- Brands with Negative Pink Tax (Women pay less) ---")
    if 'Men' in brand_gender_price.columns and 'Women' in brand_gender_price.columns:
        negative_brands = brand_gender_price[brand_gender_price['pink_tax'] < 0].sort_values('pink_tax').head(10)
        print(negative_brands)

def retailer_analysis(df):
    """Analyze products by retailer"""
    print("\n" + "=" * 80)
    print("6. RETAILER ANALYSIS")
    print("=" * 80)
    
    print("\n--- Product Count by Retailer ---")
    retailer_counts = df['retailer'].value_counts()
    print(retailer_counts)
    
    print("\n--- Average Price by Retailer ---")
    retailer_price = df.groupby('retailer')['price'].agg(['mean', 'median', 'min', 'max', 'count'])
    print(retailer_price.sort_values('mean', ascending=False))
    
    print("\n--- Pink Tax by Retailer ---")
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    retailer_gender_price = df_gender.groupby(['retailer', 'gender_target'])['price'].mean().unstack()
    if 'Men' in retailer_gender_price.columns and 'Women' in retailer_gender_price.columns:
        retailer_gender_price['pink_tax'] = retailer_gender_price['Women'] - retailer_gender_price['Men']
        retailer_gender_price['pink_tax_pct'] = (retailer_gender_price['pink_tax'] / retailer_gender_price['Men']) * 100
        print(retailer_gender_price.sort_values('pink_tax', ascending=False))

def size_analysis(df):
    """Analyze products by size"""
    print("\n" + "=" * 80)
    print("7. SIZE ANALYSIS")
    print("=" * 80)
    
    print("\n--- Most Common Sizes ---")
    size_counts = df['size'].value_counts().head(20)
    print(size_counts)
    
    print("\n--- Average Price by Size (Top 15) ---")
    size_price = df.groupby('size')['price'].agg(['mean', 'median', 'count'])
    print(size_price.sort_values('mean', ascending=False).head(15))
    
    print("\n--- Normalized Price Per Unit Statistics ---")
    if 'normalized_price_per_unit' in df.columns:
        print(df['normalized_price_per_unit'].describe())
        
        print("\n--- Normalized Price by Gender ---")
        df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
        print(df_gender.groupby('gender_target')['normalized_price_per_unit'].describe())

def ingredient_analysis(df):
    """Analyze product ingredients"""
    print("\n" + "=" * 80)
    print("8. INGREDIENT ANALYSIS")
    print("=" * 80)
    
    # Clean and extract ingredients
    df_ingredients = df[df['ingredients'].notna() & (df['ingredients'] != '')].copy()
    print(f"\nProducts with ingredient information: {len(df_ingredients)} ({len(df_ingredients)/len(df)*100:.2f}%)")
    
    if len(df_ingredients) > 0:
        # Split ingredients and count
        all_ingredients = []
        for ingredients_str in df_ingredients['ingredients']:
            if pd.notna(ingredients_str) and ingredients_str != '':
                ingredients_list = [i.strip() for i in str(ingredients_str).split(',')]
                all_ingredients.extend(ingredients_list)
        
        print("\n--- Top 20 Most Common Ingredients ---")
        ingredient_counts = Counter(all_ingredients)
        for ingredient, count in ingredient_counts.most_common(20):
            print(f"{ingredient}: {count}")
        
        print("\n--- Average Number of Ingredients per Product ---")
        df_ingredients['ingredient_count'] = df_ingredients['ingredients'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0
        )
        print(f"Mean: {df_ingredients['ingredient_count'].mean():.2f}")
        print(f"Median: {df_ingredients['ingredient_count'].median():.2f}")
        print(f"Min: {df_ingredients['ingredient_count'].min()}")
        print(f"Max: {df_ingredients['ingredient_count'].max()}")
        
        print("\n--- Ingredient Count by Category ---")
        category_ingredients = df_ingredients.groupby('category')['ingredient_count'].agg(['mean', 'median', 'count'])
        print(category_ingredients.sort_values('mean', ascending=False))

def correlation_analysis(df):
    """Analyze correlations between numerical variables"""
    print("\n" + "=" * 80)
    print("9. CORRELATION ANALYSIS")
    print("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        print("\n--- Correlation Matrix ---")
        corr_matrix = df[numerical_cols].corr()
        print(corr_matrix)
        
        print("\n--- Strong Correlations (|r| > 0.5) ---")
        # Get upper triangle of correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        
        # Find strong correlations
        strong_corr = []
        for col in upper_triangle.columns:
            for idx in upper_triangle.index:
                value = upper_triangle.loc[idx, col]
                if pd.notna(value) and abs(value) > 0.5:
                    strong_corr.append((idx, col, value))
        
        if strong_corr:
            for var1, var2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                print(f"{var1} <-> {var2}: {corr_val:.4f}")
        else:
            print("No strong correlations found")

def gender_distribution(df):
    """Analyze gender distribution"""
    print("\n" + "=" * 80)
    print("10. GENDER DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print("\n--- Product Count by Gender ---")
    gender_counts = df['gender_target'].value_counts()
    print(gender_counts)
    print(f"\nPercentages:")
    print(gender_counts / len(df) * 100)
    
    print("\n--- Gender Distribution by Category ---")
    gender_category = pd.crosstab(df['category'], df['gender_target'], normalize='index') * 100
    print(gender_category)
    
    print("\n--- Gender Distribution by Retailer ---")
    gender_retailer = pd.crosstab(df['retailer'], df['gender_target'], normalize='index') * 100
    print(gender_retailer)

def price_range_analysis(df):
    """Analyze price ranges"""
    print("\n" + "=" * 80)
    print("11. PRICE RANGE ANALYSIS")
    print("=" * 80)
    
    # Define price bins
    bins = [0, 200, 500, 1000, 2000, 5000, df['price'].max() + 1]
    labels = ['0-200', '200-500', '500-1000', '1000-2000', '2000-5000', '5000+']
    
    df['price_range'] = pd.cut(df['price'], bins=bins, labels=labels, include_lowest=True)
    
    print("\n--- Product Distribution by Price Range ---")
    price_range_counts = df['price_range'].value_counts().sort_index()
    print(price_range_counts)
    print(f"\nPercentages:")
    print(price_range_counts / len(df) * 100)
    
    print("\n--- Price Range by Gender ---")
    price_range_gender = pd.crosstab(df['price_range'], df['gender_target'], normalize='columns') * 100
    print(price_range_gender)
    
    print("\n--- Price Range by Category ---")
    price_range_category = pd.crosstab(df['price_range'], df['category'], normalize='columns') * 100
    print(price_range_category)

def outlier_analysis(df):
    """Detect and analyze outliers"""
    print("\n" + "=" * 80)
    print("12. OUTLIER ANALYSIS")
    print("=" * 80)
    
    print("\n--- Price Outliers (IQR Method) ---")
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
    print(f"Number of price outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Lower bound: ₹{lower_bound:.2f}")
    print(f"Upper bound: ₹{upper_bound:.2f}")
    
    if len(outliers) > 0:
        print("\n--- Top 10 Most Expensive Products (Outliers) ---")
        expensive_outliers = outliers.nlargest(10, 'price')[['product_name', 'brand', 'category', 'price', 'gender_target', 'retailer']]
        print(expensive_outliers)
        
        print("\n--- Top 10 Cheapest Products (Outliers) ---")
        cheap_outliers = outliers.nsmallest(10, 'price')[['product_name', 'brand', 'category', 'price', 'gender_target', 'retailer']]
        print(cheap_outliers)
    
    if 'normalized_price_per_unit' in df.columns:
        print("\n--- Normalized Price Per Unit Outliers ---")
        Q1_norm = df['normalized_price_per_unit'].quantile(0.25)
        Q3_norm = df['normalized_price_per_unit'].quantile(0.75)
        IQR_norm = Q3_norm - Q1_norm
        lower_bound_norm = Q1_norm - 1.5 * IQR_norm
        upper_bound_norm = Q3_norm + 1.5 * IQR_norm
        
        outliers_norm = df[(df['normalized_price_per_unit'] < lower_bound_norm) | 
                           (df['normalized_price_per_unit'] > upper_bound_norm)]
        print(f"Number of outliers: {len(outliers_norm)} ({len(outliers_norm)/len(df)*100:.2f}%)")

def advanced_statistics(df):
    """Perform advanced statistical tests"""
    print("\n" + "=" * 80)
    print("13. ADVANCED STATISTICAL TESTS")
    print("=" * 80)
    
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    
    # Mann-Whitney U test (non-parametric alternative to t-test)
    print("\n--- Mann-Whitney U Test (Price by Gender) ---")
    men_prices = df_gender[df_gender['gender_target'] == 'Men']['price']
    women_prices = df_gender[df_gender['gender_target'] == 'Women']['price']
    
    u_stat, p_value = stats.mannwhitneyu(women_prices, men_prices, alternative='two-sided')
    print(f"U-statistic: {u_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")
    
    # Kolmogorov-Smirnov test
    print("\n--- Kolmogorov-Smirnov Test (Distribution Similarity) ---")
    ks_stat, ks_p = stats.ks_2samp(men_prices, women_prices)
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"P-value: {ks_p:.4f}")
    if ks_p < 0.05:
        print("Result: Distributions are significantly different (p < 0.05)")
    else:
        print("Result: Distributions are similar (p >= 0.05)")
    
    # Chi-square test for category and gender association
    print("\n--- Chi-Square Test (Category vs Gender) ---")
    contingency_table = pd.crosstab(df_gender['category'], df_gender['gender_target'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_chi:.4f}")
    print(f"Degrees of freedom: {dof}")
    if p_chi < 0.05:
        print("Result: Significant association between category and gender (p < 0.05)")
    else:
        print("Result: No significant association (p >= 0.05)")
    
    # Effect size (Cohen's d)
    print("\n--- Effect Size (Cohen's d for Price Difference) ---")
    mean_diff = women_prices.mean() - men_prices.mean()
    pooled_std = np.sqrt(((len(women_prices) - 1) * women_prices.std()**2 + 
                          (len(men_prices) - 1) * men_prices.std()**2) / 
                         (len(women_prices) + len(men_prices) - 2))
    cohens_d = mean_diff / pooled_std
    print(f"Cohen's d: {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size interpretation: {effect_size}")

def key_insights(df):
    """Generate key insights summary"""
    print("\n" + "=" * 80)
    print("14. KEY INSIGHTS SUMMARY")
    print("=" * 80)
    
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    
    insights = []
    
    # Overall pink tax
    men_avg = df_gender[df_gender['gender_target'] == 'Men']['price'].mean()
    women_avg = df_gender[df_gender['gender_target'] == 'Women']['price'].mean()
    pink_tax_pct = ((women_avg - men_avg) / men_avg) * 100
    
    insights.append(f"1. On average, women's products cost ₹{women_avg - men_avg:.2f} more than men's products ({pink_tax_pct:.2f}% pink tax)")
    
    # Pair analysis
    if 'pair_id' in df.columns:
        pivot_df = df_gender.pivot_table(
            index='pair_id',
            columns='gender_target',
            values='price',
            aggfunc='first'
        ).reset_index()
        
        if 'Men' in pivot_df.columns and 'Women' in pivot_df.columns:
            pivot_df['price_difference'] = pivot_df['Women'] - pivot_df['Men']
            women_pay_more = (pivot_df['price_difference'] > 0).sum()
            total_pairs = len(pivot_df)
            
            insights.append(f"2. In {women_pay_more} out of {total_pairs} product pairs ({women_pay_more/total_pairs*100:.1f}%), women pay more")
    
    # Category with highest pink tax
    category_gender_price = df_gender.groupby(['category', 'gender_target'])['price'].mean().unstack()
    if 'Men' in category_gender_price.columns and 'Women' in category_gender_price.columns:
        category_gender_price['pink_tax'] = category_gender_price['Women'] - category_gender_price['Men']
        highest_pink_tax_cat = category_gender_price['pink_tax'].idxmax()
        highest_pink_tax_val = category_gender_price['pink_tax'].max()
        insights.append(f"3. '{highest_pink_tax_cat}' has the highest average pink tax at ₹{highest_pink_tax_val:.2f}")
    
    # Brand insights
    brand_counts = df['brand'].value_counts()
    top_brand = brand_counts.index[0]
    top_brand_count = brand_counts.iloc[0]
    insights.append(f"4. '{top_brand}' is the most featured brand with {top_brand_count} products")
    
    # Retailer insights
    retailer_counts = df['retailer'].value_counts()
    top_retailer = retailer_counts.index[0]
    top_retailer_count = retailer_counts.iloc[0]
    insights.append(f"5. '{top_retailer}' has the most products with {top_retailer_count} listings")
    
    # Price range insights
    insights.append(f"6. Product prices range from ₹{df['price'].min():.2f} to ₹{df['price'].max():.2f}")
    insights.append(f"7. The median price is ₹{df['price'].median():.2f}, while the mean is ₹{df['price'].mean():.2f}")
    
    # Category insights
    category_counts = df['category'].value_counts()
    top_category = category_counts.index[0]
    top_category_count = category_counts.iloc[0]
    insights.append(f"8. '{top_category}' is the most common category with {top_category_count} products ({top_category_count/len(df)*100:.1f}%)")
    
    print("\n" + "="*80)
    for insight in insights:
        print(f"\n{insight}")
    print("\n" + "="*80)

def export_summary(df):
    """Export summary statistics to CSV"""
    print("\n" + "=" * 80)
    print("15. EXPORTING SUMMARY REPORTS")
    print("=" * 80)
    
    # Summary by category
    category_summary = df.groupby('category').agg({
        'price': ['count', 'mean', 'median', 'min', 'max', 'std'],
        'product_id': 'count'
    }).round(2)
    category_summary.to_csv('category_summary.csv')
    print("✓ Exported: category_summary.csv")
    
    # Summary by brand
    brand_summary = df.groupby('brand').agg({
        'price': ['count', 'mean', 'median', 'min', 'max'],
        'product_id': 'count'
    }).round(2)
    brand_summary.to_csv('brand_summary.csv')
    print("✓ Exported: brand_summary.csv")
    
    # Summary by retailer
    retailer_summary = df.groupby('retailer').agg({
        'price': ['count', 'mean', 'median', 'min', 'max'],
        'product_id': 'count'
    }).round(2)
    retailer_summary.to_csv('retailer_summary.csv')
    print("✓ Exported: retailer_summary.csv")
    
    # Pink tax summary by category
    df_gender = df[df['gender_target'].isin(['Men', 'Women'])].copy()
    pink_tax_category = df_gender.groupby(['category', 'gender_target'])['price'].mean().unstack()
    if 'Men' in pink_tax_category.columns and 'Women' in pink_tax_category.columns:
        pink_tax_category['pink_tax'] = pink_tax_category['Women'] - pink_tax_category['Men']
        pink_tax_category['pink_tax_pct'] = (pink_tax_category['pink_tax'] / pink_tax_category['Men']) * 100
        pink_tax_category.to_csv('pink_tax_by_category.csv')
        print("✓ Exported: pink_tax_by_category.csv")
    
    print("\n✓ All summary reports exported successfully!")

def main():
    """Main function to run all analyses"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PINK TAX DATASET - COMPREHENSIVE EDA" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Load data
    df = load_data()
    
    # Run all analyses
    try:
        basic_info(df)
        statistical_summary(df)
        pink_tax_analysis(df)
        category_analysis(df)
        brand_analysis(df)
        retailer_analysis(df)
        size_analysis(df)
        ingredient_analysis(df)
        correlation_analysis(df)
        gender_distribution(df)
        price_range_analysis(df)
        outlier_analysis(df)
        advanced_statistics(df)
        key_insights(df)
        export_summary(df)
        
        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nAll analyses have been completed successfully.")
        print("Summary CSV files have been exported to the current directory.")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()