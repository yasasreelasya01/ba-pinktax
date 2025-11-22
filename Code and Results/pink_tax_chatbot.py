#!/usr/bin/env python3
"""
Pink Tax Chatbot - Integrated System
- Builds knowledge base from CSV (one-time)
- Runs Flask server for chatbot API
- Uses Ollama Llama3 for analysis
"""

import json
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from collections import defaultdict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_FILE = 'pink_tax_dataset_2000_inr_fixed.csv'
KNOWLEDGE_BASE_FILE = 'product_knowledge_base.json'
OLLAMA_API = 'http://localhost:11434/api/generate'
MODEL = 'llama3'

# ============================================================================
# KNOWLEDGE BASE BUILDER
# ============================================================================

def call_ollama(prompt, max_tokens=1000):
    """Call Ollama API with streaming disabled for JSON responses"""
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Please ensure Ollama is running (ollama serve)"
    except Exception as e:
        return f"ERROR: {str(e)}"


def build_knowledge_base():
    """Build knowledge base from CSV - aggregated by product type and brand"""
    
    print("\n" + "="*80)
    print("BUILDING KNOWLEDGE BASE FROM DATASET")
    print("="*80)
    
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"✓ Knowledge base already exists: {KNOWLEDGE_BASE_FILE}")
        response = input("Rebuild? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing knowledge base.")
            return
    
    # Load CSV
    print(f"\n📂 Loading dataset: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    
    # Parse pairs
    pairs = []
    pair_dict = {}
    
    for _, row in df.iterrows():
        pair_id = row['pair_id']
        if pair_id not in pair_dict:
            pair_dict[pair_id] = {}
        
        if row['gender_target'] == 'Men':
            pair_dict[pair_id]['men'] = row
        else:
            pair_dict[pair_id]['women'] = row


    # Create complete pairs
    for pair_id, data in pair_dict.items():
        if 'men' in data and 'women' in data:
            product_name_clean = data['men']['product_name']
            product_name_clean = product_name_clean.replace(' - Men', '')
            product_name_clean = re.sub(r' - \d+.*$', '', product_name_clean)

            pairs.append({
                'pair_id': pair_id,
                'product_name': product_name_clean,
                'brand': data['men']['brand'],
                'category': data['men']['category'],
                'subcategory': data['men']['subcategory'],
                'men_price': data['men']['price'],
                'women_price': data['women']['price'],
                'price_diff': data['women']['price'] - data['men']['price'],
                'price_diff_pct': ((data['women']['price'] - data['men']['price']) / data['men']['price'] * 100)
                                if data['men']['price'] > 0 else 0
            })

    
    print(f"✓ Loaded {len(pairs)} product pairs")
    
    # Group by category and subcategory
    print("\n📊 Grouping products by category and subcategory...")
    grouped = defaultdict(lambda: defaultdict(list))
    
    for pair in pairs:
        category = pair['category']
        subcategory = pair['subcategory']
        grouped[category][subcategory].append(pair)
    
    # Aggregate statistics
    knowledge_base = {}
    
    total_groups = sum(len(subcats) for subcats in grouped.values())
    current = 0
    
    print(f"\n🤖 Analyzing {total_groups} product groups with Llama3...")
    print("(This may take a few minutes...)\n")
    
    for category, subcategories in grouped.items():
        knowledge_base[category] = {}
        
        for subcategory, products in subcategories.items():
            current += 1
            print(f"[{current}/{total_groups}] Analyzing {category} > {subcategory}...")
            
            # Aggregate stats
            brands_data = defaultdict(list)
            for p in products:
                brands_data[p['brand']].append(p)
            
            # Brand analysis
            brands_analysis = {}
            for brand, brand_products in brands_data.items():
                avg_pink_tax = sum(p['price_diff'] for p in brand_products) / len(brand_products)
                avg_pink_tax_pct = sum(p['price_diff_pct'] for p in brand_products) / len(brand_products)
                
                brands_analysis[brand] = {
                    'count': len(brand_products),
                    'avg_pink_tax': round(avg_pink_tax, 2),
                    'avg_pink_tax_pct': round(avg_pink_tax_pct, 2),
                    'min_price': round(min(p['men_price'] for p in brand_products), 2),
                    'max_price': round(max(p['women_price'] for p in brand_products), 2)
                }
            
            # Find best brand (lowest pink tax)
            best_brand = min(brands_analysis.items(), key=lambda x: x[1]['avg_pink_tax_pct'])
            worst_brand = max(brands_analysis.items(), key=lambda x: x[1]['avg_pink_tax_pct'])
            
            # Overall category stats
            avg_pink_tax = sum(p['price_diff'] for p in products) / len(products)
            avg_pink_tax_pct = sum(p['price_diff_pct'] for p in products) / len(products)
            
            # Ask LLM for analysis
            prompt = f"""You are a consumer product expert analyzing gender-based pricing (Pink Tax).

Category: {category}
Product Type: {subcategory}
Total Products Analyzed: {len(products)}

PRICING DATA:
- Average Pink Tax: ₹{avg_pink_tax:.2f} ({avg_pink_tax_pct:.1f}% markup for women)
- Best Brand (lowest pink tax): {best_brand[0]} ({best_brand[1]['avg_pink_tax_pct']:.1f}% markup)
- Worst Brand (highest pink tax): {worst_brand[0]} ({worst_brand[1]['avg_pink_tax_pct']:.1f}% markup)

QUESTION: Based on your knowledge of {subcategory} products:
1. What typical differences exist between men's and women's {subcategory}?
2. Do these differences typically justify a {avg_pink_tax_pct:.1f}% price increase?
3. What is a fair/justified price difference percentage for such upgrades?
4. Is this category experiencing significant pink tax?

Provide a concise analysis in this JSON format:
{{
  "typical_differences": ["difference1", "difference2"],
  "justified_percentage": 5.0,
  "verdict": "pink_tax" or "justified" or "minimal",
  "recommendation": "brief recommendation for consumers",
  "reasoning": "2-3 sentence explanation"
}}

Respond ONLY with valid JSON, no other text."""

            llm_response = call_ollama(prompt, max_tokens=500)
            
            # Parse LLM response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    llm_analysis = json.loads(json_match.group())
                else:
                    llm_analysis = {
                        "typical_differences": ["Unable to analyze"],
                        "justified_percentage": 5.0,
                        "verdict": "unknown",
                        "recommendation": "Review product specifications carefully",
                        "reasoning": "LLM analysis unavailable"
                    }
            except:
                llm_analysis = {
                    "typical_differences": ["Unable to analyze"],
                    "justified_percentage": 5.0,
                    "verdict": "unknown",
                    "recommendation": "Review product specifications carefully",
                    "reasoning": "LLM analysis unavailable"
                }
            
            # Store in knowledge base
            knowledge_base[category][subcategory] = {
                'product_count': len(products),
                'brands': brands_analysis,
                'best_brand': best_brand[0],
                'worst_brand': worst_brand[0],
                'avg_pink_tax': round(avg_pink_tax, 2),
                'avg_pink_tax_pct': round(avg_pink_tax_pct, 2),
                'llm_analysis': llm_analysis
            }
    
    # Save knowledge base
    print(f"\n💾 Saving knowledge base to {KNOWLEDGE_BASE_FILE}...")
    with open(KNOWLEDGE_BASE_FILE, 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ KNOWLEDGE BASE BUILT SUCCESSFULLY!")
    print("="*80)
    print(f"Categories analyzed: {len(knowledge_base)}")
    print(f"Total product groups: {sum(len(subcats) for subcats in knowledge_base.values())}")
    print(f"Knowledge base saved to: {KNOWLEDGE_BASE_FILE}")


# ============================================================================
# CHATBOT QUERY HANDLER
# ============================================================================

def load_knowledge_base():
    """Load the knowledge base from JSON file"""
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return None
    
    with open(KNOWLEDGE_BASE_FILE, 'r') as f:
        return json.load(f)


def search_knowledge_base(query, knowledge_base):
    """Search knowledge base for relevant product information"""
    query_lower = query.lower()
    
    # Extract potential product type and brand from query
    results = []
    
    for category, subcategories in knowledge_base.items():
        for subcategory, data in subcategories.items():
            # Check if query matches category or subcategory
            if (category.lower() in query_lower or 
                subcategory.lower() in query_lower):
                
                # Check for brand mentions
                mentioned_brands = [brand for brand in data['brands'].keys() 
                                   if brand.lower() in query_lower]
                
                results.append({
                    'category': category,
                    'subcategory': subcategory,
                    'data': data,
                    'mentioned_brands': mentioned_brands,
                    'relevance_score': len(mentioned_brands) + 
                                      (2 if subcategory.lower() in query_lower else 0) +
                                      (1 if category.lower() in query_lower else 0)
                })
    
    # Sort by relevance
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return results


def format_chatbot_response(query, search_results, knowledge_base):
    """Format a comprehensive response for the user"""
    
    if not search_results:
        # Not in dataset - use LLM general knowledge
        prompt = f"""User is asking about: "{query}"

This product is not in our dataset. Based on your general knowledge of consumer products and pink tax:

1. What is this product typically used for?
2. Do women's versions typically cost more than men's versions?
3. What are common differences between gendered versions?
4. What would be a fair price difference?
5. Any recommendations for consumers?

Provide a helpful response in natural conversational tone."""

        llm_response = call_ollama(prompt, max_tokens=400)
        
        return {
            'found_in_dataset': False,
            'response': llm_response,
            'message': "I don't have specific pricing data for this product in my dataset, but here's what I know based on general product knowledge:",
            'suggestion': "If you're looking for products I have data on, try asking about face wash, soap, razors, deodorant, or other personal care items."
        }
    
    # Found in dataset
    top_result = search_results[0]
    data = top_result['data']
    category = top_result['category']
    subcategory = top_result['subcategory']
    
    # Build response
    response = {
        'found_in_dataset': True,
        'category': category,
        'subcategory': subcategory,
        'product_count': data['product_count'],
        'avg_pink_tax': data['avg_pink_tax'],
        'avg_pink_tax_pct': data['avg_pink_tax_pct'],
        'best_brand': data['best_brand'],
        'worst_brand': data['worst_brand'],
        'brands': data['brands'],
        'llm_analysis': data['llm_analysis']
    }
    
    # If specific brand mentioned
    if top_result['mentioned_brands']:
        brand = top_result['mentioned_brands'][0]
        response['specific_brand'] = brand
        response['brand_data'] = data['brands'][brand]
    
    return response


# ============================================================================
# FLASK API SERVER
# ============================================================================

app = Flask(__name__)
CORS(app)

knowledge_base = None


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    global knowledge_base
    
    if knowledge_base is None:
        return jsonify({
            'error': 'Knowledge base not loaded. Please run build_knowledge_base() first.'
        }), 500
    
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Search knowledge base
    results = search_knowledge_base(query, knowledge_base)
    
    # Format response
    response = format_chatbot_response(query, results, knowledge_base)
    
    return jsonify(response)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'knowledge_base_loaded': knowledge_base is not None,
        'ollama_available': not call_ollama("test").startswith("ERROR")
    })


# ============================================================================
# MAIN
# ============================================================================

def main():
    global knowledge_base
    
    print("\n" + "="*80)
    print("PINK TAX CHATBOT SYSTEM")
    print("="*80)
    
    # Check if Ollama is running
    print("\n🔍 Checking Ollama connection...")
    test_response = call_ollama("Say 'OK' if you're working", max_tokens=10)
    if test_response.startswith("ERROR"):
        print(f"❌ {test_response}")
        print("\n💡 To start Ollama:")
        print("   1. Open a new terminal")
        print("   2. Run: ollama serve")
        print("   3. Then run this script again")
        return
    print("✓ Ollama is running!")
    
    # Build or load knowledge base
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print("\n📚 Knowledge base not found. Building for the first time...")
        build_knowledge_base()
    else:
        print(f"\n📚 Loading existing knowledge base from {KNOWLEDGE_BASE_FILE}...")
        
    knowledge_base = load_knowledge_base()
    
    if knowledge_base is None:
        print("❌ Failed to load knowledge base!")
        return
    
    print(f"✓ Knowledge base loaded with {len(knowledge_base)} categories")
    
    # Start Flask server
    print("\n" + "="*80)
    print("🚀 STARTING CHATBOT SERVER")
    print("="*80)
    print("\nServer running at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/api/chat")
    print("\n💡 Open your dashboard (index.html) and use the chatbot!")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rebuild':
        # Force rebuild knowledge base
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            os.remove(KNOWLEDGE_BASE_FILE)
        build_knowledge_base()
    else:
        main()