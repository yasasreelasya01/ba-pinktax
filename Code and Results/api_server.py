#!/usr/bin/env python3
"""
Flask API for Pink Tax Dashboard with Llama 3 Integration
Provides endpoints for analyzing products and detecting pink tax
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pink_tax_analyzer import PinkTaxAnalyzer
import json
from typing import Dict, List

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize the analyzer
analyzer = PinkTaxAnalyzer()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    ollama_status = analyzer.check_ollama_status()
    return jsonify({
        "status": "healthy" if ollama_status else "degraded",
        "ollama_running": ollama_status,
        "message": "Llama 3 is ready" if ollama_status else "Ollama not running or Llama 3 not available"
    })


@app.route('/analyze/single', methods=['POST'])
def analyze_single_pair():
    """
    Analyze a single product pair
    
    Expected JSON:
    {
        "men_product": {
            "name": "...",
            "price": 100.0,
            "description": "...",
            "features": "..."
        },
        "women_product": {
            "name": "...",
            "price": 150.0,
            "description": "...",
            "features": "..."
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'men_product' not in data or 'women_product' not in data:
            return jsonify({
                "error": "Missing required fields: men_product and women_product"
            }), 400
        
        men_product = data['men_product']
        women_product = data['women_product']
        
        # Validate required fields
        for product, name in [(men_product, 'men_product'), (women_product, 'women_product')]:
            if 'price' not in product:
                return jsonify({
                    "error": f"Missing 'price' field in {name}"
                }), 400
        
        # Analyze the pair
        result = analyzer.analyze_product_pair(men_product, women_product)
        
        return jsonify({
            "success": True,
            "analysis": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple product pairs
    
    Expected JSON:
    {
        "product_pairs": [
            {
                "men_product": {...},
                "women_product": {...}
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'product_pairs' not in data:
            return jsonify({
                "error": "Missing required field: product_pairs"
            }), 400
        
        product_pairs = []
        for pair in data['product_pairs']:
            if 'men_product' not in pair or 'women_product' not in pair:
                continue
            product_pairs.append((pair['men_product'], pair['women_product']))
        
        if not product_pairs:
            return jsonify({
                "error": "No valid product pairs found"
            }), 400
        
        # Analyze all pairs
        results = analyzer.analyze_multiple_pairs(product_pairs)
        
        # Generate summary
        summary = analyzer.generate_summary_report(results)
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/analyze/from-dashboard', methods=['POST'])
def analyze_from_dashboard():
    """
    Analyze products directly from dashboard data
    
    Expected JSON:
    {
        "category": "razors",
        "subcategory": "disposable",
        "products": [
            {
                "name": "...",
                "brand": "...",
                "gender": "men" or "women",
                "price": 100.0,
                "description": "...",
                "features": "..."
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'products' not in data:
            return jsonify({
                "error": "Missing required field: products"
            }), 400
        
        products = data['products']
        
        # Separate men's and women's products
        men_products = [p for p in products if p.get('gender', '').lower() == 'men']
        women_products = [p for p in products if p.get('gender', '').lower() == 'women']
        
        if not men_products or not women_products:
            return jsonify({
                "error": "Need both men's and women's products to compare"
            }), 400
        
        # For simplicity, compare products by brand or just pair them in order
        # You can implement more sophisticated matching logic here
        product_pairs = []
        
        # Try to match by brand first
        men_by_brand = {p.get('brand', 'unknown'): p for p in men_products}
        women_by_brand = {p.get('brand', 'unknown'): p for p in women_products}
        
        common_brands = set(men_by_brand.keys()) & set(women_by_brand.keys())
        
        if common_brands:
            # Match by brand
            for brand in common_brands:
                product_pairs.append((men_by_brand[brand], women_by_brand[brand]))
        else:
            # Just pair in order
            min_len = min(len(men_products), len(women_products))
            for i in range(min_len):
                product_pairs.append((men_products[i], women_products[i]))
        
        # Analyze pairs
        results = analyzer.analyze_multiple_pairs(product_pairs)
        
        # Generate summary
        summary = analyzer.generate_summary_report(results)
        
        return jsonify({
            "success": True,
            "category": data.get('category', 'unknown'),
            "subcategory": data.get('subcategory', 'unknown'),
            "results": results,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/analyze/quick', methods=['POST'])
def quick_analyze():
    """
    Quick analysis with minimal data
    
    Expected JSON:
    {
        "men_price": 100.0,
        "women_price": 150.0,
        "product_type": "razor",
        "men_features": "optional",
        "women_features": "optional"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'men_price' not in data or 'women_price' not in data:
            return jsonify({
                "error": "Missing required fields: men_price and women_price"
            }), 400
        
        men_product = {
            "name": f"Men's {data.get('product_type', 'Product')}",
            "price": float(data['men_price']),
            "description": data.get('men_description', 'Standard product for men'),
            "features": data.get('men_features', 'Basic features')
        }
        
        women_product = {
            "name": f"Women's {data.get('product_type', 'Product')}",
            "price": float(data['women_price']),
            "description": data.get('women_description', 'Standard product for women'),
            "features": data.get('women_features', 'Basic features')
        }
        
        # Analyze
        result = analyzer.analyze_product_pair(men_product, women_product)
        
        return jsonify({
            "success": True,
            "analysis": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 Starting Pink Tax Analysis API Server")
    print("=" * 70)
    
    # Check Ollama status
    print("\n✓ Checking Ollama status...")
    if analyzer.check_ollama_status():
        print("✅ Ollama is running with Llama 3!")
    else:
        print("\n⚠️  WARNING: Ollama is not running or Llama 3 is not available")
        print("   The API will start but analyses will fail.")
        print("   Please ensure:")
        print("   1. Ollama is running: ollama serve")
        print("   2. Llama 3 is installed: ollama pull llama3")
    
    print("\n" + "=" * 70)
    print("📡 API Endpoints:")
    print("   GET  /health                    - Health check")
    print("   POST /analyze/single            - Analyze one product pair")
    print("   POST /analyze/batch             - Analyze multiple pairs")
    print("   POST /analyze/from-dashboard    - Analyze from dashboard data")
    print("   POST /analyze/quick             - Quick analysis")
    print("=" * 70)
    print("\n🌐 Server starting on http://localhost:5000\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)