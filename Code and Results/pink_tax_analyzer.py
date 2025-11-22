#!/usr/bin/env python3
"""
Pink Tax Analyzer with Llama 3 Integration - FIXED VERSION
Uses Ollama to run Llama 3 locally for intelligent product comparison and pink tax detection
"""

import json
import requests
from typing import Dict, List, Optional
import re


class PinkTaxAnalyzer:
    """Analyzes product pairs for pink tax using Llama 3 via Ollama"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the analyzer
        
        Args:
            ollama_url: Base URL for Ollama API
        """
        self.ollama_url = ollama_url
        self.model = "llama3"
        
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and Llama 3 is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m.get('name', '') for m in models]
                if any('llama3' in m.lower() for m in available_models):
                    return True
                else:
                    print("❌ Llama 3 not found. Available models:", available_models)
                    print("Run: ollama pull llama3")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
    
    def analyze_product_pair(
        self, 
        men_product: Dict[str, any], 
        women_product: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Analyze a pair of men's and women's products for pink tax
        
        Args:
            men_product: Dict with keys: name, price, description, features
            women_product: Dict with keys: name, price, description, features
            
        Returns:
            Dict with analysis results including verdict and justification
        """
        
        # Construct the prompt for Llama 3
        prompt = self._construct_analysis_prompt(men_product, women_product)
        
        # Get LLM response
        llm_response = self._query_llama(prompt)
        
        # Parse the response
        analysis = self._parse_llm_response(llm_response, men_product, women_product)
        
        return analysis
    
    def _construct_analysis_prompt(
        self, 
        men_product: Dict[str, any], 
        women_product: Dict[str, any]
    ) -> str:
        """Construct a detailed prompt for Llama 3 to analyze the products"""
        
        price_diff = women_product['price'] - men_product['price']
        price_diff_pct = (price_diff / men_product['price']) * 100 if men_product['price'] > 0 else 0
        
        prompt = f"""You are a product pricing analyst specializing in detecting "pink tax" - unjustified price differences between men's and women's products.

MEN'S PRODUCT:
- Name: {men_product.get('name', 'Unknown')}
- Price: ₹{men_product['price']:.2f}
- Description: {men_product.get('description', 'No description')}
- Features: {men_product.get('features', 'No features listed')}

WOMEN'S PRODUCT:
- Name: {women_product.get('name', 'Unknown')}
- Price: ₹{women_product['price']:.2f}
- Description: {women_product.get('description', 'No description')}
- Features: {women_product.get('features', 'No features listed')}

PRICE DIFFERENCE:
- Absolute: ₹{price_diff:.2f}
- Percentage: {price_diff_pct:.1f}%

YOUR TASK:
1. Determine if these are essentially the same base product
2. Identify any ACTUAL upgrades/differences in the women's product (e.g., added moisturizer strips, soap bars, extra blades, better materials)
3. Estimate the FAIR additional cost for those upgrades in rupees (be realistic - a soap bar might add ₹10-20, a moisturizing strip ₹5-10, etc.)
4. Compare the actual price difference with the justified price increase
5. Make a verdict: PINK_TAX (if actual difference exceeds justified amount) or FAIR_PRICING

IMPORTANT RULES:
- If women's product has NO upgrades but costs more: DEFINITELY pink tax
- If women's product has upgrades worth X rupees but costs X+Y more (where Y is significant): pink tax
- Consider packaging costs negligible unless significantly different
- "For women" or "feminine scent" alone is NOT a valid upgrade
- Be strict: Only tangible functional improvements justify price increases

CRITICAL: Respond with ONLY valid JSON. NO comments, NO extra text, NO explanations outside the JSON.
Use this EXACT format:

{{
    "base_product_same": true,
    "upgrades_in_women_product": ["list", "of", "upgrades"],
    "justified_price_increase_rs": 10.5,
    "actual_price_difference_rs": {price_diff:.2f},
    "verdict": "PINK_TAX",
    "explanation": "Your reasoning here",
    "confidence": "high"
}}

Respond with ONLY the JSON object, nothing else:"""
        
        return prompt
    
    def _query_llama(self, prompt: str) -> str:
        """Query Llama 3 via Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "top_p": 0.9,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Error querying Llama: {e}")
            return ""
    
    def _parse_llm_response(
        self, 
        llm_response: str, 
        men_product: Dict[str, any], 
        women_product: Dict[str, any]
    ) -> Dict[str, any]:
        """Parse the LLM's JSON response and add calculated fields"""
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Clean up common JSON issues from LLM responses
                # Remove JSON comments (// style)
                json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
                
                # Remove multi-line comments (/* ... */)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                
                # Remove trailing commas before closing braces/brackets  
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                
                # Handle incomplete JSON (missing closing brace)
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += '}' * (open_braces - close_braces)
                
                # Handle range values like "5-10" in justified_price_increase_rs
                # Replace with the average
                def replace_range(match):
                    nums = match.group(1)
                    if '-' in nums and not nums.startswith('-'):
                        parts = nums.split('-')
                        try:
                            avg = (float(parts[0]) + float(parts[1])) / 2
                            return f'": {avg}'
                        except:
                            return match.group(0)
                    return match.group(0)
                
                json_str = re.sub(r'": (\d+[-]\d+)', replace_range, json_str)
                
                # Try to parse
                analysis = json.loads(json_str)
            else:
                print("⚠️ Could not find JSON in LLM response")
                analysis = self._fallback_analysis(men_product, women_product)
            
            # Add calculated fields
            analysis['actual_price_difference_rs'] = women_product['price'] - men_product['price']
            analysis['actual_price_difference_pct'] = (
                (analysis['actual_price_difference_rs'] / men_product['price']) * 100 
                if men_product['price'] > 0 else 0
            )
            
            # Ensure verdict is uppercase with underscore
            if 'verdict' in analysis:
                analysis['verdict'] = analysis['verdict'].upper().replace(' ', '_')
            
            # Add product info
            analysis['men_product_name'] = men_product.get('name', 'Unknown')
            analysis['women_product_name'] = women_product.get('name', 'Unknown')
            analysis['men_price'] = men_product['price']
            analysis['women_price'] = women_product['price']
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"Cleaned JSON: {json_str[:300] if 'json_str' in locals() else 'N/A'}")
            return self._fallback_analysis(men_product, women_product)
    
    def _fallback_analysis(
        self, 
        men_product: Dict[str, any], 
        women_product: Dict[str, any]
    ) -> Dict[str, any]:
        """Fallback analysis if LLM fails"""
        
        price_diff = women_product['price'] - men_product['price']
        price_diff_pct = (price_diff / men_product['price']) * 100 if men_product['price'] > 0 else 0
        
        # Simple heuristic: >10% difference is likely pink tax
        verdict = "PINK_TAX" if price_diff_pct > 10 else "FAIR_PRICING"
        
        return {
            "base_product_same": True,
            "upgrades_in_women_product": ["Unable to analyze"],
            "justified_price_increase_rs": 0,
            "actual_price_difference_rs": price_diff,
            "actual_price_difference_pct": price_diff_pct,
            "verdict": verdict,
            "explanation": "LLM analysis failed. Using simple heuristic: >10% difference = pink tax",
            "confidence": "low",
            "men_product_name": men_product.get('name', 'Unknown'),
            "women_product_name": women_product.get('name', 'Unknown'),
            "men_price": men_product['price'],
            "women_price": women_product['price']
        }
    
    def analyze_multiple_pairs(
        self, 
        product_pairs: List[tuple]
    ) -> List[Dict[str, any]]:
        """
        Analyze multiple product pairs
        
        Args:
            product_pairs: List of (men_product, women_product) tuples
            
        Returns:
            List of analysis results
        """
        results = []
        
        print(f"\n🔍 Analyzing {len(product_pairs)} product pairs...\n")
        
        for i, (men_prod, women_prod) in enumerate(product_pairs, 1):
            print(f"[{i}/{len(product_pairs)}] Analyzing: {men_prod.get('name', 'Unknown')} vs {women_prod.get('name', 'Unknown')}")
            
            result = self.analyze_product_pair(men_prod, women_prod)
            results.append(result)
            
            # Print verdict
            verdict_emoji = "❌" if result['verdict'] == "PINK_TAX" else "✅"
            print(f"   {verdict_emoji} Verdict: {result['verdict']}")
            print(f"   💰 Price difference: ₹{result['actual_price_difference_rs']:.2f} ({result['actual_price_difference_pct']:.1f}%)")
            if result.get('justified_price_increase_rs'):
                print(f"   ⚖️  Justified increase: ₹{result['justified_price_increase_rs']:.2f}")
            print()
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Generate a summary report from multiple analyses"""
        
        if not results:
            return {"error": "No results to summarize"}
        
        pink_tax_count = sum(1 for r in results if r['verdict'] == 'PINK_TAX')
        fair_count = len(results) - pink_tax_count
        
        avg_price_diff = sum(r['actual_price_difference_rs'] for r in results) / len(results)
        avg_price_diff_pct = sum(r['actual_price_difference_pct'] for r in results) / len(results)
        
        pink_tax_cases = [r for r in results if r['verdict'] == 'PINK_TAX']
        avg_pink_tax_amount = (
            sum(r['actual_price_difference_rs'] for r in pink_tax_cases) / len(pink_tax_cases)
            if pink_tax_cases else 0
        )
        
        return {
            "total_pairs_analyzed": len(results),
            "pink_tax_detected": pink_tax_count,
            "fair_pricing": fair_count,
            "pink_tax_percentage": (pink_tax_count / len(results)) * 100,
            "average_price_difference_rs": round(avg_price_diff, 2),
            "average_price_difference_pct": round(avg_price_diff_pct, 2),
            "average_pink_tax_amount_rs": round(avg_pink_tax_amount, 2),
            "worst_offenders": sorted(
                pink_tax_cases, 
                key=lambda x: x['actual_price_difference_pct'], 
                reverse=True
            )[:3]
        }