#!/usr/bin/env python3
"""
Comprehensive LLM Evaluation for Pink Tax Analyzer
Evaluates accuracy, consistency, and response quality
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time
from collections import defaultdict
import re
from pink_tax_analyzer import PinkTaxAnalyzer

class LLMEvaluator:
    """Evaluates the Pink Tax Analyzer LLM performance"""
    
    def __init__(self):
        self.analyzer = PinkTaxAnalyzer()
        self.results = {
            'accuracy': {},
            'consistency': {},
            'response_quality': {}
        }
        
    # ========================================================================
    # 1. ACCURACY & CORRECTNESS METRICS
    # ========================================================================
    
    def evaluate_classification_accuracy(
        self, 
        gold_standard: List[Dict]
    ) -> Dict:
        """
        Evaluate classification accuracy against gold standard
        
        Args:
            gold_standard: List of dicts with keys:
                - men_product: dict
                - women_product: dict
                - true_label: "PINK_TAX" or "FAIR_PRICING"
                - actual_upgrades: list of actual product differences
                - justified_cost: actual fair cost of upgrades
        
        Returns:
            Dict with accuracy metrics
        """
        print("\n" + "="*80)
        print("1. CLASSIFICATION ACCURACY EVALUATION")
        print("="*80)
        
        predictions = []
        true_labels = []
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        print(f"\nEvaluating {len(gold_standard)} test cases...")
        
        for i, case in enumerate(gold_standard, 1):
            print(f"\r  Progress: {i}/{len(gold_standard)}", end='', flush=True)
            
            # Get LLM prediction
            result = self.analyzer.analyze_product_pair(
                case['men_product'],
                case['women_product']
            )
            
            predicted_label = result.get('verdict', 'UNKNOWN')
            true_label = case['true_label']
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
            # Calculate confusion matrix
            if true_label == 'PINK_TAX' and predicted_label == 'PINK_TAX':
                true_positives += 1
            elif true_label == 'FAIR_PRICING' and predicted_label == 'PINK_TAX':
                false_positives += 1
            elif true_label == 'FAIR_PRICING' and predicted_label == 'FAIR_PRICING':
                true_negatives += 1
            elif true_label == 'PINK_TAX' and predicted_label == 'FAIR_PRICING':
                false_negatives += 1
        
        print()  # New line after progress
        
        # Calculate metrics
        total = len(gold_standard)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        true_pink_tax_rate = recall  # Same as recall
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        false_negative_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_pink_tax_detection_rate': true_pink_tax_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
        }
        
        # Print results
        print("\n" + "-"*80)
        print("CLASSIFICATION ACCURACY RESULTS")
        print("-"*80)
        print(f"Overall Accuracy:              {accuracy:.2%}")
        print(f"Precision:                     {precision:.2%}")
        print(f"Recall:                        {recall:.2%}")
        print(f"F1 Score:                      {f1_score:.2%}")
        print(f"\nTrue Pink Tax Detection Rate:  {true_pink_tax_rate:.2%}")
        print(f"False Positive Rate:           {false_positive_rate:.2%}")
        print(f"False Negative Rate:           {false_negative_rate:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  True Negatives:  {true_negatives}")
        print(f"  False Negatives: {false_negatives}")
        
        self.results['accuracy']['classification'] = metrics
        return metrics
    
    def evaluate_justification_quality(
        self,
        gold_standard: List[Dict]
    ) -> Dict:
        """
        Evaluate quality of price justifications
        
        Args:
            gold_standard: Same format as above, must include:
                - actual_upgrades: list of real upgrades
                - justified_cost: actual fair cost
        """
        print("\n" + "="*80)
        print("2. JUSTIFICATION QUALITY EVALUATION")
        print("="*80)
        
        justified_price_errors = []
        upgrade_detection_results = []
        
        print(f"\nEvaluating {len(gold_standard)} test cases...")
        
        for i, case in enumerate(gold_standard, 1):
            print(f"\r  Progress: {i}/{len(gold_standard)}", end='', flush=True)
            
            # Skip if no actual upgrades information
            if 'actual_upgrades' not in case or 'justified_cost' not in case:
                continue
            
            result = self.analyzer.analyze_product_pair(
                case['men_product'],
                case['women_product']
            )
            
            # Justified price estimation accuracy
            predicted_justified_cost = result.get('justified_price_increase_rs', 0)
            actual_justified_cost = case['justified_cost']
            error = abs(predicted_justified_cost - actual_justified_cost)
            justified_price_errors.append(error)
            
            # Upgrade detection
            predicted_upgrades = set([u.lower().strip() for u in result.get('upgrades_in_women_product', [])])
            actual_upgrades = set([u.lower().strip() for u in case['actual_upgrades']])
            
            # Remove generic/useless predictions
            predicted_upgrades = {u for u in predicted_upgrades if u and u not in ['unable to analyze', 'no upgrades']}
            
            true_detected = len(predicted_upgrades & actual_upgrades)
            false_detected = len(predicted_upgrades - actual_upgrades)
            missed = len(actual_upgrades - predicted_upgrades)
            
            upgrade_detection_results.append({
                'completeness': true_detected / len(actual_upgrades) if actual_upgrades else 1.0,
                'precision': true_detected / len(predicted_upgrades) if predicted_upgrades else 0.0,
                'true_detected': true_detected,
                'false_detected': false_detected,
                'missed': missed
            })
        
        print()  # New line
        
        # Calculate metrics
        mae_justified_price = np.mean(justified_price_errors) if justified_price_errors else 0
        median_ae_justified_price = np.median(justified_price_errors) if justified_price_errors else 0
        
        avg_completeness = np.mean([r['completeness'] for r in upgrade_detection_results]) if upgrade_detection_results else 0
        avg_precision = np.mean([r['precision'] for r in upgrade_detection_results]) if upgrade_detection_results else 0
        
        metrics = {
            'justified_price_mae': mae_justified_price,
            'justified_price_median_ae': median_ae_justified_price,
            'upgrade_detection_completeness': avg_completeness,
            'upgrade_detection_precision': avg_precision,
            'upgrade_detection_f1': 2 * (avg_precision * avg_completeness) / (avg_precision + avg_completeness) 
                                     if (avg_precision + avg_completeness) > 0 else 0
        }
        
        print("\n" + "-"*80)
        print("JUSTIFICATION QUALITY RESULTS")
        print("-"*80)
        print(f"Justified Price MAE:            ₹{mae_justified_price:.2f}")
        print(f"Justified Price Median AE:      ₹{median_ae_justified_price:.2f}")
        print(f"\nUpgrade Detection Completeness: {avg_completeness:.2%}")
        print(f"Upgrade Detection Precision:    {avg_precision:.2%}")
        print(f"Upgrade Detection F1:           {metrics['upgrade_detection_f1']:.2%}")
        
        self.results['accuracy']['justification'] = metrics
        return metrics
    
    # ========================================================================
    # 2. CONSISTENCY & RELIABILITY METRICS
    # ========================================================================
    
    def evaluate_inter_run_consistency(
        self,
        test_cases: List[Dict],
        num_runs: int = 5
    ) -> Dict:
        """
        Evaluate consistency across multiple runs of same input
        
        Args:
            test_cases: List of product pairs to test
            num_runs: Number of times to run each test case
        """
        print("\n" + "="*80)
        print("3. INTER-RUN CONSISTENCY EVALUATION")
        print("="*80)
        
        print(f"\nRunning {len(test_cases)} test cases {num_runs} times each...")
        print(f"Total evaluations: {len(test_cases) * num_runs}")
        
        consistency_results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n  Test case {i}/{len(test_cases)}")
            
            verdicts = []
            justified_prices = []
            explanations = []
            
            for run in range(num_runs):
                print(f"\r    Run {run+1}/{num_runs}", end='', flush=True)
                
                result = self.analyzer.analyze_product_pair(
                    case['men_product'],
                    case['women_product']
                )
                
                verdicts.append(result.get('verdict', 'UNKNOWN'))
                justified_prices.append(result.get('justified_price_increase_rs', 0))
                explanations.append(result.get('explanation', ''))
            
            print()  # New line
            
            # Calculate consistency for this case
            most_common_verdict = max(set(verdicts), key=verdicts.count)
            verdict_consistency = verdicts.count(most_common_verdict) / num_runs
            
            # Check if verdict flipped
            verdict_flipped = len(set(verdicts)) > 1
            
            # Numerical stability
            justified_price_variance = np.var(justified_prices)
            justified_price_std = np.std(justified_prices)
            
            consistency_results.append({
                'verdict_consistency': verdict_consistency,
                'verdict_flipped': verdict_flipped,
                'justified_price_variance': justified_price_variance,
                'justified_price_std': justified_price_std,
                'unique_verdicts': len(set(verdicts)),
                'verdicts': verdicts,
                'justified_prices': justified_prices
            })
        
        # Calculate aggregate metrics
        avg_verdict_consistency = np.mean([r['verdict_consistency'] for r in consistency_results])
        verdict_flip_rate = np.mean([r['verdict_flipped'] for r in consistency_results])
        avg_justified_price_std = np.mean([r['justified_price_std'] for r in consistency_results])
        
        metrics = {
            'average_verdict_consistency': avg_verdict_consistency,
            'verdict_flip_rate': verdict_flip_rate,
            'average_justified_price_std': avg_justified_price_std,
            'perfectly_consistent_cases': sum(1 for r in consistency_results if r['verdict_consistency'] == 1.0),
            'total_cases': len(test_cases)
        }
        
        print("\n" + "-"*80)
        print("INTER-RUN CONSISTENCY RESULTS")
        print("-"*80)
        print(f"Average Verdict Consistency:    {avg_verdict_consistency:.2%}")
        print(f"Verdict Flip Rate:              {verdict_flip_rate:.2%}")
        print(f"Avg Justified Price Std Dev:    ₹{avg_justified_price_std:.2f}")
        print(f"Perfectly Consistent Cases:     {metrics['perfectly_consistent_cases']}/{len(test_cases)}")
        
        self.results['consistency']['inter_run'] = metrics
        return metrics
    
    def evaluate_similar_product_consistency(
        self,
        product_groups: List[Dict]
    ) -> Dict:
        """
        Evaluate consistency across similar products
        
        Args:
            product_groups: List of dicts with:
                - group_type: "brand" or "category"
                - group_name: name of brand/category
                - products: list of product pairs
        """
        print("\n" + "="*80)
        print("4. SIMILAR PRODUCT CONSISTENCY EVALUATION")
        print("="*80)
        
        group_consistency_results = []
        
        for group_idx, group in enumerate(product_groups, 1):
            print(f"\n  Evaluating group {group_idx}/{len(product_groups)}: {group['group_name']}")
            
            verdicts = []
            justified_prices = []
            
            for i, pair in enumerate(group['products'], 1):
                print(f"\r    Product {i}/{len(group['products'])}", end='', flush=True)
                
                result = self.analyzer.analyze_product_pair(
                    pair['men_product'],
                    pair['women_product']
                )
                
                verdicts.append(result.get('verdict', 'UNKNOWN'))
                justified_prices.append(result.get('justified_price_increase_rs', 0))
            
            print()  # New line
            
            # Calculate consistency within group
            most_common_verdict = max(set(verdicts), key=verdicts.count)
            verdict_consistency = verdicts.count(most_common_verdict) / len(verdicts)
            
            # Check variance in justified prices
            justified_price_cv = np.std(justified_prices) / np.mean(justified_prices) if np.mean(justified_prices) > 0 else 0
            
            group_consistency_results.append({
                'group_type': group['group_type'],
                'group_name': group['group_name'],
                'verdict_consistency': verdict_consistency,
                'justified_price_cv': justified_price_cv,
                'num_products': len(group['products']),
                'unique_verdicts': len(set(verdicts))
            })
        
        # Calculate metrics by group type
        brand_groups = [r for r in group_consistency_results if r['group_type'] == 'brand']
        category_groups = [r for r in group_consistency_results if r['group_type'] == 'category']
        
        metrics = {
            'brand_consistency': np.mean([r['verdict_consistency'] for r in brand_groups]) if brand_groups else 0,
            'category_consistency': np.mean([r['verdict_consistency'] for r in category_groups]) if category_groups else 0,
            'overall_consistency': np.mean([r['verdict_consistency'] for r in group_consistency_results]),
            'groups_evaluated': len(product_groups)
        }
        
        print("\n" + "-"*80)
        print("SIMILAR PRODUCT CONSISTENCY RESULTS")
        print("-"*80)
        print(f"Brand Consistency:              {metrics['brand_consistency']:.2%}")
        print(f"Category Consistency:           {metrics['category_consistency']:.2%}")
        print(f"Overall Consistency:            {metrics['overall_consistency']:.2%}")
        
        self.results['consistency']['similar_products'] = metrics
        return metrics
    
    # ========================================================================
    # 3. RESPONSE QUALITY METRICS
    # ========================================================================
    
    def evaluate_explanation_quality(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Evaluate quality of explanations
        """
        print("\n" + "="*80)
        print("5. EXPLANATION QUALITY EVALUATION")
        print("="*80)
        
        explanation_lengths = []
        clarity_scores = []
        evidence_citations = []
        confidence_calibration = {'high': [], 'medium': [], 'low': []}
        
        print(f"\nEvaluating {len(test_cases)} test cases...")
        
        for i, case in enumerate(test_cases, 1):
            print(f"\r  Progress: {i}/{len(test_cases)}", end='', flush=True)
            
            result = self.analyzer.analyze_product_pair(
                case['men_product'],
                case['women_product']
            )
            
            explanation = result.get('explanation', '')
            confidence = result.get('confidence', 'unknown')
            
            # Explanation length
            explanation_lengths.append(len(explanation.split()))
            
            # Clarity score (keyword-based heuristic)
            clarity_score = self._calculate_clarity_score(explanation)
            clarity_scores.append(clarity_score)
            
            # Evidence citation (does it reference specific features?)
            has_evidence = self._check_evidence_citation(
                explanation, 
                case['men_product'], 
                case['women_product']
            )
            evidence_citations.append(has_evidence)
            
            # Confidence calibration (if we have ground truth)
            if 'true_label' in case:
                correct = result.get('verdict') == case['true_label']
                if confidence in confidence_calibration:
                    confidence_calibration[confidence].append(correct)
        
        print()  # New line
        
        # Calculate metrics
        avg_length = np.mean(explanation_lengths)
        avg_clarity = np.mean(clarity_scores)
        evidence_citation_rate = np.mean(evidence_citations)
        
        # Confidence calibration
        calibration = {}
        for conf_level, correct_list in confidence_calibration.items():
            if correct_list:
                calibration[conf_level] = np.mean(correct_list)
        
        metrics = {
            'avg_explanation_length_words': avg_length,
            'avg_clarity_score': avg_clarity,
            'evidence_citation_rate': evidence_citation_rate,
            'confidence_calibration': calibration
        }
        
        print("\n" + "-"*80)
        print("EXPLANATION QUALITY RESULTS")
        print("-"*80)
        print(f"Avg Explanation Length:         {avg_length:.1f} words")
        print(f"Avg Clarity Score:              {avg_clarity:.2f}/5.0")
        print(f"Evidence Citation Rate:         {evidence_citation_rate:.2%}")
        print(f"\nConfidence Calibration:")
        for conf_level, accuracy in calibration.items():
            print(f"  {conf_level.capitalize()} confidence accuracy: {accuracy:.2%}")
        
        self.results['response_quality']['explanation'] = metrics
        return metrics
    
    def evaluate_formatting_parsability(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Evaluate JSON parsing success and schema compliance
        """
        print("\n" + "="*80)
        print("6. FORMATTING & PARSABILITY EVALUATION")
        print("="*80)
        
        json_parse_successes = 0
        schema_compliant = 0
        fallback_needed = 0
        
        required_fields = [
            'base_product_same',
            'upgrades_in_women_product',
            'justified_price_increase_rs',
            'actual_price_difference_rs',
            'verdict',
            'explanation',
            'confidence'
        ]
        
        print(f"\nEvaluating {len(test_cases)} test cases...")
        
        for i, case in enumerate(test_cases, 1):
            print(f"\r  Progress: {i}/{len(test_cases)}", end='', flush=True)
            
            result = self.analyzer.analyze_product_pair(
                case['men_product'],
                case['women_product']
            )
            
            # Check if it's a fallback result
            if result.get('confidence') == 'low' and 'Unable to analyze' in str(result.get('upgrades_in_women_product', [])):
                fallback_needed += 1
            else:
                json_parse_successes += 1
            
            # Check schema compliance
            has_all_fields = all(field in result for field in required_fields)
            if has_all_fields:
                schema_compliant += 1
        
        print()  # New line
        
        total = len(test_cases)
        metrics = {
            'json_success_rate': json_parse_successes / total,
            'schema_compliance_rate': schema_compliant / total,
            'fallback_rate': fallback_needed / total
        }
        
        print("\n" + "-"*80)
        print("FORMATTING & PARSABILITY RESULTS")
        print("-"*80)
        print(f"JSON Parse Success Rate:        {metrics['json_success_rate']:.2%}")
        print(f"Schema Compliance Rate:         {metrics['schema_compliance_rate']:.2%}")
        print(f"Fallback Rate:                  {metrics['fallback_rate']:.2%}")
        
        self.results['response_quality']['formatting'] = metrics
        return metrics
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_clarity_score(self, explanation: str) -> float:
        """
        Calculate clarity score based on heuristics
        Score from 0-5
        """
        score = 3.0  # Start with neutral
        
        # Positive indicators
        positive_keywords = [
            'because', 'since', 'therefore', 'specifically', 'for example',
            'compared to', 'upgrade', 'additional', 'feature', 'difference'
        ]
        for keyword in positive_keywords:
            if keyword in explanation.lower():
                score += 0.2
        
        # Negative indicators
        negative_keywords = [
            'unable to analyze', 'unknown', 'unclear', 'not sure', 'maybe'
        ]
        for keyword in negative_keywords:
            if keyword in explanation.lower():
                score -= 0.5
        
        # Length considerations
        word_count = len(explanation.split())
        if word_count < 10:
            score -= 0.5  # Too short
        elif word_count > 100:
            score -= 0.3  # Too verbose
        
        # Sentence structure
        sentences = explanation.split('.')
        if len(sentences) >= 2:
            score += 0.3  # Multi-sentence is good
        
        return max(0, min(5, score))  # Clamp to 0-5
    
    def _check_evidence_citation(
        self,
        explanation: str,
        men_product: Dict,
        women_product: Dict
    ) -> bool:
        """
        Check if explanation references specific product features
        """
        explanation_lower = explanation.lower()
        
        # Check for mentions of product features
        feature_keywords = [
            'blade', 'strip', 'moisturizing', 'handle', 'scent', 'packaging',
            'ingredient', 'size', 'volume', 'ml', 'g', 'formula'
        ]
        
        # Check descriptions and features
        men_desc = men_product.get('description', '').lower()
        men_features = men_product.get('features', '').lower()
        women_desc = women_product.get('description', '').lower()
        women_features = women_product.get('features', '').lower()
        
        # Does explanation mention anything specific from products?
        for keyword in feature_keywords:
            if keyword in explanation_lower:
                if (keyword in men_desc or keyword in men_features or 
                    keyword in women_desc or keyword in women_features):
                    return True
        
        return False
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def generate_report(self, output_file: str = 'llm_evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': self.analyzer.model,
            'results': self.results
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        if 'classification' in self.results['accuracy']:
            acc = self.results['accuracy']['classification']
            print(f"\n📊 Classification Performance:")
            print(f"   Accuracy: {acc['accuracy']:.2%}")
            print(f"   Precision: {acc['precision']:.2%}")
            print(f"   Recall: {acc['recall']:.2%}")
            print(f"   F1 Score: {acc['f1_score']:.2%}")
        
        if 'inter_run' in self.results['consistency']:
            cons = self.results['consistency']['inter_run']
            print(f"\n🔄 Consistency:")
            print(f"   Verdict Consistency: {cons['average_verdict_consistency']:.2%}")
            print(f"   Verdict Flip Rate: {cons['verdict_flip_rate']:.2%}")
        
        if 'formatting' in self.results['response_quality']:
            fmt = self.results['response_quality']['formatting']
            print(f"\n📝 Response Quality:")
            print(f"   JSON Success Rate: {fmt['json_success_rate']:.2%}")
            print(f"   Schema Compliance: {fmt['schema_compliance_rate']:.2%}")
            print(f"   Fallback Rate: {fmt['fallback_rate']:.2%}")
        
        print("\n" + "="*80)
        
        return report


def create_sample_gold_standard() -> List[Dict]:
    """
    Create sample gold standard test cases
    In production, these should be manually labeled
    """
    return [
        {
            'men_product': {
                'name': "Men's Gillette Razor - Basic",
                'price': 150.00,
                'description': '3-blade razor with basic handle',
                'features': '3 blades, ergonomic handle'
            },
            'women_product': {
                'name': "Women's Gillette Venus Razor",
                'price': 250.00,
                'description': '3-blade razor with moisturizing strips',
                'features': '3 blades, ergonomic handle, 2 moisturizing strips with vitamin E'
            },
            'true_label': 'FAIR_PRICING',
            'actual_upgrades': ['2 moisturizing strips', 'vitamin E'],
            'justified_cost': 80.00
        },
        {
            'men_product': {
                'name': "Men's Dove Soap",
                'price': 50.00,
                'description': 'Basic moisturizing soap',
                'features': '1/4 moisturizing cream'
            },
            'women_product': {
                'name': "Women's Dove Beauty Bar",
                'price': 85.00,
                'description': 'Moisturizing beauty bar',
                'features': '1/4 moisturizing cream, pink color'
            },
            'true_label': 'PINK_TAX',
            'actual_upgrades': [],  # Only difference is color/marketing
            'justified_cost': 0.00
        },
        {
            'men_product': {
                'name': "Men's Deodorant Stick",
                'price': 120.00,
                'description': 'Standard deodorant',
                'features': '48-hour protection, neutral scent'
            },
            'women_product': {
                'name': "Women's Deodorant Stick",
                'price': 180.00,
                'description': 'Deodorant for women',
                'features': '48-hour protection, floral scent'
            },
            'true_label': 'PINK_TAX',
            'actual_upgrades': [],  # Only scent difference
            'justified_cost': 5.00  # Maybe slight scent cost
        }
    ]


def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases for consistency testing"""
    return [
        {
            'men_product': {
                'name': "Men's Face Wash",
                'price': 200.00,
                'description': 'Daily face wash',
                'features': 'Cleanses, removes oil'
            },
            'women_product': {
                'name': "Women's Face Wash",
                'price': 280.00,
                'description': 'Daily face wash for women',
                'features': 'Cleanses, removes oil, brightening agents'
            }
        },
        {
            'men_product': {
                'name': "Men's Shampoo",
                'price': 150.00,
                'description': 'Basic shampoo',
                'features': 'Cleans hair'
            },
            'women_product': {
                'name': "Women's Shampoo",
                'price': 190.00,
                'description': 'Shampoo for women',
                'features': 'Cleans hair, adds shine'
            }
        }
    ]


def create_sample_product_groups() -> List[Dict]:
    """Create sample product groups for similarity testing"""
    return [
        {
            'group_type': 'brand',
            'group_name': 'Gillette',
            'products': [
                {
                    'men_product': {
                        'name': "Gillette Razor 1",
                        'price': 150.00,
                        'description': 'Basic razor',
                        'features': '3 blades'
                    },
                    'women_product': {
                        'name': "Gillette Venus 1",
                        'price': 220.00,
                        'description': 'Venus razor',
                        'features': '3 blades, moisture strip'
                    }
                },
                {
                    'men_product': {
                        'name': "Gillette Razor 2",
                        'price': 200.00,
                        'description': 'Premium razor',
                        'features': '5 blades'
                    },
                    'women_product': {
                        'name': "Gillette Venus 2",
                        'price': 280.00,
                        'description': 'Premium Venus',
                        'features': '5 blades, moisture strip'
                    }
                }
            ]
        }
    ]


def main():
    """Run comprehensive evaluation"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "LLM EVALUATION FOR PINK TAX ANALYZER" + " "*22 + "║")
    print("╚" + "═"*78 + "╝")
    
    # Initialize evaluator
    evaluator = LLMEvaluator()
    
    # Check if Ollama is running
    print("\n🔍 Checking Ollama status...")
    if not evaluator.analyzer.check_ollama_status():
        print("\n❌ Ollama not available. Please start Ollama first.")
        return
    
    print("✅ Ollama is running!\n")
    
    # Create sample data
    print("📝 Preparing test data...")
    gold_standard = create_sample_gold_standard()
    test_cases = create_sample_test_cases()
    product_groups = create_sample_product_groups()
    
    print(f"   Gold standard cases: {len(gold_standard)}")
    print(f"   Test cases: {len(test_cases)}")
    print(f"   Product groups: {len(product_groups)}")
    
    # Run evaluations
    try:
        # 1. Classification Accuracy
        evaluator.evaluate_classification_accuracy(gold_standard)
        
        # 2. Justification Quality
        evaluator.evaluate_justification_quality(gold_standard)
        
        # 3. Inter-run Consistency
        evaluator.evaluate_inter_run_consistency(test_cases, num_runs=3)
        
        # 4. Similar Product Consistency
        evaluator.evaluate_similar_product_consistency(product_groups)
        
        # 5. Explanation Quality
        evaluator.evaluate_explanation_quality(gold_standard + test_cases)
        
        # 6. Formatting & Parsability
        evaluator.evaluate_formatting_parsability(gold_standard + test_cases)
        
        # Generate final report
        evaluator.generate_report('llm_evaluation_report.json')  # Current directory
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        print("\n💡 Next steps:")
        print("   1. Review the evaluation report")
        print("   2. Create a larger gold standard dataset")
        print("   3. Adjust prompts based on weaknesses found")
        print("   4. Re-run evaluation to measure improvements")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()