#!/usr/bin/env python3
"""
Gold Standard Dataset Builder
Helper script to create and manage gold standard test cases for LLM evaluation
"""

import json
import csv
from typing import Dict, List
import os

class GoldStandardBuilder:
    """Helps create and manage gold standard test cases"""
    
    def __init__(self, filename: str = 'gold_standard_dataset.json'):
        self.filename = filename
        self.dataset = []
        self.load_existing()
    
    def load_existing(self):
        """Load existing gold standard if it exists"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.dataset = json.load(f)
            print(f"✅ Loaded {len(self.dataset)} existing test cases")
        else:
            print("📝 No existing dataset found. Starting fresh.")
    
    def add_case(
        self,
        men_product: Dict,
        women_product: Dict,
        true_label: str,
        actual_upgrades: List[str],
        justified_cost: float,
        notes: str = ""
    ):
        """
        Add a new test case to the gold standard
        
        Args:
            men_product: Dict with name, price, description, features
            women_product: Dict with name, price, description, features
            true_label: "PINK_TAX" or "FAIR_PRICING"
            actual_upgrades: List of actual product differences
            justified_cost: Fair cost of upgrades in INR
            notes: Optional notes about this case
        """
        
        # Validate
        if true_label not in ['PINK_TAX', 'FAIR_PRICING']:
            raise ValueError("true_label must be 'PINK_TAX' or 'FAIR_PRICING'")
        
        case = {
            'case_id': len(self.dataset) + 1,
            'men_product': men_product,
            'women_product': women_product,
            'true_label': true_label,
            'actual_upgrades': actual_upgrades,
            'justified_cost': justified_cost,
            'notes': notes,
            'price_difference': women_product['price'] - men_product['price']
        }
        
        self.dataset.append(case)
        print(f"✅ Added case #{case['case_id']}")
        
        return case
    
    def save(self):
        """Save gold standard to file"""
        with open(self.filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"💾 Saved {len(self.dataset)} cases to {self.filename}")
    
    def export_to_csv(self, csv_filename: str = 'gold_standard_dataset.csv'):
        """Export to CSV for easier editing"""
        if not self.dataset:
            print("⚠️  No data to export")
            return
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'case_id',
                'men_product_name',
                'men_price',
                'men_description',
                'men_features',
                'women_product_name',
                'women_price',
                'women_description',
                'women_features',
                'true_label',
                'actual_upgrades',
                'justified_cost',
                'price_difference',
                'notes'
            ])
            
            # Data
            for case in self.dataset:
                writer.writerow([
                    case['case_id'],
                    case['men_product'].get('name', ''),
                    case['men_product'].get('price', 0),
                    case['men_product'].get('description', ''),
                    case['men_product'].get('features', ''),
                    case['women_product'].get('name', ''),
                    case['women_product'].get('price', 0),
                    case['women_product'].get('description', ''),
                    case['women_product'].get('features', ''),
                    case['true_label'],
                    '; '.join(case['actual_upgrades']),
                    case['justified_cost'],
                    case['price_difference'],
                    case.get('notes', '')
                ])
        
        print(f"📊 Exported to {csv_filename}")
    
    def import_from_csv(self, csv_filename: str):
        """Import from CSV"""
        if not os.path.exists(csv_filename):
            print(f"❌ File not found: {csv_filename}")
            return
        
        imported = 0
        with open(csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    case = {
                        'case_id': int(row['case_id']),
                        'men_product': {
                            'name': row['men_product_name'],
                            'price': float(row['men_price']),
                            'description': row['men_description'],
                            'features': row['men_features']
                        },
                        'women_product': {
                            'name': row['women_product_name'],
                            'price': float(row['women_price']),
                            'description': row['women_description'],
                            'features': row['women_features']
                        },
                        'true_label': row['true_label'],
                        'actual_upgrades': [u.strip() for u in row['actual_upgrades'].split(';') if u.strip()],
                        'justified_cost': float(row['justified_cost']),
                        'price_difference': float(row['price_difference']),
                        'notes': row.get('notes', '')
                    }
                    
                    self.dataset.append(case)
                    imported += 1
                except Exception as e:
                    print(f"⚠️  Skipped row due to error: {e}")
        
        print(f"✅ Imported {imported} cases from CSV")
    
    def get_statistics(self):
        """Get statistics about the gold standard"""
        if not self.dataset:
            print("📊 No data available")
            return
        
        pink_tax_cases = sum(1 for c in self.dataset if c['true_label'] == 'PINK_TAX')
        fair_cases = len(self.dataset) - pink_tax_cases
        
        avg_price_diff = sum(c['price_difference'] for c in self.dataset) / len(self.dataset)
        avg_justified_cost = sum(c['justified_cost'] for c in self.dataset) / len(self.dataset)
        
        print("\n" + "="*60)
        print("GOLD STANDARD STATISTICS")
        print("="*60)
        print(f"Total cases:           {len(self.dataset)}")
        print(f"Pink tax cases:        {pink_tax_cases} ({pink_tax_cases/len(self.dataset)*100:.1f}%)")
        print(f"Fair pricing cases:    {fair_cases} ({fair_cases/len(self.dataset)*100:.1f}%)")
        print(f"Avg price difference:  ₹{avg_price_diff:.2f}")
        print(f"Avg justified cost:    ₹{avg_justified_cost:.2f}")
        print("="*60 + "\n")
    
    def create_template_csv(self, filename: str = 'gold_standard_template.csv'):
        """Create a template CSV for manual labeling"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header with instructions
            writer.writerow([
                'case_id',
                'men_product_name',
                'men_price',
                'men_description',
                'men_features',
                'women_product_name',
                'women_price',
                'women_description',
                'women_features',
                'true_label (PINK_TAX or FAIR_PRICING)',
                'actual_upgrades (semicolon separated)',
                'justified_cost (in INR)',
                'price_difference (auto-calculated)',
                'notes'
            ])
            
            # Example rows
            writer.writerow([
                1,
                "Men's Basic Razor",
                150,
                "3-blade razor with basic handle",
                "3 blades, ergonomic handle",
                "Women's Venus Razor",
                250,
                "3-blade razor with moisturizing strips",
                "3 blades, ergonomic handle, 2 moisturizing strips, vitamin E",
                "FAIR_PRICING",
                "2 moisturizing strips; vitamin E coating",
                80,
                100,
                "Moisturizing strips justify the price increase"
            ])
            
            writer.writerow([
                2,
                "Men's Dove Soap",
                50,
                "Basic moisturizing soap",
                "1/4 moisturizing cream",
                "Women's Dove Beauty Bar",
                85,
                "Moisturizing beauty bar for women",
                "1/4 moisturizing cream, pink color",
                "PINK_TAX",
                "",
                0,
                35,
                "Only difference is color and 'for women' marketing"
            ])
        
        print(f"📝 Created template: {filename}")
        print("   Fill in the rows and import using import_from_csv()")


def create_comprehensive_gold_standard():
    """Create a comprehensive gold standard with diverse cases"""
    
    builder = GoldStandardBuilder('/home/claude/gold_standard_dataset.json')
    
    # Category 1: Clear Pink Tax Cases (No Upgrades)
    print("\n📍 Adding clear pink tax cases...")
    
    builder.add_case(
        men_product={
            'name': "Men's Basic Deodorant Stick",
            'price': 120.00,
            'description': "Standard deodorant stick for men",
            'features': "48-hour protection, aluminum-free, fresh scent"
        },
        women_product={
            'name': "Women's Basic Deodorant Stick",
            'price': 175.00,
            'description': "Standard deodorant stick for women",
            'features': "48-hour protection, aluminum-free, floral scent"
        },
        true_label='PINK_TAX',
        actual_upgrades=[],
        justified_cost=5.00,  # Maybe slight scent formulation cost
        notes="Only difference is scent - clear pink tax case"
    )
    
    builder.add_case(
        men_product={
            'name': "Men's Cotton T-Shirt",
            'price': 300.00,
            'description': "100% cotton basic tee",
            'features': "100% cotton, crew neck, standard fit"
        },
        women_product={
            'name': "Women's Cotton T-Shirt",
            'price': 450.00,
            'description': "100% cotton basic tee for women",
            'features': "100% cotton, crew neck, fitted cut"
        },
        true_label='PINK_TAX',
        actual_upgrades=[],
        justified_cost=0.00,
        notes="Only difference is cut/fit - same material and quality"
    )
    
    # Category 2: Fair Pricing Cases (Real Upgrades)
    print("📍 Adding fair pricing cases...")
    
    builder.add_case(
        men_product={
            'name': "Men's Gillette Mach3 Razor",
            'price': 200.00,
            'description': "3-blade razor system",
            'features': "3 blades, ergonomic handle, basic design"
        },
        women_product={
            'name': "Women's Gillette Venus Razor",
            'price': 290.00,
            'description': "3-blade razor with moisture strips",
            'features': "3 blades, ergonomic handle, 2 moisture strips with aloe, flexible head, wider blade spacing"
        },
        true_label='FAIR_PRICING',
        actual_upgrades=['2 moisture strips with aloe', 'flexible head', 'wider blade spacing'],
        justified_cost=85.00,
        notes="Real functional upgrades justify price increase"
    )
    
    builder.add_case(
        men_product={
            'name': "Men's Basic Shampoo",
            'price': 150.00,
            'description': "Daily cleansing shampoo",
            'features': "Cleanses hair, removes oil, basic formula"
        },
        women_product={
            'name': "Women's Premium Shampoo",
            'price': 220.00,
            'description': "Cleansing and nourishing shampoo",
            'features': "Cleanses hair, removes oil, argan oil, keratin treatment, UV protection"
        },
        true_label='FAIR_PRICING',
        actual_upgrades=['argan oil', 'keratin treatment', 'UV protection'],
        justified_cost=65.00,
        notes="Additional active ingredients justify higher cost"
    )
    
    # Category 3: Borderline Cases (Minimal Upgrades)
    print("📍 Adding borderline cases...")
    
    builder.add_case(
        men_product={
            'name': "Men's Face Wash",
            'price': 200.00,
            'description': "Daily facial cleanser",
            'features': "Deep cleansing, oil control, salicylic acid"
        },
        women_product={
            'name': "Women's Face Wash",
            'price': 280.00,
            'description': "Daily facial cleanser with brightening",
            'features': "Deep cleansing, oil control, salicylic acid, vitamin C, brightening agents"
        },
        true_label='FAIR_PRICING',
        actual_upgrades=['vitamin C', 'brightening agents'],
        justified_cost=40.00,
        notes="Modest upgrades partially justify 40% price increase"
    )
    
    builder.add_case(
        men_product={
            'name': "Men's Body Lotion",
            'price': 180.00,
            'description': "Moisturizing body lotion",
            'features': "Hydrates skin, non-greasy, light scent"
        },
        women_product={
            'name': "Women's Body Lotion",
            'price': 260.00,
            'description': "Moisturizing body lotion with shimmer",
            'features': "Hydrates skin, non-greasy, floral scent, light shimmer"
        },
        true_label='PINK_TAX',
        actual_upgrades=['light shimmer'],
        justified_cost=15.00,
        notes="Shimmer is cosmetic only, doesn't justify 44% increase"
    )
    
    # Category 4: Different Product Types
    print("📍 Adding diverse product types...")
    
    builder.add_case(
        men_product={
            'name': "Men's Running Shoes",
            'price': 2500.00,
            'description': "Athletic running shoes",
            'features': "Cushioned sole, breathable mesh, standard arch support"
        },
        women_product={
            'name': "Women's Running Shoes",
            'price': 2800.00,
            'description': "Athletic running shoes for women",
            'features': "Cushioned sole, breathable mesh, enhanced arch support, lighter weight"
        },
        true_label='FAIR_PRICING',
        actual_upgrades=['enhanced arch support', 'lighter weight materials'],
        justified_cost=250.00,
        notes="Biomechanical differences require different engineering"
    )
    
    builder.add_case(
        men_product={
            'name': "Men's Multivitamin",
            'price': 500.00,
            'description': "Daily multivitamin supplement - 60 tablets",
            'features': "Essential vitamins and minerals, 60-day supply"
        },
        women_product={
            'name': "Women's Multivitamin",
            'price': 650.00,
            'description': "Daily multivitamin supplement for women - 60 tablets",
            'features': "Essential vitamins and minerals, iron, calcium, folic acid, 60-day supply"
        },
        true_label='FAIR_PRICING',
        actual_upgrades=['added iron', 'added calcium', 'folic acid'],
        justified_cost=120.00,
        notes="Women-specific nutrients justify additional cost"
    )
    
    # Category 5: Extreme Cases
    print("📍 Adding extreme cases...")
    
    builder.add_case(
        men_product={
            'name': "Men's White T-Shirt",
            'price': 250.00,
            'description': "Basic white cotton t-shirt",
            'features': "100% cotton, crew neck"
        },
        women_product={
            'name': "Women's White T-Shirt",
            'price': 600.00,
            'description': "Basic white cotton t-shirt for women",
            'features': "100% cotton, crew neck, pink stitching"
        },
        true_label='PINK_TAX',
        actual_upgrades=[],
        justified_cost=5.00,
        notes="Extreme pink tax - 140% markup for identical product with pink stitching"
    )
    
    # Save everything
    builder.save()
    builder.export_to_csv('/home/claude/gold_standard_dataset.csv')
    builder.get_statistics()
    
    print("\n✅ Gold standard dataset created!")
    print("📁 Files created:")
    print("   - gold_standard_dataset.json")
    print("   - gold_standard_dataset.csv")
    print("\n💡 You can now:")
    print("   1. Edit the CSV file to add more cases")
    print("   2. Import it back using builder.import_from_csv()")
    print("   3. Use it in llm_evaluation.py")


def main():
    """Interactive gold standard builder"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*22 + "GOLD STANDARD DATASET BUILDER" + " "*27 + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\nWhat would you like to do?")
    print("1. Create comprehensive gold standard with examples")
    print("2. Create empty template CSV for manual labeling")
    print("3. Load existing and show statistics")
    print("4. Import from CSV")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        create_comprehensive_gold_standard()
    
    elif choice == '2':
        builder = GoldStandardBuilder()
        builder.create_template_csv('/home/claude/gold_standard_template.csv')
        print("\n✅ Template created!")
        print("📝 Edit 'gold_standard_template.csv' and import it back")
    
    elif choice == '3':
        builder = GoldStandardBuilder('/home/claude/gold_standard_dataset.json')
        builder.get_statistics()
    
    elif choice == '4':
        csv_file = input("Enter CSV filename: ").strip()
        builder = GoldStandardBuilder()
        builder.import_from_csv(csv_file)
        builder.save()
        builder.get_statistics()
    
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()