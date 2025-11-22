# Pink Tax Analytics Dashboard

A comprehensive Business Analytics project that detects and quantifies gender-based pricing discrimination (Pink Tax) in the Indian consumer market through data scraping, statistical analysis, and AI-powered insights.

## 📋 Project Overview

This semester 7 academic project analyzes **2,000 product records** (1,000 matched pairs) across major Indian e-commerce platforms to identify pricing disparities between men's and women's products.

### Key Findings
- **68.7%** of product pairs show women paying higher prices
- Average markup: **₹110.79** (33.6%)
- Categories analyzed: Personal Care, Hygiene, Shaving products
- Retailers: Amazon.in, Flipkart, Nykaa, Myntra

---

##  Features

### 1. **Interactive Web Dashboard** (`index.html`)
- Glassmorphism design with responsive layout
- Real-time data filtering and search
- Chart.js visualizations
- Product comparison interface
- Statistical insights display

### 2. **Data Processing Pipeline** (`data_preprocessing.ipynb`, `something.py`)
- Multi-stage data cleaning and normalization
- Brand standardization
- Price and size unit conversion
- Intelligent product matching algorithm
- Generates matched pairs for analysis

### 3. **Comprehensive EDA** (`eda.py`)
- 15 different visualization types
- Statistical validation (t-tests, Mann-Whitney U tests)
- Category, brand, and retailer analysis
- Exports summary CSV reports

### 4. **Visualization Suite** (`viz.py`)
- Generates 15 professional publication-ready charts
- Price distribution analysis
- Pink tax heatmaps
- Brand and category comparisons
- Comprehensive dashboards

### 5. **AI-Powered Analysis** (`pink_tax_analyzer.py`)
- Uses Llama 3 via Ollama for intelligent product comparison
- Detects unjustified price differences
- Evaluates product upgrades and features
- Structured JSON output for analysis

### 6. **Flask API Server** (`api_server.py`, `pink_tax_chatbot.py`)
- RESTful API for product analysis
- Chatbot interface for queries
- Batch analysis capabilities
- Knowledge base from dataset

---

## 📁 Project Structure

```
pink-tax-analytics/
│
├── index.html                          # Main dashboard (open in browser)
├── pink_tax_dataset_2000_inr_fixed.csv # Final processed dataset
│
├── data_preprocessing.ipynb            # Jupyter notebook for data cleaning
├── something.py                        # Complete data pipeline script
├── eda.py                             # Exploratory Data Analysis
├── viz.py                             # Visualization generation
│
├── pink_tax_analyzer.py               # AI-powered analysis (Llama 3)
├── api_server.py                      # Flask API server
├── pink_tax_chatbot.py                # Chatbot system
│
├── scraper.py                         # Web scraping tool (not used - see below)
└── README.md                          # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Web browser (Chrome, Firefox, Safari)
- (Optional) Ollama with Llama 3 for AI analysis

### Quick Start

1. **View the Dashboard**
   ```bash
   # Simply open index.html in your web browser
   # No installation needed!
   ```

2. **Run Python Analysis Scripts**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   
   # Exploratory Data Analysis
   python eda.py
   
   # Generate visualizations
   python viz.py
   ```

3. **AI-Powered Analysis** (Optional)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull Llama 3
   ollama pull llama3
   
   # Start Ollama server
   ollama serve
   
   # Run analyzer
   python pink_tax_analyzer.py
   ```

4. **Flask API Server** (Optional)
   ```bash
   pip install flask flask-cors requests
   
   python api_server.py
   # Server runs on http://localhost:5000
   ```

---

## 📊 Dataset Information

### Source
Synthetic dataset based on real pink tax research findings from Indian e-commerce platforms.

**Note**: Initial attempts at web scraping faced significant anti-bot protection. The project pivoted to synthetic data generation based on real market research, allowing focus on algorithm development and statistical analysis.

### Dataset Specifications
- **Total Records**: 2,000 products
- **Product Pairs**: 1,000 matched pairs
- **Categories**: Personal Care, Hygiene, Shaving
- **Brands**: 20+ major brands (Dove, Gillette, Nivea, etc.)
- **Retailers**: Amazon.in, Flipkart, Nykaa, Myntra

### Data Fields
- Product ID, Pair ID
- Product Name, Brand, Category, Subcategory
- Gender Target (Men/Women)
- Price (INR), Size, Normalized Price Per Unit
- Retailer, Description, Ingredients

---

## 🔬 Methodology

### 1. Data Collection
Synthetic data generation based on:
- Real pink tax research findings
- Actual market price patterns
- Genuine product specifications

### 2. Data Processing
Five-stage pipeline:
1. **EDA & Audit**: Data quality assessment
2. **Cleaning & Normalization**: Standardize formats
3. **Deduplication**: Remove duplicates
4. **Gender Filtering**: Keep only Men/Women products
5. **Product Matching**: Pair similar products

### 3. Statistical Analysis
- **T-tests**: Compare price means
- **Mann-Whitney U tests**: Non-parametric comparison
- **Effect size**: Cohen's d calculation
- **Confidence intervals**: 95% CI for differences

### 4. AI Analysis (Optional)
- **Model**: Llama 3 via Ollama
- **Approach**: Evaluate product upgrades vs price differences
- **Metrics**: Precision, Recall, Consistency, Accuracy

---

## 📈 Key Insights

### Overall Statistics
- 68.7% of products show women paying more
- Average pink tax: ₹110.79 (33.6% markup)
- Max price difference: ₹500+
- 20+ brands analyzed

### Category Breakdown
| Category | Avg Pink Tax | % Products Affected |
|----------|--------------|---------------------|
| Personal Care | ₹120.50 | 72% |
| Shaving | ₹95.30 | 65% |
| Hygiene | ₹85.20 | 60% |

### Top Offenders
- Highest individual markup: 150%+
- Brands with most pink tax instances identified
- Retailer comparison available in dashboard

---

## 🎯 Usage Examples

### Dashboard
1. Open `index.html` in browser
2. Explore different tabs:
   - **Overview**: Key statistics and insights
   - **Product Search**: Find and compare products
   - **Analytics**: Visual analysis with charts
   - **Detailed View**: Full dataset table

### Python Analysis
```python
# Run EDA
python eda.py
# Outputs: Console analysis + CSV summaries

# Generate visualizations
python viz.py
# Outputs: 15 PNG charts in visualizations/ folder
```

### AI Analysis
```python
from pink_tax_analyzer import PinkTaxAnalyzer

analyzer = PinkTaxAnalyzer()

# Analyze a product pair
result = analyzer.analyze_product_pair(
    men_product={"name": "Men's Razor", "price": 150, ...},
    women_product={"name": "Women's Razor", "price": 250, ...}
)

print(result['verdict'])  # PINK_TAX or FAIR_PRICING
print(result['explanation'])
```

---

## 🧪 Evaluation Framework

### LLM Analyzer Performance
- **Recall**: 100% (catches all pink tax cases)
- **Consistency**: 100% (same results on re-runs)
- **Precision Challenge**: 66.67% (false positive rate)
- **Main Issue**: Systematic undervaluation of product upgrades

### Systematic Bias Identified
The LLM tends to underestimate the value of:
- Moisturizing strips
- Ergonomic improvements
- Additional blades/features

This is documented and addressed through prompt engineering iterations.

---

## 📚 Academic Context

**Course**: Business Analytics (Semester 7)  
**Objective**: Demonstrate comprehensive BA methodologies  
**Skills Showcased**:
- Data collection and processing
- Statistical analysis and hypothesis testing
- Data visualization and storytelling
- Web development (dashboard)
- AI/ML integration (LLM analysis)
- API development

---

## 🔮 Future Enhancements

1. **Real-time Scraping**: Overcome anti-bot protections
2. **Extended Categories**: Electronics, clothing, toys
3. **Temporal Analysis**: Track price changes over time
4. **Mobile App**: Native iOS/Android dashboard
5. **ML Predictions**: Predict pink tax likelihood
6. **Improved LLM Prompts**: Better upgrade valuation

---

## ⚠️ Limitations

1. **Synthetic Data**: Based on research, not live scraping
2. **Scope**: Limited to specific categories
3. **Time Period**: Snapshot, not longitudinal
4. **LLM Precision**: Known bias in upgrade valuation
5. **Scale**: 2,000 products (expandable)

