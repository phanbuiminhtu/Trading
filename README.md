# Vietnam Stock Analysis AI Tool

This project uses two AI agents working together to analyze Vietnam’s stock market.  
One agent provides **technical & financial analysis**, while the second agent collects and summarizes **news from the internet** to support investment decisions.

---

## 🤖 AI Agents

### 🔹 1. Technical & Financial Analysis Agent
- Performs technical analysis on Vietnam stocks  
- Interprets financial data, ratios, and company performance  
- Helps evaluate whether a stock is worth investing in  
- Supports chart, pattern, and trend analysis

### 🔹 2. News Collection Agent
- Collects the latest news from the internet related to Vietnam stock market  
- Summarizes relevant articles for each stock  
- Helps identify catalysts, risks, and market sentiment  
- Uses SerpAPI as the search engine bridge

Together, these two agents generate a more complete and meaningful stock evaluation.

---

## 🚀 Features

- Automated Vietnam stock analysis (technical + fundamental)
- AI-powered news gathering and summarization
- SERP-based web search for latest updates
- Easy to run locally with `.env` configuration
- Modular design — simple to extend or integrate into other systems

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/phanbuiminhtu/Trading.git
cd Trading
```
### 2. Install required libraries
`pip install -r requirements.txt`

### Environment Variables
Create a .env file in the project root and add:
```
GENAI_API_KEY="YOUR_API_KEY"
SERP_API_KEY="YOUR_API_KEY"
```
### 3. Usage

Open the project folder in VS Code, then run `AINewsAgent.py` to collect news URLs and `AIFinancialAnalystAgent.py` to analyze financial data. Simply change the stock symbol inside `main("STOCK")` at the bottom of each file to analyze a different company.

