import os
from dotenv import load_dotenv
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from llama_parse import LlamaParse
import yfinance as yf

load_dotenv()

# --- 1. The History Hunter (New!) ---
class WebSearchTool(BaseTool):
    name: str = "Search Web for Company History"
    description: str = "Useful for finding 10-year history, milestones, major acquisitions, and controversies. Input: A specific query like 'Reliance Industries major acquisitions list' or 'TCS major failures 2015'."

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

# --- 2. The Financial Fetcher ---
class FinancialsTool(BaseTool):
    name: str = "Fetch Financial Segments"
    description: str = "Fetches business segment data and revenue split. Input: Ticker (e.g., RELIANCE.NS)"

    def _run(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            # We specifically look for business summary and sector info
            info = stock.info
            summary = {
                "Industry": info.get("industry", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Business Summary": info.get("longBusinessSummary", "N/A"),
                "Website": info.get("website", "N/A")
            }
            return f"Company Basic Profile:\n{summary}"
        except Exception as e:
            return f"Error fetching profile: {e}"

# --- 3. The PDF Reader (Source of Truth) ---
class PDFReadTool(BaseTool):
    name: str = "Read Annual Report"
    description: str = "Reads the Annual Report to find Export/Import data and Management Discussion. Input: Filename (e.g., 'reliance.pdf')."

    def _run(self, pdf_name: str) -> str:
        file_path = os.path.join("annual_reports", pdf_name)
        if not os.path.exists(file_path):
            return "File not found."
        
        try:
            parser = LlamaParse(
                result_type="markdown",
                api_key=os.environ.get("LLAMA_CLOUD_API_KEY")
            )
            # We read the first 10k chars which usually contains the Corporate Overview
            documents = parser.load_data(file_path)
            return documents[0].text[:15000] 
        except Exception as e:
            return f"Error reading PDF: {e}"