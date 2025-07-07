from mcp.server.fastmcp import FastMCP
import yfinance as yf
# import pandas as pd

mcp = FastMCP(name="Finance MCP Server")

@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'][0]
        return f"The latest closing price of {ticker.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {e}"

@mcp.tool()
def calculate_interest(principal: float, rate: float, years: float) -> str:
    """Calculate simple interest."""
    try:
        interest = principal * (rate / 100) * years
        total = principal + interest
        return f"Simple interest on ${principal} at {rate}% for {years} years is ${interest:.2f}. Total amount: ${total:.2f}."
    except Exception as e:
        return f"Error calculating interest: {e}"

@mcp.tool()
def retrieve_compliance_docs(query: str) -> str:
    """Stub for compliance document retrieval."""
    return f"Compliance documents related to '{query}' would be retrieved here."

@mcp.tool()
def compare_stock(ticker1: str, ticker2: str) -> str:
    """
    Compare the latest closing prices of two stocks.
    Returns a summary string.
    """
    try:
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        hist1 = stock1.history(period="1d")
        hist2 = stock2.history(period="1d")
        price1 = hist1['Close'][0]
        price2 = hist2['Close'][0]
        diff = price1 - price2
        diff_pct = (diff / price2) * 100 if price2 != 0 else 0

        return (f"Latest closing price of {ticker1.upper()}: ${price1:.2f}\n"
                f"Latest closing price of {ticker2.upper()}: ${price2:.2f}\n"
                f"Difference: ${diff:.2f} ({diff_pct:.2f}%)")
    except Exception as e:
        return f"Error comparing stocks {ticker1} and {ticker2}: {e}"

@mcp.tool()
def historical_data(ticker: str, period: str = "1mo") -> str:
    """
    Fetch historical stock data for a ticker over a given period.
    Period examples: '1d', '5d', '1mo', '3mo', '1y', '5y', 'max'
    Returns a summary of the data.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return f"No historical data found for {ticker} over period '{period}'."
        
        # Prepare a concise summary (e.g., last 5 rows)
        summary = hist[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_string()
        return f"Historical data for {ticker.upper()} over period '{period}':\n{summary}"
    except Exception as e:
        return f"Error fetching historical data for {ticker}: {e}"

if __name__ == "__main__":
    mcp.run()
