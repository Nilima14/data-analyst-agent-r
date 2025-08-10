import trafilatura
import requests
from bs4 import BeautifulSoup


def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    # Send a request to the website
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)
    return text if text is not None else ""


def scrape_wikipedia_table(url: str) -> list:
    """
    Scrape tables from Wikipedia pages
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all tables with wikitable class
    tables = soup.find_all('table', class_='wikitable')
    
    table_data = []
    for table in tables:
        rows = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            rows.append(headers)
        
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                rows.append(cells)
        
        table_data.append(rows)
    
    return table_data
