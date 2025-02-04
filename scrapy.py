import requests
from bs4 import BeautifulSoup
import json

def scrape_url(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text content from the page
        page_content = soup.get_text(separator='\n', strip=True)

        # Return the scraped content
        return page_content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

def save_content(content, filename):
    try:
        # Save the content to a file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Content saved to {filename}")
    except IOError as e:
        print(f"Error saving the file: {e}")

def main():
    # URL to scrape
    url = input("Enter the URL to scrape: ")

    # Scrape the content
    scraped_content = scrape_url(url)

    if scraped_content:
        # Save the scraped content to a file
        filename = "scraped_content.txt"
        save_content(scraped_content, filename)

if __name__ == "__main__":
    main()