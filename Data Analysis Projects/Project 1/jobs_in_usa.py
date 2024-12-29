import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Navigate through the structure
    main = soup.find('main', class_='v-main bg-white desktop-main')
    if main:
        container = main.find('div', class_='v-container v-locale--is-ltr')
        if container:
            row = container.find('div', class_='v-row')
            if row:
                column = row.find('div', class_='v-col-lg-7 v-col-xl-5 v-col-9 mobile-column')
                if column:
                    jobs_container = column.find('div', id='jobs-container')
                    if jobs_container:
                        table = jobs_container.find('table', class_='jobs-desktop-table')
                        if table:
                            jobs_data = []
                            for tr in table.find_all('tr'):
                                tds = tr.find_all('td')
                                if len(tds) >= 6:
                                    job_data = {
                                        "Job Title": tds[0].text.strip(),
                                        "Company": tds[1].text.strip(),
                                        "City": tds[2].text.strip(),
                                        "Country": tds[3].text.strip(),
                                        "Remote": tds[4].text.strip(),
                                        "Date": tds[5].text.strip()
                                    }
                                    jobs_data.append(job_data)
                            return jobs_data
    
    print(f"No data found on {url}")
    return []

def scrape_multiple_pages(base_url, num_pages):
    all_jobs_data = []
    
    for page in range(1, num_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Scraping page {page}...")
        page_data = scrape_page(url)
        all_jobs_data.extend(page_data)
        
        # Add a small delay to be respectful to the server
        time.sleep(2)
    
    return all_jobs_data

# Usage
base_url = "https://jobs-in-data.com/"
num_pages = 100
all_data = scrape_multiple_pages(base_url, num_pages)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_data)
df.to_csv("jobs_data.csv", index=False)
print(f"Scraped {len(all_data)} job listings and saved to jobs_data.csv")