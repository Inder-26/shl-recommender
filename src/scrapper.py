"""
SHL Product Catalog Scraper
Scrapes Individual Test Solutions with correct URL format
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        return session
    
    def get_page(self, url: str, timeout: int = 60) -> BeautifulSoup:
        """Fetch and parse a page"""
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    
    def _normalize_url(self, href: str) -> str:
        """
        Normalize URL to consistent format with /solutions/products/
        """
        if not href:
            return ""
        
        # Make absolute
        if href.startswith('/'):
            url = f"https://www.shl.com{href}"
        elif href.startswith('http'):
            url = href
        else:
            url = f"https://www.shl.com/{href}"
        
        # Ensure /solutions/products/ format
        if '/products/product-catalog/' in url and '/solutions/products/' not in url:
            url = url.replace('/products/product-catalog/', '/solutions/products/product-catalog/')
        
        # Ensure trailing slash
        if not url.endswith('/'):
            url = url + '/'
        
        return url
    
    def _extract_from_table(self, table) -> list:
        """Extract assessments from a table"""
        assessments = []
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all('td')
            
            if len(cells) >= 4:
                link = cells[0].find('a')
                if not link:
                    continue
                
                name = link.get_text(strip=True)
                href = link.get('href', '')
                
                if not name or not href:
                    continue
                
                # Normalize URL
                assessment_url = self._normalize_url(href)
                
                # Get other fields
                remote = self._check_yes(cells[1]) if len(cells) > 1 else False
                adaptive = self._check_yes(cells[2]) if len(cells) > 2 else False
                test_type = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                
                assessments.append({
                    'name': name,
                    'url': assessment_url,
                    'remote_testing': remote,
                    'adaptive_irt': adaptive,
                    'test_type': test_type,
                    'description': f"SHL assessment for evaluating {name}. Test type: {test_type}"
                })
        
        return assessments
    
    def _check_yes(self, cell) -> bool:
        """Check if cell indicates Yes/True"""
        if not cell:
            return False
        html = str(cell).lower()
        return any(x in html for x in ['-yes', 'check', 'icon--check'])
    
    def scrape_page(self, start: int = 0) -> list:
        """
        Scrape a single page
        First page has 2 tables - we want Table 2 (Individual Test Solutions)
        Other pages have only Individual Test Solutions
        """
        url = f"{self.base_url}?start={start}&type=1"
        
        try:
            soup = self.get_page(url)
            tables = soup.find_all('table')
            
            if not tables:
                return []
            
            if start == 0:
                # First page: use Table 2 (index 1) if available
                if len(tables) >= 2:
                    print(f"  First page: Using Table 2 (Individual Tests)")
                    return self._extract_from_table(tables[1])
                else:
                    return self._extract_from_table(tables[0])
            else:
                # Subsequent pages: use first table
                return self._extract_from_table(tables[0])
                
        except Exception as e:
            print(f"  Error at start={start}: {e}")
            return []
    
    def scrape_all(self) -> list:
        """Scrape all Individual Test Solutions"""
        all_assessments = []
        start = 0
        step = 12
        consecutive_empty = 0
        
        print("\n" + "="*60)
        print("SCRAPING SHL CATALOG - Individual Test Solutions")
        print("="*60 + "\n")
        
        while consecutive_empty < 3:
            print(f"  Page start={start}")
            
            assessments = self.scrape_page(start)
            
            if assessments:
                all_assessments.extend(assessments)
                print(f"    Found {len(assessments)} | Total: {len(all_assessments)}")
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                print(f"    Empty page ({consecutive_empty}/3)")
            
            start += step
            time.sleep(0.5)
            
            if start > 600:
                break
        
        # Remove duplicates by URL
        seen = set()
        unique = []
        for a in all_assessments:
            url_key = a['url'].lower().rstrip('/')
            if url_key not in seen:
                seen.add(url_key)
                unique.append(a)
        
        print("\n" + "="*60)
        print(f"SCRAPING COMPLETE")
        print(f"  Total unique assessments: {len(unique)}")
        print("="*60)
        
        return unique
    
    def save(self, assessments: list, path: str = 'data/raw/shl_catalog.json'):
        """Save to JSON and CSV"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {path}")
        
        import pandas as pd
        df = pd.DataFrame(assessments)
        csv_path = path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Print stats
        print("\nTest Type Distribution:")
        type_counts = {}
        for a in assessments:
            t = a['test_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {t}: {count}")
        
        # Verify URL format
        print("\nSample URLs:")
        for a in assessments[:3]:
            print(f"  {a['url']}")


def main():
    """Main function"""
    print("SHL CATALOG SCRAPER")
    print("="*60)
    
    scraper = SHLScraper()
    assessments = scraper.scrape_all()
    
    if assessments:
        scraper.save(assessments)
        
        if len(assessments) >= 377:
            print(f"\nSUCCESS: Got {len(assessments)} Individual Test Solutions")
        else:
            print(f"\nWARNING: Got {len(assessments)} assessments (expected 377+)")
    else:
        print("\nERROR: No assessments scraped")


if __name__ == "__main__":
    main()