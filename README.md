#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PLASU Scholar Profile Scraper
============================
A tool for analyzing research output and interests at Plateau State University
through Google Scholar profiles.

License: MIT
Author: Emmanuel Nnadi
"""

import logging
import re
import string
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from wordcloud import WordCloud

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

def random_delay(min_seconds=2, max_seconds=5):
    """Introduce a random delay to mimic human behavior."""
    time.sleep(random.uniform(min_seconds, max_seconds))

class ScholarProfileScraper:
    """A web scraper for collecting and analyzing Google Scholar profiles."""

    def __init__(self, headless: bool = True) -> None:
        """
        Initialize the scraper with browser settings.

        Args:
            headless (bool): Whether to run browser in headless mode
        """
        self.setup_driver(headless)
        self.verified_domains = ['@plasu.edu.ng']
        self.seen_urls = set()
        self.custom_stopwords = {
            'research', 'study', 'analysis', 'using', 'based',
            'approach', 'towards', 'case', 'development',
            'evaluation', 'method'
        }

    def setup_driver(self, headless: bool) -> None:
        """
        Configure and initialize the Chrome WebDriver.

        Args:
            headless (bool): Whether to run browser in headless mode
        """
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless=new')  # Use the latest headless mode
        else:
            chrome_options.add_argument('--start-maximized')  # For debugging

        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/112.0.0.0 Safari/537.36'
        )

        # Optional: Disable extensions and automation flags
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        })
                    """
                },
            )
            self.wait = WebDriverWait(self.driver, 15)
            logging.info("Chrome WebDriver initialized successfully.")
        except WebDriverException as e:
            logging.error(f"Failed to initialize WebDriver: {e}")
            raise

    def is_plasu_affiliated(self, card) -> bool:
        """
        Check if a profile card belongs to a PLASU affiliate.

        Args:
            card: Selenium WebElement representing a profile card

        Returns:
            bool: True if affiliated with PLASU, False otherwise
        """
        try:
            affiliation = card.find_element(By.CLASS_NAME, 'gs_ai_aff').text.lower()
            email = card.find_element(By.CLASS_NAME, 'gs_ai_eml').text.lower()

            plasu_indicators = [
                'plateau state university',
                'plasu',
                '@plasu.edu.ng',
                'bokkos'
            ]

            return any(indicator in affiliation or indicator in email
                      for indicator in plasu_indicators)
        except NoSuchElementException:
            logging.warning("Affiliation or email element not found in profile card.")
            return False
        except Exception as e:
            logging.error(f"Error checking PLASU affiliation: {e}")
            return False

    def find_plasu_profiles(self) -> List[Dict[str, Any]]:
        """
        Search for and collect PLASU scholar profiles.

        Returns:
            List[Dict[str, Any]]: List of profile data dictionaries
        """
        logging.info("Starting search for PLASU profiles...")

        search_terms = [
            "plateau state university",
            "plasu.edu.ng",
            "PLASU Bokkos"
        ]

        all_profiles = []

        for search_term in search_terms:
            search_url = (
                f"https://scholar.google.com/citations"
                f"?view_op=search_authors&hl=en&mauthors={search_term.replace(' ', '+')}"
            )

            self.driver.get(search_url)
            logging.info(f"Navigated to search URL: {search_url}")
            random_delay(3, 6)  # Random delay after navigation

            try:
                self.wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 'gsc_1usr'))
                )
                logging.info(f"Profiles loaded for search term: {search_term}")

                page = 1
                while True:
                    profile_cards = self.driver.find_elements(By.CLASS_NAME, 'gsc_1usr')

                    logging.info(f"Processing page {page} with {len(profile_cards)} profiles.")
                    for card in profile_cards:
                        try:
                            if not self._process_profile_card(card, all_profiles):
                                continue
                            random_delay(1, 2)  # Short delay between profile processing
                        except Exception as e:
                            logging.error(f"Error processing profile card: {e}")
                            continue

                    if not self._navigate_to_next_page(profile_cards):
                        break

                    page += 1
                    random_delay(2, 4)  # Delay before next page

            except TimeoutException:
                logging.warning(f"No profiles found for search term: {search_term}")
                continue
            except WebDriverException as e:
                logging.error(f"WebDriverException during profile search: {e}")
                continue

        logging.info(f"Found total of {len(all_profiles)} PLASU profiles")
        return all_profiles

    def _process_profile_card(self, card, all_profiles: List) -> bool:
        """
        Process a single profile card and add to collection if valid.

        Args:
            card: Selenium WebElement representing a profile card
            all_profiles: List to store profile data

        Returns:
            bool: True if profile was processed successfully
        """
        try:
            profile_url = card.find_element(
                By.CSS_SELECTOR, '.gs_ai_name a'
            ).get_attribute('href')
        except NoSuchElementException:
            logging.warning("Profile URL not found.")
            return False

        if profile_url in self.seen_urls:
            logging.info(f"Duplicate profile found: {profile_url}")
            return False

        if not self.is_plasu_affiliated(card):
            logging.info(f"Non-PLASU profile skipped: {profile_url}")
            return False

        try:
            name = card.find_element(By.CLASS_NAME, 'gs_ai_name').text
        except NoSuchElementException:
            name = "N/A"

        try:
            citations = card.find_element(By.CLASS_NAME, 'gs_ai_cby').text
            citations_count = int(re.search(r'Cited by (\d+)', citations).group(1)) if citations else 0
        except (NoSuchElementException, AttributeError):
            citations_count = 0

        all_profiles.append({
            'name': name,
            'profile_url': profile_url,
            'citations': citations_count
        })

        self.seen_urls.add(profile_url)
        logging.info(f"Found PLASU profile: {name}")
        return True

    def _navigate_to_next_page(self, current_cards) -> bool:
        """
        Attempt to navigate to the next page of results.

        Args:
            current_cards: Current page's profile cards

        Returns:
            bool: True if navigation successful, False otherwise
        """
        try:
            next_button = self.driver.find_element(
                By.XPATH, "//button[@aria-label='Next']"
            )

            if next_button.is_enabled():
                next_button.click()
                self.wait.until(EC.staleness_of(current_cards[0]))
                logging.info("Navigated to the next page.")
                return True
            else:
                logging.info("Next button is disabled. End of pages.")
        except (NoSuchElementException, ElementClickInterceptedException) as e:
            logging.info(f"Navigation ended: {e}")

        return False

    def get_research_interests(self, profile_url: str) -> List[str]:
        """
        Extract research interests and publication titles from a profile.

        Args:
            profile_url (str): URL of the scholar's profile

        Returns:
            List[str]: List of research interests and publication titles
        """
        interests = []
        try:
            self.driver.get(profile_url)
            self.wait.until(EC.presence_of_element_located((By.ID, 'gsc_prf_i')))
            logging.info(f"Loaded profile page: {profile_url}")
            random_delay(2, 5)

            try:
                interests_element = self.driver.find_element(By.ID, 'gsc_prf_int')
                extracted_interests = [interest.strip() for interest in interests_element.text.split(',')]
                interests.extend(extracted_interests)
                logging.info(f"Extracted research interests: {extracted_interests}")
            except NoSuchElementException:
                logging.warning(f"Research interests not found for {profile_url}")

            try:
                publication_elements = self.driver.find_elements(
                    By.CLASS_NAME, 'gsc_a_at'
                )
                extracted_publications = [pub.text for pub in publication_elements]
                interests.extend(extracted_publications)
                logging.info(f"Extracted publication titles: {len(extracted_publications)} items")
            except NoSuchElementException:
                logging.warning(f"Publications not found for {profile_url}")

        except TimeoutException:
            logging.error(f"Timeout while loading profile: {profile_url}")
        except WebDriverException as e:
            logging.error(f"WebDriverException occurred: {e}")
        except Exception as e:
            logging.error(f"Unexpected error getting research interests: {e}")

        return interests

    def get_publications(self, profile_url: str) -> List[Dict[str, Any]]:
        """
        Get publication data for a profile.

        Args:
            profile_url (str): URL of the scholar's profile

        Returns:
            List[Dict[str, Any]]: List of publication data dictionaries
        """
        publications = []
        current_year = datetime.now().year

        try:
            self.driver.get(profile_url)
            self.wait.until(EC.presence_of_element_located((By.ID, 'gsc_a_b')))
            logging.info(f"Loaded publications section for: {profile_url}")
            random_delay(2, 4)

            while self._load_more_publications():
                pass

            pub_rows = self.driver.find_elements(By.CLASS_NAME, 'gsc_a_tr')
            logging.info(f"Found {len(pub_rows)} publications.")

            for row in pub_rows:
                pub_data = self._extract_publication_data(row, current_year)
                if pub_data:
                    publications.append(pub_data)
                random_delay(0.5, 1.5)  # Short delay between processing publications

        except TimeoutException:
            logging.error(f"Timeout while loading publications for: {profile_url}")
        except WebDriverException as e:
            logging.error(f"WebDriverException occurred: {e}")
        except Exception as e:
            logging.error(f"Error getting publications: {e}")

        return publications

    def _load_more_publications(self) -> bool:
        """
        Click 'Show more' button to load additional publications.

        Returns:
            bool: True if more publications were loaded
        """
        try:
            show_more = self.wait.until(
                EC.element_to_be_clickable((By.ID, 'gsc_bpf_more'))
            )

            if show_more.is_displayed():
                show_more.click()
                self.wait.until(EC.staleness_of(show_more))
                logging.info("Clicked 'Show more' to load additional publications.")
                random_delay(1, 2)
                return True

        except (TimeoutException, ElementClickInterceptedException):
            logging.info("No more publications to load or 'Show more' button not found.")

        return False

    def _extract_publication_data(
        self, row, current_year: int
    ) -> Dict[str, Any]:
        """
        Extract publication data from a single row.

        Args:
            row: Selenium WebElement representing a publication row
            current_year (int): Current year for filtering

        Returns:
            Dict[str, Any]: Publication data dictionary
        """
        try:
            year_text = row.find_element(By.CLASS_NAME, 'gsc_a_y').text.strip()

            if year_text and year_text.isdigit():
                year = int(year_text)

                if year >= current_year - 4:
                    citations_text = row.find_element(
                        By.CLASS_NAME, 'gsc_a_c'
                    ).text.strip()

                    citations = int(citations_text) if citations_text.isdigit() else 0

                    return {
                        'year': year,
                        'citations': citations
                    }

        except NoSuchElementException:
            logging.warning("Year or citations element not found in publication row.")
        except ValueError:
            logging.warning("Non-integer value found for citations.")

        return None

    def process_all_profiles(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Process all PLASU profiles and collect research areas.

        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: Profile data and research areas
        """
        profiles = self.find_plasu_profiles()
        all_research_areas = []

        if not profiles:
            logging.warning("No PLASU profiles found!")
            return [], []

        for idx, profile in enumerate(profiles, start=1):
            logging.info(f"Processing profile {idx}/{len(profiles)}: {profile['name']}")

            publications = self.get_publications(profile['profile_url'])
            profile['publications'] = publications

            research_interests = self.get_research_interests(profile['profile_url'])
            all_research_areas.extend(research_interests)
            profile['research_interests'] = research_interests

            random_delay(2, 5)  # Delay between profile processing

        return profiles, all_research_areas

    def create_wordcloud(self, research_areas: List[str], base_filename: str) -> None:
        """
        Create and save a word cloud visualization of research areas.

        Args:
            research_areas (List[str]): List of research areas and titles
            base_filename (str): Base name for output file
        """
        text = ' '.join(research_areas).lower()
        stop_words = set(stopwords.words('english')).union(self.custom_stopwords)

        tokens = [
            word for word in word_tokenize(text)
            if word not in string.punctuation
            and word not in stop_words
            and len(word) > 2
        ]

        processed_text = ' '.join(tokens)

        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            max_words=100,
            collocations=False,
            contour_width=3,
            contour_color='steelblue'
        ).generate(processed_text)

        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('PLASU Research Areas Word Cloud', pad=20, size=20)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(
            f'{base_filename}_wordcloud_{timestamp}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logging.info("Word cloud created and saved.")

    def save_data(
        self,
        data: List[Dict[str, Any]],
        research_areas: List[str],
        base_filename: str = 'plasu_scholar_data'
    ) -> None:
        """
        Save all collected data and generate visualizations.

        Args:
            data (List[Dict[str, Any]]): Profile data
            research_areas (List[str]): Research areas data
            base_filename (str): Base name for output files
        """
        if not data:
            logging.warning("No data to save")
            return

        self.create_publication_citation_graph(data, base_filename)
        self.create_wordcloud(research_areas, base_filename)

        df_profiles = pd.DataFrame(data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_profiles.to_csv(f'{base_filename}_profiles_{timestamp}.csv', index=False)

        logging.info(f"Data saved with timestamp: {timestamp}")

    def create_publication_citation_graph(self, data: List[Dict[str, Any]], base_filename: str) -> None:
        """
        Create and save a publication citation graph visualization.

        Args:
            data (List[Dict[str, Any]]): Profile data containing publications
            base_filename (str): Base name for output file
        """
        citation_data = {}

        for profile in data:
            for pub in profile.get('publications', []):
                year = pub.get('year')
                citations = pub.get('citations', 0)
                if year:
                    citation_data.setdefault(year, 0)
                    citation_data[year] += citations

        if not citation_data:
            logging.warning("No citation data available to plot.")
            return

        # Sort the data by year
        sorted_years = sorted(citation_data.keys())
        sorted_citations = [citation_data[year] for year in sorted_years]

        plt.figure(figsize=(12, 6))
        plt.plot(sorted_years, sorted_citations, marker='o', linestyle='-', color='b')
        plt.xlabel('Year')
        plt.ylabel('Total Citations')
        plt.title('Total Citations Over the Years for PLASU Scholars')
        plt.grid(True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(
            f'{base_filename}_citation_graph_{timestamp}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logging.info("Publication citation graph created and saved.")

    def close(self) -> None:
        """
        Close the Selenium WebDriver.
        """
        self.driver.quit()
        logging.info("WebDriver closed.")

def main():
    scraper = ScholarProfileScraper(headless=True)
    try:
        profiles, research_areas = scraper.process_all_profiles()
        scraper.save_data(profiles, research_areas)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()

