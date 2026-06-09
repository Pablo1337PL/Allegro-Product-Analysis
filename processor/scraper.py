import logging
import random
import time

from bs4 import BeautifulSoup

from scraper.browser import _apply_symlink_fix

log = logging.getLogger(__name__)


class DescriptionScraper:
    """
    Selenium-based scraper for Allegro offer description text.
    One browser window is shared across all calls — create once, call many times,
    then call close() when done.
    """

    def __init__(self) -> None:
        _apply_symlink_fix()

        import undetected_geckodriver as uc
        from selenium.webdriver.firefox.options import Options

        opts = Options()
        opts.set_preference("intl.accept_languages", "pl-PL,pl,en-US,en")

        self._driver = uc.Firefox(options=opts)
        self._driver.set_window_size(1920, 1080)

    def scrape_description(self, offer_url: str, delay: float = 2.0) -> str:
        try:
            self._driver.get(offer_url)
            time.sleep(random.uniform(delay * 0.6, delay * 1.4))

            html = self._driver.page_source
            soup = BeautifulSoup(html, "lxml")

            for attrs in [
                {"data-box-name": "description"},
                {"data-testid": "description-section"},
            ]:
                desc = soup.find("div", attrs=attrs)
                if desc:
                    text = desc.get_text(separator=" ", strip=True)
                    if text:
                        return text

        except Exception as exc:
            log.warning("Scrape failed for %s: %s", offer_url, exc)

        return ""

    def close(self) -> None:
        try:
            self._driver.quit()
        except Exception:
            pass
