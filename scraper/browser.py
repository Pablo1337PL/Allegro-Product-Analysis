import logging
import os
import shutil

log = logging.getLogger(__name__)


def _apply_symlink_fix() -> None:
    """
    undetected_geckodriver copies the Firefox directory with shutil.copytree.
    On Arch Linux, /usr/lib/firefox/libonnxruntime.so is a dangling symlink,
    which causes the copy to fail unless symlinks=True is passed.
    """
    try:
        import undetected_geckodriver as uc

        def _patched(self):
            if not os.path.exists(self._undetected_path):
                shutil.copytree(self._firefox_path, self._undetected_path, symlinks=True)
            return self._undetected_path

        uc.Firefox._create_undetected_firefox_directory = _patched
    except ImportError:
        pass


def new_driver():
    """
    Return a Selenium WebDriver backed by undetected-geckodriver (real Firefox).

    Headless mode is intentionally not used — DataDome passes headful Firefox
    but blocks headless. The browser window appears on the desktop during the
    scrape and closes when the run finishes.
    """
    _apply_symlink_fix()

    try:
        import undetected_geckodriver as uc
        from selenium.webdriver.firefox.options import Options
    except ImportError as exc:
        raise RuntimeError(
            "undetected-geckodriver and selenium are required — "
            "run: pip install selenium undetected-geckodriver"
        ) from exc

    opts = Options()
    opts.set_preference("intl.accept_languages", "pl-PL,pl,en-US,en")

    driver = uc.Firefox(options=opts)
    driver.set_window_size(1920, 1080)
    return driver
