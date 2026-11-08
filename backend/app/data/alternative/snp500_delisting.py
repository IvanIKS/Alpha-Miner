"""S&P 500 index-deletion data from S&P Dow Jones Indices press releases.

The canonical page (``spglobal.com/en/press/press-release``) sits behind Akamai
Bot Manager, which blocks non-browser clients with a 403 regardless of headers.
S&P DJI syndicates the identical "... Set to Join S&P 500 ..." announcements
through PR Newswire, which is reachable with a plain HTTP client, so we read the
S&P Dow Jones Indices newsroom there instead.

Two environment notes:
  * PR Newswire articles contain the reconstitution table with columns
    Effective Date | Index Name | Action | Company Name | Ticker | GICS Sector.
  * On machines behind a TLS-inspecting middlebox (corporate proxy / AV HTTPS
    shield), ``requests`` fails cert verification because it uses the bundled
    ``certifi`` store rather than the OS store. ``truststore`` fixes this by
    delegating verification to the OS trust store; we inject it on import if it
    is available (harmless elsewhere).
"""

import html
import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

import requests

try:  # Trust the OS cert store (handles corporate TLS interception).
    import truststore
    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore is an optional convenience
    pass

from app.data.data_provider import WebpageDataProvider

# S&P Dow Jones Indices newsroom on PR Newswire (the %26 is a literal '&').
NEWSROOM_URL = "https://www.prnewswire.com/news/s%26p-dow-jones-indices/"

# Phrase that identifies an S&P 500 index-reconstitution announcement.
JOIN_TITLE_MARKER = "Set to Join S&P 500"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_REQUEST_TIMEOUT = 30

# How many newsroom pages to page through before giving up on an old date.
_MAX_LISTING_PAGES = 40


class SnP500DelistingData:
    def __init__(self,
                 name,
                 symbol,
                 press_release_date,
                 effective_date,
                 gics_sector):
        self.name = name
        self.symbol = symbol
        self.press_release_date = press_release_date
        self.effective_date = effective_date
        self.gics_sector = gics_sector

    def __repr__(self):
        return (f"SnP500DelistingData(name={self.name!r}, "
                f"symbol={self.symbol!r}, "
                f"press_release_date={self.press_release_date!r}, "
                f"effective_date={self.effective_date!r}, "
                f"gics_sector={self.gics_sector!r})")


class _LinkCollector(HTMLParser):
    """Collects every <a> on a listing page as (href, inner_text).

    PR Newswire renders each release as a single anchor whose text holds both
    the publish date (``<small>Oct 30, 2026, 17:51 ET</small>``) and the title,
    so the two never need to be paired across elements.
    """

    def __init__(self):
        super().__init__()
        self.links = []  # list of (href, text)
        self._in_a = False
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._in_a = True
                self._href = href
                self._text = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_a:
            text = " ".join("".join(self._text).split())
            self.links.append((self._href, text))
            self._in_a = False
            self._href = None
            self._text = []

    def handle_data(self, data):
        if self._in_a:
            self._text.append(data)


class _TableExtractor(HTMLParser):
    """Extracts every <table> on a page as a list of rows of cell text."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self._depth = 0
        self._rows = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self._depth == 0:
                self._rows = []
            self._depth += 1
        elif self._depth > 0:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th") and self._row is not None:
                self._cell = []

    def handle_endtag(self, tag):
        if self._depth > 0:
            if tag in ("td", "th") and self._cell is not None:
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell = None
            elif tag == "tr" and self._row is not None:
                self._rows.append(self._row)
                self._row = None
            elif tag == "table":
                self._depth -= 1
                if self._depth == 0:
                    self.tables.append(self._rows)
                    self._rows = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_DATE_FORMATS = (
    "%b %d, %Y",   # Oct 30, 2026
    "%B %d, %Y",   # October 30, 2026
    "%b. %d, %Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
)

# Matches the date substring inside listing/table text.
_DATE_PATTERN = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\.?\s+\d{1,2},\s*\d{4})",
    re.IGNORECASE,
)


def _parse_date_text(text: str):
    text = text.strip().strip(",")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _extract_date(text: str):
    match = _DATE_PATTERN.search(text)
    return _parse_date_text(match.group(0)) if match else None


def _normalize_date(value) -> date:
    """Coerce a date/datetime/ISO-ish string to a ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _parse_date_text(str(value))
    if parsed is None:
        raise ValueError(f"Unrecognized date: {value!r}")
    return parsed


def _header_index(header_row, *keywords):
    """Return the column index whose header contains all keywords (case-insensitive)."""
    for idx, cell in enumerate(header_row):
        low = cell.lower()
        if all(k in low for k in keywords):
            return idx
    return None


class SnP500DelistingDataProvider(WebpageDataProvider):
    def __init__(self):
        super().__init__(NEWSROOM_URL)

    def fetch_data(self, date) -> list[SnP500DelistingData]:
        """Return S&P 500 deletions announced in the release published on ``date``.

        Finds the newsroom entry whose title contains "Set to Join S&P 500" and
        whose publish date equals ``date``, then parses that release's
        reconstitution table for rows that are both a ``Deletion`` and from the
        ``S&P 500`` (MidCap 400 / SmallCap 600 / S&P 100 rows are ignored).
        Returns ``[]`` if no such release exists.
        """
        target = _normalize_date(date)

        article_url = self._find_article_url(target)
        if article_url is None:
            return []

        article_html = self._get(article_url)
        return self._parse_deletions(article_html, press_release_date=target)

    # --- fetching -----------------------------------------------------------

    def _get(self, url: str) -> str:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text

    # --- listing page -------------------------------------------------------

    def _find_article_url(self, target: date):
        """Page through the newsroom (newest-first) to find the release URL."""
        for page in range(1, _MAX_LISTING_PAGES + 1):
            url = self.base_url if page == 1 else f"{self.base_url}?page={page}"
            parser = _LinkCollector()
            parser.feed(self._get(url))

            releases = [
                (href, text)
                for href, text in parser.links
                if "/news-releases/" in href and JOIN_TITLE_MARKER.lower() in text.lower()
            ]
            if not releases:
                # No release cards at all -> past the last populated page.
                if not any("/news-releases/" in href for href, _ in parser.links):
                    return None
                continue

            passed_target = False
            for href, text in releases:
                entry_date = _extract_date(text)
                if entry_date == target:
                    return self._absolute_url(href)
                if entry_date is not None and entry_date < target:
                    passed_target = True

            # Listing is newest-first; once we're past the target date, stop.
            if passed_target:
                return None
        return None

    def _absolute_url(self, href: str) -> str:
        if href.startswith(("http://", "https://")):
            return href
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(self.base_url))
        if href.startswith("//"):
            return urlsplit(self.base_url).scheme + ":" + href
        if href.startswith("/"):
            return origin + href
        return self.base_url.rstrip("/") + "/" + href

    # --- article page -------------------------------------------------------

    def _parse_deletions(self, article_html: str, press_release_date: date):
        """Return S&P 500 'Deletion' rows from the announcement table.

        The announcement table has columns: Effective Date, Index Name, Action,
        Company Name, Ticker, GICS Sector.
        """
        extractor = _TableExtractor()
        extractor.feed(article_html)

        results = []
        for rows in extractor.tables:
            if len(rows) < 2:
                continue
            header = [html.unescape(c) for c in rows[0]]

            i_index = _header_index(header, "index")
            i_name = _header_index(header, "company name")
            i_action = _header_index(header, "action")
            i_ticker = _header_index(header, "ticker")
            i_effective = _header_index(header, "effective")
            i_sector = _header_index(header, "gics") or _header_index(header, "sector")

            # Not the reconstitution table if the key columns are absent.
            if i_index is None or i_action is None or i_ticker is None:
                continue

            needed = [c for c in (i_index, i_action, i_ticker, i_effective, i_sector)
                      if c is not None]
            for row in rows[1:]:
                if max(needed) >= len(row):
                    continue
                cells = [html.unescape(c) for c in row]
                index_name = cells[i_index].lower()
                action = cells[i_action].lower()

                if "deletion" not in action:
                    continue
                # Only the S&P 500 (exclude MidCap 400 / SmallCap 600 / S&P 100).
                if "s&p 500" not in index_name.replace("s&p500", "s&p 500"):
                    continue

                effective = None
                if i_effective is not None:
                    effective = _parse_date_text(cells[i_effective]) or cells[i_effective]
                sector = cells[i_sector] if i_sector is not None else None

                results.append(SnP500DelistingData(
                    name=cells[i_name],
                    symbol=cells[i_ticker],
                    press_release_date=press_release_date,
                    effective_date=effective,
                    gics_sector=sector,
                ))
        return results
