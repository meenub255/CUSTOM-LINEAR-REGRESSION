import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

from playwright.sync_api import sync_playwright


BASE_URL = "https://housing.com/in/buy/thrissur/house-fid/"
EXTRA_SEED_URLS = [
    "https://housing.com/in/buy/thrissur/resale-house-fid/",
    "https://housing.com/in/buy/thrissur/ready-to-move-house-fid/",
    "https://housing.com/in/buy/thrissur/house-without-brokerage-fid/",
    "https://housing.com/in/buy/thrissur/1bhk-house-fid/",
    "https://housing.com/in/buy/thrissur/2bhk-house-fid/",
    "https://housing.com/in/buy/thrissur/3bhk-house-fid/",
]


def parse_price_lacs(text: str) -> float | None:
    match = re.search(r"₹\s*([\d.]+)\s*(L|Cr)", text.replace(",", ""), re.I)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value * 100 if unit == "cr" else value


def parse_bhk(title: str) -> float | None:
    match = re.search(r"([\d.]+)\s*BHK", title, re.I)
    return float(match.group(1)) if match else None


def parse_locality(title: str) -> str:
    match = re.search(r"Independent House\s+in\s+(.+)", title, re.I)
    if not match:
        return ""
    return re.sub(r",\s*Thrissur$", "", match.group(1), flags=re.I).strip()


def parse_area_sqft(lines: Iterable[str]) -> float | None:
    for line in lines:
        match = re.fullmatch(r"([\d,.]+)\s*sq\.ft", line, re.I)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def parse_avg_price_per_sqft(lines: Iterable[str]) -> str:
    for line in lines:
        if line.lower().startswith("avg. price:"):
            cleaned = re.sub(r"^avg\. price:\s*", "", line, flags=re.I).strip()
            return cleaned.replace("₹", "").strip()
    return ""


def parse_possession_status(lines: list[str]) -> str:
    for index, line in enumerate(lines):
        if line.lower() == "possession status" and index > 0:
            return lines[index - 1]
    return ""


def extract_articles(page) -> list[dict]:
    return page.evaluate(
        """
        () => Array.from(document.querySelectorAll('article')).map((article) => {
          const textLines = article.innerText
            .split('\\n')
            .map((line) => line.trim())
            .filter(Boolean);
          const titleLine = textLines.find((line) => /BHK\\s+Independent House\\s+in/i.test(line)) || '';
          const anchor = Array.from(article.querySelectorAll('a[href]')).find((a) => {
            const href = a.getAttribute('href') || '';
            return href.includes('/in/buy/resale/page/');
          });
          return {
            title: titleLine,
            href: anchor ? anchor.href : '',
            textLines,
          };
        })
        """
    )


def detect_total_pages(page) -> int:
    hrefs = page.evaluate(
        """
        () => Array.from(document.querySelectorAll('a[href]')).map((a) => a.href)
        """
    )
    page_numbers = []
    for href in hrefs:
        match = re.search(r"[?&]page=(\d+)", href)
        if match:
            page_numbers.append(int(match.group(1)))
    return max([1, *page_numbers])


def discover_locality_seed_urls(page) -> list[str]:
    hrefs = page.evaluate(
        """
        () => Array.from(document.querySelectorAll('a[href]')).map((a) => a.href)
        """
    )
    locality_urls = []
    for href in hrefs:
        if re.fullmatch(r"https://housing\\.com/in/buy/thrissur/[^/?]+-gid/house-fid/?", href):
            locality_urls.append(href.rstrip("/") + "/")
    return sorted(set(locality_urls))


def normalize_paginated_url(base_url: str, page_number: int) -> str:
    if page_number == 1:
        return base_url
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}page={page_number}"


def open_listing_page(page, url: str, timeout_ms: int = 30000) -> bool:
    page.goto(url, wait_until="load", timeout=timeout_ms)
    try:
        page.locator("article").first.wait_for(state="visible", timeout=timeout_ms)
        page.wait_for_timeout(1000)
        return True
    except Exception:
        title = page.title()
        body_text = page.locator("body").inner_text(timeout=5000)[:500]
        print(
            f"Skipping page without visible listings: {url}\nTitle: {title}\nPreview: {body_text[:180]}",
            file=sys.stderr,
        )
        return False


def scrape_listings(
    max_pages: int | None = None,
    headless: bool = False,
    include_extra_seeds: bool = True,
) -> list[dict]:
    rows = []
    seen_urls = set()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        if not open_listing_page(page, BASE_URL):
            browser.close()
            return rows

        seed_urls = [BASE_URL]
        if include_extra_seeds:
            seed_urls.extend(EXTRA_SEED_URLS)
            seed_urls.extend(discover_locality_seed_urls(page))

        processed_seed_urls = set()

        for seed_url in seed_urls:
            seed_url = seed_url.rstrip("/") + "/"
            if seed_url in processed_seed_urls:
                continue
            processed_seed_urls.add(seed_url)

            if not open_listing_page(page, seed_url):
                continue

            total_pages = detect_total_pages(page)
            if max_pages is not None:
                total_pages = min(total_pages, max_pages)

            for page_number in range(1, total_pages + 1):
                page_url = normalize_paginated_url(seed_url, page_number)
                if not open_listing_page(page, page_url):
                    continue

                for listing in extract_articles(page):
                    url = listing["href"]
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = listing["title"]
                    lines = listing["textLines"]
                    price_line = next((line for line in lines if line.startswith("₹")), "")

                    row = {
                        "BHK": parse_bhk(title),
                        "Area_SqFt": parse_area_sqft(lines),
                        "City": "Thrissur",
                        "Price_Lacs": parse_price_lacs(price_line),
                        "Locality": parse_locality(title),
                        "Avg_Price_Per_SqFt": parse_avg_price_per_sqft(lines),
                        "Possession_Status": parse_possession_status(lines),
                        "Listing_Title": title,
                        "Listing_URL": url,
                        "Source": "Housing.com",
                        "Page_Number": page_number,
                        "Seed_URL": seed_url,
                    }

                    if row["BHK"] and row["Area_SqFt"] and row["Price_Lacs"] and row["Locality"]:
                        rows.append(row)

        browser.close()

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "BHK",
        "Area_SqFt",
        "City",
        "Price_Lacs",
        "Locality",
        "Avg_Price_Per_SqFt",
        "Possession_Status",
        "Listing_Title",
        "Listing_URL",
        "Source",
        "Page_Number",
        "Seed_URL",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape Thrissur house listings into CSV.")
    parser.add_argument(
        "--output",
        default="thrissur_house_prices.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit for smaller test scrapes.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium in headless mode. Housing.com may block this.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = scrape_listings(max_pages=args.max_pages, headless=args.headless)
    output_path = Path(args.output)
    write_csv(rows, output_path)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
