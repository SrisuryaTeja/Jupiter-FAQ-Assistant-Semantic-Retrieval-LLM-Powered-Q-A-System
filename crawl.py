import asyncio
import json
from playwright.async_api import async_playwright
import nest_asyncio
nest_asyncio.apply()

HELP_URL = "https://community.jupiter.money/c/help/27"
TAGS_URL = "https://community.jupiter.money/tags"


async def get_topic_urls_from_category(page):
    topic_urls = set()
    await page.goto(HELP_URL)

    try:
        await page.wait_for_selector("a.title", timeout=10000)
    except:
        print("No topics found in help category")
        return topic_urls

    previous_height = 0
    while True:
        # Scroll down
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(3)  # Increased wait time

        # Extract current topic URLs
        topics = await page.query_selector_all("a.title")
        for t in topics:
            href = await t.get_attribute("href")
            if href:
                # Handle both absolute and relative URLs
                if href.startswith('http'):
                    topic_urls.add(href)
                else:
                    topic_urls.add("https://community.jupiter.money" + href)

        # Check if new content was loaded
        current_height = await page.evaluate("document.body.scrollHeight")
        if current_height == previous_height:
            break
        previous_height = current_height
    return topic_urls


async def get_tags(page):
    try:
        await page.goto(TAGS_URL)
        await page.wait_for_selector(".discourse-tag.box", timeout=10000)
        tag_links = await page.query_selector_all(".discourse-tag.box")
        tag_urls = []

        for tag in tag_links:
            href = await tag.get_attribute("href")
            if href:
                if href.startswith('http'):
                    tag_urls.append(href)
                else:
                    tag_urls.append("https://community.jupiter.money" + href)

        return tag_urls
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return []


async def get_topic_urls_from_tags(page, tag_urls):
    topic_urls = set()

    for tag_url in tag_urls:
        print(f"Scrolling through tag: {tag_url}")
        await page.goto(tag_url)
        try:
            await page.wait_for_selector("a.title", timeout=10000)
        except:
            print(f"No topics found in tag: {tag_url}")
            continue

        previous_height = 0
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            topics = await page.query_selector_all("a.title")
            for t in topics:
                href = await t.get_attribute("href")
                if href:
                    if href.startswith("http"):
                        topic_urls.add(href)
                    else:
                        topic_urls.add("https://community.jupiter.money" + href)

            current_height = await page.evaluate("document.body.scrollHeight")
            if current_height == previous_height:
                break
            previous_height = current_height

    return topic_urls


async def extract_user_info(topic_body_element):
    """
    Extract user information from a topic body element

    Args:
        topic_body_element: The .topic-body element to search within

    Returns:
        dict: User object with 'name' and 'title' fields
    """
    user_obj = {
        "name": "",
        "title": "User"  # Default value
    }

    try:
        # Extract name from "first full-name" class
        name_element = await topic_body_element.query_selector(".first.full-name a")
        if name_element:
            user_obj["name"] = (await name_element.text_content()).strip()

        # Extract title from "user-title" class (optional)
        title_element = await topic_body_element.query_selector(".user-title")
        if title_element:
            user_obj["title"] = (await title_element.text_content()).strip()

    except Exception as e:
        print(f"Error extracting user info: {e}")

    return user_obj


async def scrape_topic(page, url):
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_selector(".topic-body", timeout=15000)

        # Extract title
        title = ""
        try:
            title_element = await page.wait_for_selector(".fancy-title", timeout=5000)
            title = await title_element.text_content()
        except:
            # Try alternative title selector
            try:
                title_element = await page.query_selector("h1")
                if title_element:
                    title = await title_element.text_content()
            except:
                pass

        # Extract all topic bodies (posts/replies)
        topic_bodies = await page.query_selector_all(".topic-body")
        posts = []

        for index, topic_body in enumerate(topic_bodies):
            try:
                # Extract user info for this post
                user_info = await extract_user_info(topic_body)

                # Extract post content
                post_text = ""
                try:
                    cooked_element = await topic_body.query_selector(".cooked")
                    if cooked_element:
                        post_text = await cooked_element.text_content()
                except:
                    pass

                # Determine post type
                post_type = "question" if index == 0 else "reply"

                if user_info["name"] and post_text.strip():
                    post_data = {
                        "user": user_info,
                        "text": post_text.strip(),
                        "post_type": post_type,
                        "post_index": index
                    }
                    posts.append(post_data)

            except Exception as e:
                print(f"Error extracting post {index} from {url}: {e}")
                continue

        # Extract links from all posts
        links = []
        try:
            link_elements = await page.query_selector_all(".topic-body .cooked a")
            for link in link_elements:
                href = await link.get_attribute("href")
                if href:
                    links.append(href)
        except:
            pass

        # Extract images from all posts
        images = []
        try:
            image_elements = await page.query_selector_all(".topic-body .cooked img")
            for img in image_elements:
                src = await img.get_attribute("src")
                if src:
                    images.append(src)
        except:
            pass

        # Extract tags
        tags = []
        try:
            tag_elements = await page.query_selector_all(".title-wrapper .discourse-tag.box")
            for tag in tag_elements:
                tag_text = await tag.text_content()
                if tag_text:
                    tags.append(tag_text.strip())
        except:
            pass

        # Separate question and replies for backward compatibility
        question_text = ""
        replies = []

        for post in posts:
            if post["post_type"] == "question":
                question_text = post["text"]
            else:
                replies.append({
                    "user": post["user"]["name"],
                    "text": post["text"]
                })

        return {
            "title": title.strip() if title else "",
            "url": url,
            "text": question_text,  # Original question text
            "images": images,
            "links": links,
            "replies": replies,  # Simplified replies for backward compatibility
            "posts": posts,  # Detailed posts with user objects
            "tags": tags
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


async def main(test_mode=False, max_pages=5):
    """
    Main scraping function

    Args:
        test_mode (bool): If True, limits scraping to max_pages for testing
        max_pages (int): Maximum number of pages to scrape in test mode
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            # Stability improvements
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )

        # Set up page with better error handling
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        # Set user agent to avoid detection
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

        try:
            if test_mode:
                print(f"🧪 TEST MODE: Limiting to {max_pages} pages")

            print("Fetching topic URLs from category /c/help/27...")
            help_urls = await get_topic_urls_from_category(page)
            print(f"Found {len(help_urls)} URLs from help category")

            print("Fetching tag URLs from /tags...")
            tag_urls = await get_tags(page)
            print(f"Found {len(tag_urls)} tag URLs")

            print("Fetching topic URLs from tag pages...")
            tag_topic_urls = await get_topic_urls_from_tags(page, tag_urls)
            print(f"Found {len(tag_topic_urls)} URLs from tags")

            all_urls = list(help_urls.union(tag_topic_urls))

            # Apply test mode limitation
            if test_mode:
                all_urls = all_urls[:max_pages]
                print(f"🧪 TEST MODE: Limited to {len(all_urls)} pages")
            else:
                print(f"Total unique topic URLs found: {len(all_urls)}")

            results = []
            failed_urls = []

            for i, url in enumerate(all_urls):
                print(f"Scraping {i + 1}/{len(all_urls)}: {url}")
                try:
                    result = await scrape_topic(page, url)
                    if result:
                        results.append(result)
                    else:
                        failed_urls.append(url)
                except Exception as e:
                    print(f"Failed to scrape {url}: {e}")
                    failed_urls.append(url)

                # Rate limiting
                if i % 10 == 0:
                    await asyncio.sleep(2)

        except Exception as e:
            print(f"Error in main execution: {e}")

        finally:
            await browser.close()

        # Save results with test mode indicator
        output_filename = "faq_data_test.json" if test_mode else "faq_data_raw.json"
        failed_filename = "failed_urls_test.json" if test_mode else "failed_urls.json"

        if results:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Successfully scraped {len(results)} topics → {output_filename}")

        if failed_urls:
            with open(failed_filename, "w", encoding="utf-8") as f:
                json.dump(failed_urls, f, ensure_ascii=False, indent=2)
            print(f"Failed to scrape {len(failed_urls)} URLs → {failed_filename}")

# Test function


async def run_test():
    """Run scraper in test mode with only 5 pages"""
    print("=" * 50)
    print("🧪 RUNNING TEST MODE - 5 PAGES ONLY")
    print("=" * 50)
    await main(test_mode=True, max_pages=5)

if __name__ == "__main__":
    import sys

    # Check if test mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(run_test())
    else:
        # Run full scraping
        print("=" * 50)
        print("🚀 RUNNING FULL SCRAPING MODE")
        print("=" * 50)
        asyncio.run(main())
