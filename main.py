import re, os, tiktoken
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, JsonCssExtractionStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Load environment variables from .env file
load_dotenv()
app = FastAPI()

def get_llm_config():
    """Lấy cấu hình LLM từ biến môi trường với fallback tự động"""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    
    # Thử provider được chỉ định trước
    config = _get_provider_config(provider)
    if config:
        print(f"[INFO] Using LLM provider: {provider}")
        return config
    
    # Nếu provider được chỉ định không có key, thử các provider khác
    print(f"[WARN] No API key for {provider}, trying other providers...")
    
    # Thứ tự ưu tiên fallback
    fallback_providers = ["gemini", "openai", "claude"]
    if provider in fallback_providers:
        fallback_providers.remove(provider)
    
    for fallback_provider in fallback_providers:
        config = _get_provider_config(fallback_provider)
        if config:
            print(f"[INFO] Fallback to {fallback_provider} (no key for {provider})")
            return config
    
    # Nếu không có provider nào có key
    print(f"[ERROR] No API key found for any provider")
    return None

def _get_provider_config(provider):
    """Lấy cấu hình cho provider cụ thể"""
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        return {
            "provider": "openai",
            "url": f"{base_url}/v1/chat/completions",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            "model": model
        }
    
    elif provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        return {
            "provider": "claude",
            "url": f"{base_url}/v1/messages",
            "headers": {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            "model": model
        }
    
    else:  # gemini (default)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        return {
            "provider": "gemini",
            "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
            "headers": {"Content-Type": "application/json"},
            "model": model,
            "api_key": api_key
        }

async def clean_and_format_chunk(md_input: str, require_ai: bool = False) -> str:
    """Làm sạch và format nội dung trong Markdown chunk bằng LLM.
    - Khi require_ai=True: bắt buộc lấy kết quả từ AI (retry nhiều lần). Nếu không có kết quả, raise lỗi.
    - Khi require_ai=False: cố gắng dùng AI, nếu lỗi sẽ trả về nội dung gốc.
    """
    if not md_input or len(md_input.strip()) < 30:
        return md_input

    try:
        import aiohttp, asyncio

        # Lấy cấu hình LLM từ biến môi trường
        config = get_llm_config()
        if not config:
            msg = f"[WARN] No API key found for any LLM provider"
            print(msg)
            if require_ai:
                raise RuntimeError(f"No API key available for any LLM provider")
            return md_input
        
        provider = config["provider"]

        url = config["url"]
        headers = config["headers"]
        model = config["model"]

        short_prompt = """Clean this Markdown chunk. Output ONLY the cleaned content (no code fences, no comments).

Delete: loaders, nav/breadcrumb/pagination/read-more/see-more, social/share widgets, copyright notices, translator helpers, CAPTCHA/anti-bot pages.

Preserve: real content; [links](url), ![images](url), [video](url), plain URLs, code blocks & inline code.

Headings: max one H1; enforce proper H2→H4 hierarchy (promote/demote as needed).

Formatting: collapse excess whitespace; keep meaningful blocks together, keep contact form (BUT DELETE form submit buttons); remove extra spaces around between words.

Listings: only “- Title” + nearest read-more link.

Convert (e.g., pricing, plan, process,...) blocks into a Markdown table; infer column headers; add a leading auto-numbered column; preserve links; use <br> for in-cell breaks; no hallucination.

CHUNK:"""

        prompt = f"{short_prompt}\n{md_input}"

        # Tạo payload theo provider
        if provider == "gemini":
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 20000
                }
            }
            # Thêm API key vào URL cho Gemini
            url = f"{url}?key={config['api_key']}"
            
        elif provider == "openai":
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 20000
            }
            
        elif provider == "claude":
            payload = {
                "model": model,
                "max_tokens": 20000,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

        timeout = aiohttp.ClientTimeout(total=300)
        max_ai_retries = 3
        backoff_ms = [0, 500, 1000]

        for attempt in range(max_ai_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            print(f"[WARN] AI request status={response.status} on attempt {attempt+1}/{max_ai_retries}")
                            if attempt < max_ai_retries - 1:
                                await asyncio.sleep(backoff_ms[attempt] / 1000)
                                continue
                            if require_ai:
                                raise RuntimeError(f"AI request failed with status {response.status}")
                            return md_input

                        result = await response.json()
                        
                        # Parse response theo provider
                        if provider == "gemini":
                            if "candidates" in result and result["candidates"]:
                                normalized = (result["candidates"][0]
                                                .get("content", {})
                                                .get("parts", [{}])[0]
                                                .get("text", "")).strip()
                        elif provider == "openai":
                            if "choices" in result and result["choices"]:
                                normalized = result["choices"][0].get("message", {}).get("content", "").strip()
                        elif provider == "claude":
                            if "content" in result and result["content"]:
                                normalized = result["content"][0].get("text", "").strip()
                        else:
                            normalized = ""

                        if normalized:
                            # Clean up markdown code blocks
                            if normalized.startswith('```markdown'):
                                normalized = normalized.replace('```markdown', '', 1)
                            if normalized.startswith('```'):
                                normalized = normalized.replace('```', '', 1)
                            if normalized.endswith('```'):
                                normalized = normalized[:-3]
                            normalized = normalized.strip()

                            if normalized:
                                return normalized
                            else:
                                print(f"[WARN] Empty AI content on attempt {attempt+1}/{max_ai_retries}")
                                if attempt < max_ai_retries - 1:
                                    await asyncio.sleep(backoff_ms[attempt] / 1000)
                                    continue
                                if require_ai:
                                    raise RuntimeError("AI returned empty content")
                                return md_input
                        else:
                            print(f"[WARN] No valid response from {provider} on attempt {attempt+1}/{max_ai_retries}")
                            if attempt < max_ai_retries - 1:
                                await asyncio.sleep(backoff_ms[attempt] / 1000)
                                continue
                            if require_ai:
                                raise RuntimeError(f"AI response has no valid content")
                            return md_input
            except Exception as e:
                print(f"[WARN] AI call error on attempt {attempt+1}/{max_ai_retries}: {e}")
                if attempt < max_ai_retries - 1:
                    await asyncio.sleep(backoff_ms[attempt] / 1000)
                    continue
                if require_ai:
                    raise
                return md_input
        # Should never reach here
        return md_input
    except Exception as e:
        print(f"[WARN] Heading normalization failed: {str(e)}.")
        if require_ai:
            raise
        return md_input

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def test():
    config = get_llm_config()
    return {
        "message": "Crawl API is working!", 
        "status": "ok", 
        "current_llm": config["provider"],
        "endpoints": {"POST /": "crawl website"},
        "how_to_change_llm": "Set LLM_PROVIDER environment variable to: gemini, openai, or claude"
    }

@app.post("/")
async def crawl(request_data: dict, request: Request):
    try:
        url = request_data.get("url", "https://vnexpress.net/chu-tich-quoc-hoi-nghien-cuu-sua-mot-so-dieu-cua-hien-phap-trong-thang-3-4856862.html")
        # Giới hạn tham số để tránh crawl quá lớn
        max_pages = min(request_data.get("max_pages", 10), 30)  # Theo client default
        max_depth = min(request_data.get("max_depth", 3), 3)   # Tăng depth lên 3
        force_engine = request_data.get("force_engine", None)
        use_fit_markdown = request_data.get("use_fit_markdown", True)  # Use fit_markdown for cleaner content
        extraction_schema = request_data.get("extraction_schema", None)  # Optional JSON CSS schema

        segment = request_data.get("segment", False)
        segment_size = request_data.get("segment_size", 1200)
        word_count_threshold = request_data.get("word_count_threshold", 5)  # Giảm threshold để giữ nội dung ngắn
        wait_for = request_data.get("wait_for", None)  # e.g. "networkidle" or "css:.main-loaded" or "js:() => window.loaded"
        log_pre_llm = request_data.get("log_pre_llm", False)
        include_pre_llm_in_response = request_data.get("include_pre_llm_in_response", False)
        use_normalize_headings = request_data.get("normalize_headings", True)  # Bật chuẩn hóa heading mặc định
        require_ai = request_data.get("require_ai", True)  # Bắt buộc có kết quả AI để tiếp tục
        preserve_content = request_data.get("preserve_content", True)  # Giữ nguyên nội dung, không lọc quá mức
        
        # Legacy chunking/lazy options (no-op now)
        use_chunking = request_data.get("use_chunking", True)  # Bật mặc định theo client
        chunking_type = request_data.get("chunking_type", "regex")
        chunk_size = request_data.get("chunk_size", 768)  # Theo client default
        chunk_overlap = request_data.get("chunk_overlap", 768)  # Theo client default
        enable_lazy_load = request_data.get("enable_lazy_load", True)
        scroll_delay = request_data.get("scroll_delay", 2000)
        max_scrolls = request_data.get("max_scrolls", 1)
        
        print(f"[INFO] Starting crawl: {url} | Max Pages: {max_pages} | Max Depth: {max_depth} | Use Fit Markdown: {use_fit_markdown}")
        print(f"[INFO] Segment: {segment} (size={segment_size}) | wait_for={wait_for} | wct={word_count_threshold}")
        
        # Thêm timeout cho crawl operation (5 phút) và giám sát disconnect để hủy
        import asyncio
        timeout_seconds = 300
        max_retries = 3

        async def run_crawl_with_retries():
            for attempt in range(max_retries):
                try:
                    return await asyncio.wait_for(
                        main(
                            url, max_pages, max_depth, force_engine, use_fit_markdown,
                            use_chunking, chunking_type, chunk_size, chunk_overlap,
                            enable_lazy_load, scroll_delay, max_scrolls, extraction_schema,
                            segment, segment_size, word_count_threshold, wait_for,
                            log_pre_llm, include_pre_llm_in_response,
                            use_normalize_headings, preserve_content, require_ai
                        ),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        print(f"[WARN] Crawl timeout on attempt {attempt + 1}/{max_retries}, retrying...")
                        await asyncio.sleep(5)
                        continue
                    print(f"[ERROR] Crawl timeout after {max_retries} attempts")
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"[WARN] Crawl failed on attempt {attempt + 1}/{max_retries}: {str(e)}, retrying...")
                        await asyncio.sleep(5)
                        continue
                    raise

        async def monitor_disconnect():
            try:
                while True:
                    # Nếu client ngắt kết nối (đóng/reload tab), trả về True
                    if await request.is_disconnected():
                        print("[CANCEL] Client disconnected - will cancel crawl task")
                        return True
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                # Task monitor bị hủy khi crawl xong trước
                return False
            except Exception:
                # Không chặn luồng nếu có lỗi theo dõi
                return False

        crawl_task = asyncio.create_task(run_crawl_with_retries())
        monitor_task = asyncio.create_task(monitor_disconnect())

        done, pending = await asyncio.wait({crawl_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

        if monitor_task in done and monitor_task.result() is True:
            # Hủy crawl nếu client ngắt
            crawl_task.cancel()
            await asyncio.gather(crawl_task, return_exceptions=True)
            # Dọn task monitor (đã done)
            await asyncio.gather(monitor_task, return_exceptions=True)
            return {"message": "Crawl cancelled by client", "result": [], "cancelled": True}

        # Nếu crawl xong trước
        if crawl_task in done:
            result = crawl_task.result()
            # Dọn monitor
            if not monitor_task.done():
                monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
        else:
            # Monitor kết thúc nhưng không phải disconnect → vẫn chờ crawl
            result = await crawl_task
            # Đảm bảo monitor đã được dọn
            await asyncio.gather(monitor_task, return_exceptions=True)
        
        print(f"[SUCCESS] Crawl completed with {len(result)} chunks")
        return {"message": "Crawl completed", "result": result}
    except Exception as e:
        print(f"[ERROR] Crawl failed: {str(e)}")
        return {"message": "Crawl failed", "error": str(e), "result": []}

"""Minimal API using Crawl4AI core configs only (BrowserConfig, CrawlerRunConfig)."""

async def main(url="https://vnexpress.net/chu-tich-quoc-hoi-nghien-cuu-sua-mot-so-dieu-cua-hien-phap-trong-thang-3-4856862.html", 
                 max_pages=5, max_depth=3, force_engine=None, use_fit_markdown=True,
                               use_chunking=True, chunking_type="regex", chunk_size=768, chunk_overlap=768,  # Theo client default
               enable_lazy_load=True, scroll_delay=2000, max_scrolls=5, extraction_schema=None,
                               segment=False, segment_size=1200, word_count_threshold=5, wait_for=None,  # Giảm threshold
                               log_pre_llm=False, include_pre_llm_in_response=False,
               use_normalize_headings=True, preserve_content=True, require_ai: bool = True):

    # Normalize seed URL: remove fragment (after #)
    try:
        from urllib.parse import urlsplit, urlunsplit
        sp = urlsplit(url)
        # Remove fragment and normalize trailing slash (keep only for root)
        path = sp.path or '/'
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')
        url = urlunsplit((sp.scheme, sp.netloc, path, sp.query, ''))
    except Exception:
        pass

    # BrowserConfig per docs - chỉ sử dụng tham số cơ bản
    # Apply force_engine if provided (e.g., chromium, stealth, undetected)
    if force_engine:
        try:
            browser_cfg = BrowserConfig(headless=True, engine=str(force_engine))
            print(f"[INFO] Using BrowserConfig(headless=True, engine={force_engine})")
        except Exception as e:
            print(f"[WARN] Invalid force_engine={force_engine}, fallback default. Error: {e}")
            browser_cfg = BrowserConfig(headless=True)
    else:
        browser_cfg = BrowserConfig(headless=True)
        print("[INFO] Using BrowserConfig(headless=True)")
    content_filter = None
    print("[INFO] LLMContentFilter is temporarily disabled")

    # Optional JsonCssExtractionStrategy per docs
    extraction_strategy = None
    if extraction_schema:
        try:
            extraction_strategy = JsonCssExtractionStrategy(extraction_schema)
            print("[INFO] Using JsonCssExtractionStrategy")
        except Exception as e:
            print(f"[WARN] Invalid extraction_schema: {e}. Skipping.")



    md_generator = DefaultMarkdownGenerator(
        content_source="cleaned_html",
        content_filter=content_filter,
        options={
            "ignore_links": False,
            "ignore_images": False,
            "escape_html": False,
            "body_width": 0,
            "preserve_formatting": True,  # Giữ nguyên định dạng
            "preserve_tables": True,      # Giữ nguyên bảng
            "preserve_lists": True,       # Giữ nguyên danh sách
            "fit_markdown": bool(use_fit_markdown),  # Bật fit_markdown nếu yêu cầu
        }
    )

    # Deep crawl strategy if requested
    deep_crawl_strategy = None
    if (max_pages and max_pages > 1) or (max_depth and max_depth > 1):
        # Build FilterChain per docs: allow same domain, deny fragments (#)
        allow_domain = None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.netloc:
                host = parsed.netloc.lower().replace(':80', '').replace(':443', '')
                allow_domain = host
        except Exception:
            allow_domain = None

        fc = None
        if allow_domain:
            # URLPatternFilter with wildcard patterns: allow https only, deny query strings
            fc = FilterChain([
                URLPatternFilter(patterns=[f"https://{allow_domain}/*"]),
            ])

        deep_crawl_strategy = BestFirstCrawlingStrategy(
            max_depth=max(0, int(max_depth) - 1),
            include_external=False,
            max_pages=int(max_pages),
            filter_chain=fc
        )
    
    # Thêm wait_for JS tối giản: chuyển background-image -> <img>, và a[href^="javascript:void"] bọc ảnh -> ảnh
    default_wait_js = r'''js:(async () => {
  try {
    // Lazy-load scroll controls (injected by Python)
    const _maxScrolls = %MAX_SCROLLS%;
    const _scrollDelay = %SCROLL_DELAY%;
    // Extract background images into <img> (inline style, computed style, and common data-* fallbacks)
    const _extractBgUrl = (el) => {
      try {
        // 1) Inline style
        const inline = (el.getAttribute('style') || '').match(/background-image\s*:\s*url\(("|'|)(.*?)\1\)/i);
        if (inline && inline[2]) return inline[2];
        // 2) Computed style
        const cs = window.getComputedStyle(el);
        if (cs && cs.backgroundImage && cs.backgroundImage.startsWith('url(')) {
          const m = cs.backgroundImage.match(/url\(("|'|)?(.*?)(\1)?\)/i);
          if (m && m[2]) return m[2];
        }
        // 3) Common data-* attributes
        const attrs = ['data-bg','data-background','data-src','data-srcset'];
        for (const a of attrs) {
          const v = el.getAttribute(a);
          if (v) {
            // if srcset-like, take first URL
            const first = v.split(',')[0].trim().split(' ')[0];
            if (first) return first;
          }
        }
      } catch(e) {}
      return null;
    };

    document.querySelectorAll('[style*="background-image"], .swiper-slide, .slick-slide, .carousel-item').forEach(el => {
      try {
        if (el.querySelector('img')) return; // already has image
        const url = _extractBgUrl(el);
        if (!url) return;
        const im = document.createElement('img');
        im.setAttribute('src', url);
        const alt = el.getAttribute('aria-label') || el.getAttribute('title') || 'slide image';
        im.setAttribute('alt', alt);
        el.insertBefore(im, el.firstChild);
      } catch(e) {}
    });

    // Replace javascript:void(0) anchors that wrap images with the image itself
    document.querySelectorAll('a[href^="javascript:void"]').forEach(a => {
      try {
        const img = a.querySelector('img');
        if (img && a.parentNode) a.replaceWith(img);
      } catch(e) {}
    });

    // Optional virtual scroll for lazy-load
    const _sleep = (ms) => new Promise(r => setTimeout(r, ms));
    if (%ENABLE_LAZY_LOAD%) {
      let lastH = 0;
      for (let i = 0; i < _maxScrolls; i++) {
        try { window.scrollTo(0, document.body.scrollHeight || 999999); } catch(e) {}
        await _sleep(_scrollDelay);
        const h = document.body.scrollHeight || 0;
        if (h && Math.abs(h - lastH) < 16) break; // converged
        lastH = h;
      }
    }
  } catch(e) {}
  return true;
})'''

    # Replace lazy-load controls
    default_wait_js = default_wait_js.replace('%ENABLE_LAZY_LOAD%', 'true' if enable_lazy_load else 'false')
    default_wait_js = default_wait_js.replace('%MAX_SCROLLS%', str(int(max_scrolls)))
    default_wait_js = default_wait_js.replace('%SCROLL_DELAY%', str(int(scroll_delay)))

    effective_wait_for = wait_for if (wait_for is not None and str(wait_for).strip() != '') else default_wait_js
    
    # CrawlerRunConfig for both single and deep crawl
    crawl_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=word_count_threshold,
        wait_for=effective_wait_for,
        wait_for_timeout=25000,
        excluded_tags=['script', 'style', 'noscript', 'header', 'footer'],  # Giữ lại 'form', 'input', 'label', 'select', 'textarea', 'fieldset', 'legend'
        deep_crawl_strategy=deep_crawl_strategy,
        verbose=True
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # Optional pre-LLM crawl to log raw markdown before LLM filtering
        pre_payload = None
        if log_pre_llm:
            pre_md_gen = DefaultMarkdownGenerator(
                content_source="cleaned_html",
                content_filter=None,
                options={
                    "ignore_links": False,
                    "ignore_images": False,
                    "escape_html": False,
                    "body_width": 0,
                }
            )
            pre_cfg = CrawlerRunConfig(
                markdown_generator=pre_md_gen,
                extraction_strategy=extraction_strategy,
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=word_count_threshold,
                wait_for=effective_wait_for,
                wait_for_timeout=15000,
            verbose=True
        )
            print("[INFO] Pre-LLM crawl (no content_filter) ...")
            pre_res = await crawler.arun(url=url, config=pre_cfg)
            if pre_res and getattr(pre_res, 'success', False):
                pre_md_obj = getattr(pre_res, 'markdown', '')
                pre_md = getattr(pre_md_obj, 'fit_markdown', '') or str(pre_md_obj)
                print(f"[PRE-LLM] Markdown length: {len(pre_md)}")
                # Cắt log tránh spam console
                snippet = (pre_md[:1000] + '...') if len(pre_md) > 1000 else pre_md
                print(f"[PRE-LLM] Markdown snippet:\n{snippet}")
                if include_pre_llm_in_response:
                    pre_payload = {
                        "content": pre_md or "",
                        "type": "text",
                        "source_url": url,
                        "stage": "pre_llm"
                    }
            else:
                print(f"[PRE-LLM] Failed: {getattr(pre_res, 'error_message', 'unknown error')}")

        print("[INFO] Starting crawl (with configured content_filter)...")
        result = await crawler.arun(url=url, config=crawl_config)
        print("[INFO] Crawl completed")

    # Handle single or deep results
    if isinstance(result, list):
        pages = []
        seen_urls = set()
        from urllib.parse import urlsplit, urlunsplit
        def _normalize_url(u: str) -> str:
            try:
                sp = urlsplit(u)
                # remove fragment; normalize path trailing slash (keep only for root)
                path = sp.path or '/'
                if path != '/' and path.endswith('/'):
                    path = path.rstrip('/')
                host = (sp.netloc or '').lower().replace(':80', '').replace(':443', '')
                return urlunsplit((sp.scheme, host, path, '', ''))
            except Exception:
                return u

        for i, res in enumerate(result):
            if not getattr(res, 'success', False):
                continue
            res_url = getattr(res, 'url', None) or ''
            norm_url = _normalize_url(res_url)
            if norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)
            md_obj = getattr(res, 'markdown', '')
            md_text = ''
            if md_obj:
                fm = getattr(md_obj, 'fit_markdown', '')
                md_text = fm if (isinstance(fm, str) and len(fm.strip()) > 10) else str(md_obj)
            # Hậu xử lý trước khi chunk (không dùng AI)
            md_text = await postprocess_markdown(md_text, normalize_headings_enabled=False, preserve_content=preserve_content, require_ai=require_ai)
            
            # XÓA DÒNG TRỐNG TRƯỚC CHUNKING LẦN 1 + SỬA LINK BỊ CẮT NGANG
            import re
            md_text = re.sub(r'\n{3,}', '\n\n', md_text)
            # Nối ']' xuống dòng với '(' để tránh cắt markdown link
            md_text = re.sub(r'\]\s*\n\s*\(', '](', md_text)
            # Xóa newline bên trong ngoặc URL: ( ... )
            def _join_newlines_in_parens(m):
                return '(' + re.sub(r'\s*\n\s*', '', m.group(1)) + ')'
            md_text = re.sub(r'\(([^)]*?)\)', _join_newlines_in_parens, md_text)
            md_text = md_text.strip()
            
            # Gọi AI một lần trên toàn bộ nội dung đã làm sạch
            combined_content = md_text
            if use_normalize_headings and combined_content.strip():
                try:
                    combined_content = await clean_and_format_chunk(combined_content, require_ai=require_ai)
                except Exception as e:
                    print(f"[WARN] Full-content AI processing failed: {str(e)}")
            import re
            combined_content = re.sub(r'\n+', '\n', combined_content).strip()

            # CHUNKING LẦN 2: Chia lại theo cùng logic
            token_limit = max(1, int(chunk_size) + int(chunk_overlap))
            reserve_tokens = 136
            other_limit = max(1, token_limit - reserve_tokens)
            final_chunks = chunk_by_tokens_variable_limits(combined_content, token_limit, other_limit)

            # THÊM BỐI CẢNH: prepend_heading_context với []
            scanned_prefix = ""
            annotated_chunks = []
            for idx, ch in enumerate(final_chunks):
                ch2 = prepend_heading_context(scanned_prefix, ch) if idx > 0 else ch
                annotated_chunks.append(ch2)
                scanned_prefix += (ch2 + "\n")

            if not annotated_chunks:
                continue

            for idx, ch in enumerate(annotated_chunks):
                item = {
                    "content": ch,
                    "type": "text",
                    "source_url": getattr(res, 'url', url),
                    "chunk_index": idx,
                    "total_chunks": len(annotated_chunks),
                    "depth": getattr(res, 'metadata', {}).get('depth', 0),
                    "engine_used": "browser",
                    "content_type": "regular_markdown",
                    "chunking_type": "token"
                }
                if extraction_strategy is not None and getattr(res, 'extracted_content', None) is not None and idx == 0:
                    item["extracted_content"] = res.extracted_content
                pages.append(item)
        
        # Giữ nguyên, không dedup để tăng lượng thông tin
        return pages

    if result.success:
        markdown_obj = getattr(result, 'markdown', '')
        markdown_content = ''
        if markdown_obj:
            # Prefer fit_markdown if available and non-empty, else str(markdown)
            fm = getattr(markdown_obj, 'fit_markdown', '')
            markdown_content = fm if (isinstance(fm, str) and len(fm.strip()) > 10) else str(markdown_obj)

        # Hậu xử lý trước khi chunk (không dùng AI)
        markdown_content = await postprocess_markdown(markdown_content, normalize_headings_enabled=False, preserve_content=preserve_content, require_ai=require_ai)

        # XÓA DÒNG TRỐNG TRƯỚC CHUNKING LẦN 1 + SỬA LINK BỊ CẮT NGANG
        import re
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
        # Nối ']' xuống dòng với '(' để tránh cắt markdown link
        markdown_content = re.sub(r'\]\s*\n\s*\(', '](', markdown_content)
        # Xóa newline bên trong ngoặc URL: ( ... )
        def _join_newlines_in_parens(m):
            return '(' + re.sub(r'\s*\n\s*', '', m.group(1)) + ')'
        markdown_content = re.sub(r'\(([^)]*?)\)', _join_newlines_in_parens, markdown_content)
        markdown_content = markdown_content.strip()

        # BỎ CHUNKING LẦN 1: Gọi AI một lần trên toàn bộ nội dung đã làm sạch
        combined_content = markdown_content
        if use_normalize_headings and combined_content.strip():
            try:
                combined_content = await clean_and_format_chunk(combined_content, require_ai=require_ai)
            except Exception as e:
                print(f"[WARN] Full-content AI processing failed: {str(e)}")
        # Loại bỏ tất cả dòng trống thừa
        import re
        combined_content = re.sub(r'\n+', '\n', combined_content).strip()

        # CHUNKING LẦN 2: Chia lại theo cùng logic
        token_limit = max(1, int(chunk_size) + int(chunk_overlap))
        reserve_tokens = 136
        other_limit = max(1, token_limit - reserve_tokens)
        final_chunks = chunk_by_tokens_variable_limits(combined_content, token_limit, other_limit)

        # THÊM BỐI CẢNH: prepend_heading_context với []
        scanned_prefix = ""
        annotated_chunks = []
        for idx, ch in enumerate(final_chunks):
            ch2 = prepend_heading_context(scanned_prefix, ch) if idx > 0 else ch
            annotated_chunks.append(ch2)
            scanned_prefix += (ch2 + "\n")

        out_payloads = []
        if pre_payload is not None:
            out_payloads.append(pre_payload)
        if not annotated_chunks:
            return out_payloads
        for idx, ch in enumerate(annotated_chunks):
            item = {
                "content": ch,
                "type": "text",
                "source_url": url,
                "chunk_index": idx,
                "total_chunks": len(annotated_chunks),
                "depth": getattr(result, 'metadata', {}).get('depth', 0) if hasattr(result, 'metadata') else 0,
                "engine_used": "browser",
                "content_type": "regular_markdown",
                "chunking_type": "token"
            }
            if extraction_strategy is not None and getattr(result, 'extracted_content', None) is not None and idx == 0:
                item["extracted_content"] = result.extracted_content

            out_payloads.append(item)
        
        # Giữ nguyên, không dedup để tăng lượng thông tin
        return out_payloads
    else:
        print(f"[ERROR] Crawl failed: {result.error_message}")
        return [{"error": result.error_message, "type": "error"}]

def tokenize_len(text: str, encoding_name: str = "o200k_base") -> int:
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return len(text or "")

def _heading_match(line: str):
    import re
    m = re.match(r'^(#{1,6})\s+(.+)', line.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None

def _build_stack_until(lines, end_idx: int):
    """Xây stack heading theo quy tắc markdown tới vị trí end_idx (inclusive)."""
    stack = []  # list[(level, title)]
    for i, raw in enumerate(lines[: max(0, end_idx + 1) ]):
        hm = _heading_match(raw)
        if not hm:
            continue
        level, title = hm
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return stack

def compute_boundary_heading_chain(prefix_markdown: str, max_level: int = 3):
    """Lấy chuỗi heading cha→con áp dụng cho ranh giới chunk.
    - Xác định dòng trống gần nhất (blank line) tính từ cuối prefix.
    - Tìm heading xuất hiện ngay TRƯỚC dòng trống đó (gần nhất).
    - Xây stack cha→con tới chính heading đó; trả về các heading với cấp ≤ max_level.
    """
    if not prefix_markdown:
        return []
    lines = prefix_markdown.split('\n')
    # Tìm chỉ số dòng trống gần nhất từ cuối lên
    blank_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            blank_idx = i
            break
    # Tìm heading gần nhất trước blank_idx
    target_idx = None
    start_scan = (blank_idx - 1) if blank_idx is not None else len(lines) - 1
    for j in range(start_scan, -1, -1):
        if _heading_match(lines[j]):
            target_idx = j
            break
    if target_idx is None:
        return []
    stack = _build_stack_until(lines, target_idx)
    return [(lvl, title) for (lvl, title) in stack if lvl <= max_level]

def prepend_heading_context(prefix_markdown: str, chunk: str) -> str:
    """Nếu chunk không bắt đầu bằng heading (#/##/###), chèn heading gần nhất đã xuất hiện trước đó.
    Chèn đầy đủ hệ cấp cha-con (ví dụ có cả # và ##, sẽ chèn cả hai), và bọc trong [] như yêu cầu.
    """
    import re
    if not chunk:
        return chunk
    # Luôn chèn bối cảnh: [H1 Title] và tiếp theo [H2], [H3] nếu có, không kèm dấu '#'
    chain = compute_boundary_heading_chain(prefix_markdown, max_level=3)
    title_h1 = None
    for lvl, title in chain:
        if lvl == 1:
            title_h1 = title
            break
    if not title_h1:
        # Fallback: lấy H1 từ chính chunk nếu dòng đầu là H1
        m = re.match(r'^\s*#\s+(.+)', chunk.lstrip())
        if m:
            title_h1 = m.group(1).strip()
    if not title_h1 and not chain:
        return chunk
    titles = []
    if title_h1:
        titles.append(title_h1)
    # Thêm H2/H3 từ chain theo thứ tự, tránh trùng với H1
    for lvl, title in chain:
        if lvl >= 2 and lvl <= 3:
            titles.append(title)
    # Xóa trùng liên tiếp nếu có
    dedup = []
    for t in titles:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    prefix_block = "\n".join(f"[{t}]" for t in dedup if t)
    if not prefix_block:
        return chunk
    body = chunk.lstrip('\n')
    return f"{prefix_block}\n{body}"

async def postprocess_markdown(text: str, normalize_headings_enabled: bool = True, preserve_content: bool = True, require_ai: bool = False) -> str:
    """Hậu xử lý markdown: chuẩn hóa heading, danh sách, khoảng trắng, giữ nguyên code block/table.
    - Giới hạn heading tối đa ###
    - Chuẩn hóa bullet dùng '* '
    - Rút gọn dòng trống thừa
    - Chuẩn hóa heading bằng LLM (nếu enabled)
    """
    if not text:
        return ""
    import re

    # Loại bỏ ký tự không xác định (PUA, control) ở cấp chuỗi trước
    def _is_pua(ch: str) -> bool:
        o = ord(ch)
        return (0xE000 <= o <= 0xF8FF) or (0xF0000 <= o <= 0xFFFFD) or (0x100000 <= o <= 0x10FFFD)

    cleaned_chars = []
    for ch in text:
        if ch in ['\n', '\t']:
            cleaned_chars.append(ch)
            continue
        if _is_pua(ch):
            continue
        if ord(ch) < 32:
            continue
        cleaned_chars.append(ch)
    text = ''.join(cleaned_chars)

    lines = text.split('\n')
    out = []
    in_code = False
    code_fence = None
    for raw in lines:
        line = raw
        # code fence toggle
        if re.match(r'^```', line.strip()):
            in_code = not in_code
            code_fence = code_fence or '```'
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        # normalize headings depth to max ###
        m = re.match(r'^(#{1,6})(\s+)(.+)$', line)
        if m:
            hashes = '#' * min(len(m.group(1)), 3)
            line = f"{hashes} {m.group(3).strip()}"
            out.append(line)
            continue
        # normalize bullet markers to '* '
        if re.match(r'^\s*[-+•·‣◦▪▹–—]\s+', line):
            content = re.sub(r'^\s*[-+•·‣◦▪▹–—]\s+', '', line)
            line = f"* {content.strip()}"
        # collapse excessive spaces inside line (avoid breaking tables/links)
        if not re.match(r'^\s*\|', line):
            line = re.sub(r'\s+', ' ', line).rstrip()

        # Lọc bỏ các link rỗng và nội dung không hữu ích
        # Drop links with empty text: [](...)
        if re.search(r'\[\s*\]\([^)]+\)', line):
            line = re.sub(r'\[\s*\]\([^)]+\)', '', line).strip()
            if not line:
                continue

        # Nếu preserve_content=True, bỏ qua việc lọc quá mức
        if not preserve_content:
            # Drop captcha lines like "3 + 15 ="
            if re.fullmatch(r'\s*\d+\s*\+\s*\d+\s*=\s*', line):
                continue

            # Drop single stray letters (chỉ bỏ nếu dòng chỉ có 1 chữ cái)
            if re.fullmatch(r'^\s*[a-zA-Z]\s*$', line):
                continue

            # Drop bullets with empty link text: "* [](...)" or "* [ ](...)"
            if re.fullmatch(r'\s*[*-]\s*\[\s*\]\s*\([^)]+\)\s*', line):
                continue

            # Drop lines that are ONLY pagination links like [Trước][Tiếp theo][1][2]
            # Nhưng giữ lại nếu có nội dung khác
            link_texts = re.findall(r'\[([^\]]*)\]\([^)]+\)', line)
            if link_texts and len(line.strip()) < 100:  # Chỉ áp dụng cho dòng ngắn
                normalized = [t.strip().lower() for t in link_texts]
                if all(t.isdigit() or t in {'trước', 'tiếp theo', 'prev', 'next'} for t in normalized):
                    continue
        out.append(line)

    # collapse multiple blank lines -> keep some spacing for readability
    text2 = '\n'.join(out)
    text2 = re.sub(r'\n{3,}', '\n\n', text2)  # Giữ lại 2 dòng trống thay vì 1
    
    text2 = text2.strip()
    
    # Chuẩn hóa heading bằng LLM nếu được bật
    if normalize_headings_enabled and text2:
        try:
            text2 = await clean_and_format_chunk(text2, require_ai=require_ai)
        except Exception as e:
            print(f"[WARN] Heading normalization failed in postprocess: {str(e)}")
    
    return text2

def deduplicate_chunks(chunks: list) -> list:
    """Loại bỏ các chunk trùng lặp dựa trên nội dung."""
    if not chunks:
        return chunks
    
    seen_content = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Chuẩn hóa nội dung để so sánh (loại bỏ khoảng trắng, chuyển lowercase)
        normalized_content = re.sub(r'\s+', ' ', chunk.get('content', '')).strip().lower()
        
        # Bỏ qua chunk rỗng hoặc quá ngắn (tăng ngưỡng từ 50 lên 100)
        if len(normalized_content) < 100:
            continue
            
        # Kiểm tra trùng lặp - chỉ loại bỏ nếu nội dung giống hệt 90% trở lên
        is_duplicate = False
        for seen in seen_content:
            if len(normalized_content) > 0 and len(seen) > 0:
                # Tính độ tương đồng đơn giản
                shorter = min(len(normalized_content), len(seen))
                longer = max(len(normalized_content), len(seen))
                if shorter / longer > 0.9:  # 90% tương đồng
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_content.add(normalized_content)
            unique_chunks.append(chunk)
    
    print(f"[INFO] Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
    return unique_chunks



def tidy_chunk_edges(text: str) -> str:
    """Làm sạch mép chunk để tránh dòng rời rạc (bullet/marker không có nội dung)."""
    if not text:
        return ""
    lines = text.split('\n')

    def is_orphan_marker(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        bullet_class = r"\*|\-|\+|•|·|‣|◦|▪|▹|–|—"
        if re.fullmatch(rf"(?:{bullet_class})(?:\s*\[\s*\])?", s):
            return True
        if re.fullmatch(r"\d+[\.)]", s):
            return True
        return False

    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and is_orphan_marker(lines[0]):
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and is_orphan_marker(lines[-1]):
        lines.pop()

    return '\n'.join(lines)

def _find_safe_cut_pos(window: str) -> int:
    """Tìm vị trí cắt an toàn trong đoạn window để tránh cắt vào giữa markdown link/image/code.
    Trả về chỉ số trong window (>=0) hoặc -1 nếu không tìm thấy.
    Ưu tiên: '\n\n' > '\n' > kết câu ('. ', '! ', '? ') > trước '[' khi cặp '[](...)' chưa đóng.
    """
    import re
    # 1) Ưu tiên ngắt tại dòng trống
    dbl_nl = window.rfind('\n\n')
    if dbl_nl != -1:
        return dbl_nl + 1  # cắt ngay sau newline đầu tiên của cặp để giữ 1 newline

    # 2) Tránh cắt bên trong code fence
    fence_count = window.count('```')
    if fence_count % 2 == 1:
        # đang ở trong code fence, cố gắng cắt ở newline gần nhất
        nl_in_code = window.rfind('\n')
        return nl_in_code

    # 3) Tránh cắt bên trong markdown link/image: [...](...)
    last_open_bracket = window.rfind('[')
    last_close_paren = window.rfind(')')
    last_open_paren = window.rfind('(')
    if last_open_bracket != -1 and (last_close_paren == -1 or last_close_paren < last_open_bracket):
        # có '[' mà chưa thấy ')' sau đó → tránh cắt, lùi về newline trước '['
        nl_before_bracket = window.rfind('\n', 0, last_open_bracket)
        if nl_before_bracket != -1:
            return nl_before_bracket
        # nếu không có newline, đừng cắt trong cửa sổ này
        return -1

    # 4) data:image hoặc URL trong ngoặc còn mở
    if 'data:image' in window and (last_open_paren > last_close_paren):
        nl_before_paren = window.rfind('\n', 0, last_open_paren)
        if nl_before_paren != -1:
            return nl_before_paren
        return -1

    # 5) ranh giới tự nhiên: newline hoặc kết câu
    candidates = [window.rfind('\n')]
    for mark in ['. ', '! ', '? ', '。', '！', '？']:
        candidates.append(window.rfind(mark))
    cut = max(candidates)
    return cut

def chunk_by_tokens_variable_limits(text: str, first_limit: int, other_limit: int, encoding_name: str = "o200k_base") -> list:
    """Chia theo token với limit khác nhau: chunk đầu dùng first_limit, các chunk sau dùng other_limit."""
    if not text:
        return []
    try:
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
        chunks = []
        start = 0
        n = len(tokens)
        idx = 0
        while start < n:
            limit = first_limit if idx == 0 else other_limit
            tentative_end = min(start + limit, n)
            chunk_tokens = tokens[start:tentative_end]
            chunk_text = enc.decode(chunk_tokens)

            # Cắt ở ranh giới an toàn trong cửa sổ nhỏ: tránh cắt giữa link/image/code
            if tentative_end < n and chunk_text:
                window_size = 320
                window = chunk_text[-window_size:] if len(chunk_text) > window_size else chunk_text
                cut_pos = _find_safe_cut_pos(window)
                if cut_pos >= 0:
                    safe_text = chunk_text[: len(chunk_text) - len(window) + cut_pos + 1]
                    # đảm bảo không tạo chunk rỗng
                    if safe_text.strip():
                        chunk_tokens = enc.encode(safe_text)
                        chunk_text = safe_text

            chunk_text = tidy_chunk_edges(chunk_text)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

            start = start + len(chunk_tokens)
            if start >= n:
                break
            idx += 1
        return chunks
    except Exception:
        # Fallback theo ký tự với tỉ lệ ước lượng tokens≈chars/4
        avg_token_chars = 4
        chunks = []
        start = 0
        n = len(text)
        idx = 0
        while start < n:
            limit_chars = (first_limit if idx == 0 else other_limit) * avg_token_chars
            end = min(start + limit_chars, n)
            chunk_text = text[start:end]
            chunk_text = tidy_chunk_edges(chunk_text)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            if end >= n:
                break
            start = end
            idx += 1
        return chunks

# Run the async main function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8881)

