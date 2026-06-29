# Ethics for `StealthClient`

This module is a tool. Tools are neutral; their use is not.

## Rules of thumb

1. **Public data only.** Do not scrape behind login walls without
   explicit permission from the site operator.
2. **Respect `robots.txt`.** Read it before scraping. If the operator
   has marked your target path `Disallow`, find a different source.
3. **Rate limit conservatively.** The default 1 req/s per host is for
   scraping at a hobby scale. For larger scale, write to the operator
   first, ask if they have an API, and negotiate a rate.
4. **Identify yourself when possible.** A `User-Agent` like
   `"YourProject/1.0 (https://your-site.example/contact)"` lets the
   operator reach you if there's an issue. Some operators block any
   non-browser UA; in those cases impersonate a browser via
   `impersonate="chrome124"`.
5. **Cache aggressively.** Re-fetching the same URL is wasteful for the
   operator and slow for you. Use the built-in `DiskCache`.
6. **Don't republish copyrighted content.** Use scraping to build
   derived data (statistics, summaries) that you can publish under fair
   use; do not mirror articles, prices, or other proprietary content.
7. **Stop scraping when asked.** If you receive a cease-and-desist or
   the site adds a `noindex` or a personalised block, stop.

## Legal landscape (US, EU)

The legal status of scraping public data is jurisdiction-dependent and
evolving:

- **US**: hiQ Labs v. LinkedIn (2022) clarified that scraping
  *public* data does not violate the CFAA. Terms of service
  violations are still actionable.
- **EU**: GDPR applies if you process personal data, even if it's
  public. You typically need a lawful basis.
- **Both jurisdictions**: copyright still applies to scraped content.

For commercial use, consult a lawyer.

## What this module does NOT do

- Solve CAPTCHAs (use 2Captcha / Anti-Captcha if you need to)
- Defeat browser-fingerprinting (escalate to `playwright-stealth`)
- Bypass paywalls (illegal in most cases anyway)
- Hide your identity (use a real proxy, not just curl_cffi)

The module is for *legitimate* scraping that is slowed or blocked by
overzealous bot detection. It is not a hacking toolkit.
