"""URL -> lexical feature vector. Pure functions, no network lookups."""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link", "zip", "country", "support"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at"}
KEYWORDS = ("login", "verify", "secure", "account", "update", "confirm", "banking", "paypal", "signin", "password", "wallet")
# Free site/page builders and tunnels that are widely abused to host phishing pages.
FREE_HOSTING = (
    "firebaseapp.com", "web.app", "pages.dev", "workers.dev", "netlify.app", "vercel.app", "weebly.com",
    "weeblysite.com", "wixsite.com", "godaddysites.com", "blogspot.com", "github.io", "herokuapp.com",
    "glitch.me", "repl.co", "000webhostapp.com", "myportfolio.com", "webwave.dev", "ngrok.io", "ipfs.io",
    "r2.dev", "framer.app", "carrd.co", "webflow.io", "sites.google.com", "forms.gle",
)
# Brands commonly impersonated in phishing; flagged when they appear in the host of an unrelated domain.
BRANDS = (
    "paypal", "google", "microsoft", "apple", "amazon", "facebook", "netflix", "instagram", "whatsapp",
    "linkedin", "dropbox", "adobe", "outlook", "office365", "chase", "wellsfargo", "bankofamerica",
    "dhl", "usps", "docusign", "metamask", "coinbase", "binance", "steam", "icloud",
)
SCRIPT_EXTS = (".php", ".asp", ".aspx", ".jsp", ".exe", ".zip", ".scr")

# Original v1 features first, then v2 additions.
FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_digits",
    "has_ip", "has_at_symbol", "num_subdomains", "uses_https",
    "suspicious_tld", "is_shortened", "domain_entropy", "has_suspicious_keyword",
    "host_digit_ratio", "has_digit_in_word", "num_query_params", "query_length", "path_depth",
    "longest_token_len", "longest_digit_run", "has_port", "is_punycode", "free_hosting",
    "brand_mismatch", "script_extension", "num_percent_encoded", "tld_length", "path_entropy",
]

# Plain-English explanations: name -> (sentence when feature pushes toward phishing, when away)
EXPLANATIONS = {
    "url_length": ("The URL is unusually long", "The URL is short and simple"),
    "hostname_length": ("The domain name is unusually long", "The domain name is short"),
    "path_length": ("The URL path is unusually long", "The URL path is short"),
    "num_dots": ("The hostname contains many dots", "The hostname has few dots"),
    "num_hyphens": ("The URL contains many hyphens", "The URL has few or no hyphens"),
    "num_underscores": ("The URL contains underscores", "The URL has no underscores"),
    "num_slashes": ("The URL has a deeply nested path", "The URL path is shallow"),
    "num_digits": ("The URL contains many digits", "The URL contains few digits"),
    "has_ip": ("The URL uses an IP address instead of a domain name", "The URL uses a normal domain name"),
    "has_at_symbol": ("The URL contains an '@' symbol, which can hide the real destination", "The URL has no '@' symbol"),
    "num_subdomains": ("The URL has many subdomains", "The URL has few subdomains"),
    "uses_https": ("The URL does not use HTTPS", "The URL uses HTTPS"),
    "suspicious_tld": ("The domain uses a TLD commonly abused by phishing sites", "The domain uses an ordinary TLD"),
    "is_shortened": ("The URL uses a link shortener that hides its destination", "The URL is not a shortened link"),
    "domain_entropy": ("The domain looks random-generated", "The domain looks like natural text"),
    "has_suspicious_keyword": ("The URL contains words like 'login' or 'verify'", "The URL has no credential-bait keywords"),
    "host_digit_ratio": ("The domain name has an unusually high share of digits", "The domain name has few digits"),
    "has_digit_in_word": ("The domain mixes digits into words (like 'amaz0n')", "The domain has no digits inside words"),
    "num_query_params": ("The URL has many query parameters", "The URL has few query parameters"),
    "query_length": ("The URL has a long query string", "The URL has a short or no query string"),
    "path_depth": ("The URL path has many levels", "The URL path has few levels"),
    "longest_token_len": ("The URL contains a very long random-looking token", "The URL has no unusually long tokens"),
    "longest_digit_run": ("The URL contains a long run of digits", "The URL has no long runs of digits"),
    "has_port": ("The URL uses a non-standard port", "The URL uses the default port"),
    "is_punycode": ("The domain uses punycode, which can disguise look-alike characters", "The domain does not use punycode"),
    "free_hosting": ("The page is hosted on a free site-builder platform often abused by phishers", "The page is not on a free hosting platform"),
    "brand_mismatch": ("The domain names a well-known brand but is not that brand's site", "The domain does not imitate a well-known brand"),
    "script_extension": ("The URL points at a script or executable file type", "The URL does not end in a script file type"),
    "num_percent_encoded": ("The URL contains many percent-encoded characters", "The URL has few encoded characters"),
    "tld_length": ("The domain ending is unusually long", "The domain ending has a typical length"),
    "path_entropy": ("The URL path looks random-generated", "The URL path looks like natural text"),
}


def _parse(url: str):
    url = url.strip()
    full = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url
    try:
        parsed = urlparse(full)
    except ValueError:  # e.g. malformed IPv6 literal
        parsed = urlparse("http://")
    return url, parsed


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict:
    raw, p = _parse(url)
    try:
        host = (p.hostname or "").lower()
    except ValueError:
        host = ""
    scheme_given = "://" in raw
    path = p.path or ""
    has_ip = _is_ip(host)
    labels = [l for l in host.split(".") if l]
    tld = labels[-1] if len(labels) > 1 else ""
    registered = ".".join(labels[-2:]) if len(labels) >= 2 else host
    # Hostname-based features skip the registered domain and TLD for subdomain count.
    num_subdomains = 0 if has_ip else max(len(labels) - 2, 0)
    haystack = (host + path).lower()
    query = p.query or ""
    try:
        port = p.port
    except ValueError:
        port = None
    sld = labels[-2] if len(labels) >= 2 else host
    on_free_host = any(host == d or host.endswith("." + d) for d in FREE_HOSTING)
    brand_hit = (not has_ip) and any(b in host and b not in sld for b in BRANDS)
    # A brand inside a free-hosting name (e.g. paypal.weebly.com) is also a mismatch.
    brand_hit = brand_hit or (not has_ip and any(b in host and on_free_host for b in BRANDS))
    in_word = any(host[i - 1].isalpha() and host[i].isdigit() and host[i + 1].isalpha() for i in range(1, len(host) - 1))
    runs = [len(t) for t in re.split("[^A-Za-z0-9]+", raw) if t]
    digit_runs = [len(t) for t in re.split("[^0-9]+", raw) if t]
    return {
        "url_length": len(raw),
        "hostname_length": len(host),
        "path_length": len(path),
        "num_dots": host.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/") - (2 if scheme_given else 0),
        "num_digits": sum(ch.isdigit() for ch in raw),
        "has_ip": int(has_ip),
        "has_at_symbol": int("@" in raw),
        "num_subdomains": num_subdomains,
        "uses_https": int(p.scheme == "https" and scheme_given),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortened": int(registered in SHORTENERS),
        "domain_entropy": round(shannon_entropy(host), 4),
        "has_suspicious_keyword": int(any(k in haystack for k in KEYWORDS)),
        "host_digit_ratio": round(sum(ch.isdigit() for ch in host) / len(host), 4) if host else 0.0,
        "has_digit_in_word": int(in_word and not has_ip),
        "num_query_params": len([q for q in query.split("&") if q]),
        "query_length": len(query),
        "path_depth": len([s for s in path.split("/") if s]),
        "longest_token_len": max(runs, default=0),
        "longest_digit_run": max(digit_runs, default=0),
        "has_port": int(port is not None and port not in (80, 443)),
        "is_punycode": int("xn--" in host),
        "free_hosting": int(on_free_host),
        "brand_mismatch": int(brand_hit),
        "script_extension": int(path.lower().endswith(SCRIPT_EXTS)),
        "num_percent_encoded": raw.count("%"),
        "tld_length": len(tld),
        "path_entropy": round(shannon_entropy(path), 4),
    }


def feature_vector(url: str) -> list:
    f = extract_features(url)
    return [f[n] for n in FEATURE_NAMES]


def normalize_for_model(url: str) -> str:
    """Canonicalise a URL before feature extraction: add a missing scheme, drop a leading 'www.'.

    'www.' is removed because it is a dataset artefact (about half of the benign training
    URLs have it, almost no phishing URLs do) rather than a real signal. The path and
    query are kept.
    """
    raw = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    return re.sub(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)www\.", r"\1", raw, flags=re.I)


BINARY_FEATURES = {
    "has_ip", "has_at_symbol", "uses_https", "suspicious_tld", "is_shortened", "has_suspicious_keyword",
    "has_digit_in_word", "has_port", "is_punycode", "free_hosting", "brand_mismatch", "script_extension",
}

# Short noun phrases used when a numeric feature pushes the score in the less obvious direction.
FEATURE_LABELS = {
    "url_length": "The URL length", "hostname_length": "The domain name length", "path_length": "The path length",
    "num_dots": "The number of dots in the hostname", "num_hyphens": "The number of hyphens",
    "num_underscores": "The number of underscores", "num_slashes": "The number of slashes",
    "num_digits": "The number of digits", "num_subdomains": "The number of subdomains",
    "domain_entropy": "The randomness of the domain", "host_digit_ratio": "The share of digits in the domain",
    "num_query_params": "The number of query parameters", "query_length": "The query string length",
    "path_depth": "The path depth", "longest_token_len": "The longest token length",
    "longest_digit_run": "The longest run of digits", "num_percent_encoded": "The number of encoded characters",
    "tld_length": "The domain ending length", "path_entropy": "The randomness of the path",
}


def explain(name: str, value: float, median: float, toward_phishing: bool) -> str:
    """Plain-English reason for one feature's effect, worded to match which side of typical it is on."""
    if name in BINARY_FEATURES:
        return EXPLANATIONS[name][0 if toward_phishing else 1]
    label = FEATURE_LABELS[name]
    high = value > median
    if toward_phishing:
        return EXPLANATIONS[name][0] if high else f"{label} is lower than is typical for legitimate sites"
    return EXPLANATIONS[name][1] if not high else f"{label} is higher than is typical, which favours legitimate sites"
