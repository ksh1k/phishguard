"""URL -> lexical feature vector. Pure functions, no network lookups."""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link", "zip", "country", "support"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at"}
KEYWORDS = ("login", "verify", "secure", "account", "update", "confirm", "banking", "paypal", "signin", "password", "wallet")

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_digits",
    "has_ip", "has_at_symbol", "num_subdomains", "uses_https",
    "suspicious_tld", "is_shortened", "domain_entropy", "has_suspicious_keyword",
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
    }


def feature_vector(url: str) -> list:
    f = extract_features(url)
    return [f[n] for n in FEATURE_NAMES]
