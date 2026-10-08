import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import FEATURE_NAMES, extract_features, feature_vector, shannon_entropy


def test_feature_keys_match_names():
    assert list(extract_features("https://google.com").keys()) == FEATURE_NAMES
    assert len(feature_vector("https://google.com")) == len(FEATURE_NAMES)


def test_obvious_phish():
    f = extract_features("http://secure-paypal-login.verify-account.tk/update/confirm?id=123")
    assert f["suspicious_tld"] == 1
    assert f["has_suspicious_keyword"] == 1
    assert f["uses_https"] == 0
    assert f["num_hyphens"] >= 3


def test_google_is_clean():
    f = extract_features("https://www.google.com/")
    assert f["uses_https"] == 1
    assert f["has_ip"] == 0
    assert f["suspicious_tld"] == 0
    assert f["has_suspicious_keyword"] == 0
    assert f["num_subdomains"] == 1


def test_shortener():
    assert extract_features("https://bit.ly/3abcXYZ")["is_shortened"] == 1
    assert extract_features("https://example.com/bit.ly")["is_shortened"] == 0


def test_ip_literal():
    assert extract_features("http://192.168.1.10/login.php")["has_ip"] == 1
    assert extract_features("http://[2001:db8::1]/x")["has_ip"] == 1
    assert extract_features("http://192.168.1.10/login.php")["num_subdomains"] == 0


def test_long_subdomain_chain():
    f = extract_features("https://a.b.c.d.e.example.com/")
    assert f["num_subdomains"] == 5
    assert f["num_dots"] == 6


def test_at_symbol():
    assert extract_features("http://google.com@evil.com/")["has_at_symbol"] == 1


def test_scheme_missing():
    f = extract_features("example.com/path")
    assert f["hostname_length"] == len("example.com")
    assert f["uses_https"] == 0


def test_garbage_does_not_crash():
    for u in ["", "   ", "http://", "http://[bad", "not a url at all"]:
        assert list(extract_features(u).keys()) == FEATURE_NAMES


def test_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == 2.0
