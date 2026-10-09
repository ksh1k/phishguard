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


def test_normalize_for_model():
    from features import normalize_for_model
    assert normalize_for_model("https://www.Example.com/a/b?x=1") == "https://Example.com/a/b?x=1"
    assert normalize_for_model("http://evil.tk/login") == "http://evil.tk/login"
    assert normalize_for_model("example.com/x") == "http://example.com/x"
    assert normalize_for_model("https://wwwx.example.com/") == "https://wwwx.example.com/"


def test_v2_free_hosting_and_brand():
    assert extract_features("https://citizens-online.firebaseapp.com/")["free_hosting"] == 1
    assert extract_features("https://example.com/")["free_hosting"] == 0
    assert extract_features("http://paypal.evil-site.com/login")["brand_mismatch"] == 1
    assert extract_features("https://www.paypal.com/signin")["brand_mismatch"] == 0
    assert extract_features("https://paypal.weebly.com/")["brand_mismatch"] == 1


def test_v2_digit_in_word_and_runs():
    assert extract_features("http://amaz0n-verify.com/")["has_digit_in_word"] == 1
    assert extract_features("http://amazon.com/")["has_digit_in_word"] == 0
    f = extract_features("http://x.com/a/b/c.php?id=123456&u=1")
    assert f["longest_digit_run"] == 6
    assert f["num_query_params"] == 2
    assert f["path_depth"] == 3
    assert f["script_extension"] == 1


def test_v2_port_punycode():
    assert extract_features("http://example.com:8080/")["has_port"] == 1
    assert extract_features("https://example.com:443/")["has_port"] == 0
    assert extract_features("http://xn--pple-43d.com/")["is_punycode"] == 1
