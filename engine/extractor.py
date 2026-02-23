from __future__ import annotations

import base64
import ipaddress
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlparse

import numpy as np
import requests
import tldextract
import whois
from ipwhois import IPWhois


_EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_BASE64_REGEX = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_PASSWORD_INPUT_REGEX = re.compile(
    r"<input[^>]*type=['\"]password['\"][^>]*>", re.IGNORECASE
)


_KNOWN_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
}


_BRAND_KEYWORDS = {
    "paypal",
    "microsoft",
    "google",
    "apple",
    "facebook",
    "instagram",
    "netflix",
    "amazon",
    "bankofamerica",
    "chase",
}


_NETWORK_TIMEOUT_SECONDS = 3.0
_MAX_REDIRECTS = 5


def _shannon_entropy(value: str) -> float:
    """Compute Shannon entropy for a string.

    Args:
        value: Input string.

    Returns:
        Entropy value in bits.
    """
    if not value:
        return 0.0
    length = float(len(value))
    counts: Dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(entropy)


def _ratio(value: str, characters: str) -> float:
    """Compute the ratio of given characters in a string.

    Args:
        value: Input string.
        characters: Characters to count.

    Returns:
        Ratio in the range [0, 1].
    """
    if not value:
        return 0.0
    total = sum(value.count(ch) for ch in characters)
    return float(total) / float(len(value))


def _is_email(value: str) -> bool:
    """Check whether a string looks like an email address."""
    return bool(_EMAIL_REGEX.search(value))


def _looks_base64(value: str) -> bool:
    """Heuristically determine whether a string is Base64 encoded."""
    if len(value) < 8 or len(value) % 4 != 0:
        return False
    if not _BASE64_REGEX.match(value):
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception:
        return False
    if not decoded:
        return False
    printable = sum(32 <= b <= 126 for b in decoded)
    ratio = printable / float(len(decoded))
    return ratio > 0.7


def _host_from_url(url: str) -> str:
    """Extract hostname from URL."""
    parsed = urlparse(url)
    return parsed.hostname or ""


def _uses_standard_ip(host: str) -> bool:
    """Determine whether hostname is a standard IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def _uses_encoded_ip(host: str) -> bool:
    """Detect hexadecimal, octal, or decimal encoded IPv4 forms."""
    if not host:
        return False
    numeric_patterns = [
        (r"^0x[0-9a-fA-F]+$", 16),
        (r"^[0-9]+$", 10),
        (r"^0[0-7]+$", 8),
    ]
    for pattern, base in numeric_patterns:
        if re.match(pattern, host):
            try:
                value = int(host, base=base)
                if 0 <= value <= 0xFFFFFFFF:
                    ipaddress.ip_address(value)
                    return True
            except Exception:
                continue
    return False


def _domain_from_url(url: str) -> str:
    """Extract registrable domain from URL."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    extracted = tldextract.extract(host)
    if not extracted.domain or not extracted.suffix:
        return host
    return f"{extracted.domain}.{extracted.suffix}"


def _domain_age_days(domain: str) -> float:
    """Compute domain age in days using WHOIS."""
    if not domain:
        return 0.0
    try:
        record = whois.whois(domain)
    except Exception:
        return 0.0
    created = record.creation_date
    if isinstance(created, list):
        created = created[0]
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created)
        except Exception:
            return 0.0
    if not isinstance(created, datetime):
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - created
    return float(delta.days)


def _asn_trust_score(host: str) -> float:
    """Derive a simple ASN trust score for a hostname."""
    try:
        ip = ipaddress.ip_address(host)
    except Exception:
        return 0.5
    try:
        lookup = IPWhois(str(ip))
        data = lookup.lookup_rdap(timeout=_NETWORK_TIMEOUT_SECONDS)
    except Exception:
        return 0.5
    asn_description = str(data.get("asn_description") or "").lower()
    score = 0.5
    if any(keyword in asn_description for keyword in ("hosting", "bulletproof", "vpn", "proxy")):
        score -= 0.2
    if any(keyword in asn_description for keyword in ("google", "amazon", "microsoft", "cloudflare")):
        score += 0.1
    return float(max(0.0, min(1.0, score)))


def _redirect_chain_metrics(url: str) -> Dict[str, Any]:
    """Resolve redirect chain for a URL."""
    session = requests.Session()
    try:
        response = session.get(
            url,
            allow_redirects=True,
            timeout=_NETWORK_TIMEOUT_SECONDS,
        )
    except Exception:
        return {"final_url": url, "chain_length": 0, "uses_shortener": False}
    history = response.history or []
    chain_length = min(len(history), _MAX_REDIRECTS)
    final_url = response.url or url
    parsed_final = urlparse(final_url)
    host = parsed_final.hostname or ""
    uses_shortener = False
    if host:
        extracted = tldextract.extract(host)
        fqdn = f"{extracted.domain}.{extracted.suffix}" if extracted.domain and extracted.suffix else host
        uses_shortener = fqdn in _KNOWN_SHORTENERS
    return {
        "final_url": final_url,
        "chain_length": float(chain_length),
        "uses_shortener": uses_shortener,
    }


def _brand_jacking_flag(url: str) -> bool:
    """Detect potential brand-jacking in subdomains or domains."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    extracted = tldextract.extract(host)
    domain = extracted.domain.lower() if extracted.domain else ""
    subdomain = extracted.subdomain.lower() if extracted.subdomain else ""
    for brand in _BRAND_KEYWORDS:
        if brand in subdomain:
            return True
        if brand in domain and domain != brand:
            return True
    return False


def _password_field_on_low_reputation(html: Optional[str], domain_age: float) -> bool:
    """Detect password fields on low-reputation sites."""
    if html is None:
        return False
    if domain_age <= 0 or domain_age > 365.0:
        return False
    return bool(_PASSWORD_INPUT_REGEX.search(html))


@dataclass
class FeatureExtractionResult:
    """Container for extracted phishing detection features.

    Attributes:
        vector: Numeric feature vector suitable for model ingestion.
        feature_names: Ordered list of feature names corresponding to vector.
        metadata: Optional additional metadata for explainability.
    """

    vector: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Convert URLs and associated data into numerical feature vectors."""

    feature_names: List[str] = [
        "url_length",
        "url_entropy",
        "dot_density",
        "hyphen_density",
        "ratio_dollar",
        "ratio_at",
        "ratio_underscore",
        "ratio_question",
        "ratio_equal",
        "uses_standard_ip",
        "uses_encoded_ip",
        "asn_trust_score",
        "domain_age_days",
        "has_email_in_params",
        "has_base64_param_value",
        "has_open_redirect_param",
        "redirect_chain_length",
        "uses_known_shortener",
        "brand_jacking_flag",
        "password_field_on_low_reputation",
    ]

    def extract(self, url: str, html: Optional[str] = None) -> FeatureExtractionResult:
        """Extract phishing detection features for a single URL.

        Args:
            url: URL to analyze.
            html: Optional rendered HTML content for behavioral features.

        Returns:
            FeatureExtractionResult containing the numeric vector and metadata.
        """
        url = url.strip()
        parsed = urlparse(url)
        full = url
        host = parsed.hostname or ""
        domain = _domain_from_url(url)

        url_length = float(len(full))
        url_entropy = _shannon_entropy(full)
        dot_density = _ratio(full, ".")
        hyphen_density = _ratio(full, "-")
        ratio_dollar = _ratio(full, "$")
        ratio_at = _ratio(full, "@")
        ratio_underscore = _ratio(full, "_")
        ratio_question = _ratio(full, "?")
        ratio_equal = _ratio(full, "=")

        standard_ip = _uses_standard_ip(host)
        encoded_ip = _uses_encoded_ip(host)
        asn_score = _asn_trust_score(host) if standard_ip else 0.5
        domain_age = _domain_age_days(domain)

        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        has_email_in_params = any(_is_email(value) for _, value in query_pairs)
        has_base64_param_value = any(_looks_base64(value) for _, value in query_pairs)
        open_redirect_keywords = {"next", "url", "redirect", "redir", "return", "dest"}
        has_open_redirect_param = False
        for key, value in query_pairs:
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in open_redirect_keywords):
                if value.startswith("http://") or value.startswith("https://"):
                    has_open_redirect_param = True
                    break

        redirect_metrics = _redirect_chain_metrics(url)
        redirect_chain_length = float(redirect_metrics["chain_length"])
        uses_shortener = bool(redirect_metrics["uses_shortener"])
        brand_flag = _brand_jacking_flag(url)
        password_low_rep = _password_field_on_low_reputation(html, domain_age)

        values = np.array(
            [
                url_length,
                url_entropy,
                dot_density,
                hyphen_density,
                ratio_dollar,
                ratio_at,
                ratio_underscore,
                ratio_question,
                ratio_equal,
                float(standard_ip),
                float(encoded_ip),
                asn_score,
                domain_age,
                float(has_email_in_params),
                float(has_base64_param_value),
                float(has_open_redirect_param),
                redirect_chain_length,
                float(uses_shortener),
                float(brand_flag),
                float(password_low_rep),
            ],
            dtype=float,
        )

        metadata: Dict[str, Any] = {
            "url": url,
            "host": host,
            "domain": domain,
            "final_url": redirect_metrics["final_url"],
            "has_email_in_params": has_email_in_params,
            "has_base64_param_value": has_base64_param_value,
            "has_open_redirect_param": has_open_redirect_param,
            "brand_jacking_flag": brand_flag,
            "password_field_on_low_reputation": password_low_rep,
            "domain_age_days": domain_age,
            "asn_trust_score": asn_score,
        }

        return FeatureExtractionResult(
            vector=values,
            feature_names=list(self.feature_names),
            metadata=metadata,
        )

