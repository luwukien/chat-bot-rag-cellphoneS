"""Validate storage requests against the indexed catalog, without model calls."""

import re
import unicodedata


CAPACITY = re.compile(r"\b(\d+)\s*(GB|TB)\b", re.IGNORECASE)


def normalize(text):
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\b(?:ip|iphon)\b", "iphone", text)
    text = re.sub(r"\b(?:pm|promax)\b", "pro max", text)
    text = re.sub(r"\bpr\b", "pro", text)
    return " ".join(text.split())


def capacity_value(match):
    return int(match[1]) * (1024 if match[2].upper() == "TB" else 1)


class ProductCapacityCatalog:
    def __init__(self, chunks=()):
        self.families = {}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            name = meta.get("product_name", "")
            match = CAPACITY.search(name)
            if not match:
                continue
            family = re.sub(r"^Điện thoại\s+", "", name[:match.start()], flags=re.I).strip()
            key = normalize(family)
            variants = self.families.setdefault(key, {"name": family, "capacities": {}})
            value = capacity_value(match)
            variant = variants["capacities"].setdefault(value, {
                "label": f"{match[1]}{match[2].upper()}", "sources": [],
            })
            if meta.get("type") == "variants":
                variant["sources"].append(chunk)

    def resolve_price_query(self, query):
        """Return a grounded correction only for a single-product price question."""
        normalized = normalize(query)
        requested = list(CAPACITY.finditer(query))
        if len(requested) != 1 or not re.search(r"\b(?:gia|bao nhieu|bn tien)\b", normalized):
            return None
        # Keep comparisons and additional intents in the regular RAG flow.
        if re.search(r"\b(?:va|voi|vs|so sanh|hay|hoac|cau hinh|camera|chip|pin|mau|bao hanh|tra gop|doi tra)\b|[&;]", normalized):
            return None
        remaining = normalized
        matches = []
        for key in sorted(self.families, key=len, reverse=True):
            pattern = r"\b" + re.escape(key) + r"\b(?!\s+(?:pro|max|plus|mini|air|ultra)\b)"
            if re.search(pattern, remaining):
                matches.append(self.families[key])
                remaining = re.sub(pattern, " ", remaining)
        if len(matches) != 1:
            return None
        family = matches[0]
        capacities = family["capacities"]
        requested_value = capacity_value(requested[0])
        if requested_value in capacities:
            return None

        label = f"{requested[0][1]}{requested[0][2].upper()}"
        answer = f"{family['name']} không có phiên bản {label} trong dữ liệu hiện có."
        # Recommend the nearest indexed capacity with a recorded price.
        for value in sorted(capacities, key=lambda value: (abs(value - requested_value), value)):
            variant = capacities[value]
            prices = set()
            sources = []
            for source in variant["sources"]:
                found = re.findall(r"giá bán là\s+(\d+(?:\.\d{3})*)\s*₫", source["text"], flags=re.I)
                amounts = {int(price.replace(".", "")) for price in found}
                if amounts:
                    prices.update(amounts)
                    sources.append(source)
            if not prices:
                continue
            low, high = min(prices), max(prices)
            low_text = f"{low:,}".replace(",", ".") + "₫"
            high_text = f"{high:,}".replace(",", ".") + "₫"
            price_text = low_text if low == high else f"từ {low_text} đến {high_text} tùy màu"
            answer += f" Bản {variant['label']} có giá {price_text}. Giá này là giá đã thu thập và có thể thay đổi."
            return answer, sources

        labels = ", ".join(capacities[value]["label"] for value in sorted(capacities))
        return answer + f" Các dung lượng được ghi nhận: {labels}. Chưa có giá bán để đối chiếu.", []
