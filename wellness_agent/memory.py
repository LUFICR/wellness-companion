import re
from datetime import datetime
from .utils.storage import load_json, save_json, now_iso, days_since, merge_dicts
from .config import get_user_memory_path, MEMORY_CATEGORIES


class MemorySystem:
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.path = get_user_memory_path(user_id)
        self.memory = self._load()

    def _load(self):
        data = load_json(self.path)
        if not data:
            data = {
                "user_id": self.user_id,
                "facts": [],
                "trust_score": 30,
                "pillar_coverage": {},
                "last_updated": now_iso(),
                "session_count": 0,
                "avoided_pillars": {},
                "deprioritized_pillars": []
            }
        return data

    def save(self):
        self.memory["last_updated"] = now_iso()
        save_json(self.path, self.memory)

    def add_fact(self, category, key, value, confidence=70, source="conversation"):
        existing = self.get_fact(key)
        if existing:
            return self.update_fact(key, value, confidence, source)
        fact = {
            "category": category,
            "key": key,
            "value": value,
            "confidence": confidence,
            "source": source,
            "last_updated": now_iso(),
            "resolved": False
        }
        self.memory["facts"].append(fact)
        self._update_pillar_coverage(key, confidence)
        self.save()
        return fact

    def update_fact(self, key, value, confidence=None, source="conversation"):
        for fact in self.memory["facts"]:
            if fact["key"] == key:
                old_value = fact["value"]
                fact["value"] = value
                if confidence:
                    fact["confidence"] = confidence
                fact["source"] = source
                fact["last_updated"] = now_iso()
                self._update_pillar_coverage(key, fact["confidence"])
                self.save()
                return {"action": "update", "old_value": old_value, "new_value": value, "fact": fact}
        return None

    def get_fact(self, key):
        for fact in self.memory["facts"]:
            if fact["key"] == key:
                return fact
        return None

    def get_facts_by_category(self, category):
        return [f for f in self.memory["facts"] if f["category"] == category]

    def get_facts_by_pillar(self, pillar):
        return [f for f in self.memory["facts"] if pillar in f["key"] or pillar in f.get("tags", [])]

    def get_all_facts(self):
        return self.memory["facts"]

    def get_pillar_coverage(self):
        return self.memory.get("pillar_coverage", {})

    def _update_pillar_coverage(self, key, confidence):
        from .config import PILLARS
        for pillar in PILLARS:
            if pillar in key:
                coverage = self.memory.setdefault("pillar_coverage", {})
                if pillar not in coverage:
                    coverage[pillar] = {"confidence": 0, "last_updated": None, "fact_count": 0}
                coverage[pillar]["fact_count"] = coverage[pillar].get("fact_count", 0) + 1
                coverage[pillar]["confidence"] = max(coverage[pillar]["confidence"], confidence)
                coverage[pillar]["last_updated"] = now_iso()
                break

    def get_known_pillars(self):
        coverage = self.get_pillar_coverage()
        return {p: v for p, v in coverage.items() if v.get("fact_count", 0) > 0}

    def get_unknown_pillars(self):
        known = self.get_known_pillars()
        from .config import PILLARS
        return [p for p in PILLARS if p not in known]

    def get_pillar_recency(self, pillar):
        coverage = self.get_pillar_coverage()
        if pillar in coverage and coverage[pillar].get("last_updated"):
            return days_since(coverage[pillar]["last_updated"])
        return 999

    def get_trust_score(self):
        return self.memory.get("trust_score", 30)

    def adjust_trust_score(self, delta):
        self.memory["trust_score"] = max(0, min(100, self.memory.get("trust_score", 30) + delta))
        self.save()

    def mark_avoided_pillar(self, pillar):
        avoided = self.memory.setdefault("avoided_pillars", {})
        avoided[pillar] = avoided.get(pillar, 0) + 1
        if avoided[pillar] >= 2:
            if pillar not in self.memory.setdefault("deprioritized_pillars", []):
                self.memory["deprioritized_pillars"].append(pillar)
        self.save()

    def get_avoided_pillars(self):
        return self.memory.get("avoided_pillars", {})

    def get_deprioritized_pillars(self):
        return self.memory.get("deprioritized_pillars", [])

    def get_emotional_history(self, limit=10):
        facts = self.get_facts_by_category("emotional_history")
        return sorted(facts, key=lambda f: f.get("last_updated", ""), reverse=True)[:limit]

    def get_habit_trends(self):
        habits = self.get_facts_by_category("habits")
        trends = {}
        for h in habits:
            key_base = h["key"].split("_trend")[0] if "_trend" in h["key"] else h["key"]
            if key_base not in trends:
                trends[key_base] = []
            trends[key_base].append(h)
        return trends

    def extract_facts_from_message(self, message):
        extracted = []
        message_lower = message.lower()

        sleep_patterns = [
            ("sleep_hours", r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s*)?sleep', 80),
            ("sleep_quality", r'sleep\s*(?:quality|was|is)\s*(bad|poor|terrible|okay|good|great|amazing)', 70),
            ("bedtime", r'(?:went to|hit the|to)\s*bed\s*(?:at\s*)?(\d+\s*(?::\d{2})\s*(?:am|pm)?)', 60),
        ]

        for key, pattern, conf in sleep_patterns:
            match = re.search(pattern, message_lower)
            if match:
                value = match.group(1) if match.groups() else "mentioned"
                extracted.append(("habits", key, value, conf))

        mood_patterns = [
            (r"\b(i'm?\s*(?:feeling|feel)\s*(sad|depressed|down|unhappy|miserable))\b", "mood_state", 75),
            (r"\b(i'm?\s*(?:feeling|feel)\s*(anxious|worried|nervous|stressed))\b", "mood_state", 75),
            (r"\b(i'm?\s*(?:feeling|feel)\s*(happy|good|great|okay|fine))\b", "mood_state", 60),
            (r"\b(i'm?\s*(?:feeling|feel)\s*(lonely|alone))\b", "mood_state", 75),
            (r"\b(i'm?\s*(?:feeling|feel)\s*(tired|exhausted|drained))\b", "mood_state", 65),
        ]

        for pattern, key, conf in mood_patterns:
            match = re.search(pattern, message_lower)
            if match:
                value = match.group(1)
                extracted.append(("emotional_history", key, value, conf))
                break

        work_patterns = [
            (r"\b(work|job|career)\s*(is|has been)\s*(stressful|busy|hectic|overwhelming)\b", "work_stress", 75),
            (r"\b(deadlines?|overwork|burnout)\b", "work_stress", 80),
        ]
        for pattern, key, conf in work_patterns:
            if re.search(pattern, message_lower):
                extracted.append(("emotional_history", key, "high", conf))
                break

        if "exercise" in message_lower or "workout" in message_lower or "gym" in message_lower:
            nums = re.findall(r'\b(\d+)\s*(?:times?|days?|x)\b', message_lower)
            value = f"{nums[0]}x/week" if nums else "mentioned"
            extracted.append(("habits", "exercise_frequency", value, 65))

        if "eat" in message_lower or "food" in message_lower or "diet" in message_lower or "meal" in message_lower:
            extracted.append(("habits", "nutrition_mentioned", "true", 50))

        if "stress" in message_lower or "stressed" in message_lower:
            extracted.append(("emotional_history", "stress_level", "elevated", 70))

        return extracted

    def get_session_summary(self):
        return {
            "user_id": self.user_id,
            "trust_score": self.get_trust_score(),
            "facts_count": len(self.memory["facts"]),
            "known_pillars": list(self.get_known_pillars().keys()),
            "unknown_pillars": self.get_unknown_pillars(),
            "session_count": self.memory.get("session_count", 0),
            "last_updated": self.memory.get("last_updated", "")
        }

    def increment_session(self):
        self.memory["session_count"] = self.memory.get("session_count", 0) + 1
        self.save()
