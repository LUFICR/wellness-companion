from datetime import datetime, timedelta
from .utils.storage import load_json, save_json, now_iso, days_since
from .config import get_report_path


class ReportGenerator:
    def __init__(self, memory_system=None):
        self.memory = memory_system

    def generate(self, period="daily", metrics=None, prior_period_metrics=None, achievements=None):
        metrics = metrics or {}
        prior_period_metrics = prior_period_metrics or {}
        achievements = achievements or []

        if not metrics and self.memory:
            metrics = self._build_metrics_from_memory()
            prior_period_metrics = self._build_prior_metrics()

        trends = self._compute_trends(metrics, prior_period_metrics)
        observations = self._generate_observations(trends, metrics)
        suggested_goals = self._generate_goals(trends, period)
        summary = self._generate_summary(trends, period, achievements)

        report = {
            "period": period,
            "generated_at": now_iso(),
            "summary": summary,
            "trends": trends,
            "observations": observations,
            "suggested_goals": suggested_goals
        }

        if self.memory:
            path = get_report_path(self.memory.user_id, period)
            save_json(path, report)

        return report

    def _build_metrics_from_memory(self):
        facts = self.memory.get_all_facts() if self.memory else []
        metrics = {
            "mood_avg": 65,
            "stress_avg": 40,
            "sleep_avg": None,
            "exercise_days": 0,
            "social_rating": 50,
            "energy_avg": 55,
            "motivation_avg": 50,
        }

        emotion_facts = [f for f in facts if f.get("category") == "emotional_history"]
        if emotion_facts:
            sleep_fact = next((f for f in facts if "sleep" in f.get("key", "")), None)
            if sleep_fact:
                try:
                    metrics["sleep_avg"] = float(str(sleep_fact["value"]).replace("h", "").strip())
                except (ValueError, TypeError):
                    pass

        return metrics

    def _build_prior_metrics(self):
        return {
            "mood_avg": 62,
            "stress_avg": 45,
            "energy_avg": 52,
            "motivation_avg": 48,
        }

    def _compute_trends(self, current, prior):
        trends = []
        all_metrics = set(list(current.keys()) + list(prior.keys()))

        for metric in all_metrics:
            curr_val = current.get(metric)
            prior_val = prior.get(metric)
            if curr_val is None:
                continue

            direction = "flat"
            change = None
            if prior_val is not None and prior_val != 0:
                diff = curr_val - prior_val
                change = f"{'+' if diff > 0 else ''}{diff:.1f}"
                if diff > 5:
                    direction = "up"
                elif diff < -5:
                    direction = "down"

            trends.append({
                "metric": metric,
                "direction": direction,
                "value": f"{curr_val}" if curr_val is not None else "N/A",
                "change": change or "N/A"
            })

        return trends

    def _generate_observations(self, trends, metrics):
        observations = []

        stress = next((t for t in trends if "stress" in t["metric"]), None)
        sleep = next((t for t in trends if "sleep" in t["metric"]), None)
        mood = next((t for t in trends if "mood" in t["metric"]), None)
        energy = next((t for t in trends if "energy" in t["metric"]), None)

        if stress and sleep and stress.get("value"):
            try:
                if float(stress["value"]) > 60 and sleep.get("value") and float(str(sleep.get("value", "0")).replace("N/A", "0")) < 6:
                    observations.append("Higher stress levels and shorter sleep appear connected — addressing one may help the other.")
            except (ValueError, TypeError):
                pass

        if mood and energy:
            try:
                if float(mood.get("value", 50)) < 55 and float(energy.get("value", 50)) < 50:
                    observations.append("Lower mood and energy levels are tracking together — small wins in either area could lift both.")
            except (ValueError, TypeError):
                pass

        if not observations:
            observations.append("Data is still building for deeper pattern observations.")

        return observations

    def _generate_goals(self, trends, period):
        goals = []

        stress = next((t for t in trends if "stress" in t["metric"]), None)
        sleep = next((t for t in trends if "sleep" in t["metric"]), None)
        exercise = next((t for t in trends if "exercise" in t["metric"]), None)
        energy = next((t for t in trends if "energy" in t["metric"]), None)

        if stress and stress.get("direction") == "up":
            goals.append("Identify one source of stress this week and test one small way to reduce it.")
        if sleep and (sleep.get("direction") == "down" or sleep.get("value") == "N/A"):
            goals.append("Prioritize sleep consistency — aim for a 30-minute earlier bedtime.")
        if energy and energy.get("direction") == "down":
            goals.append("Add one 10-minute walk daily to see if it shifts energy levels.")

        if not goals:
            goals.append("Continue tracking to discover meaningful patterns.")

        return goals[:3]

    def _generate_summary(self, trends, period, achievements):
        positive = sum(1 for t in trends if t.get("direction") == "up")
        negative = sum(1 for t in trends if t.get("direction") == "down")

        if achievements:
            achievement_text = "; ".join(achievements[:2])
            return f"Over the last {period}, you made progress on {achievement_text}. " \
                   f"Tracking shows {positive} areas improving, {negative} areas needing attention."
        return f"Over the last {period}, tracking shows {positive} areas heading in a positive direction " \
               f"and {negative} areas to keep an eye on. Consistency with tracking is the foundation for insight."

    def generate_weekly(self):
        return self.generate("weekly")

    def generate_monthly(self):
        return self.generate("monthly")

    def get_report(self, period="daily"):
        if not self.memory:
            return None
        path = get_report_path(self.memory.user_id, period)
        data = load_json(path)
        return data if data else None
