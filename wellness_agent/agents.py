from .emotion_engine import EmotionEngine
from .memory import MemorySystem
from .root_cause import RootCauseAnalyzer
from .question_planner import QuestionPlanner
from .routine_generator import RoutineGenerator
from .reports import ReportGenerator
from .conversation_planner import ConversationPlanner


class AgentRegistry:
    def __init__(self, user_id="default"):
        self.memory = MemorySystem(user_id)
        self.emotion_engine = EmotionEngine(self.memory)
        self.planner = ConversationPlanner(self.memory)
        self.question_planner = QuestionPlanner(self.memory)
        self.root_cause_analyzer = RootCauseAnalyzer(self.memory)
        self.routine_generator = RoutineGenerator(self.memory)
        self.report_generator = ReportGenerator(self.memory)

    def get_agent(self, name):
        registry = {
            "emotion_detection": self.emotion_engine.analyze,
            "memory_manager": self.extract_and_store,
            "root_cause_engine": self.root_cause_analyzer.analyze,
            "question_planner": self.question_planner.generate_question,
            "routine_generator": self.routine_generator.generate,
            "report_generator": self.report_generator.generate,
            "conversation_planner": self.planner.select_target_pillar,
            "reflection_agent": self.reflection_response,
        }
        return registry.get(name)

    def extract_and_store(self, message, emotion_result=None):
        if not message:
            return []

        facts = self.memory.extract_facts_from_message(message)

        if emotion_result and emotion_result.get("primary_emotion") != "neutral":
            facts.append({
                "action": "add",
                "category": "emotional_history",
                "key": f"emotion_{emotion_result['primary_emotion']}",
                "value": emotion_result["primary_emotion"],
                "confidence": emotion_result.get("confidence", 60),
                "source": "conversation"
            })

        stored = []
        for fact in facts:
            if isinstance(fact, dict):
                stored.append(self.memory.add_fact(
                    fact.get("category", "identity"),
                    fact.get("key", "unknown"),
                    fact.get("value", "mentioned"),
                    fact.get("confidence", 50),
                    fact.get("source", "conversation")
                ))
            elif len(fact) >= 4:
                category, key, value, confidence = fact[0], fact[1], fact[2], fact[3]
                source = fact[4] if len(fact) > 4 else "conversation"
                result = self.memory.add_fact(category, key, value, confidence, source)
                stored.append(result)

        return stored

    def reflection_response(self, state_info=None):
        state_info = state_info or {}
        if state_info.get("routine_created"):
            return "You've built a solid plan today. How do you feel about the steps you've set up?"
        if state_info.get("insight_delivered"):
            return "It sounds like today brought some useful clarity. Anything you want to hold onto from this conversation?"
        return "We've covered a lot today. How are you feeling about what came up?"
