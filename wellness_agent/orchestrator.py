from .agents import AgentRegistry
from .state_machine import ConversationStateMachine
from .config import PRODUCT_CONTEXT, APP_NAME
from .utils.storage import load_json, save_json, now_iso
from .config import get_user_session_path
from .llm_service import GroqLLM
import itertools
import warnings

SHORT_DEFLECTIONS = frozenset([
    "no", "idk", "nah", "nope", "not sure", "i don't know",
    "i dunno", "i don't think so", "not really", "maybe",
    "i guess", "whatever", "fine", "okay", "ok",
])

# Message variants per state — cycles through these to avoid verbatim repeats
_MESSAGE_VARIANTS = {
    "free_conversation": [
        "I'm here to listen. Tell me more about that.",
        "I'm listening — what's on your mind?",
        "Go on, I'm right here with you.",
        "Thanks for sharing. What else comes to mind?",
    ],
    "rapport_building": [
        "I appreciate you sharing. How's your day been, genuinely?",
        "Thanks for telling me that. How are you doing right now?",
        "I'm glad you're here. How's your day going so far?",
    ],
    "avoidance_detection": [
        "No pressure — want to keep talking, switch topics, or just check in later?",
        "We can go at whatever pace works for you. What feels best right now?",
        "Whenever you're ready — we can explore something, take a break, or just chat.",
    ],
    "soft_exploration": [
        "Sometimes it's hard to put into words. Want to start anywhere?",
        "No need to have the perfect answer. What comes to mind first?",
        "We don't need a big topic — a small thing works too. What's one thought?",
    ],
    "insight_generation": [
        "Does that resonate with you? We can explore it more or think about what might help.",
        "What do you make of that? Does it feel true to your experience?",
        "How does that land with you? We can sit with it or move toward next steps.",
    ],
    "follow_up": [
        "It's good to talk with you again. How has your week been since our last conversation?",
        "Welcome back. How have things been since we last checked in?",
        "Great to see you again. What's been happening since we talked?",
    ],
    "default": [
        "I hear you. Can you tell me a bit more about that?",
        "Thanks for saying that. Want to unpack it a little more?",
        "I'm following. What else is on your mind about this?",
    ],
}


class Orchestrator:
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.agents = AgentRegistry(user_id)
        self.state_machine = ConversationStateMachine(self.agents.memory)
        self.context = PRODUCT_CONTEXT.format(APP_NAME=APP_NAME)
        self.last_turns = []
        self.current_pillar = None
        self.current_insight = None
        self.current_routine = None
        self.last_question = None
        self.avoidance_count = 0
        self._exit_offered = False
        self._exit_consumed = False  # one-shot: never re-trigger exit after handling
        self._last_user_message = ""
        self._last_response_text = None
        self._last_response_state = None
        self._repeat_count = 0
        self._response_cyclers = {k: itertools.cycle(v) for k, v in _MESSAGE_VARIANTS.items()}
        self.session_path = get_user_session_path(user_id)
        self.llm = GroqLLM()
        self._load_session()

    def process_message(self, user_message):
        turn_result = {
            "user_message": user_message,
            "risk_detected": False,
            "emotion": None,
            "state": None,
            "response": None,
            "options": None,
            "insight": None,
            "routine": None,
            "route": [],
            "memory_updates": [],
            "llm_used": self.llm.is_available()
        }

        # Phase 5: Emotion extraction (LLM + rule hybrid)
        emotion = self._analyze_emotion(user_message)
        turn_result["emotion"] = emotion

        if emotion.get("risk_flag"):
            turn_result["risk_detected"] = True
            turn_result["route"] = ["risk_protocol"]
            turn_result["response"] = self._risk_response(emotion.get("risk_reason", ""))
            turn_result["state"] = self.state_machine.get_state_info()
            self._save_turn(user_message, turn_result)
            return turn_result

        self._last_user_message = user_message

        # ─── Hard Avoidance Counter (deterministic, not LLM) ───
        if user_message.strip():
            words = user_message.strip().split()
            is_deflecting = (
                len(words) < 3
                or user_message.strip().lower() in SHORT_DEFLECTIONS
            )
            if is_deflecting:
                self.avoidance_count += 1
            else:
                self.avoidance_count = 0

        if self.avoidance_count == 2 and self.state_machine.current_state in (
            "guided_discovery", "pillar_selection", "deep_investigation"
        ):
            self.state_machine.set_state("rapport_building")

        # ─── Handle reply to exit-offer message ───
        if self._exit_offered and user_message.strip():
            self._exit_offered = False
            self._exit_consumed = True
            self.avoidance_count = 0
            msg_lower = user_message.strip().lower()
            if msg_lower in ("yes", "yep", "sure", "ok", "okay"):
                turn_result["response"] = "Sounds good — I'll check in tomorrow. Take care."
                turn_result["state"] = self.state_machine.get_state_info()
                self._save_turn(user_message, turn_result)
                return turn_result
            else:
                turn_result["response"] = "No worries — want to talk about something else instead, or just chat casually for now?"
                turn_result["options"] = ["Talk about something", "Just chat", "Maybe later"]
                turn_result["state"] = self.state_machine.get_state_info()
                self._save_turn(user_message, turn_result)
                return turn_result

        # Phase 4: Memory extraction (LLM-powered)
        memory_updates = self._extract_memory(user_message, emotion)
        turn_result["memory_updates"] = memory_updates

        if emotion.get("avoidance", 0) > 60:
            self.agents.memory.adjust_trust_score(-3)
        elif emotion.get("engagement", 0) > 60:
            self.agents.memory.adjust_trust_score(2)

        # Phase 3: State transition
        prev_state = self.state_machine.current_state
        next_state = self.state_machine.transition(emotion, user_message)
        if prev_state == "greeting" and next_state != "greeting":
            self.agents.memory.increment_session()
        turn_result["state"] = self.state_machine.get_state_info()

        # Phase 2: LLM-powered routing
        route = self._decide_route(emotion, user_message)
        turn_result["route"] = route

        # Generate response (returns {"text": "...", "options": [...]})
        resp_data = self._generate_response(route, emotion, user_message)
        turn_result["response"] = resp_data["text"]
        turn_result["options"] = resp_data["options"]

        self._save_turn(user_message, turn_result)
        self.last_turns.append({
            "user": user_message,
            "assistant": resp_data["text"],
            "state": next_state,
            "emotion": {k: v for k, v in emotion.items() if k in ("primary_emotion", "emotional_intensity", "risk_flag")}
        })
        if len(self.last_turns) > 10:
            self.last_turns = self.last_turns[-10:]

        return turn_result

    # ─── LLM-Integrated Phase 5: Emotion ──────────────────────

    def _analyze_emotion(self, user_message):
        llm_result = self.llm.extract_emotion(user_message, self.last_turns)
        if llm_result and llm_result.get("primary_emotion"):
            return llm_result
        return self.agents.emotion_engine.analyze(user_message, self.last_turns[-3:])

    # ─── LLM-Integrated Phase 4: Memory ───────────────────────

    def _extract_memory(self, user_message, emotion):
        existing = self.agents.memory.get_session_summary()
        llm_facts = self.llm.extract_memory(user_message, existing)

        if llm_facts:
            stored = []
            for fact in llm_facts:
                if isinstance(fact, dict) and fact.get("key") and fact.get("value"):
                    result = self.agents.memory.add_fact(
                        category=fact.get("category", "identity"),
                        key=fact["key"],
                        value=fact["value"],
                        confidence=min(95, fact.get("confidence", 70)),
                        source=fact.get("source", "conversation")
                    )
                    stored.append(result)
            if stored:
                return stored

        return self.agents.extract_and_store(user_message, emotion)

    # ─── Phase 2: LLM-Powered Routing ─────────────────────────

    def _decide_route(self, emotion, user_message):
        state = self.state_machine.current_state
        state_info = self.state_machine.get_state_info()

        # Base: always run these
        route = ["emotion_detection", "memory_manager"]

        # State-based routing rules (always applied)
        if state in ("guided_discovery", "pillar_selection", "deep_investigation"):
            route.append("question_planner")
            if state == "deep_investigation" and self.current_pillar:
                route.append("root_cause_engine")
        if state == "routine_planning" and self.current_insight:
            route.append("routine_generator")
        if state == "insight_generation" and not state_info.get("insight_delivered"):
            if self.current_pillar:
                route.append("root_cause_engine")

        # LLM can supplement but not override state-based routing
        if self.llm.is_available() and user_message.strip():
            mem_snapshot = {
                "state": state,
                "pillar": self.current_pillar,
                "trust_score": self.agents.memory.get_trust_score(),
                "insight_delivered": state_info.get("insight_delivered"),
                "routine_created": state_info.get("routine_created"),
            }
            llm_result = self.llm.route_turn(user_message, state, mem_snapshot, self.last_turns)
            if llm_result and "route" in llm_result:
                extra = [a for a in llm_result["route"] if a not in route and a != "risk_protocol"]
                route.extend(extra)

        return route

    # ─── Response Generation (LLM-enhanced) ───────────────────

    def _generate_response(self, route, emotion, user_message):
        state = self.state_machine.current_state
        text = ""
        options = None

        if state == "greeting":
            session_num = self.agents.memory.memory.get("session_count", 1)
            known = self.agents.memory.get_known_pillars()
            opts = self.agents.question_planner.generate_question(
                "mood", "greeting", memory_context={})
            options = opts.get("options") or opts.get("response_options")
            if session_num <= 1:
                text = "Hey there. I'm your wellness companion. I'm here to help you understand yourself better — no judgment, no agenda. How are you feeling today?"
            else:
                pillars_known = list(known.keys())
                if pillars_known:
                    text = f"Welcome back. Last time we touched on {pillars_known[0]}. How have things been since we talked?"
                else:
                    text = "Welcome back. I'm glad you're here. What's on your mind today?"

        elif "question_planner" in route and self.avoidance_count == 1 and not self._exit_consumed:
            # Force choice with concrete options (never re-ask open-ended)
            text = "Which of these areas feels most relevant right now?"
            base = ["😴 Sleep", "💼 Work", "💛 Relationships", "😰 Stress"]
            if self.current_pillar:
                highlighted = [p.title() for p in [self.current_pillar] if p]
                others = [b for b in base if b.split()[1].lower() != (self.current_pillar or "").lower()][:2]
                options = highlighted + others + ["Something else", "Let me explain"]
            else:
                options = base + ["Something else", "Let me explain"]

        elif self.avoidance_count >= 3 and not self._exit_offered and not self._exit_consumed:
            text = "Totally okay — I'm here whenever you want to dig into something. Want me to just check in tomorrow instead?"
            options = ["Yes", "No"]
            self._exit_offered = True

        elif state == "rapport_building" and self.avoidance_count == 2 and not self._exit_consumed:
            text = "No pressure at all. Want to just chat about your day instead?"
            options = ["Sure", "Not really", "Maybe later"]

        elif "question_planner" in route:
            q_data = self._generate_question(emotion)
            text = q_data["question_text"]
            options = q_data.get("response_options")

        elif "root_cause_engine" in route and self.current_pillar and "question_planner" not in route:
            text = self._generate_insight()
            options = ["Yes, that's it", "Partly", "Not quite"]

        elif "routine_generator" in route:
            text = self._generate_routine_suggestion()
            options = ["All of them", "Just one", "Not right now"]

        elif state == "reflection":
            llm_close = self.llm.generate_reflection(
                self.state_machine.get_state_info(), self.last_turns)
            text = llm_close or self.agents.reflection_agent(self.state_machine.get_state_info())
            options = ["Good for today", "One more thing"]

        elif state == "follow_up":
            text = next(self._response_cyclers["follow_up"])
            options = ["Better", "Same", "Rougher"]

        elif state == "free_conversation":
            text = next(self._response_cyclers["free_conversation"])
            options = None

        elif state == "rapport_building":
            text = next(self._response_cyclers["rapport_building"])
            options = ["🙂 Good", "😑 Meh", "Rough one"]

        elif state == "avoidance_detection":
            text = next(self._response_cyclers["avoidance_detection"])
            options = ["Keep talking", "Switch topics", "Check in later"]

        elif state == "soft_exploration":
            text = next(self._response_cyclers["soft_exploration"])
            options = ["A specific thing", "Just talking helps", "Not sure yet"]

        elif state == "insight_generation":
            text = next(self._response_cyclers["insight_generation"])
            options = ["Yes, that's it", "Partly", "Not quite"]

        else:
            text = next(self._response_cyclers["default"])
            options = None

        # ─── Repetition safeguard ───
        if text == self._last_response_text and state == self._last_response_state:
            self._repeat_count += 1
            warnings.warn(
                f"[REPEAT] State '{state}' produced verbatim repeat #{self._repeat_count} "
                f"(user: {user_message!r}) — route was {route}"
            )
            # Force a variant from a different tone to break the loop
            forced = next(self._response_cyclers.get(state, self._response_cyclers["default"]))
            if forced != text:
                text = forced
            else:
                text = next(self._response_cyclers["default"])
        else:
            self._repeat_count = 0
            self._last_response_text = text
            self._last_response_state = state

        return {"text": text, "options": options}

    # ─── LLM-Integrated Phase 7: Questions ────────────────────

    def _generate_question(self, emotion):
        if self.state_machine.current_state == "deep_investigation" and self.current_pillar:
            pillar = self.current_pillar
        else:
            known = self.agents.memory.get_known_pillars()
            unknown = self.agents.memory.get_unknown_pillars()
            planner_result = self.agents.planner.select_target_pillar(
                known_pillars=known,
                unknown_pillars=unknown,
                current_state=self.state_machine.current_state,
                latest_emotion_scores=emotion,
                user_message=self._last_user_message
            )
            pillar = planner_result["target_pillar"]
            if self.state_machine.current_state in ("guided_discovery", "pillar_selection"):
                self.state_machine.select_pillar(pillar)
                self.current_pillar = pillar
                # Reset deep count on new pillar
                self.agents.question_planner.reset_deep_count()

        # Try LLM-generated question first
        llm_q = self.llm.generate_question(
            target_pillar=pillar,
            current_state=self.state_machine.current_state,
            memory_context={"trust_score": self.agents.memory.get_trust_score(),
                            "pillar": pillar,
                            "recent_topic": self._last_user_message}
        )
        if llm_q and llm_q.get("question_text"):
            result = {
                "question_text": llm_q["question_text"],
                "response_options": llm_q.get("response_options") or self._fallback_options_for_state()
            }
            self.last_question = result
            return result

        # Fallback to template
        q_data = self.agents.question_planner.generate_question(
            target_pillar=pillar,
            current_state=self.state_machine.current_state,
            memory_context={"trust_score": self.agents.memory.get_trust_score()}
        )
        result = {
            "question_text": q_data.get("question_text") or q_data.get("text", ""),
            "response_options": q_data.get("response_options") or q_data.get("options") or self._fallback_options_for_state()
        }
        self.last_question = result
        return result

    def _fallback_options_for_state(self):
        state = self.state_machine.current_state
        opts = {
            "guided_discovery": ["Yes, that's it", "Also something else", "Not really that"],
            "pillar_selection": ["This one", "Something else", "Let me explain"],
            "deep_investigation": ["Tell me more", "Something else", "Let me explain"],
            "routine_planning": ["All of them", "Just one", "Not right now"],
        }
        return opts.get(state, ["Yes", "No", "Let me explain"])

    # ─── LLM-Integrated Phase 8: Root Cause ───────────────────

    def _generate_insight(self):
        facts = self.agents.memory.get_facts_by_pillar(self.current_pillar)
        emotions = self.agents.memory.get_emotional_history()
        habits = self.agents.memory.get_habit_trends()

        llm_result = self.llm.analyze_root_cause(self.current_pillar, facts, emotions, habits)
        if llm_result and llm_result.get("likely_root_cause"):
            result = llm_result
        else:
            result = self.agents.root_cause_analyzer.analyze(
                pillar=self.current_pillar,
                memory_facts=facts,
                emotion_history=emotions,
                habit_trends=habits
            )

        self.current_insight = result
        chain_summary = "; ".join(
            f"{o['observation']} ({o['confidence']}% confidence)"
            for o in result.get("chain", [])
        )
        response = f"Based on what you've shared, here's what I'm noticing: {chain_summary}. "
        response += f"The pattern that stands out most is: {result['likely_root_cause']}. "
        response += "This is based on what you've told me — it's not a diagnosis, just a way of connecting the dots."

        self.state_machine.mark_insight_delivered()
        return response

    # ─── LLM-Integrated Phase 9: Routines ─────────────────────

    def _generate_routine_suggestion(self):
        if not self.current_insight:
            self.current_insight = {"likely_root_cause": "building on what we've discussed", "probability": 50}

        facts = self.agents.memory.get_all_facts()
        llm_routine = self.llm.generate_routine(
            root_cause_or_goal=self.current_insight["likely_root_cause"],
            memory_facts=facts,
            past_adherence={},
            constraints={}
        )

        if llm_routine and llm_routine.get("actions"):
            routine = llm_routine
        else:
            routine = self.agents.routine_generator.generate(
                root_cause_or_goal=self.current_insight["likely_root_cause"],
                memory_facts=facts
            )

        self.current_routine = routine
        self.state_machine.mark_routine_created()
        actions_text = "\n".join(
            f"• {a['action']} ({a['time_of_day']}) — {a['why']}"
            for a in routine.get("actions", [])
        )
        return f"Here's a small plan based on what we've explored:\n{actions_text}\n\nWe can check in on how it's going. Sound good?"

    # ─── Crisis Response (LLM-enhanced) ───────────────────────

    def _risk_response(self, reason):
        llm_response = self.llm.generate_crisis_response(self._last_user_message, reason)
        if llm_response:
            return llm_response
        return ("I'm really glad you told me this. What you're feeling matters, and you're not alone. "
                "Please reach out to a crisis service — they're trained to help right now. "
                "In the US, you can call or text 988 for immediate support. You matter, and help is available.")

    # ─── Persistence ──────────────────────────────────────────

    def _save_turn(self, user_message, turn_result):
        session = load_json(self.session_path)
        if not session:
            session = {"user_id": self.user_id, "turns": [], "created": now_iso()}
        turn = {
            "timestamp": now_iso(),
            "user_message": user_message,
            "response": turn_result.get("response"),
            "state": turn_result.get("state"),
            "emotion_summary": {
                "primary": turn_result.get("emotion", {}).get("primary_emotion"),
                "intensity": turn_result.get("emotion", {}).get("emotional_intensity"),
                "risk": turn_result.get("risk_detected")
            }
        }
        session["turns"].append(turn)
        session["last_updated"] = now_iso()
        save_json(self.session_path, session)

    def _load_session(self):
        data = load_json(self.session_path)
        if data and "turns" in data:
            last_turns = data["turns"][-10:]
            self.last_turns = [
                {"user": t.get("user_message"), "assistant": t.get("response"),
                 "state": t.get("state"), "emotion": t.get("emotion_summary", {})}
                for t in last_turns
            ]

    def get_summary(self):
        return {
            "user": self.user_id,
            "state": self.state_machine.get_state_info(),
            "memory": self.agents.memory.get_session_summary(),
            "current_pillar": self.current_pillar,
            "trust_score": self.agents.memory.get_trust_score(),
            "last_turns_count": len(self.last_turns),
            "llm_available": self.llm.is_available()
        }

    def reset_state(self):
        self.state_machine = ConversationStateMachine(self.agents.memory)
        self.current_pillar = None
        self.current_insight = None
        self.current_routine = None
        self.last_question = None
        self.avoidance_count = 0
        self._exit_offered = False
        self._exit_consumed = False
        self._last_response_text = None
        self._last_response_state = None
        self._repeat_count = 0
        self._response_cyclers = {k: itertools.cycle(v) for k, v in _MESSAGE_VARIANTS.items()}
