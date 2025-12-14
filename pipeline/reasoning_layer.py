import re
from structures import PipelineState

class ReasoningLayer:
    def process(self, state: PipelineState) -> PipelineState:
        # 1. Fact Extraction (Highest Priority)
        # Prevents "I hate pasta" from being treated as a Denial/Contradiction
        if state.features["has_fact_pattern"] or state.intent == "FACT_TEACH":
            m_pref = re.search(r"\bi (like|love|hate|dislike)s? (.+)", state.raw_text, re.IGNORECASE)
            if m_pref:
                verb = m_pref.group(1).lower()
                obj = m_pref.group(2).strip()
                key = f"preference:{obj}"
                
                existing_val = state.user_profile.facts.get(key)
                if existing_val and existing_val != verb:
                    state.plan.append({
                        "op": "CONFIRM_FACT", 
                        "key": key, 
                        "val": verb, 
                        "old_val": existing_val, 
                        "obj_name": obj
                    })
                else:
                    state.plan.append({"op": "SAVE_FACT", "key": key, "val": verb})
                return state

            m_attr = re.search(r"my (\w+) is (.+)", state.raw_text, re.IGNORECASE)
            if m_attr:
                key = m_attr.group(1).lower()
                val = m_attr.group(2).strip()
                existing_val = state.user_profile.facts.get(key)
                if existing_val and existing_val != val:
                     state.plan.append({"op": "CONFIRM_FACT", "key": key, "val": val, "old_val": existing_val, "obj_name": key})
                else:
                    state.plan.append({"op": "SAVE_FACT", "key": key, "val": val})
                return state

        # 2. Self-Correction (Only if NOT a fact teach)
        if state.features["is_contradiction"] and state.intent != "FACT_TEACH":
            state.thought_trace.append("Reasoning: Contradiction detected. Initiating repair.")
            state.plan.append({"op": "REPAIR_MEMORY"})
            return state

        # 3. Smalltalk / Emotion / Affirmation
        # If the bot doesn't know the intent but the user is just saying "Cool" or "Thanks"
        if (state.act == "AFFIRM" or state.act == "EMOTION") and state.intent == "UNKNOWN":
             state.plan.append({"op": "RESPOND", "text": "Glad to hear it!", "style": "SMALLTALK"})
             return state

        # 4. Compound Query
        if state.features["is_compound"] and state.intent == "QA_SEARCH":
            parts = state.raw_text.split(" and ")
            if len(parts) == 2:
                state.plan.append({"op": "RETRIEVE", "query_text": parts[0]})
                state.plan.append({"op": "RETRIEVE", "query_text": parts[1]})
                state.plan.append({"op": "SYNTHESIZE"}) 
                return state

        # 5. Default Plan
        if state.intent == "QA_SEARCH" or state.act == "REQUEST_INFO":
            state.plan.append({"op": "RETRIEVE", "query_vector": state.query_vector})
        elif state.intent == "FAREWELL":
            state.plan.append({"op": "EXIT"})
        elif state.intent == "GREETING":
            state.plan.append({"op": "RESPOND", "text": "Hello! I am ready."})
        elif state.intent == "META" and "wipe" in state.norm_text:
            state.plan.append({"op": "WIPE"})
        else:
            state.plan.append({"op": "FALLBACK_TEACH"})
            
        return state