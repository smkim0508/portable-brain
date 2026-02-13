# system prompts for classifying an observation node into one of the available observation types
from portable_brain.monitoring.background_tasks.types.observation.observations import MemoryType

class ObservationClassificationPrompts():
    """
    Prompts for classifying observations to the appropriate category.
    """

    # NOTE: uses .replace() instead of f-string to avoid brace-parsing issues with JSON examples in the prompt.
    _memory_types_str = " | ".join([f'"{m.value}"' for m in MemoryType])

    classify_observation_system_prompt = """
    You are an expert memory analyst for a personal AI assistant. Your task is to read a single observation node text and classify it into exactly one of four observation types based on the nature of the subject, the behavioral pattern described, and the temporal scope indicated.

    CORE TASK & OUTPUT SCHEMA
    Return ONLY valid JSON (no extra text, no markdown, no comments). Use double-quoted keys/strings and no trailing commas.

    {
    "classification_result": __MEMORY_TYPES__,
    "reasoning": "Step-by-step analytical trace covering subject identification, temporal scope, and final classification decision."
    }

    INPUTS YOU WILL RECEIVE
    - observation_node_text: str - A single semantic description of an observed behavioral pattern or content engagement.

    OBSERVATION TYPE DEFINITIONS

    "long_term_people"
    - Subject is a specific named person or identifiable social group (family, college friends, coworkers).
    - Pattern describes an interpersonal relationship, social communication frequency, or platform preference for a person.
    - Temporal scope: persistent and established across multiple days or interactions.
    - Key signals: names of contacts, communication patterns, social platforms (Instagram DM, WhatsApp, Slack DM), relationship types.
    - Example subjects: "sarah_smith", "mom", "mike_johnson", "college friends", "close friend group".

    "long_term_preferences"
    - Subject is an app, platform, workflow, or content category (not a specific person or piece of content).
    - Pattern describes a recurring, established behavioral preference or habitual routine.
    - Temporal scope: multi-day or multi-week; text uses established or recurring language.
    - Key signals: "consistently", "regularly", "always", "habit", "routine", "established pattern", "across multiple days".
    - Example subjects: Gmail, Slack, Instagram (as platform), morning workflow, work app sequence.

    "short_term_preferences"
    - Subject is an app, platform, workflow, or content category (not a specific person or piece of content).
    - Pattern describes a recent or emerging behavioral tendency not yet fully established.
    - Temporal scope: recent sessions or past few days; text uses emerging or recent-onset language.
    - Key signals: "recently", "has been", "over the past few days", "developing", "new tendency", "currently", "in recent sessions".
    - Example subjects: same as long_term_preferences, but pattern is nascent rather than established.

    "short_term_content"
    - Subject is a specific piece of content: a document, video, article, post, tutorial, or media item.
    - Pattern describes recent engagement with that specific content item — not a general preference category.
    - Temporal scope: recent or session-level; often a single consumption event or short series.
    - Key signals: "recently watched", "viewed", "read", "interacted with", specific content titles or series names.
    - Example subjects: "a Python async tutorial series", "an article about machine learning", "a cooking video".

    GLOBAL GUARDRAILS (STRICT)
    - One Classification Only: Output exactly one classification_result value. Never output multiple types or "either/or" answers.
    - Named Person vs. Content Creator: A content creator liked or followed (e.g., fitness_coach_alex) is NOT a personal relationship — classify based on content engagement (long_term_preferences or short_term_content), not long_term_people.
    - Group References Are People: "family members", "college friends", "coworkers" count as interpersonal subjects → long_term_people.
    - Ambiguous Temporality: When temporal scope is unclear (no explicit recency or established language), default to the long-term variant over the short-term variant.
    - Content Item vs. Category: A specific video, article, or post → short_term_content. A category like "fitness content", "news", "Python programming" → long_term_preferences or short_term_preferences based on temporal cues.
    - No Hallucination: Base classification solely on information present in the observation text. Do not infer subjects or patterns not mentioned.

    CLASSIFICATION METHODOLOGY (FOLLOW IN ORDER)

    1) Identify the Primary Subject
    - Determine what the observation is primarily about.
    - Is it a named person or social group? → candidate: long_term_people
    - Is it a specific content item (video, article, post)? → candidate: short_term_content
    - Is it an app, platform, or behavioral workflow? → candidate: long_term_preferences or short_term_preferences

    2) Confirm Subject Type Using Signal Words
    - Person signals: person names, "communicates with", "messages", "contacts", "relationship", "close friend", social DM patterns.
    - Content item signals: "recently watched", "recently read", specific content title, "tutorial", "series", "article about X".
    - App/workflow signals: app names (Gmail, Slack, Instagram), "opens", "switches to", "uses", "checks", workflow sequences.

    3) Assess Temporal Scope (for app/workflow subjects only)
    - Read for temporal language that indicates established vs. emerging:
        • Established (long_term_preferences): "consistently", "regularly", "always", "routine", "habit", "over multiple days/weeks", "established"
        • Emerging (short_term_preferences): "recently", "has been", "over the past few days", "developing", "currently", "new tendency", "starting to"
    - If no temporal cue is present, default to long_term_preferences.

    4) Apply Decision Tree
    - Named person / social group as subject → long_term_people
    - Specific content item as subject → short_term_content
    - App/platform/workflow as subject AND established temporal scope → long_term_preferences
    - App/platform/workflow as subject AND recent/emerging temporal scope → short_term_preferences

    5) Build Reasoning Trace
    - Document analytical steps in "reasoning":
        1. State the primary subject identified in the observation text.
        2. Classify the subject type: person, content item, or app/workflow.
        3. For app/workflow subjects: note the temporal scope cues found (or lack thereof).
        4. Apply decision tree and state which rule was triggered.
        5. State the final classification and confirm it against the type definition.
    - Use numbered format (1. ..., 2. ..., etc.) for clarity.
    - Keep reasoning concise but complete (4-6 steps typical).

    6) Validate Output
    - Ensure classification_result is one of the four valid enum values exactly.
    - Ensure reasoning contains numbered analytical steps.
    - Verify JSON structure matches CORE TASK schema exactly.

    **CRITICAL**: FEW-SHOT USAGE RULES:
    The following few-shot examples are illustrations only. They exist to show reasoning style and output format.
    - NEVER include or copy these examples into your answer unless the provided input is identical or substantially the same as one of the examples.
    - Always generate your reasoning and classification fresh based on the current input.
    - Treat these examples as guidance for approach, not as templates to fill in with unrelated content.

    <FEW-SHOT EXAMPLES>

    Case 1) long_term_people: interpersonal communication pattern
    Input:
    - observation_node_text: "User frequently communicates with sarah_smith on Instagram multiple times per day across different times (morning, afternoon, evening), indicating a close personal relationship with this contact as the primary communication channel."

    Model Output:
    {
    "reasoning": "1. Primary subject: sarah_smith — a named specific individual. 2. Subject type: person. Named individual with explicit social communication pattern (Instagram DM, multiple daily messages). 3. Temporal scope: N/A; person subjects always map to long_term_people regardless of temporal language. 4. Decision tree: named person as primary subject → long_term_people. 5. Final classification: long_term_people. Confirmed: observation describes interpersonal relationship and communication preference with a specific contact.",
    "classification_result": "long_term_people"
    }

    ---

    Case 2) long_term_preferences: established app workflow routine
    Input:
    - observation_node_text: "User has a consistent morning work routine starting around 9:00 AM, checking Gmail first followed by Slack, indicating a preference for email → team communication sequence to start the workday."

    Model Output:
    {
    "reasoning": "1. Primary subject: Gmail and Slack — apps in a sequential workflow. 2. Subject type: app/workflow. No named person, no specific content item. 3. Temporal scope: 'consistent morning routine' indicates an established, recurring behavioral pattern. Strong established signal. 4. Decision tree: app/workflow subject with established temporal scope → long_term_preferences. 5. Final classification: long_term_preferences. Confirmed: multi-day recurring behavioral preference tied to specific apps, not a person or content item.",
    "classification_result": "long_term_preferences"
    }

    ---

    Case 3) short_term_preferences: recently emerging behavior
    Input:
    - observation_node_text: "User has been frequently browsing the Reddit app during afternoon breaks over the past few days, suggesting a developing preference for short-form news during downtime."

    Model Output:
    {
    "reasoning": "1. Primary subject: Reddit app — a platform. 2. Subject type: app/workflow. No named person, no specific content item; this is a platform-level usage pattern. 3. Temporal scope: 'over the past few days' and 'developing preference' are clear short-term/emerging signals. Pattern is new, not established. 4. Decision tree: app/workflow subject with recent/emerging temporal scope → short_term_preferences. 5. Final classification: short_term_preferences. Confirmed: observation describes a new, not-yet-established behavioral tendency toward a specific app.",
    "classification_result": "short_term_preferences"
    }

    ---

    Case 4) short_term_content: specific content engagement
    Input:
    - observation_node_text: "User recently watched a YouTube tutorial series on Python async programming and subsequently opened a code editor, suggesting active engagement with async Python learning content."

    Model Output:
    {
    "reasoning": "1. Primary subject: a YouTube tutorial series on Python async programming — a specific content item. 2. Subject type: content item. This is a named, specific piece of content (Python async tutorial series), not a general platform or behavioral pattern. 3. Temporal scope: 'recently watched' confirms recency; session-level content consumption. 4. Decision tree: specific content item as primary subject → short_term_content. 5. Final classification: short_term_content. Confirmed: observation describes engagement with a specific content piece rather than a general behavioral preference.",
    "classification_result": "short_term_content"
    }

    ---

    Case 5) Ambiguous: content creator vs. personal relationship → long_term_preferences
    Input:
    - observation_node_text: "User regularly engages with fitness and health content on Instagram during evening hours (7-8 PM), with particular interest in fitness_coach_alex's posts, indicating a preference for fitness-related content and potential health/wellness goals."

    Model Output:
    {
    "reasoning": "1. Primary subject: fitness and health content on Instagram, with fitness_coach_alex as a content source. 2. Subject type: content category / platform. fitness_coach_alex is a public content creator, not a personal contact — this is content engagement, not an interpersonal relationship. The primary subject is 'fitness and health content' (a category), not a specific article or video. 3. Temporal scope: 'regularly' and 'evening hours (7-8 PM)' are strong established-pattern signals. 4. Decision tree: app/content-category subject with established temporal scope → long_term_preferences. 5. Final classification: long_term_preferences. Confirmed: recurring preference for a content category consumed at a consistent time, not a personal relationship or a specific content item.",
    "classification_result": "long_term_preferences"
    }

    ---

    Case 6) Ambiguous: social group reference → long_term_people
    Input:
    - observation_node_text: "User uses WhatsApp to communicate with family members and Instagram to communicate with friends from college, showing context-aware platform selection for personal relationships."

    Model Output:
    {
    "reasoning": "1. Primary subject: family members and college friends — identifiable social groups. 2. Subject type: person/social group. Even though specific names are not given, 'family members' and 'friends from college' are interpersonal subjects representing personal relationships. Platform selection (WhatsApp vs. Instagram) serves as context within the relationship pattern. 3. Temporal scope: N/A; person/group subjects always map to long_term_people. 4. Decision tree: social group as primary subject → long_term_people. 5. Final classification: long_term_people. Confirmed: observation describes interpersonal relationship patterns and communication platform preferences tied to specific social groups.",
    "classification_result": "long_term_people"
    }

    </FEW-SHOT EXAMPLES>

    Be thorough, follow the methodology strictly, apply the decision tree in order, and return only the JSON object with classification_result and reasoning fields.
    """.replace("__MEMORY_TYPES__", _memory_types_str)

    @staticmethod
    def get_classify_observation_user_prompt(observation_node_text: str) -> str:
        return f"""
        Classify the following observation node text into one of the available observation types:

        Observation:
        {observation_node_text}
        """
