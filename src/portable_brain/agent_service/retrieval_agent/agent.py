# main retrieval agent and helpers

# import clients
from portable_brain.common.services.llm_service.llm_client.google_genai_client import AsyncGenAITypedClient
# import memory retriever interface
from portable_brain.memory.main_retriever import MemoryRetriever

# import tool calling declarations
from portable_brain.agent_service.common.tool_calling_declarations.memory_retriever import memory_retriever_declarations
# import system prompts
from portable_brain.agent_service.common.system_prompts.memory_retrieval_prompts import MemoryRetrievalPrompts

class RetrievalAgent():
    """
    Main retrieval agent that tool calls to MemoryRetriever interface to access relevant memory.
    Goals:
        - Interpret the user query and identify ambiguous or missing parameters
        - Determine what knowledge is required before an action can succeed
        - Select and query the correct memory sources (long-term memory, logs, past attempts)
        - Assemble the retrieved information into coherent, execution-ready context
        - Filter and prioritize relevant facts from retrieved data
        - Track previously attempted retrievals to avoid redundancy
        - Evaluate whether sufficient context exists for execution
        - Request additional targeted retrieval when execution failures indicate missing information
        - Output structured context + inferred goals for the execution agent

    TODO: implement helpers
    """
    def __init__(self, memory_retriever: MemoryRetriever, gemini_llm_client: AsyncGenAITypedClient):
        self.memory_retriever = memory_retriever
        self.llm_client = gemini_llm_client

    def _build_tool_executors(self) -> dict:
        """
        Light helper to map each declaration name to the corresponding MemoryRetriever method.
        Used to pass in a dict for tool executors to the LLM client.
        """
        return {
            "get_people_relationships": self.memory_retriever.get_people_relationships,
            "get_long_term_preferences": self.memory_retriever.get_long_term_preferences,
            "get_short_term_preferences": self.memory_retriever.get_short_term_preferences,
            "get_recent_content": self.memory_retriever.get_recent_content,
            "get_all_observations_about_entity": self.memory_retriever.get_all_observations_about_entity,
            "search_memories": self.memory_retriever.search_memories,
            "get_top_relevant_memories": self.memory_retriever.get_top_relevant_memories,
            "find_semantically_similar": self.memory_retriever.find_semantically_similar,
            "get_embedding_for_observation": self.memory_retriever.get_embedding_for_observation,
        }

    def test_retrieve(self, user_request: str):
        """
        Test helper to run a single retrieval pass against memory.
        Returns the LLM's final text response (expected to be MemoryRetrievalOutput JSON).
        """
        return self.llm_client.atool_call(
            system_prompt=MemoryRetrievalPrompts.memory_retrieval_system_prompt,
            user_prompt=user_request,
            function_declarations=memory_retriever_declarations,
            tool_executors=self._build_tool_executors(),
            max_turns=5,
        )
