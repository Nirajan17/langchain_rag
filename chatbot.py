from langgraph.graph import MessagesState, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from config import SYSTEM_PROMPT

class Chatbot:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.workflow = self._setup_workflow()
        
    def _setup_workflow(self):
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        workflow.add_edge("model", "model")
        return workflow.compile(checkpointer=MemorySaver())
    
    def _call_model(self, state: MessagesState):
        last_message = state["messages"][-1].content
        
        retrieval_chain = {
            "context": self.retriever.get_relevant_documents(last_message),
            "question": last_message
        }
        
        prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        response = (prompt | self.llm).invoke(retrieval_chain)
        
        return {"messages": state["messages"] + [HumanMessage(content=response)]}
    
    def chat(self):
        while True:
            try:
                user_input = input("Your Query (type 'exit' to end) :: \n")
                if user_input.lower() == 'exit':
                    break
                
                response = self.workflow.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={
                        "configurable": {"thread_id": "1"},
                        "recursion_limit": 100
                    },
                )
                print("\nResponse:", response["messages"][-1].content, "\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}") 