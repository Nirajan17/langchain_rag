import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog
import threading
from model_setup import initialize_model, initialize_embeddings
from document_processor import load_and_process_documents
from config import SYSTEM_PROMPT
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Chat Assistant")
        self.root.geometry("800x600")
        
        # Initialize chatbot components
        self.setup_initial_state()
        
        # Create GUI elements
        self.create_widgets()
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_initial_state(self):
        """Initialize basic components without document loading"""
        try:
            self.llm = initialize_model()
            self.embeddings = initialize_embeddings()
            self.retriever = None
            self.current_pdf = None
            self.app = None
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Error initializing components: {str(e)}")
            self.root.destroy()
    
    def setup_chatbot(self, pdf_path):
        """Setup chatbot with specific PDF"""
        try:
            self.retriever = load_and_process_documents(self.embeddings, pdf_path)
            
            # Setup workflow
            workflow = StateGraph(state_schema=MessagesState)
            workflow.add_node("model", self._call_model)
            workflow.add_edge(START, "model")
            workflow.add_edge("model", "model")
            self.app = workflow.compile(checkpointer=MemorySaver())
            
            # Enable chat interface
            self.input_field.config(state='normal')
            self.send_button.config(state='normal')
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.insert(tk.END, f"PDF loaded: {pdf_path}\nYou can now start chatting!\n\n")
            self.chat_display.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("PDF Loading Error", f"Error loading PDF: {str(e)}")
            
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # PDF Upload button
        self.upload_button = ttk.Button(
            main_frame,
            text="Upload PDF",
            command=self.upload_pdf
        )
        self.upload_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            height=20,
            font=("Arial", 10)
        )
        self.chat_display.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        
        # Input area
        self.input_field = ttk.Entry(
            main_frame,
            font=("Arial", 10),
            state='disabled'  # Initially disabled
        )
        self.input_field.grid(row=2, column=0, sticky="ew", padx=(0, 5))
        self.input_field.bind("<Return>", lambda e: self.send_message())
        
        # Send button
        self.send_button = ttk.Button(
            main_frame,
            text="Send",
            command=self.send_message,
            state='disabled'  # Initially disabled
        )
        self.send_button.grid(row=2, column=1, sticky="e")
        
        # Configure grid weights for main_frame
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Please upload a PDF to begin")
        self.status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
    
    def upload_pdf(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")],
            title="Select a PDF file"
        )
        if file_path:
            self.status_var.set("Loading PDF...")
            self.root.update()
            
            # Start PDF processing in a separate thread
            threading.Thread(
                target=self._process_pdf,
                args=(file_path,),
                daemon=True
            ).start()
    
    def _process_pdf(self, file_path):
        try:
            self.setup_chatbot(file_path)
            self.current_pdf = file_path
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set("Error loading PDF")
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
            
    def _call_model(self, state: MessagesState):
        last_message = state["messages"][-1].content
        
        retrieval_chain = {
            "context": self.retriever.get_relevant_documents(last_message),
            "question": last_message
        }
        
        prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        response = (prompt | self.llm).invoke(retrieval_chain)
        
        return {"messages": state["messages"] + [HumanMessage(content=response)]}
    
    def display_message(self, message, sender):
        self.chat_display.configure(state='normal')
        if sender == "user":
            self.chat_display.insert(tk.END, "You: " + message + "\n\n", "user")
        else:
            self.chat_display.insert(tk.END, message + "\n\n", "assistant")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
    
    def process_response(self, user_input):
        try:
            self.status_var.set("Processing...")
            response = self.app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {"thread_id": "1"},
                    "recursion_limit": 100
                },
            )
            
            # Get the response and update the GUI
            assistant_response = response["messages"][-1].content
            self.root.after(0, self.display_message, f"Assistant: {assistant_response}", "system")
            self.status_var.set("Ready")
            
        except Exception as e:
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def send_message(self):
        if not self.current_pdf:
            messagebox.showwarning("Warning", "Please upload a PDF first!")
            return
            
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Display user message
        self.display_message(user_input, "user")
        
        # Process in separate thread
        threading.Thread(
            target=self.process_response,
            args=(user_input,),
            daemon=True
        ).start()

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 