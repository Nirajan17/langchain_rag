# PDF Chat Assistant

A Python-based intelligent chatbot that can read and answer questions about PDF documents using LangChain and Hugging Face models.

## Features

- 🤖 Powered by Mistral-7B-Instruct-v0.2 model
- 📄 PDF document processing and analysis
- 💬 Interactive chat interface with PDF upload capability
- 🔍 Context-aware responses
- 🧠 Conversation memory
- 🎯 Accurate document retrieval
- 📁 Support for custom PDF uploads

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-chat-assistant.git
cd pdf-chat-assistant
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Hugging Face API token:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Usage

### GUI Version
1. Launch the application:
```bash
python gui.py
```

2. Using the GUI:
   - Click the "Upload PDF" button to select your PDF file
   - Wait for the PDF to be processed
   - Once loaded, the chat interface will be enabled
   - Type your questions and press Enter or click Send
   - View the AI's responses in the chat display

### CLI Version
To use the command-line interface:

```bash
python main.py
```

### Adding Your Own PDFs
1. Place your PDF file in the project root directory
2. Update the `PDF_FILE_PATH` in `config.py` to match your PDF filename

### Supported File Types
- PDF documents (.pdf)

## Project Structure

```
pdf-chat-assistant/
├── main.py           # CLI application entry point
├── gui.py           # GUI application entry point
├── config.py        # Configuration settings
├── model_setup.py   # Model initialization
├── document_processor.py  # PDF processing
├── chatbot.py       # Core chatbot logic
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Components

- **main.py**: Command-line interface implementation
- **gui.py**: Graphical user interface implementation
- **config.py**: Central configuration management
- **model_setup.py**: Model initialization and setup
- **document_processor.py**: PDF document processing
- **chatbot.py**: Core chatbot implementation

## Technical Details

### Models Used
- **LLM**: Mistral-7B-Instruct-v0.2
- **Embeddings**: HuggingFace Embeddings

### Key Libraries
- LangChain
- Hugging Face Transformers
- tkinter (GUI)
- PyPDF

## Requirements

- Python 3.8+
- Hugging Face API token
- Sufficient RAM for model loading
- PDF document(s) for analysis

## Error Handling

The application includes comprehensive error handling for:
- Missing PDF files
- API authentication issues
- Model loading failures
- Runtime processing errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain team for the framework
- Hugging Face for model hosting
- Mistral AI for the language model

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Future Improvements

- [ ] Support for multiple PDF documents simultaneously
- [ ] PDF preview functionality
- [ ] Document summary generation
- [ ] Enhanced error handling
- [ ] More customization options
- [ ] Export chat history
- [ ] Document summarization feature
- [ ] Multi-language support
- [ ] Support for other document formats (docx, txt, etc.)

## Security Notes

- Keep your Hugging Face API token secure
- Don't share sensitive PDFs
- Be cautious with input validation