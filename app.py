import gradio as gr
from src.load_config import LoadConfig
from src.components.openAIWrapper import OpenAIWrapper
from src.pipeline.RAGPipeline import RAGPipeline  # Import your RAG pipeline class

# Load configuration
config_loader = LoadConfig()

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    top_k=5,
)


def generate_response(message, history, selected_model="GPT"):
    """
    Generate a response using the RAG pipeline and update the chat history.
    """
    try:
        # Call the RAG pipeline to generate a response
        result = rag_pipeline.response(query=message, model=selected_model)
        bot_response = result["answer"]
    except Exception as e:
        bot_response = (
            f"Sorry, an error occurred while generating the response: {str(e)}"
        )

    # Update chat history with user message and assistant response
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_response})

    return history, ""


# Create Gradio chatbot UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        placeholder="<strong>I am your assistant</strong><br>Ask me a question...",
        bubble_full_width=True,
        height=500,
        avatar_images=("assets/Images/User.png", "assets/Images/AI.png"),
        show_label=False,
        type="messages", 
    )

    with gr.Row():
        dropdown = gr.Dropdown(
            label="Select LLM model",
            choices=["GPT", "Llama3"],
            value="GPT",
            interactive=True,
        )

    with gr.Row():
        input_box = gr.Textbox(placeholder="Type your message here...", scale=15)
        submit_button = gr.Button("Send")

    # Define the click event for the submit button
    submit_button.click(
        generate_response,
        inputs=[input_box, chatbot, dropdown],
        outputs=[chatbot, input_box],
    )

# Launch the Gradio app
demo.launch(share=False)  # Set share=True to make it public
