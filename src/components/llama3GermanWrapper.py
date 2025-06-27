import torch
from src.load_config import LoadConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

config_loader = LoadConfig()


class Llama3GermanWrapper:
    def __init__(
        self,
        model_name=config_loader.llm_model_name,
        device=config_loader.device,
    ):
        """
        Initialize the Llama3-DiscoLeo model and tokenizer.

        :param model_name: The name of the model to load from Hugging Face.
        :param device: The device to run the model on (e.g., "cuda" or "cpu").
        """
        self.model_name = model_name
        self.device = device

        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def rag_text_generator(
        self,
        prompt,
        max_new_tokens=512,
    ):
        """
        Generate a response from the model based on the given prompt.
        """
        # Prepare the messages in chat format
        messages = [
            {"role": "system", "content": config_loader.llm_system_role},
            {"role": "user", "content": prompt},
        ]

        # Apply the chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the input and move to the appropriate device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate the response
        generated_ids = self.model.generate(
            model_inputs.input_ids, max_new_tokens=max_new_tokens
        )

        # Decode the generated tokens to a string
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response

    def raw_text_generator(
        self,
        prompt,
        max_new_tokens=512,
    ):
        """
        Generate a response from the model based on the given prompt.
        """
        # Tokenize the input and move to the appropriate device
        model_inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        # Generate the response
        generated_ids = self.model.generate(
            model_inputs.input_ids, max_new_tokens=max_new_tokens
        )

        # Decode the generated tokens to a string
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response


# Example usage
if __name__ == "__main__":
    # Initialize the  model
    config_loader = LoadConfig()
    llama3_german = Llama3GermanWrapper(
        model_name=config_loader.embedding_models["llama3_german"]["model_name"],
        device=config_loader.device,
    )

    # Define a prompt
    prompt = "Schreibe ein Essay über die Bedeutung der Energiewende für Deutschlands Wirtschaft"

    # Generate a response
    response = llama3_german.rag_text_generator(prompt)

    # Print the response
    print("Generated Response:", response)
