import gc
import torch
from src.load_config import LoadConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

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
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"  # Force CPU for compatibility

        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # def text_generator(
    #     self,
    #     prompt,
    #     max_new_tokens=512,
    # ):

    #     # Prepare the messages in chat format
    #     messages = [
    #         {"role": "system", "content": config_loader.llm_system_role},
    #         {"role": "user", "content": prompt},
    #     ]

    #     # Apply the chat template
    #     text = self.tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )

    #     # Tokenize the input and move to the appropriate device
    #     model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

    #     # Generate the response
    #     generated_ids = self.model.generate(
    #         model_inputs.input_ids,
    #     )

    #     # Decode the generated tokens to a string
    #     response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
    #         0
    #     ]

    #     return response

    def rag_text_generator(
        self,
        prompt,
    ):
        """
        Generate a response from the model based on the given prompt.
        """
        print("Generating response using rag_text_generator...")
        messages = [
            {"role": "system", "content": config_loader.llm_system_role},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and move to device correctly
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        generated_ids = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        input_text = self.tokenizer.decode(
            model_inputs["input_ids"][0], skip_special_tokens=True
        )
        generated_response = output_text[len(input_text) :].strip()

        return generated_response

    def raw_text_generator(
        self,
        prompt,
        max_new_tokens=512,
    ):
        """
        Generate a response from the model using raw prompt text.
        """
        print("Generating response using raw_text_generator...")

        messages = [
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and move to device correctly
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        generated_ids = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        input_text = self.tokenizer.decode(
            model_inputs["input_ids"][0], skip_special_tokens=True
        )
        generated_response = output_text[len(input_text) :].strip()

        return generated_response


# Example usage
if __name__ == "__main__":
    # Initialize the model
    config_loader = LoadConfig()
    llama3_german = Llama3GermanWrapper()

    # Define a prompt
    prompt = "Schreibe ein Essay über die Bedeutung der Energiewende für Deutschlands Wirtschaft"

    # Generate a response
    response = llama3_german.rag_text_generator(prompt)

    # Print the response
    print("Generated Response:", response)
