from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import argparse

def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM using MLX")
    parser.add_argument("--model", type=str, default="mlx-community/LFM2-1.2B-8bit", help="Model to use")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    model, tokenizer = load(args.model)
    print("Model loaded. Type 'quit' or 'exit' to stop.")

    messages = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "user", "content": user_input})
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenizer=False, add_generation_prompt=True
        )

        
        sampler = make_sampler(temp=args.temp)
        print("Assistant: ", end="", flush=True)
        response_text = ""
        for response in stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            sampler=sampler, 
            max_tokens=args.max_tokens
        ):
            text = response.text
            print(text, end="", flush=True)
            response_text += text
        print() # Newline after generation completes
        
        messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
