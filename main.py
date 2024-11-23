import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.title("AI Code Generator")
st.write("Enter a prompt to generate code using the model.")

prompt = st.text_area("Prompt", "Write a quick sort algorithm.")

device = "cuda"
model_path = "01-ai/Yi-Coder-9B-Chat"

def load_model():
    try:
        # Ensure the user knows to install sentencepiece if required
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()
        return tokenizer, model
    except ImportError as e:
        st.error(f"Missing dependency: {e}. Please install it using `pip install sentencepiece`.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

tokenizer, model = load_model()

if st.button("Generate Code"):
    st.write("Generating code...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with st.spinner("Processing..."):
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024,
                eos_token_id=tokenizer.eos_token_id  
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.subheader("Generated Code")
        st.code(response, language="python")
    except Exception as e:
        st.error(f"Error during generation: {e}")
