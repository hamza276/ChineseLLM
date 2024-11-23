import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Streamlit app setup
st.title("AI Code Generator")
st.write("Enter a prompt to generate code using the model.")

# Input field for user prompt
prompt = st.text_area("Prompt", "Write a quick sort algorithm.")

# Device and model setup
device = "cuda"  # You can switch to 'cpu' if CUDA is not available
model_path = "01-ai/Yi-Coder-9B-Chat"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()
    return tokenizer, model

tokenizer, model = load_model()

# Generate code when the button is clicked
if st.button("Generate Code"):
    st.write("Generating code...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

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

    # Display the generated code
    st.subheader("Generated Code")
    st.code(response, language="python")
