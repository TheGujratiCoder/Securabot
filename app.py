import streamlit as st

# Set Streamlit config FIRST
st.set_page_config(page_title="TinyLLaMA Cybersecurity Chatbot", page_icon="🤖")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer (cached)
@st.cache_resource
def load_model():
    model_path = r"./final_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🤖 TinyLLaMA Cybersecurity Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    # Add user input to chat history
    st.session_state.chat_history.append(("🧑 You", user_input))

    # Generate response
    prompt = user_input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text.replace(prompt, "").strip()

    # Add bot response to chat history
    st.session_state.chat_history.append(("🤖 Bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")
