import streamlit as st
from transformers import pipeline

# Load a pre-trained or fine-tuned model
@st.cache_resource
def load_model():
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    return generator

# Streamlit UI
st.title("Generative AI Screenplay Writer 🎥")
st.write("Generate screenplay scripts based on your cues and situations!")

# Inputs
cue = st.text_input("Enter a cue (e.g., 'A romantic scene at a café'):")
situation = st.text_area("Enter a situation (e.g., 'The characters meet for the first time over coffee.'):")

if st.button("Generate Script"):
    if cue and situation:
        model = load_model()
        prompt = f"CUE: {cue}\nSituation: {situation}\nScreenplay:"
        result = model(prompt, max_length=200, num_return_sequences=1)
        st.subheader("Generated Screenplay")
        st.text(result[0]["generated_text"])
    else:
        st.warning("Please fill out both fields!")

# Footer
st.markdown("Built with [Hugging Face Transformers](https://huggingface.co/) and Streamlit.")
