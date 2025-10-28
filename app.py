"""
Urdu Conversational Chatbot - Streamlit App
RTL-supported web interface for the trained Transformer model
"""

import streamlit as st
import torch
import sys
from pathlib import Path
import time

st.set_page_config(
    page_title="Urdu Chatbot",
    page_icon="chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

    .urdu-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
        font-size: 18px;
        line-height: 2;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 16px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .urdu-title {
        text-align: center;
        direction: rtl;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 2em;
        color: #667eea;
        margin-bottom: 20px;
    }

    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }

    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
    }

    .stat-label {
        color: #666;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and tokenizer from saved files"""
    model_path = Path("models/best_model.pt")
    vocab_path = Path("models/vocabulary")

    if not model_path.exists():
        st.error("Model file not found at models/best_model.pt")
        st.stop()

    if not (vocab_path / "tokenizer.pkl").exists():
        st.error("Tokenizer not found at models/vocabulary/tokenizer.pkl")
        st.stop()

    from model_loader import load_transformer_model, load_tokenizer

    tokenizer = load_tokenizer(vocab_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_transformer_model(model_path, device=device)
    model.eval()

    return model, tokenizer, device


def generate_response(user_input, model, tokenizer, device, strategy='greedy', beam_width=3):
    """Generate response from the model"""
    try:
        from inference import generate_response as model_generate
        response = model_generate(
            model=model,
            tokenizer=tokenizer,
            input_text=user_input,
            max_len=50,
            decoding_strategy=strategy,
            beam_width=beam_width,
            device=device
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main application"""

    model, tokenizer, device = load_model()

    st.markdown('<div class="main-title">Urdu Conversational Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="urdu-title">اردو گفتگو چیٹ بوٹ</div>', unsafe_allow_html=True)

    st.success("Model loaded successfully!")

    with st.sidebar:
        st.markdown("### Settings")

        decoding_strategy = st.radio(
            "Decoding Strategy",
            options=['greedy', 'beam'],
            index=0,
            help="Greedy is faster, Beam Search produces better quality"
        )

        if decoding_strategy == 'beam':
            beam_width = st.slider(
                "Beam Width",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of candidates to consider"
            )
        else:
            beam_width = 3

        st.markdown("---")

        st.markdown("### Model Information")

        st.metric("Status", "Active")
        st.metric("Parameters", "~10M")
        st.metric("Embedding Dim", "256")
        st.metric("Attention Heads", "2")

        st.markdown("---")

        st.markdown("### Session Statistics")

        if 'message_count' not in st.session_state:
            st.session_state.message_count = 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.message_count)
        with col2:
            st.metric("Language", "Urdu")

        st.markdown("---")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.message_count = 0
            st.rerun()

        st.markdown("---")

        st.markdown("### About")
        st.markdown("""
        <div class="info-box">
        Built with PyTorch Transformers and Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Chat Interface")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    chat_container = st.container()

    with chat_container:
        if len(st.session_state.conversation_history) == 0:
            st.markdown("""
            <div class="info-box">
            <h4>Welcome to Urdu Chatbot!</h4>
            <p>Type your message in Urdu to start the conversation:</p>
            <ul>
                <li>آپ کیسے ہیں؟</li>
                <li>السلام علیکم</li>
                <li>میں کیا کر سکتا ہوں؟</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        for user_msg, bot_msg in st.session_state.conversation_history:
            st.markdown(
                f'<div class="user-message">You: {user_msg}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="bot-message">Bot: {bot_msg}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            "Your message:",
            key="user_input",
            placeholder="Type in Urdu...",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")

    if send_button and user_input.strip():
        with st.spinner('Generating response...'):
            response = generate_response(
                user_input=user_input,
                model=model,
                tokenizer=tokenizer,
                device=device,
                strategy=decoding_strategy,
                beam_width=beam_width
            )

            st.session_state.conversation_history.append((user_input, response))
            st.session_state.message_count += 1

            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with PyTorch and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
