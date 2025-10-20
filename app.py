"""
Urdu Conversational Chatbot - Streamlit App
Deployment-ready application with RTL support
"""

import streamlit as st
import torch
import sys
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Urdu Chatbot | اردو چیٹ بوٹ",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Urdu RTL support and styling
st.markdown("""
<style>
    /* Import Urdu font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

    /* Urdu text styling */
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
        font-size: 18px;
        line-height: 2;
    }

    /* Chat message styling */
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

    /* Input box styling */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 16px;
    }

    /* Button styling */
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

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Title styling */
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

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }

    /* Stats display */
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
    """
    Load the trained model and tokenizer
    Returns None if model not found (for testing without model)
    """
    try:
        # Add model loading code here
        # This is a placeholder that returns None when model isn't available
        model_path = Path("models/best_model.pt")
        vocab_path = Path("models/vocabulary")

        if not model_path.exists():
            return None, None, "Model not found. Please upload the trained model to 'models/' folder."

        # When you have the model, uncomment and modify this:
        # from model_loader import load_transformer_model, load_tokenizer
        # model = load_transformer_model(model_path)
        # tokenizer = load_tokenizer(vocab_path)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = model.to(device)
        # model.eval()
        # return model, tokenizer, None

        return None, None, "Model loading not implemented yet."

    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def generate_response(user_input, model, tokenizer, strategy='greedy', beam_width=3):
    """
    Generate response from the model
    Placeholder function for when model is not loaded
    """
    if model is None:
        # Demo responses when model is not available
        demo_responses = {
            "آپ کیسے ہیں؟": "میں ٹھیک ہوں، شکریہ! آپ کیسے ہیں؟",
            "السلام علیکم": "وعلیکم السلام! خوش آمدید",
            "آپ کا نام کیا ہے؟": "میں ایک اردو چیٹ بوٹ ہوں",
            "شکریہ": "خوش آمدید! کوئی اور سوال؟",
            "خدا حافظ": "اللہ حافظ! پھر ملیں گے"
        }

        # Return demo response or default
        response = demo_responses.get(user_input, "معاف کیجیے، میں سمجھ نہیں پایا۔ براہ کرم دوبارہ کوشش کریں۔")
        return response

    # When you have the model, implement actual inference here:
    # try:
    #     from inference import generate_response as model_generate
    #     response = model_generate(
    #         model=model,
    #         tokenizer=tokenizer,
    #         input_text=user_input,
    #         max_len=50,
    #         decoding_strategy=strategy,
    #         beam_width=beam_width,
    #         device=device
    #     )
    #     return response
    # except Exception as e:
    #     return f"خرابی: {str(e)}"

    return "Model inference not implemented yet."


def main():
    """Main application"""

    # Load model (or get error message)
    model, tokenizer, error = load_model()

    # Header
    st.markdown('<div class="main-title">💬 Urdu Conversational Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="urdu-title">اردو گفتگو چیٹ بوٹ</div>', unsafe_allow_html=True)

    # Show model status
    if error:
        st.warning(f"⚠️ {error}")
        st.info("🔧 The app is running in **DEMO MODE** with sample responses. Upload your trained model to enable full functionality.")
    else:
        st.success("✅ Model loaded successfully!")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings | ترتیبات")

        # Decoding strategy
        decoding_strategy = st.radio(
            "Decoding Strategy",
            options=['greedy', 'beam'],
            index=0,
            help="Greedy is faster, Beam Search produces better quality"
        )

        # Beam width (only for beam search)
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

        # Model information
        st.markdown("### 📊 Model Info | ماڈل کی معلومات")

        if model is None:
            st.info("**Status:** Demo Mode")
            st.write("Upload trained model to enable full features")
        else:
            st.metric("Status", "Active ✅")
            st.metric("Parameters", "~10M")
            st.metric("Embedding Dim", "512")
            st.metric("Attention Heads", "2")

        st.markdown("---")

        # Statistics
        st.markdown("### 📈 Session Stats | اعداد و شمار")

        if 'message_count' not in st.session_state:
            st.session_state.message_count = 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.message_count)
        with col2:
            st.metric("Language", "اردو")

        st.markdown("---")

        # Clear conversation
        if st.button("🗑️ Clear Chat | صاف کریں", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.message_count = 0
            st.rerun()

        st.markdown("---")

        # About section
        st.markdown("### ℹ️ About | معلومات")
        st.markdown("""
        <div class="info-box">
        Built with ❤️ using:
        <br>• PyTorch Transformers
        <br>• Multi-Head Attention
        <br>• Streamlit Framework
        </div>
        """, unsafe_allow_html=True)

        # Links
        st.markdown("### 🔗 Links | روابط")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/YOUR_USERNAME/urdu-chatbot)")
        st.markdown("[![Documentation](https://img.shields.io/badge/Docs-Read-blue)](https://github.com/YOUR_USERNAME/urdu-chatbot)")

    # Main chat interface
    st.markdown("### 💬 Chat Interface | گفتگو")

    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Display conversation history
    chat_container = st.container()

    with chat_container:
        if len(st.session_state.conversation_history) == 0:
            st.markdown("""
            <div class="info-box">
            <h4>👋 خوش آمدید! Welcome!</h4>
            <p>Start the conversation in Urdu. Here are some examples:</p>
            <ul>
                <li>آپ کیسے ہیں؟ (How are you?)</li>
                <li>آپ کا نام کیا ہے؟ (What is your name?)</li>
                <li>السلام علیکم (Greetings)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        for i, (user_msg, bot_msg) in enumerate(st.session_state.conversation_history):
            # User message
            st.markdown(
                f'<div class="user-message">👤 آپ: {user_msg}</div>',
                unsafe_allow_html=True
            )

            # Bot message
            st.markdown(
                f'<div class="bot-message">🤖 بوٹ: {bot_msg}</div>',
                unsafe_allow_html=True
            )

    # Input area
    st.markdown("---")

    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            "Your message (اپنا پیغام یہاں لکھیں):",
            key="user_input",
            placeholder="اردو میں لکھیں... (Type in Urdu)",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("Send ➤", use_container_width=True, type="primary")

    # Process input
    if send_button and user_input.strip():
        with st.spinner('جواب تیار کیا جا رہا ہے... Generating response...'):
            # Simulate processing time
            time.sleep(0.5)

            # Generate response
            response = generate_response(
                user_input=user_input,
                model=model,
                tokenizer=tokenizer,
                strategy=decoding_strategy,
                beam_width=beam_width
            )

            # Add to conversation history
            st.session_state.conversation_history.append((user_input, response))
            st.session_state.message_count += 1

            # Rerun to update display
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with ❤️ using PyTorch and Streamlit</p>
        <p style='direction: rtl; font-family: "Noto Nastaliq Urdu", serif;'>
            پائی ٹارچ اور سٹریم لِٹ سے بنایا گیا
        </p>
        <p><a href='https://github.com/YOUR_USERNAME/urdu-chatbot' target='_blank'>View on GitHub</a> |
        <a href='https://github.com/YOUR_USERNAME/urdu-chatbot/blob/main/README.md' target='_blank'>Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
