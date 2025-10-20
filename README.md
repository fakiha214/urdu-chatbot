# 💬 Urdu Conversational Chatbot - Streamlit App

A beautiful, RTL-supported Urdu chatbot interface built with Streamlit.

## 🌟 Features

- ✅ **RTL Support** - Proper right-to-left text rendering for Urdu
- ✅ **Beautiful UI** - Gradient-based modern design
- ✅ **Demo Mode** - Works without trained model (sample responses)
- ✅ **Conversation History** - Keeps track of chat sessions
- ✅ **Customizable Settings** - Greedy vs Beam Search decoding
- ✅ **Session Statistics** - Track message count
- ✅ **Responsive Design** - Works on all screen sizes

## 🚀 Quick Start

### Local Development

1. **Clone this repository**:
```bash
git clone https://github.com/YOUR_USERNAME/urdu-chatbot-streamlit.git
cd urdu-chatbot-streamlit
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
streamlit run app.py
```

4. **Open in browser**:
   - The app will automatically open at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Urdu chatbot Streamlit app"

# Add remote (create repo on GitHub first)
git remote add origin https://github.com/YOUR_USERNAME/urdu-chatbot-streamlit.git

# Push
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10
5. Click **"Deploy"**

Your app will be live at: `https://YOUR_USERNAME-urdu-chatbot-streamlit.streamlit.app`

## 🤖 Adding Your Trained Model

Currently, the app runs in **DEMO MODE** with sample responses. To add your trained model:

### Step 1: Prepare Model Files

After training your model on Google Colab or locally, you'll have:
- `best_model.pt` (model checkpoint)
- `tokenizer.pkl` (vocabulary)

### Step 2: Add to Repository

Create a `models/` folder:
```bash
mkdir models
```

Add your files:
```
streamlit_app/
├── models/
│   ├── best_model.pt
│   └── vocabulary/
│       └── tokenizer.pkl
├── app.py
├── requirements.txt
└── README.md
```

### Step 3: Update Model Loading

In `app.py`, the `load_model()` function (lines 145-172) has placeholders. Uncomment and modify:

```python
@st.cache_resource
def load_model():
    try:
        model_path = Path("models/best_model.pt")
        vocab_path = Path("models/vocabulary")

        if not model_path.exists():
            return None, None, "Model not found."

        # Uncomment when you have the model:
        from model_loader import load_transformer_model, load_tokenizer
        model = load_transformer_model(model_path)
        tokenizer = load_tokenizer(vocab_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        return model, tokenizer, None

    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"
```

### Step 4: Update Inference

In `generate_response()` function (lines 175-210), uncomment the actual inference code:

```python
def generate_response(user_input, model, tokenizer, strategy='greedy', beam_width=3):
    if model is None:
        # Demo responses (current mode)
        ...

    # Uncomment for actual inference:
    try:
        from inference import generate_response as model_generate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return f"خرابی: {str(e)}"
```

### Step 5: Add Helper Modules

Create `model_loader.py` to load your model:

```python
import torch
from pathlib import Path
import pickle

def load_tokenizer(vocab_path):
    """Load the tokenizer"""
    with open(Path(vocab_path) / 'tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

def load_transformer_model(model_path):
    """Load the trained Transformer model"""
    # Import your model architecture
    from models.transformer import Transformer

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Initialize model with same config as training
    model = Transformer(
        src_vocab_size=checkpoint['config']['src_vocab_size'],
        tgt_vocab_size=checkpoint['config']['tgt_vocab_size'],
        d_model=512,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=2048,
        max_seq_length=100,
        dropout=0.1
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
```

### Step 6: Update requirements.txt

Add any additional dependencies:
```txt
streamlit>=1.28.0
torch>=2.0.0
# Add if needed:
# sacrebleu>=2.3.1
# rouge-score>=0.1.2
```

### Step 7: Push Updates

```bash
git add .
git commit -m "Add trained model"
git push
```

Streamlit Cloud will automatically redeploy!

## 📁 Project Structure

```
streamlit_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── .gitignore             # Files to ignore
├── models/                # (Add later) Trained model files
│   ├── best_model.pt
│   └── vocabulary/
│       └── tokenizer.pkl
└── model_loader.py        # (Add later) Model loading utilities
```

## 🎨 Customization

### Change Color Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"        # Purple gradient
backgroundColor = "#ffffff"      # White
secondaryBackgroundColor = "#f5f7fa"  # Light gray
textColor = "#262730"           # Dark gray
```

### Change Urdu Font

In `app.py`, modify the CSS import (line 24):

```python
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
```

Available Urdu fonts:
- Noto Nastaliq Urdu (current)
- Jameel Noori Nastaleeq
- Alvi Nastaleeq

### Add More Demo Responses

In `generate_response()` function, add to `demo_responses` dict:

```python
demo_responses = {
    "آپ کیسے ہیں؟": "میں ٹھیک ہوں، شکریہ! آپ کیسے ہیں؟",
    "السلام علیکم": "وعلیکم السلام! خوش آمدید",
    # Add more here:
    "آپ کا نام کیا ہے؟": "میں ایک اردو چیٹ بوٹ ہوں",
    "شکریہ": "خوش آمدید! کوئی اور سوال؟",
}
```

## 🔧 Troubleshooting

### Issue: Urdu Text Not Displaying

**Solution**:
- Install Urdu fonts on your system
- Ensure UTF-8 encoding in browser
- Clear browser cache

### Issue: Model Not Loading

**Solution**:
- Check `models/` folder exists
- Verify file paths in `load_model()`
- Check model file size (GitHub has 100MB limit)
- For large models, use [Git LFS](https://git-lfs.github.com/)

### Issue: Streamlit Cloud Deployment Fails

**Solution**:
- Check `requirements.txt` has all dependencies
- Verify Python version compatibility
- Check Streamlit Cloud logs for errors
- Ensure no large files (>100MB) without LFS

### Issue: Slow Response Time

**Solution**:
- Use greedy decoding (faster than beam search)
- Reduce model size
- Use Streamlit Cloud's caching (`@st.cache_resource`)

## 📊 Performance

**Demo Mode**:
- Response time: <100ms
- No GPU required

**With Model**:
- Response time: 1-5 seconds (CPU)
- Response time: 0.2-1 seconds (GPU)
- Model size: ~45MB

## 🌍 Sharing Your App

Once deployed on Streamlit Cloud, share your app:

1. **Get public URL**: `https://YOUR_USERNAME-urdu-chatbot-streamlit.streamlit.app`
2. **Share on social media**
3. **Embed in website**: Streamlit provides embed code
4. **Custom domain**: Available on paid plans

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check Streamlit documentation: https://docs.streamlit.io

## 🎯 Future Enhancements

- [ ] Add model to repository
- [ ] Implement actual inference
- [ ] Add conversation export (download chat)
- [ ] Add voice input support
- [ ] Add response rating system
- [ ] Add multi-turn context memory
- [ ] Add user authentication
- [ ] Add conversation analytics

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Urdu font: [Google Fonts - Noto Nastaliq Urdu](https://fonts.google.com/noto/specimen/Noto+Nastaliq+Urdu)
- Icons: Unicode emoji

---

**Current Status**: ✅ Ready to deploy in DEMO MODE

**Next Step**: Add your trained model from the main urdu_chatbot project!

---

Made with ❤️ for the Urdu language community
