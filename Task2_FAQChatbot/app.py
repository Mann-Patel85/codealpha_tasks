import os
import string
import json
import tempfile
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# --- 1. Robust Dataset Loading & Enrichment ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'faq_data.csv')

def load_data():
    try:
        df = pd.read_csv(CSV_PATH, quotechar='"')
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["question", "answer"])

df = load_data()

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Initialize TF-IDF Vectorizer
if not df.empty:
    vectorizer = TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['question'])
else:
    vectorizer = None
    tfidf_matrix = None

# Intent Rules for Common Greetings / Conversational Queries
GREETINGS = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good evening", "good afternoon"}
THANK_YOUS = {"thank you", "thanks", "thx", "appreciate it", "thank you so much"}
GOODBYES = {"bye", "goodbye", "see you", "cya", "exit"}

# --- 2. Intelligent Response Engine ---
def get_bot_response(user_input, history):
    if df.empty or vectorizer is None:
        return "⚠️ System Error: FAQ Database not loaded properly."
    
    clean_input = user_input.strip().lower()
    if not clean_input:
        return "Please type a question or select one of the quick options below."

    # Direct Intent Handling
    if clean_input in GREETINGS:
        return (
            "👋 **Hello and welcome to TechHub Support!**\n\n"
            "I can help you with questions about:\n"
            "- 📦 **Orders & Tracking**\n"
            "- 🛡️ **Product Warranties**\n"
            "- 💳 **EMI & Payment Options**\n"
            "- 🔄 **Returns & Refunds**\n\n"
            "How can I assist you today?"
        )
    
    if clean_input in THANK_YOUS:
        return "✨ You're very welcome! If you have any other questions, feel free to ask anytime. Happy shopping at TechHub!"

    if clean_input in GOODBYES:
        return "👋 Goodbye! Thank you for visiting TechHub. Have a fantastic day!"

    # TF-IDF Matching
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)[0]
    
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    # High Confidence Match
    if best_score >= 0.40:
        confidence_pct = int(best_score * 100)
        answer = df.iloc[best_idx]['answer']
        return f"{answer}\n\n`Confidence: {confidence_pct}%`"
    
    # Moderate Confidence with Suggested Queries
    elif best_score >= 0.22:
        top_indices = similarities.argsort()[-3:][::-1]
        suggestions = [f"- *{df.iloc[idx]['question'].title()}*" for idx in top_indices if similarities[idx] > 0.15]
        suggestions_text = "\n".join(suggestions)
        
        fallback_msg = (
            "🤔 I am not completely sure, but here is what might help:\n\n"
            f"{df.iloc[best_idx]['answer']}\n\n"
            "**Did you mean one of these?**\n"
            f"{suggestions_text}\n\n"
            "If not, contact our support at **1800-123-4567** or `support@techhub.com`."
        )
        return fallback_msg
    
    # Fallback when no match
    else:
        return (
            "🤖 I couldn't find an exact answer for that query in our knowledge base.\n\n"
            "📞 **Need immediate help?**\n"
            "- **Helpline:** 1800-123-4567 (Mon-Sat, 10 AM - 8 PM)\n"
            "- **Email Support:** `support@techhub.com`\n"
            "- **Live Agent:** Visit your TechHub account dashboard."
        )

# Export Chat History Function
def export_chat(history):
    if not history:
        return None, "Chat is currently empty."
    
    temp_dir = tempfile.gettempdir()
    log_path = os.path.join(temp_dir, "techhub_chat_transcript.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== TechHub AI Assistant Chat Transcript ===\n\n")
        for user_msg, bot_msg in history:
            f.write(f"User : {user_msg}\n")
            f.write(f"Agent: {bot_msg}\n")
            f.write("-" * 50 + "\n")
            
    return log_path, "✅ Chat log exported successfully!"

# --- 3. Modern UI with Quick Suggestion Chips ---
custom_css = """
.chat-container {max-width: 880px; margin: auto;}
.badge-title {text-align: center; color: #0284c7; font-weight: 700; margin-bottom: 2px;}
.badge-desc {text-align: center; color: #64748b; font-size: 14px; margin-bottom: 16px;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"), css=custom_css, title="TechHub AI Assistant") as demo:
    with gr.Column(elem_classes="chat-container"):
        gr.Markdown("# 🛒 TechHub AI Customer Support", elem_classes="badge-title")
        gr.Markdown("Instant 24/7 answers for Orders, Warranties, Returns, EMI, and Delivery.", elem_classes="badge-desc")
        
        chatbot = gr.Chatbot(height=420, show_copy_button=True, bubble_full_width=False)
        
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about your order, return, warranty, EMI...",
                scale=7,
                lines=1
            )
            send_btn = gr.Button("Send 💬", variant="primary", scale=1)

        # Quick Suggestion Chips
        gr.Markdown("##### ⚡ Quick Questions:")
        with gr.Row():
            btn1 = gr.Button("📦 Where is my order?", size="sm")
            btn2 = gr.Button("🛡️ Laptop warranty?", size="sm")
            btn3 = gr.Button("💳 EMI options?", size="sm")
            btn4 = gr.Button("🔄 Return policy?", size="sm")
            btn5 = gr.Button("📞 Contact support", size="sm")

        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")
            export_btn = gr.Button("📥 Export Chat Log", variant="secondary", size="sm")
            export_file = gr.File(label="Download Transcript", scale=1)

        # Event Handlers
        def user_send(user_message, history):
            if not user_message.strip():
                return "", history
            history = history or []
            bot_reply = get_bot_response(user_message, history)
            history.append((user_message, bot_reply))
            return "", history

        def quick_send(text, history):
            history = history or []
            bot_reply = get_bot_response(text, history)
            history.append((text, bot_reply))
            return history

        msg_input.submit(user_send, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(user_send, [msg_input, chatbot], [msg_input, chatbot])

        btn1.click(lambda h: quick_send("where is my order?", h), [chatbot], [chatbot])
        btn2.click(lambda h: quick_send("laptop warranty", h), [chatbot], [chatbot])
        btn3.click(lambda h: quick_send("emi options", h), [chatbot], [chatbot])
        btn4.click(lambda h: quick_send("return policy", h), [chatbot], [chatbot])
        btn5.click(lambda h: quick_send("contact support", h), [chatbot], [chatbot])

        clear_btn.click(lambda: [], None, chatbot)
        export_btn.click(export_chat, [chatbot], [export_file, msg_input])

        gr.Markdown("---")
        gr.Markdown("<div style='text-align: center; color: #94a3b8; font-size: 12px;'>Powered by Scikit-Learn TF-IDF & Cosine Similarity | Built by Mann Patel (CodeAlpha 2025)</div>")

if __name__ == "__main__":
    demo.launch()