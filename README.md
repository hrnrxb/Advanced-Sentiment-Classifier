# 🧠 Advanced Sentiment Classifier: RoBERTa + BiLSTM + Attention

Dive into the nuances of language with this **powerful sentiment analysis model** that goes beyond simple positive/negative detection. Built with the robust `roberta-base` transformer, enhanced by a Bidirectional LSTM (BiLSTM) for sequential understanding, and refined with an attention mechanism, this model can truly **handle sarcasm, irony, and even mixed opinions** in text.

---

## ✅ Features: What Makes This Model Stand Out?

This isn't your average sentiment analyzer. It's engineered to understand the subtleties of human expression:

* **Master of Nuance:**
    * Handles complex sentences with remarkable accuracy, such as:
        * “If I had a dollar for every cliché in this film, I’d be rich.” ➜ 🔴 **Negative** (Recognizes sarcasm)
        * “It’s fine. Just fine. Totally... fine.” ➜ 🔴 **Negative** (Detects understated negativity/irony)
* **Robust Training:** Initially trained on the **IMDB (20K samples)** dataset, making it highly effective for movie reviews and similar textual data. It's also **extensible to other challenging datasets** like SST-5 (fine-grained sentiment) or dedicated Sarcasm datasets.
* **Multilingual Ready:** Designed for flexibility! Easily switch to `xlm-roberta` to **add support for multiple languages, including Persian**, opening up new possibilities for diverse text analysis.
* **Instant Interaction:** Comes with a **ready-to-use Gradio interface**, allowing anyone to test the model with their own text in real-time.
* **Hassle-Free Deployment:** Fully **deployable on Hugging Face Spaces**, providing a free and easy way to share your model with the world.

---

## 🔗 Quick Links: Explore the Model

* 🔍 **Model Weights:**
    * [📦 **Roberta-BiLSTM-Attention Model on Hugging Face**](https://huggingface.co/hrnrxb/roberta-bilstm-attention-sentiment) - Download the model weights and tokenizer for your own projects.
* 🚀 **Live Demo** (Gradio app):
    * [🌐 **Try it on Hugging Face Spaces**](https://huggingface.co/spaces/hrnrxb/roberta-bilstm-attention-sentiment) - Experience the model's capabilities live in your browser!

---

## 📦 Run Locally: Get Started in Minutes

Want to run this powerful sentiment classifier on your machine? Follow these simple steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/roberta-bilstm-attention-sentiment](https://github.com/your-username/roberta-bilstm-attention-sentiment)
    cd roberta-bilstm-attention-sentiment
    ```

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

    ```bash
    python3 -m venv env
    source env/bin/activate  # On macOS/Linux
    # On Windows: .\env\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Gradio application:**
    ```bash
    python app.py
    ```
    After running, the Gradio interface will launch, usually accessible at `http://127.0.0.1:7860/` in your web browser.
