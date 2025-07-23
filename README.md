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

## 📚 Key Research Papers

This project is built upon foundational concepts and architectures from these influential research papers. Understanding these works is crucial for grasping the model's design and capabilities.

* **RoBERTa: A Robustly Optimized BERT Pretraining Approach**
    * **Authors:** Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
    * **Link:** [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
    * *This paper introduces RoBERTa, an optimized version of BERT, which forms the backbone of the sentiment classifier for robust contextual embeddings.*

* **Long Short-Term Memory (LSTM) Networks**
    * **Key Paper:** [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Original LSTM paper)
        * **Authors:** Sepp Hochreiter, Jürgen Schmidhuber
        * *This foundational paper introduced the Long Short-Term Memory (LSTM) recurrent neural network architecture, which is extended by Bidirectional LSTMs to process sequences in both forward and backward directions for richer context.*
    * **Concept:** *Bidirectional LSTMs (BiLSTM) enhance LSTMs by processing sequences in both forward and backward directions, capturing long-range dependencies and richer contextual information from both past and future contexts within a sequence.*
    * **Link (General BiLSTM Overview):** [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (A classic blog post providing an intuitive understanding of LSTMs and their variants like BiLSTMs.)
    * *The BiLSTM layer in this model enhances sequential understanding beyond what transformers alone might capture for specific tasks, leveraging the strengths of recurrent architectures.*

* **Neural Machine Translation by Jointly Learning to Align and Translate**
    * **Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
    * **Link:** [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
    * *This seminal paper introduced an attention mechanism that allows neural networks to focus on specific parts of the input sequence when producing an output. This concept is highly relevant to how attention layers enhance sequence modeling tasks, including those built upon BiLSTMs.*

* **👑Attention Is All You Need**
    * **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
    * **Link:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    * *This groundbreaking paper introduced the Transformer architecture, which relies solely on attention mechanisms, demonstrating their power for sequence transduction tasks. While RoBERTa is a transformer, the explicit attention layer in this project (e.g., applied to BiLSTM outputs) draws inspiration from the broader concept of attention.*

* **Deep Learning for Sentiment Analysis: A Review**
    * **Authors:** L Zhang, S Wang, B Liu
    * **Link:** [Chapter 12 in *Handbook of Sentiment Analysis*, Springer, 2017](https://scholar.google.com/scholar?q=Deep+Learning+for+Sentiment+Analysis:+A+Survey+Erik+Cambria)
    * *TA comprehensive survey providing a broad overview of deep learning techniques applied to sentiment analysis, offering context for various architectural choices and highlighting the evolution of models in this field.*

---

## 📦 Run Locally: Get Started in Minutes

Want to run this powerful sentiment classifier on your machine? Follow these simple steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hrnrxb/roberta-bilstm-attention-sentiment](https://github.com/hrnrxb/roberta-bilstm-attention-sentiment) # لینک صحیح ریپازیتوری خودتان
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
