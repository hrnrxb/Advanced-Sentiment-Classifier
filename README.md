# 🧠 Advanced Sentiment Classifier: RoBERTa + BiLSTM + Attention

A powerful sentiment analysis model that can handle sarcasm, irony, and mixed opinions.  
Built with `roberta-base`, BiLSTM, and attention mechanism.

---

## ✅ Features

- Handles complex sentences like:
  - “If I had a dollar for every cliché in this film, I’d be rich.” ➜ 🔴 Negative  
  - “It’s fine. Just fine. Totally... fine.” ➜ 🔴 Negative

- Trained on IMDB (20K samples) – extensible to SST-5 or Sarcasm datasets
- Multilingual-ready (switch to `xlm-roberta` to add Persian support)
- Ready-to-use Gradio interface
- Deployable on Hugging Face Spaces

---

## 🔗 Links

- 🔍 **Model weights**:  
  [📦 Roberta-BiLSTM-Attention Model on google drive](https://huggingface.co/hrnrxb/roberta-bilstm-attention-sentiment)

- 🚀 **Live demo** (Gradio app):  
  [🌐 Try it on Hugging Face Spaces](https://huggingface.co/spaces/hrnrxb/roberta-bilstm-attention-sentiment)

---

## 📦 Run locally

```bash
git clone https://github.com/your-username/roberta-bilstm-attention-sentiment
cd roberta-bilstm-attention-sentiment
pip install -r requirements.txt
python app.py
