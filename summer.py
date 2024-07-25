import transformers
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 1. Dataset Preparation
dataset = load_dataset("cnn_dailymail", '3.0.0')  
# You can replace this with your custom dataset

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def preprocess_function(examples):
    model_inputs = tokenizer(examples["article"], max_length=1024, truncation=True)
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 2. Model Selection & Training
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()

# 3. Evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# 4. Enhancements (Optional)
# - You can explore techniques like beam search for more diverse outputs.
# - Implement ROUGE and BLEU metrics for quantitative evaluation.
# - Experiment with different models like T5 or Pegasus for comparison.

# Example Inference
article_text = """[Your sample article text here]"""
input_ids = tokenizer(article_text, return_tensors="pt").input_ids
output = model.generate(input_ids)
summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(summary)
