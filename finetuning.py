import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,  # 변경됨
    Seq2SeqTrainingArguments,  # 변경됨
    Seq2SeqTrainer,  # 변경됨
    DataCollatorForSeq2Seq  # 변경됨
)

# Hugging Face 토큰 설정
os.environ["HUGGINGFACE_TOKEN"] = "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"

# 더 작은 모델 선택
model_name = "google/flan-t5-small"  # 변경됨
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# CSV 파일에서 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# DataFrame을 Dataset으로 변환
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = examples["input_text"]  # CSV 파일의 열 이름에 맞게 수정
    targets = examples["target_text"]  # CSV 파일의 열 이름에 맞게 수정
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 데이터셋 전처리
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 트레이닝 인자 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_steps=500,
    save_steps=1000,
    warmup_steps=500,
    prediction_loss_only=True,
)

# Trainer 초기화 및 학습
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# 모델 파인튜닝
trainer.train()

# 파인튜닝된 모델 저장
trainer.save_model("./finetuned_flan_t5_small_model")