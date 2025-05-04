from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import LayoutLMv2TokenizerFast

app = FastAPI()
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")

class TokenizeRequest(BaseModel):
    text: List[str]
    boxes: List[List[int]]  

@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    if len(req.text) != len(req.boxes):
        return {"error": "Length of text and boxes must match."}

    encoding = tokenizer(
        req.text,
        boxes=req.boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    return {
        "input_ids": encoding["input_ids"].tolist(),
        "attention_mask": encoding["attention_mask"].tolist(),
        "token_type_ids": encoding["token_type_ids"].tolist(),
        "bbox": encoding["bbox"].tolist()
    }
