import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def restore_text(text):
    """Restores missing words in a historical text using BERT."""
    masked_text = text.replace("___", "[MASK]") 
    inputs = tokenizer(masked_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs).logits

    masked_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    for idx in masked_indices:
        predicted_token_id = outputs[0, idx].argmax(dim=-1).item()
        predicted_word = tokenizer.decode([predicted_token_id])
        masked_text = masked_text.replace("[MASK]", predicted_word, 1)

    return masked_text

extracted_text = "The ancient ___ was built in the year 12__."
restored_text = restore_text(extracted_text)

print("\n🔍 Restored Text:\n", restored_text)      
