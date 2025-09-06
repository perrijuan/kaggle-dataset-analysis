# Swin Fine-Tuned (luarAssembly/swin_finetune)

Fine-tuning de Swin Transformer para classificação de imagens.  
Modelo hospedado no Hugging Face: https://huggingface.co/luarAssembly/swin_finetune
<img width="1060" height="304" alt="image" src="https://github.com/user-attachments/assets/89ae11ab-999b-46a4-8e9c-1ce3933e86e0" />

## Visão Rápida
- Arquitetura: Swin Transformer (variante base/tiny — ajuste)
- Tarefa: Classificação de imagens
- Framework: PyTorch + 🤗 Transformers
- Entrada: imagem RGB
- Saída: lista de labels com probabilidades
- Formato: `AutoImageProcessor` + `AutoModelForImageClassification`

## Instalação
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # ajuste se CPU
pip install transformers pillow
```

## Uso (Pipeline)
```python
from transformers import pipeline
clf = pipeline("image-classification", model="luarAssembly/swin_finetune")
print(clf("exemplo.jpg"))
```

## Uso (Processor + Modelo)
```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_id = "luarAssembly/swin_finetune"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

img = Image.open("exemplo.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)[0]
top = torch.topk(probs, 3)
for score, idx in zip(top.values, top.indices):
    print(model.config.id2label[idx.item()], float(score))
```

## Labels
```python
from transformers import AutoModelForImageClassification
m = AutoModelForImageClassification.from_pretrained("luarAssembly/swin_finetune")
print(m.config.id2label)
```

## Download Offline
```bash
huggingface-cli download luarAssembly/swin_finetune --local-dir ./swin_model
```

## Continuação de Fine-Tuning
```python
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("luarAssembly/swin_finetune")
# Se mudar número de classes:
# import torch
# model.classifier = torch.nn.Linear(model.classifier.in_features, NOVO_NUM)
```

## Exportação (TorchScript)
```python
import torch
model.eval()
dummy = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, dummy)
traced.save("swin_finetune_ts.pt")
```

## Métricas (placeholder)
- accuracy(val): ...
- f1_macro: ...
- loss_final: ...
(Preencha quando disponível.)

## Input / Pré-processamento
- Resize / center crop (interno pelo processor)
- Normalização padrão ImageNet (`image_mean`, `image_std`)
- Espera tensor [B, 3, 224, 224] (ou tamanho interno do variant)



## Licença
GPL(3.0)



