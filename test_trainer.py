import transformers
print(transformers.__file__)
from transformers import TrainingArguments
args = TrainingArguments(output_dir='test', evaluation_strategy='epoch')
print(args)