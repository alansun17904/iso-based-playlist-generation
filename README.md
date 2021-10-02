# Learning Playlist-Mood Representations According to the ISO-Principle
Generating playlists using deep learning based on the ISO principle.


# Setup
Download the textual sentimental analysis model DistillBERT from HugginFace.
The configuration for the model is shown below:

```
{
  "_num_labels": 2,
  "activation": "gelu",
  "architectures": [
    "DistilBertForSequenceClassification"
  ],
  "attention_dropout": 0.1,
  "bad_words_ids": null,
  "bos_token_id": null,
  "decoder_start_token_id": null,
  "dim": 768,
  "do_sample": false,
  "dropout": 0.1,
  "early_stopping": false,
  "eos_token_id": null,
  "finetuning_task": "sst-2",
  "hidden_dim": 3072,
  "id2label": {
    "0": "NEGATIVE",
    "1": "POSITIVE"
  },
  "initializer_range": 0.02,
  "is_decoder": false,
  "is_encoder_decoder": false,
  "label2id": {
    "NEGATIVE": 0,
    "POSITIVE": 1
  },
  "length_penalty": 1.0,
  "max_length": 20,
  "max_position_embeddings": 512,
  "min_length": 0,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "no_repeat_ngram_size": 0,
  "num_beams": 1,
  "num_return_sequences": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pad_token_id": 0,
  "prefix": null,
  "pruned_heads": {},
  "qa_dropout": 0.1,
  "repetition_penalty": 1.0,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "task_specific_params": null,
  "temperature": 1.0,
  "tie_weights_": true,
  "top_k": 50,
  "top_p": 1.0,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 30522
}
```

Please put this in `models/sentiment` directory.

