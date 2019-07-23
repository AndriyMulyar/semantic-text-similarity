from pytorch_transformers import BertTokenizer
import torch


def _truncate_seq_pair(tokens_a, tokens_b, max_length: int):
    """
    Copied exactly from: https://github.com/huggingface/pytorch-pretrained-BERT/blob/78462aad6113d50063d8251e27dbaadb7f44fbf0/examples/extract_features.py#L150
    Truncates a sequence pair in place to the maximum length.
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def bert_sentence_pair_preprocessing(data: list, tokenizer: BertTokenizer, max_sequence_length=128):
    """
    Pre-processes an array of sentence pairs for input into bert. Sentence pairs are expected to be processed
    as given in data.py.

    Each sentence pair is tokenized and concatenated together by the [SEP] token for input into BERT

    :return: three tensors: [data_size, input_ids], [data_size, token_type_ids], [data_size, attention_mask]
    """

    max_bert_input_length = 0
    for sentence_pair in data:

        sentence_1_tokenized, sentence_2_tokenized = tokenizer.tokenize(sentence_pair['sentence_1']), tokenizer.tokenize(sentence_pair['sentence_2'])
        _truncate_seq_pair(sentence_1_tokenized, sentence_2_tokenized, max_sequence_length - 3) #accounting for positioning tokens


        max_bert_input_length = max(max_bert_input_length, len(sentence_1_tokenized) + len(sentence_2_tokenized) + 3)
        sentence_pair['sentence_1_tokenized'] = sentence_1_tokenized
        sentence_pair['sentence_2_tokenized'] = sentence_2_tokenized

    dataset_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_scores = torch.empty((len(data), 1), dtype=torch.float)

    for idx, sentence_pair in enumerate(data):
        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in sentence_pair['sentence_1_tokenized']:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in sentence_pair['sentence_2_tokenized']:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_masks = [1] * len(input_ids)
        while len(input_ids) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)


        dataset_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
        dataset_token_type_ids[idx] = torch.tensor(input_type_ids, dtype=torch.long)
        dataset_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)
        if 'similarity' not in sentence_pair or sentence_pair['similarity'] is None:
            dataset_scores[idx] = torch.tensor(float('nan'), dtype=torch.float)
        else:
            dataset_scores[idx] = torch.tensor(sentence_pair['similarity'], dtype=torch.float)


    return dataset_input_ids, dataset_token_type_ids, dataset_attention_masks, dataset_scores