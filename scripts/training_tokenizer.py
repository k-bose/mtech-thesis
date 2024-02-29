import sentencepiece as spm
import os
from constants import ABSOLUTE_PARENT


def train_sp_lang(input_file, tokenizer_path, vocab_size):
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=tokenizer_path,
        model_type='bpe',
        vocab_size=vocab_size,
        pad_piece='[PAD]',
        pad_id='0',
        unk_piece='[UNK]',
        unk_id='1',
        bos_piece='[SOS]',
        bos_id='2',
        eos_piece='[EOS]',
        eos_id='3'
    )
    return tokenizer_path


# Directory to save scripts outputs
working_dir = f"{ABSOLUTE_PARENT}/files/outputs"
# Tokenizer file name
tokenizer_file_name = f"sp_{{0}}_bpe_{{1}}"

# Assigning configurations/parameters
ds_file = f"{ABSOLUTE_PARENT}/files/datasets/iitb_hi_train.txt"
lang = 'hi'  # Language
vocab_size_lst = ['1024', '2048', '3000', '3700', '4096', '5000', '6000', '7000', '8192', '9000', '10000',
                  '11000', '12000', '13000', '14000', '15000', '16384', '18000', '20000',
                  '25000', '32768', '65536', '131072']  # List of vocab sizes

for vocab_size in vocab_size_lst:
    tokenizer_path = f"{working_dir}/{tokenizer_file_name.format(lang, vocab_size)}"
    if os.path.exists(f"{tokenizer_path}.model"):
        print(f"Tokenizer {tokenizer_file_name.format(lang, vocab_size)}.model exists.")
    else:
        print(f"Tokenizer {tokenizer_file_name.format(lang, vocab_size)}"
              f".model does not exist. Generating tokenizer from scratch.")
        tokenizer = train_sp_lang(ds_file, tokenizer_path, vocab_size)
