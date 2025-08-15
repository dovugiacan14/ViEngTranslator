import argparse
import torch 
from torch.utils.data import DataLoader


from src.train_setup import train_seq2seq_model, train_model
from src.utils import TranslationPreprocessor
from src.dataloader import TranslationDataset, collate_fn

from models.transformers import TransformerModel
from models.seq2seq import EncoderLSTM, AttentionDecoderLSTM
from config import Seq2SeqTrainConfig, TransformerTrainConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-model",
        type=str,
        default="seq2seq",
        help="the problem name",
        choices=[
            "seq2seq",
            "transformers"
        ],
    )

    parser.add_argument(
        "--eng_data_path",
        "-eng_path",
        type=str,
        default="datasets/en_sents.txt",
    )

    parser.add_argument(
        "--vi_data_path",
        "-vi_path",
        type=str,
        default="datasets/vi_sents.txt"
    )

    parser.add_argument(
        "--n_epoch", "-epoch", type=int, default=10, help="number epochs"
    )

    parser.add_argument(
        "--test_size", "-test", type=float, default=0.2
    )

    parser.add_argument(
        "--val_size", "-val", type=float, default=0.5
    )

    return parser.parse_args()

def main(args):
    # load data 
    with open(args.eng_data_path, 'r', encoding='utf-8') as file: 
        eng_sentences = [line.strip() for line in file if line.strip()]
    
    with open(args.vi_data_path, 'r', encoding='utf-8') as vi_file: 
        vi_sentences = [line.strip() for line in vi_file if line.strip()]
    

    print(eng_sentences[2:4])
    print(vi_sentences[2:4])
    print(f"Total records: {len(vi_sentences)}")

    # pre-processing data
    processor = TranslationPreprocessor()
    data_info = processor.prepare_datasets(vi_sentences, eng_sentences, args.test_size, args.val_size)

    # build dataloader 
    train_dataset = TranslationDataset(data_info.get('train'))
    val_dataset = TranslationDataset(data_info.get('val'))
    test_dataset = TranslationDataset(data_info.get('test'))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle= True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle= True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle= True, collate_fn=collate_fn)
    
    if args.model_name == "seq2seq":
        params = Seq2SeqTrainConfig()
        encoder = EncoderLSTM(
            vocab_size_src= data_info.get('vocab_size_src'), 
            hidden_size= params.hidden_size,
            num_layers= params.num_layers, 
            dropout= params.dropout
        )
        decoder = AttentionDecoderLSTM(
            hidden_size= params.hidden_size, 
            vocab_size_target= data_info.get('vocab_size_src'), 
            num_layers= params.num_layers, 
            dropout= params.dropout
        )

        train_seq2seq_model(encoder, decoder, train_loader, params)
    else: 
        params = TransformerTrainConfig()
        model = TransformerModel(
            src_vocab_size=data_info.get('vocab_size_src'), 
            tgt_vocab_size=data_info.get('vocab_size_src'), 
            config=params
        )

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Config - d_model: {params.d_model}, nhead: {params.nhead}")
        print(f"Config - epochs: {params.n_epochs}, lr: {params.learning_rate}")

        train_model(
            model= model,
            train_loader= train_loader,
            val_loader= val_loader, 
            config= params
        )


if __name__ == "__main__":
    args = parse_argument()
    main(args)

