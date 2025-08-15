import torch


class Seq2SeqTrainConfig:
    def __init__(self):
        self.n_epochs = 2
        self.learning_rate = 1e-3
        self.teacher_forcing_ratio = 0.5
        self.max_length = 20
        self.SOS_token = 2
        self.EOS_token = 3
        self.optimizer = torch.optim.Adam
        self.ignore_index = 0  # PAD token ids
        self.clip_grad_norm = 5.0
        self.hidden_size = 256
        self.num_layers = 3
        self.dropout = 0.1
        self.model_save_path = "seq2seq_attn.pth"
        self.loss_log_path = "loss_history.txt"


class TransformerTrainConfig:
    def __init__(self):
        # training hyper-parameters
        self.n_epochs = 2
        self.learning_rate = 1e-3
        self.teacher_forcing_ratio = 0.5
        self.clip_grad_norm = 5.0
        self.betas = (0.9, 0.98)
        self.eps = 1e-9

        # model architecture
        self.d_model = 512
        self.nhead = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1

        # sequence parameters
        self.max_length = 20
        self.SOS_token = 2
        self.EOS_token = 3
        self.ignore_index = 0  # PAD token id

        # optimizer
        self.optimizer = torch.optim.Adam

        # save paths
        self.model_save_path = "transformer_model.pth"
        self.loss_log_path = "transformer_loss_history.txt"
