import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

# Compositional Network
class SentModel(pl.LightningModule):
    # TODO compositional network init
    def __init__(self, encoder_name, encoder_hidden_size,
                 arg_comp_hidden_size, arg_comp_output_size,
                 event_comp_hidden_size, event_comp_output_size,
                 margin):
        super().__init__()
        self.__dict__.update(locals())
        # Sentence Bert
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        self.encoder = SBERT(self.encoder)

        # Argument composition
        arg_hidden_size = self.arg_comp_hidden_size
        arg_output_size = self.arg_comp_output_size
        self.arg_comp_prior = nn.Sequential(MLP(self.encoder_hidden_size, arg_output_size, arg_hidden_size))
        self.arg_comp_posterior = nn.Sequential(MLP(self.encoder_hidden_size, arg_output_size, arg_hidden_size))

                                      #MLP(arg_output_size, arg_output_size, arg_hidden_size))

        # Event composition
        event_hidden_size = self.event_comp_hidden_size
        event_output_size = self.event_comp_output_size
        self.event_comp = nn.Sequential(MLP(arg_output_size*2, event_output_size, event_hidden_size),
                                        MLP(event_output_size, 1, event_hidden_size))


        self.loss = CLOSS(self.margin)

    # TODO compositional network forward
    def forward(self, data):
        # move to gpu
        anchor = {key: data["anchor"][key].to(self.device) for key in data["anchor"].keys() if
                  key in ["input_ids", "attention_mask"]}
        positive = {key: data["positive"][key].to(self.device) for key in data["positive"].keys() if
                    key in ["input_ids", "attention_mask"]}
        negative = {key: data["negative"][key].to(self.device) for key in data["negative"].keys() if
                    key in ["input_ids", "attention_mask"]}
        # encode
        # event-level encoding
        anchor = self.encoder(anchor)
        positive = self.encoder(positive)
        negative = self.encoder(negative)

        # argument composition
        anchor = self.arg_comp_prior(anchor)
        positive = self.arg_comp_posterior(positive)
        negative = self.arg_comp_posterior(negative)

        # event concatenation
        pp_pair = torch.cat((anchor, positive), dim=-1)
        pn_pair = torch.cat((anchor, negative), dim=-1)

        # event composition
        pp_pair = self.event_comp(pp_pair)
        pn_pair = self.event_comp(pn_pair)

        # loss
        loss = self.loss(pp_pair, pn_pair)

        return loss

    def pair_foward(self, data):
        # encode
        anchor = {key: data["anchor"][key].to(self.device) for key in data["anchor"].keys() if
                  key in ["input_ids", "attention_mask"]}
        positive = {key: data["positive"][key].to(self.device) for key in data["positive"].keys() if
                    key in ["input_ids", "attention_mask"]}

        # argument concatenation
        anchor = torch.cat(anchor, dim=-1)
        positive = torch.cat(positive, dim=-1)

        # argument composition
        anchor = self.arg_comp(anchor)
        positive = self.arg_comp(positive)

        # event concatenation
        pp_pair = torch.cat((anchor, positive), dim=-1)

        # event composition
        pp_pair = F.sigmoid(self.event_comp(pp_pair))

        return pp_pair

    # solely caulculate similarity
    def calculate_similarity(self, anchor, positive):
        self.encoder.eval()
        self.arg_comp.eval()
        self.event_comp.eval()

        # tokenize
        anchor = self.encoder.parse_event(anchor)
        positive = self.encoder.parse_event(positive)

        # tokenize
        anchor = [self.encoder.tokenizer(anchor[i]) for i in range(len(anchor))]
        positive = [self.encoder.tokenizer(positive[i]) for i in range(len(positive))]

        return self.pair_foward(anchor, positive)

    # mini-batch calculate similarity
    def super_calculate_similarity(self, search_loader, event_num=5):
        self.encoder.eval()
        self.arg_comp.eval()
        self.event_comp.eval()
        # search loop
        sim_list = []
        for data in tqdm(search_loader, total=len(search_loader)):
            anchor = [{key: data["anchor"][i][key].to(config["DEVICE"]) for key in data["anchor"][i].keys() if
                       key in ["input_ids", "attention_mask"]} for i in range(len(data["anchor"]))]
            positive = [{key: data["positive"][i][key].to(config["DEVICE"]) for key in data["positive"][i].keys() if
                         key in ["input_ids", "attention_mask"]} for i in range(len(data["positive"]))]
            # forward
            sim_list.append(self.pair_foward(anchor, positive).squeeze().cpu().numpy())
        sim_list = np.concatenate(sim_list, axis=0)
        return sim_list



class SBERT(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input):
        return self.mean_pooling(self.encoder(**input), input["attention_mask"])




class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class CLOSS(nn.Module):
    def __init__(self, m=1.0):
        super().__init__()
        self.m = m

    def forward(self, pp_pair, pn_pair):
        # Loss
        basic_loss = F.sigmoid(pp_pair) - F.sigmoid(pn_pair) + self.m
        loss = torch.max(torch.zeros_like(basic_loss), basic_loss).mean()

        return loss
