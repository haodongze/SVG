import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedTrans(nn.Module):
    """
        original code is from https://github.com/yuleiniu/rva (CVPR, 2019)
        They used tanh and sigmoid, but we used tanh and LeakyReLU for non-linear transformation function
    """
    def __init__(self, in_dim, out_dim):
        super(GatedTrans, self).__init__()
        self.embed_y = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Tanh()
        )
        self.embed_g = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x_in):
        x_y = self.embed_y(x_in)
        x_g = self.embed_g(x_in)
        x_out = x_y * x_g

        return x_out

class TopicAggregation(nn.Module):
    def __init__(self, hparams):
        super(TopicAggregation, self).__init__()
        self.hparams = hparams

        self.ques_emb = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            )
        )
        self.hist_emb = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        self.topic_gate = nn.Sequential(
            nn.Linear(hparams.word_embedding_size * 2, hparams.word_embedding_size * 2),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, curr_q_word_embed, curr_q_word_encoded, accu_h_word_embed, accu_h_word_encoded,
                            accu_h_not_pad, context_matching_score):
        """
        Attention between ith Question and all history
        """
        bs, sl_q, bilstm = curr_q_word_encoded.size()
        _, num_r, sl_h, _ = accu_h_word_embed.size()
        lstm = self.hparams.lstm_hidden_size
        word_embedding = self.hparams.word_embedding_size

        # non-linear transformation
        curr_q_feat = self.ques_emb(curr_q_word_encoded)
        curr_q_feat = curr_q_feat.unsqueeze(1).repeat(1,num_r,1,1).reshape(bs*num_r,sl_q,lstm)

        accu_h_feat = self.hist_emb(accu_h_word_encoded)
        accu_h_feat = accu_h_feat.reshape(bs*num_r, sl_h, lstm)

        qh_dot_score = torch.bmm(curr_q_feat, accu_h_feat.permute(0, 2, 1))

        accu_h_not_pad = accu_h_not_pad.reshape(bs*num_r,sl_h).unsqueeze(1)
        qh_score = qh_dot_score * accu_h_not_pad
        h_mask = (accu_h_not_pad.float() - 1.0) * 10000.0
        qh_score = self.softmax(qh_score + h_mask)   # bs*num_r sl_q sl_h

        # (bs*num_r sl_q sl_h 1) * (bs*num_r 1 sl_h bilstm)  => sum(dim=2) => bs*num_r sl_q bilstm
        qh_topic_att = qh_score.unsqueeze(-1) * accu_h_word_embed.reshape(bs*num_r,sl_h,word_embedding).unsqueeze(1)
        qh_topic_att = torch.sum(qh_topic_att, dim=2)
        qh_topic_att = qh_topic_att.reshape(bs,num_r,sl_q, word_embedding)

        # attention features
        hist_qatt_embed = torch.sum(context_matching_score.view(bs, num_r, 1, 1) * qh_topic_att, dim=1)

        hist_ques_word_feat = torch.cat((curr_q_word_embed, hist_qatt_embed), dim=-1)
        topic_gate = self.topic_gate(hist_ques_word_feat)  # bs, sl_q, 600
        topic_aware_feat = topic_gate * hist_ques_word_feat  # bs, sl_q, 600

        return topic_aware_feat

class ATT_MODULE(nn.Module):
    """docstring for ATT_MODULE"""

    def __init__(self, hparams):
        super(ATT_MODULE, self).__init__()

        self.V_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.img_feature_size,
                hparams.lstm_hidden_size
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.word_embedding_size *2,
                hparams.lstm_hidden_size
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                1
            )
        )

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, img, ques):
        # input
        # img - shape: (batch_size, num_rounds, num_proposals, img_feature_size)
        # ques - shape: (batch_size, num_rounds, word_embedding_size)
        # output
        # att - shape: (batch_size, num_rounds, num_proposals)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        num_proposals = img.size(2)

        # img_embed = img.view(-1, img.size(-1))  # shape: (batch_size * num_proposals, img_feature_size)
        img_embed = self.V_embed(img)  # shape: (batch_size, num_proposals, lstm_hidden_size)
        # img_embed = img_embed.view(batch_size, num_proposals,
        #                            img_embed.size(-1))  # shape: (batch_size, num_proposals, lstm_hidden_size)
        # img_embed = img_embed.unsqueeze(1).repeat(1, num_rounds, 1,
        #                                           1)  # shape: (batch_size, num_rounds, num_proposals, lstm_hidden_size)

        ques_embed = ques.view(-1, ques.size(-1))  # shape: (batch_size * num_rounds, word_embedding_size)
        ques_embed = self.Q_embed(ques_embed)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.view(batch_size, num_rounds,
                                     ques_embed.size(-1))  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_proposals,
                                                    1)  # shape: (batch_size, num_rounds, num_proposals, lstm_hidden_size)

        att_embed = F.normalize(img_embed * ques_embed, p=2,
                                dim=-1)  # (batch_size, num_rounds, num_proposals, lstm_hidden_size)
        att_embed = self.att(att_embed).squeeze(-1)  # (batch_size, num_rounds, num_proposals)
        att = self.softmax(att_embed)  # shape: (batch_size, num_rounds, num_proposals)

        mf_context = torch.sum(att.unsqueeze(-1) * img, dim=2)  # bs, num_r, 4096

        return mf_context

class Q_ATT(nn.Module):
    """Self attention module of questions."""

    def __init__(self, hparams):
        super(Q_ATT, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques_word, ques_word_encoded, ques_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim)
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2)
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max)
        # output: img_att (batch_size, num_rounds, embed_dim)
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = self.embed(ques_word_encoded)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim) 

        att = self.att(ques_norm).squeeze(-1)  # shape: (batch_size, num_rounds, quen_len_max)
        # ignore <pad> word
        att = self.softmax(att)
        att = att * ques_not_pad  # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True)  # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2)  # shape: (batch_size, num_rounds, rnn_dim)

        return feat, att

class TextAttImage(nn.Module):
    def __init__(self, hparams):
        super(TextAttImage, self).__init__()

        # image
        self.image_emb = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.img_feature_size,
                hparams.lstm_hidden_size
            ),
        )
        self.context_matching_emb = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            )
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(hparams.lstm_hidden_size, 1)
        )

        self.softmax = nn.Softmax(dim=-1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, image, context_matching, img_mask=None):
        # image = image.unsqueeze(1).repeat(1, context_matching.size(1), 1, 1)
        bs, num_r, num_p, mf_topic_feat_size = image.size()

        # fused_feat = torch.cat((mf_topic, context_matching.unsqueeze(2).repeat(1,1,num_p,1)), dim=-1)
        mf_feat = self.image_emb(image)

        cm_feat = self.context_matching_emb(context_matching)
        cm_feat = cm_feat.unsqueeze(2).repeat(1, 1, num_p, 1)  # bs,num_r, num_p, lstm

        att_feat = F.normalize(mf_feat * cm_feat, p=2, dim=-1)
        att_feat = self.att(att_feat).squeeze(-1)  # bs, num_r, np

        if img_mask is not None:
            att_feat = att_feat * img_mask  # 1 or 0
            sf_mask = (img_mask.float() - 1.0) * 10000.0
            att_feat += sf_mask
        att_feat = self.softmax(att_feat)  # bs, num_r, np

        mf_context = torch.sum(att_feat.unsqueeze(-1) * image, dim=2)  # bs, num_r, 4096

        return mf_context

class Q_ATT_H(nn.Module):
    def __init__(self, hparams):
        super(Q_ATT_H, self).__init__()
        self.hparams = hparams

        # non-linear transformation
        self.ques_emb = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            )
        )

        self.hist_emb = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            )
        )

        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(hparams.lstm_hidden_size, 1),
        )

        self.softmax = nn.Softmax(dim=-1)

        self.context_gate = nn.Sequential(
            nn.Linear((hparams.lstm_hidden_size) * 2,
                      (hparams.lstm_hidden_size) * 2),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, curr_q_sent, accu_h_sent):
        """
		Context Matching between t-th question and dialog history
		"""
        bs, nr, bilstm = accu_h_sent.size()  # hist

        curr_q_feat = self.ques_emb(curr_q_sent).repeat(1, nr, 1)
        accu_h_feat = self.hist_emb(accu_h_sent)

        att_score = self.att(curr_q_feat * accu_h_feat).squeeze(-1)  # element_wise multiplication -> attention
        att_score = self.softmax(att_score)
        hist_qatt_feat = (accu_h_sent * att_score.unsqueeze(-1)).sum(1,
                                                                     keepdim=True)  # weighted sum : question-relevant dialog history

        return hist_qatt_feat, att_score