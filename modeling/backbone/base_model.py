import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_layer import Embedding


class BaseRec(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        
        self.dataset = dataset
        self.args = args

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.user_list = torch.LongTensor([i for i in range(self.num_users)]).cuda()
        self.item_list = torch.LongTensor([i for i in range(self.num_items)]).cuda()
    
    def forward(self):
        raise NotImplementedError
    
    def forward_multi_items(self, batch_user, batch_items):
        """forward when we have multiple items for a user

        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_items : 2-D LongTensor (batch_size x k)

        Returns
        -------
        score : 2-D FloatTensor (batch_size x k)
        """
        raise NotImplementedError
    
    def get_all_ratings(self):
        raise NotImplementedError

    def get_ratings(self, batch_user):
        raise NotImplementedError
    
    def get_all_embedding(self):
        raise NotImplementedError
    
    def get_user_item_ratings(self, batch_user, item_idx):
        raise NotImplementedError
    
    def get_loss(self, output):
        """Compute the loss function with the model output

        Parameters
        ----------
        output : 
            model output (results of forward function)

        Returns
        -------
        loss : float
        """
        pos_score, neg_score = output[0], output[1]
        pos_score = pos_score.expand_as(neg_score)  # batch_size, num_ns
        loss = -F.logsigmoid(pos_score - neg_score).mean(1).sum()
        return loss


class BaseGCN(BaseRec):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
    
    def computer(self):
        """
        propagate methods
        """
        raise NotImplementedError

    def get_all_pre_embedding(self):
        """get total embedding of users and items before convolution

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        raise NotImplementedError
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_pos_item : 1-D LongTensor (batch_size)
        batch_neg_item : 2-D LongTensor (batch_size, num_ns)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        all_users, all_items = self.computer()

        u = all_users[batch_user]
        i = all_items[batch_pos_item]
        j = all_items[batch_neg_item]
        
        pos_score = (u * i).sum(dim=1, keepdim=True)    # batch_size, 1
        neg_score = torch.bmm(j, u.unsqueeze(-1)).squeeze(-1)       # batch_size, num_ns

        return pos_score, neg_score
    
    def get_user_embedding(self, batch_user):
        all_users, all_items = self.computer()
        users = all_users[batch_user]
        return users
    
    def get_item_embedding(self, batch_item):
        all_users, all_items = self.computer()
        items = all_items[batch_item]
        return items

    def get_all_post_embedding(self):
        """get total embedding of users and items after convolution

        Returns
        -------
        users : 2-D FloatTensor (num. users x dim)
        items : 2-D FloatTensor (num. items x dim)
        """
        all_users, all_items = self.computer()
        return all_users, all_items

    def get_all_embedding(self):
        return self.get_all_post_embedding()
    
    def forward_multi_items(self, batch_user, batch_items):
        """forward when we have multiple items for a user

        Parameters
        ----------
        batch_user : 1-D LongTensor (batch_size)
        batch_items : 2-D LongTensor (batch_size x k)

        Returns
        -------
        score : 2-D FloatTensor (batch_size x k)
        """
        all_users, all_items = self.computer()
        
        u = all_users[batch_user]		# batch_size x dim
        i = all_items[batch_items]		# batch_size x k x dim
        
        score = torch.bmm(i, u.unsqueeze(-1)).squeeze(-1)   # batch_size, k
        
        return score
    
    def get_all_ratings(self):
        users, items = self.get_all_post_embedding()
        score_mat = torch.matmul(users, items.T)
        return score_mat

    def get_ratings(self, batch_user):
        users, items = self.get_all_post_embedding()
        users = users[batch_user]
        score_mat = torch.matmul(users, items.T)
        return score_mat
    
    def get_user_item_ratings(self, batch_user, item_idx):
        # user_items: 用户要计算的item num_batch_user X num_batch_item
        users, items = self.get_all_post_embedding()
        users = users[batch_user] # num_user X dim

        # score = []
        # for i in range(users.shape(0)):
        #     score = []
        #     for j in range(user_item.shape(1)):
        #         ans.append(torch.matmul(users[i],items[user_item[i][j]]))
        #     score.append(ans)

        score = torch.matmul(users.unsqueeze(1), items[item_idx].transpose(1, 2)) # num_batch_user X num_batch_item
        return score.squeeze(1)

        


class BaseCTR(nn.Module):
    def __init__(self, args, feature_stastic):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.L2_weight = args.L2
        self.num_fields = len(feature_stastic) - 1
        self.embedding_layer = Embedding(self.embedding_dim, feature_stastic)
        self.embedding_layer_dim = self.embedding_dim * self.num_fields
    
    def forward_embed(self, sparse_input, dense_input=None):
        """Return the output of the embedding layer
        """
        dense_input = self.embedding_layer(sparse_input)
        return dense_input

    def forward(self, sparse_input, dense_input=None):
        dense_input = self.embedding_layer(sparse_input)
        logits = self.FeatureInteraction(dense_input, sparse_input)
        return logits
    
    def FeatureInteraction(self, dense_input, sparse_input):
        raise NotImplementedError

    def L2_Loss(self, weight):
        if weight == 0:
            return 0
        loss = 0
        for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            loss += torch.norm(param, p=2)
        return loss * weight
    
    def get_loss(self, logits, label):
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), label.squeeze(-1).float()) + self.L2_Loss(self.L2_weight)
        return loss
