class MRRD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "mrrd"
        self.K = args.mrrd_K
        self.L = args.mrrd_L
        self.T = args.mrrd_T
        self.mxK = args.mrrd_mxK
        self.no_sort = args.no_sort
        self.beta = args.mrrd_beta      # weight of rest of topk predictions
        self.loss_type = args.loss_type
        self.sample_rank = args.sample_rank
        self.tau = args.mrrd_tau
        self.gamma = args.mrrd_gamma    # weight of uninteresting predictions
        # self.test_generalization = args.mrrd_test_type

        # For interesting item
        # if self.loss_type in ["ce", "listnet"]:
        #     self.mxK = self.K
        # if self.test_generalization == 1 or self.test_generalization == 2:
        #     self.topk_scores, self.topk_dict = self.get_topk_dict(self.mxK)
        #     if self.test_generalization == 1:
        #         f_test_topk_dict = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_dict_train.pkl")
        #         f_test_topk_score = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_score_train.pkl")
        #     else:
        #         f_test_topk_dict = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_dict_test.pkl")
        #         f_test_topk_score = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_score_test.pkl")
        #     if not os.path.exists(f_test_topk_dict) or not os.path.exists(f_test_topk_score):
        #         self.test_topk_dict = {}
        #         self.test_topk_score = {}
        #         train_dict = self.dataset.train_dict
        #         for u in range(self.num_users):
        #             if self.test_generalization == 1:
        #                 if u not in train_dict:
        #                     continue
        #                 test_topk_dict = train_dict[u][:100].long().cuda()
        #             else:
        #                 if u not in valid_dict or u not in test_dict:
        #                     continue
        #                 test_topk_dict = torch.concat([valid_dict[u], test_dict[u]]).long().cuda()

        #             test_topk_score = self.teacher.forward_multi_items(torch.tensor([u]).long().cuda(), test_topk_dict.unsqueeze(0))[0]
        #             idx = torch.argsort(test_topk_score, descending=True)
        #             self.test_topk_dict[u] = test_topk_dict[idx]
        #             self.test_topk_score[u] = test_topk_score[idx]
        #         dump_pkls((self.test_topk_dict, f_test_topk_dict), (self.test_topk_score, f_test_topk_score))
        #     else:
        #         _, self.test_topk_dict, self.test_topk_score = load_pkls(f_test_topk_dict, f_test_topk_score)
        # elif self.test_generalization == 3 or self.test_generalization == 5:
        #     self.test_K = 100
        #     topk_scores, topk_dict = self.get_topk_dict(self.mxK + self.test_K)
        #     f_train_idx = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"train_idx_{self.mxK}_{self.test_K}.npy")
        #     f_test_idx = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_idx_{self.mxK}_{self.test_K}.npy")
        #     if not os.path.exists(f_train_idx) or os.path.exists(f_test_idx):
        #         train_idx = torch.zeros(self.num_users, self.mxK).long()
        #         test_idx = torch.zeros(self.num_users, self.test_K).long()
        #         for u in range(self.num_users):
        #             tr_idx, te_idx = torch.utils.data.random_split(torch.arange(self.mxK + self.test_K), [self.mxK, self.test_K])
        #             train_idx[u], test_idx[u] = torch.tensor(tr_idx).sort()[0].long(), torch.tensor(te_idx).sort()[0].long()
        #         os.makedirs(os.path.dirname(f_train_idx), exist_ok=True)
        #         os.makedirs(os.path.dirname(f_test_idx), exist_ok=True)
        #         np.save(f_train_idx, train_idx.cpu().numpy())
        #         np.save(f_test_idx, test_idx.cpu().numpy())
        #     else:
        #         train_idx = torch.from_numpy(np.load(f_train_idx)).long()
        #         test_idx = torch.from_numpy(np.load(f_test_idx)).long()
        #     self.topk_scores, self.topk_dict = torch.zeros(self.num_users, self.mxK).cuda(), torch.zeros(self.num_users, self.mxK).long().cuda()
        #     self.test_topk_score, self.test_topk_dict = torch.zeros(self.num_users, self.test_K).cuda(), torch.zeros(self.num_users, self.test_K).long().cuda()
        #     for u in range(self.num_users):
        #         self.topk_scores[u], self.topk_dict[u] = topk_scores[u][train_idx[u]], topk_dict[u][train_idx[u]]
        #         self.test_topk_score[u], self.test_topk_dict[u] = topk_scores[u][test_idx[u]], topk_dict[u][test_idx[u]]
        # elif self.test_generalization == 4:
        #     self.test_K = 100
        #     self.topk_scores, self.topk_dict = self.get_topk_dict(self.mxK)
        #     f_test_topk_dict = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_dict_{self.mxK}_{self.test_K}.pkl")
        #     f_test_topk_score = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_score_{self.mxK}_{self.test_K}.pkl")
        #     if os.path.exists(f_test_topk_dict) and os.path.exists(f_test_topk_score):
        #         _, self.test_topk_dict, self.test_topk_score = load_pkls(f_test_topk_dict, f_test_topk_score)
        #     else:
        #         self.test_topk_dict = torch.zeros(self.num_users, self.test_K).long().cuda()
        #         self.test_topk_score = torch.zeros(self.num_users, self.test_K).long().cuda()
        #         indices = torch.multinomial(torch.ones_like(self.topk_scores), self.test_K, replacement=False).sort(-1)[0]
        #         for u in range(self.num_users):
        #             self.test_topk_dict[u] = self.topk_dict[u][indices[u]]
        #             self.test_topk_score[u] = self.topk_scores[u][indices[u]]
        #         dump_pkls((self.test_topk_dict, f_test_topk_dict), (self.test_topk_score, f_test_topk_score))
        # else:
        self.topk_scores, self.topk_dict = self.get_topk_dict(self.mxK)

        if self.sample_rank:
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)
        else:
            self.ranking_mat = torch.exp(self.topk_scores / self.tau)

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0
        self.mask.requires_grad = False

    def get_topk_dict(self, mxK):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            topk_scores, topk_dict = torch.topk(inter_mat, mxK, dim=-1)
        return topk_scores.cuda(), topk_dict.cuda()
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        return interesting_samples, uninteresting_samples

    @torch.no_grad()
    def generalization_error(self):
        gen_errors = []
        user_list = torch.arange(self.num_users).cuda()
        for _ in range(5):
            errs = []
            if self.test_generalization == 1 or self.test_generalization == 2:
                for u in range(self.num_users):
                    if u not in self.test_topk_dict:
                        continue
                    if self.sample_rank:
                        ranking_list = torch.exp(-(torch.arange(len(self.test_topk_dict[u])) + 1) / self.T)
                    else:
                        ranking_list = torch.exp(self.test_topk_score[u] / self.tau)
                    samples = torch.multinomial(ranking_list, len(self.test_topk_dict[u]), replacement=False)
                    interesting_items_u = self.test_topk_dict[u][samples]
                    S1 = self.student.forward_multi_items(torch.tensor([u]).long().cuda(), interesting_items_u.unsqueeze(0))
                    above = S1.sum(-1)
                    below = S1.flip(-1).exp().cumsum(-1).log().sum(-1)
                    loss = -(above - below)
                    errs.append(loss)
            elif self.test_generalization == 3 or self.test_generalization == 4:
                if self.sample_rank:
                    ranking_list = torch.exp(-(torch.arange(self.test_K) + 1) / self.T)
                    ranking_mat = ranking_list.repeat(self.num_users, 1)
                else:
                    ranking_mat = torch.exp(self.test_topk_score / self.tau)
                samples = torch.multinomial(ranking_mat, self.test_K, replacement=False)
                interesting_items = torch.zeros((self.num_users, self.test_K)).long().cuda()
                for u in range(self.num_users):
                    interesting_items[u] = self.test_topk_dict[u][samples[u]]
                bs = self.args.batch_size
                for i in range(math.ceil(self.num_users / bs)):
                    batch_user = user_list[bs * i: bs * (i + 1)]
                    interesting_items_u = torch.index_select(interesting_items, 0, batch_user)
                    S1 = self.student.forward_multi_items(batch_user, interesting_items_u)
                    above = S1.sum(-1)
                    below = S1.flip(-1).exp().cumsum(-1).log().sum(-1)
                    loss = -(above - below)
                    errs.append(loss)
            loss = torch.concat(errs).mean().item()
            gen_errors.append(loss)
        err =  sum(gen_errors) / len(gen_errors)
        return err
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        output = self.student(batch_user, batch_pos_item, batch_neg_item)
        base_loss = self.student.get_loss(output)
        kd_loss = self.get_loss(batch_user, batch_pos_item, batch_neg_item)
        if self.test_generalization > 0:
            loss = self.lmbda * kd_loss
        else:
            loss = base_loss + self.lmbda * kd_loss
        return loss, base_loss.detach(), kd_loss.detach()
    
    @torch.no_grad()
    def plot_statistics(self, epoch):
        ce_errs, sigma_errs = [], []
        user_list = torch.arange(self.num_users).cuda()
        for _ in range(5):
            ce_err, sigma_err = [], []
            bs = self.args.batch_size
            for i in range(math.ceil(self.num_users / bs)):
                batch_user = user_list[bs * i: bs * (i + 1)]
                K = self.test_K // 2
                randidx = torch.randperm(self.test_K)[:K]
                test_items = self.test_topk_dict[batch_user][:, randidx]
                T = self.test_topk_score[batch_user][:, randidx]
                S = self.student.forward_multi_items(batch_user, test_items)
                expT = torch.exp(T / self.tau)  # bs, K
                prob_T1 = expT.unsqueeze(-1) / torch.sum(expT, dim=-1, keepdim=True).unsqueeze(-1)    # bs, K, 1
                Z_T2 = expT.sum(-1, keepdim=True).unsqueeze(1).repeat(1, K, K)   # bs, K, K
                Z_T2 = Z_T2 - expT.unsqueeze(-1)
                # make diag of prob_T2 0
                prob_T2 = expT.unsqueeze(1) / Z_T2 # bs, K, K
                prob_T2 -= torch.diag_embed(torch.diagonal(prob_T2, dim1=1, dim2=2), dim1=1, dim2=2)
                prob_T = prob_T1 * prob_T2  # bs, K, K
                expS = torch.exp(S / self.tau)
                log_prob_S1 = torch.log(expS.unsqueeze(-1) / torch.sum(expS, dim=-1, keepdim=True).unsqueeze(-1)) # bs, K, 1
                Z_S2 = expS.sum(-1, keepdim=True).unsqueeze(1).repeat(1, K, K)   # bs, K, K
                Z_S2 = Z_S2 - expS.unsqueeze(-1)
                Z_S2 = torch.maximum(Z_S2, torch.tensor(1e-4))
                log_prob_S2 = torch.log(expS.unsqueeze(1) / Z_S2)  # bs, K, K
                log_prob_S = log_prob_S1 + log_prob_S2  # bs, K, K
                loss_all = -(prob_T * log_prob_S).sum(-1).sum(-1)   # bs

                prob_T = torch.softmax(T / self.tau, dim=-1)
                loss_ce = F.cross_entropy(S / self.tau, prob_T, reduction='none') # bs
                ce_err.append(loss_ce)
                sigma_err.append(loss_all - loss_ce)
            loss_ce = torch.concat(ce_err).mean().item()
            loss_sigma = torch.cat(sigma_err).mean().item()
            ce_errs.append(loss_ce)
            sigma_errs.append(loss_sigma)
        ce_errs, sigma_errs = np.array(ce_errs), np.array(sigma_errs)
        mlflow.log_metrics({"sigma_expectation_pow2":np.power(sigma_errs.mean(), 2), "ce_expectation_pow2":np.power(ce_errs.mean(), 2), "sigma_variance":np.var(sigma_errs, ddof=1), "cov_sigma_ce":np.cov(sigma_errs, ce_errs, ddof=1)[0, 1]}, step=epoch // 5)

    def do_something_in_each_epoch(self, epoch):
        # if 1 <= self.test_generalization <= 4:
        #     if epoch % 5 == 0:
        #         err = self.generalization_error()
        #         mlflow.log_metric("gen_error", err, step=epoch // 5)
        # elif self.test_generalization >=5:
        #     if epoch % 5 == 0:
        #         self.plot_statistics(epoch)
        
        with torch.no_grad():
            if self.loss_type == "rrd":
                # interesting items
                self.interesting_items = torch.zeros((self.num_users, self.K))

                # sampling
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False)
                
                if not self.no_sort:
                    samples = samples.sort(dim=1)[0]
                
                for user in range(self.num_users):
                    self.interesting_items[user] = self.topk_dict[user][samples[user]]

                self.interesting_items = self.interesting_items.cuda()

            # uninteresting items
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)

    def rrd_all_loss(self, S1, S2, Stop):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))
        Stop = torch.minimum(Stop, torch.tensor(80., device=Stop.device))
        above = S1.sum(-1)
        below1 = S1.flip(-1).exp().cumsum(-1)    # exp() of interesting_prediction results in inf
        below3 = Stop.exp().sum(-1, keepdims=True) - S1.exp().sum(-1, keepdims=True)
        below3 = torch.maximum(below3, torch.tensor(0., device=below3.device))
        below2 = S2.exp().sum(-1, keepdims=True)
        below = (below1 + self.gamma * below2 + self.beta * below3).log().sum(-1)
        loss = -(above - below).sum()
        return loss
    
    def neg_loss(self, logit_S_itemT, logit_S_uninteresting):
        above = torch.log(logit_S_itemT.exp().sum(-1))
        below = torch.log(logit_S_itemT.exp().sum(-1) + torch.exp(logit_S_uninteresting).sum(-1))
        loss = -(above - below).sum()
        return loss
    
    def ce_loss(self, S, T):
        if self.sample_rank:
            ranking_list = -(torch.arange(self.mxK) + 1) / self.T
            ranking_mat = ranking_list.repeat(len(T), 1).cuda()
            prob_T = torch.softmax(ranking_mat, dim=-1)     # bs, mxK
        else:
            prob_T = torch.softmax(T / self.tau, dim=-1)
        loss = F.cross_entropy(S / self.tau, prob_T, reduction='sum')
        return loss

    def list2_loss(self, S, T):
        S = torch.minimum(S, torch.tensor(60., device=S.device))
        if self.sample_rank:
            ranking_list = -(torch.arange(self.mxK) + 1) / self.T
            ranking_mat = ranking_list.repeat(len(T), 1).cuda()
            expT = torch.exp(ranking_mat)   # bs, mxK
        else:
            expT = torch.exp(T / self.tau)  # bs, mxK
        prob_T1 = expT.unsqueeze(-1) / torch.sum(expT, dim=-1, keepdim=True).unsqueeze(-1)    # bs, mxK, 1
        Z_T2 = expT.sum(-1, keepdim=True).unsqueeze(1).repeat(1, self.mxK, self.mxK)   # bs, mxK, mxK
        Z_T2 = Z_T2 - expT.unsqueeze(-1)
        prob_T2 = expT.unsqueeze(1) / Z_T2 # bs, mxK, mxK
        # make diag of prob_T2 0
        prob_T2 -= torch.diag_embed(torch.diagonal(prob_T2, dim1=1, dim2=2), dim1=1, dim2=2)
        prob_T = prob_T1 * prob_T2  # bs, mxK, mxK
        expS = torch.exp(S / self.tau)
        log_prob_S1 = torch.log(expS.unsqueeze(-1) / torch.sum(expS, dim=-1, keepdim=True).unsqueeze(-1)) # bs, mxK, 1
        Z_S2 = expS.sum(-1, keepdim=True).unsqueeze(1).repeat(1, self.mxK, self.mxK)   # bs, mxK, mxK
        Z_S2 = Z_S2 - expS.unsqueeze(-1)
        Z_S2 = torch.maximum(Z_S2, torch.tensor(1e-4))
        log_prob_S2 = torch.log(expS.unsqueeze(1) / Z_S2)  # bs, mxK, mxK
        log_prob_S = log_prob_S1 + log_prob_S2  # bs, mxK, mxK
        loss = -(prob_T * log_prob_S).sum()
        return loss
    
    def ce_all_loss(self, S, T, S2):
        if self.loss_type == "listnet":
            loss = self.list2_loss(S, T)
        else:
            loss = self.ce_loss(S, T)
        if self.gamma > 0:
            loss += self.gamma * self.neg_loss(S, S2)
        return loss
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        # if self.loss_type in ["ce", "listnet"]:
        #     uninteresting_items = torch.index_select(self.uninteresting_items, 0, users).type(torch.LongTensor).cuda()
        #     uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        #     topk_prediction_S = self.student.forward_multi_items(users, self.topk_dict[users])
        #     topk_prediction_T = self.topk_scores[users]
        #     loss = self.ce_all_loss(topk_prediction_S, topk_prediction_T, uninteresting_prediction)
        # else:
        interesting_items, uninteresting_items = self.get_samples(users)
        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        topk_prediction = self.student.forward_multi_items(users, self.topk_dict[users])

        loss = self.rrd_all_loss(interesting_prediction, uninteresting_prediction, topk_prediction)
        return loss
