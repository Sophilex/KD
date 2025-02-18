import torch
def get_user_item_ratings(batch_user, user_item):
        # user_items: 用户要计算的item
        users, items = torch.tensor([
    [10, 20, 15, 30, 25],
    [5, 10, 15, 20, 25],
    [30, 25, 20, 15, 10]
]),torch.tensor([
    [10, 20, 15, 30, 25],
    [5, 10, 15, 20, 25],
    [30, 25, 20, 15, 10]
])
        users = users[batch_user] # num_user X dim
        
        # score = []
        # for i in range(users.shape(0)):
        #     score = []
        #     for j in range(user_item.shape(1)):
        #         ans.append(torch.matmul(users[i],items[user_item[i][j]]))
        #     score.append(ans)

        score = torch.matmul(users, items[user_item].transpose(1, 2))
        return score
batch_user = torch.tensor([0,1])
ui = torch.tensor([
    [0,2],
    [1,2]
])
ans = get_user_item_ratings(batch_user, ui)
print(ans)

# up = torch.tensor([[0, 2, 3], [0, 1, 2], [0, 3, 4]])

# inter_mat[[0,1,2],up] = 0
# print(inter_mat)


# inter_mat[bt, :] += up**2

# print(inter_mat)

