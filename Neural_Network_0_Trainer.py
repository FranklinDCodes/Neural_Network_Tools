from neural_network_tools import Network
from random import randint
from statistics import median
from datetime import datetime

# generate input for network
def generate_input(type = None):

    input = [0, 0, 0,
             0, 0, 0,
             0, 0, 0]

    ans = [0, 0, 0]

    col_list = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    row_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # left, right, top, bottom, or all
    if type is None:
        side = randint(0, 4)
    else:
        side = type

    if side == 0:
        col_num = randint(1, 2)
        columns = col_list[:col_num]
        for col in columns:
            for ind in col:
                input[ind] = 1
        ans[0] = 1
    elif side == 1:
        col_num = randint(1, 2)
        columns = col_list[-col_num:]
        for col in columns:
            for ind in col:
                input[ind] = 1
        ans[0] = 1
    elif side == 2:
        row_num = randint(1, 2)
        rows = row_list[:row_num]
        for row in rows:
            for ind in row:
                input[ind] = 1
        ans[1] = 1
    elif side == 3:
        row_num = randint(1, 2)
        rows = row_list[-row_num:]
        for row in rows:
            for ind in row:
                input[ind] = 1
        ans[1] = 1
    else:
        fill = randint(0, 1)
        for i in range(len(input)):
            input[i] = fill
        ans[2] = 1
    
    return input, ans

# calculate median loss of network
def get_median_loss(iters, network):
    startInd = 0
    # ind == num, (add, count)
    nums = []
    for i in range(iters):
        inp, ans = generate_input()
        loss = network.forwardPass(inp, ans)
        nums.append(loss)
    return median(nums)

# init network
net = Network("Neural_Network_0", 2, 9, 3, [6, 6], 1, .5)

# caluculate loss before training
stLoss = get_median_loss(100, net)


# start times
start = datetime.now()
batchTime = datetime.now()

# open training loop
for i in range(10000):

    # get example
    pattern, solution = generate_input()

    # train
    net.backPass(pattern, solution, 0.01)

    # print average time every hundred
    if not i % 100 and i != 0:
        print(i)
        print(f'Average time for group : {(datetime.now() - batchTime)/100}')
        batchTime = datetime.now()

    # backup database every thousand
    if not i % 1000 and i != 0:
        net.databaseBackup()

# loss comparison
loss = get_median_loss(100, net)
print(f'Start loss : {stLoss}')
print(f'current loss : {loss}')

print(f"Process finished after {datetime.now() - start}")