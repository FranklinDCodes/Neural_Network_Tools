from neural_network_tools import Network
from random import randint

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

"""
NEURAL NETWORK 0

Structure | 9 - 6 - 6 - 3
Variables | 123 

Network takes a 3x3 square of either 0s or 1s as input
It has been trained 10,000 times to output based on the patter that the 0s and 1s make
    Pattern           | Ouput Neuron Values
    Vertical Stripe   | 1 0 0
    Horizontal Stripe | 0 1 0
    Filled square     | 0 0 1

"""

# initialize network
net = Network("Neural_Network_0", 2, 9, 3, [6, 6], 1, .5)

for i in range(5):

    # generate one of each input type
    sq, ans = generate_input(i)

    # print input
    print('INPUT')
    print(sq[:3])
    print(sq[3:6])
    print(sq[6:9])

    # pass it into network
    net.forwardPass(sq, ans)

    # collect output values
    output = ""
    output_vals = []
    for out in net.outputLayer.neurons:
        output += str(out.value) + " | "
        output_vals.append(out.value)

    # print output
    print("\nOUTPUT")
    print(output[:-2])
    answers = ['vertical', "horizontal", "fill"]
    print(str(round((1 - abs(max(output_vals) - 1))*100, 2)) + f"% {answers[output_vals.index(max(output_vals))]}")
    print()
    print()
