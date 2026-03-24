w1 = 1
w2 = -1
threshold = 1

def mp_neuron(A, B):
    #Return 1 if (w1*A + w2*B) >= threshold, else 0.
    net_input = (w1 * A) + (w2 * B)
    return 1 if net_input >= threshold else 0

def main():
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    print("A B | Output")
    print("--------------")
    for A, B in inputs:
        output = mp_neuron(A, B)
        print(f"{A} {B} |   {output}")

if __name__ == "__main__":
    main()
