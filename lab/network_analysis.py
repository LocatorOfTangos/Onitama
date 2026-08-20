import torch
import random
import tqdm
from sys import argv
from onitama.opponents.NN.architecture.ValueNet1 import ValueNet1
from onitama.opponents.NN.VN1_train import encode_state_for_value_net
from onitama.Boardstate import Boardstate
from onitama.Deck import Deck

def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = ValueNet1().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # JIT-trace on the target device, same as production workers
    try:
        dummy_b = torch.empty(1, 4, 5, 5, device=device)
        dummy_c = torch.empty(1, 3, 16,  device=device)
        model = torch.jit.trace(model, (dummy_b, dummy_c))
        model.eval()
    except Exception as e:
        print(f"  JIT trace failed ({e}); using eager.")
    return model

def test_condition(gen_boardstate, model, test, iter=10000):
    mean_value = 0.0
    stdev_value = 0.0
    sum_value = 0.0
    sum_sq_value = 0.0
    for i in range(iter):
        state = gen_boardstate()
        bt, ct = encode_state_for_value_net(state)
        with torch.no_grad():
            value = model(bt[None, :, :, :], ct[None, :, :]).item()
            sum_value += value
            sum_sq_value += value * value
    
    mean_value = sum_value / iter
    stdev_value = ((sum_sq_value / iter) - (mean_value ** 2)) ** 0.5
    print(f"{test}\nMean value: {1000*mean_value:.1f}, Standard deviation: {1000*stdev_value:.1f}")

def main(model):
    def gen1():
        return Boardstate()
    test_condition(gen1, model, "Initial condition")
    
    def gen2():
        return Boardstate(red_start=True, red_students=[None,(1,0),(3,0),(4,0)])
    test_condition(gen2, model, "Down corner student")
    
    def gen3():
        return Boardstate(red_start=True, red_students=[(0,0),None,(3,0),(4,0)])
    test_condition(gen3, model, "Down center student")
    
    def gen4():
        idx = random.randint(0, 3)
        students = [(0,0),(1,0),(3,0),(4,0)]
        students[idx] = None
        return Boardstate(red_start=True, red_students=students)
    test_condition(gen4, model, "Down 1 student")
    
    def gen5():
        idx1, idx2 = random.sample(range(0,4),2)
        students = [(0,0),(1,0),(3,0),(4,0)]
        students[idx1] = None
        students[idx2] = None
        return Boardstate(red_start=True, red_students=students)
    test_condition(gen5, model, "Down 2 students")
    
    def gen6():
        idx1, idx2, idx3 = random.sample(range(0,4),3)
        students = [(0,0),(1,0),(3,0),(4,0)]
        students[idx1] = None
        students[idx2] = None
        students[idx3] = None
        return Boardstate(red_start=True, red_students=students)
    test_condition(gen6, model, "Down 3 students")

    def gen7():
        return Boardstate(red_start=True, red_students=[None,None,None,None])
    test_condition(gen7, model, "Down 4 students")

def main2(model):
    card_sum = {card : 0 for card in Deck.DECK}
    card_count = {card : 0 for card in Deck.DECK}
    for _ in tqdm.tqdm(range(1000000)):
        state = Boardstate(red_start=True)
        cards = state.red_cards
        bt, ct = encode_state_for_value_net(state)
        with torch.no_grad():
            value = model(bt[None, :, :, :], ct[None, :, :]).item()
        for card in cards:
            card_sum[card] += value
            card_count[card] += 1

    total_avg = sum(card_sum.values()) / sum(card_count.values())
    print("Card value analysis:")
    print(f"Overall average value: {1000*total_avg:.1f}")
    for card in Deck.DECK:
        if card_count[card] > 0:
            avg_value = card_sum[card] / card_count[card]
            print(f"{card.name}: Average delta {1000*(avg_value - total_avg):.1f} over {card_count[card]} samples")
        else:
            print(f"{card.name}: No samples")

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python network_analysis.py <ckpt_path>")
        exit(1)
    ckpt_path = argv[1]
    device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, device)
    #main(model = model)
    main2(model = model)


