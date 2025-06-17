import torch
import numpy as np

def compute_deviation(model, input_seq, real_seq):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(input_seq).unsqueeze(0).float())[0].numpy()
    diff = np.abs(pred - real_seq)
    return np.mean(diff, axis=1)  # per timestep deviation
def compute_deviation_score(model, input_seq, real_seq):
    """
    Compute the deviation score between the model's prediction and the real sequence.
    Args:
        model: The trained model to use for prediction.
        input_seq (np.ndarray): The input sequence for the model.
        real_seq (np.ndarray): The real sequence to compare against.
    Returns:
        float: The deviation score.
    """
    deviation = compute_deviation(model, input_seq, real_seq)
    score = np.mean(deviation)  # average deviation score
    return score