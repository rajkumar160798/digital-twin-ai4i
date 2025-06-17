def run_scenario(model, sequence, injection_fn=None):
    if injection_fn:
        faulty = injection_fn(sequence.copy())
    else:
        faulty = sequence.copy()

        deviation = compute_deviation(model, sequence, faulty)
        return faulty, deviation
    
    def compute_deviation(model, sequence, faulty):
        """
        Computes the deviation between the original and faulty sequences using the model.
        Replace this stub with the actual implementation.
        """
        # Example stub: return a dummy deviation value
        return 0
