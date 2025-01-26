from .fit_eval import get_gof

def gof_based_simulator(target_motion, sim_motion, metrics: list,
                        individual_score: float, mean_score: float = 65,
                        max_iterations: int = 100):
    """ #TODO Simulate ground motions based on a level of GOF"""
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        sim_motion.model.simulate(1)
        sim_motion.update()
        gof_scores = get_gof(target_motion, sim_motion, metrics)

        _mean_all = sum(gof_scores.values()) / len(metrics)

        if _mean_all >= mean_score:
            if individual_score:
                if all(gof_scores[metric] >= individual_score for metric in metrics):
                    return gof_scores
            else:
                return gof_scores
    raise RuntimeError("Failed to achieve the desired GoF thresholds."
                       f"Last GoF scores: {gof_scores}")
