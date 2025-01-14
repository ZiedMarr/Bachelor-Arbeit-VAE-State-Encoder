from stable_baselines3.common.callbacks import BaseCallback
import os

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            save_path = os.path.join(self.save_path, f"ppo_model_{self.n_calls}")
            self.model.save(save_path)
            if self.verbose:
                print(f"Saved model to {save_path}")
        return True
